"""Measurements of the released GTEx CSV, not a training selector.

The caller supplies already loaded tokenizers and, optionally, chromosome sizes.
Counting performs no model, network, cache, filtering, or output operations.
Separate readers/writers retain completed JSON audits without replacing evidence.
Missing sequences and hypothetical length policies remain separate observations;
this audit cannot recover which rows the authors actually trained on.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from .isoformer_data_checks import (
    DATA_CHECKS_VERSION, GTEX_TISSUES, GTExDataChecks, validate_data_checks,
)


RNA_LENGTH_LIMITS = (12000, 12282, 12288)
PROTEIN_LENGTH_LIMITS = (1024, 1200, 2047, 2048)
TOKEN_LIMITS = (1024, 2048)
CHECK_EVERY = 97
METHOD_VERSION = "isoformer-cohort-v1-exact-markers"
_IDENTITY_FIELDS = (
    "source_sha256", "source_bytes", "dataset_revision", "model_revision",
    "vocabulary_sha256", "chromosome_sizes_sha256", "method_version", "transformers",
)
_PRODUCER_FIELDS = frozenset({
    "started_utc", "finished_utc", "python", "host", "implementation_sha256", "source_url",
    "data_checks_implementation_sha256",
})
_RNA_RUNS = re.compile(r"[ACGT]+|N")
_RNA_CANONICAL = re.compile(r"[ACGTN]*")
_REQUIRED = ("transcript_id_gtex", "split", "RNA", "Protein")


def is_missing_sequence(sequence):
    """Recognize exact lowercase missing markers; uppercase sequences stay intact."""
    return sequence is None or sequence in {"", "nan", "none", "null"}


def rna_fast_token_count(sequence):
    """Count released NT 6-mers, single-base remainders/N, and one CLS token.

    Unsupported characters return None so the real tokenizer decides their meaning.
    No uppercase conversion, whitespace removal, U-to-T conversion, or truncation
    is performed. An empty string has one token; a missing CSV field is not passed
    here by the audit and is reported separately.
    """
    if _RNA_CANONICAL.fullmatch(sequence) is None:
        return None
    return 1 + sum(
        1 if run.group() == "N" else len(run.group()) // 6 + len(run.group()) % 6
        for run in _RNA_RUNS.finditer(sequence)
    )


def protein_fast_token_count(sequence, alphabet):
    """Count one vocabulary token per character, plus released CLS and EOS."""
    return len(sequence) + 2 if set(sequence).issubset(alphabet) else None


def _actual_count(tokenizer, sequence):
    result = tokenizer(sequence, add_special_tokens=True, truncation=False, padding=False)
    ids = result["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        raise ValueError("Tokenizers must return one unbatched input_ids sequence")
    return len(ids)


def _protein_alphabet(tokenizer):
    return frozenset(token for token in tokenizer.get_vocab() if len(token) == 1)


def validate_tokenizers(rna_tokenizer, protein_tokenizer):
    """Check fast-count assumptions against supplied tokenizers on invented strings.

    This checks the counting contract, not pretrained weights or biological validity.
    The audit calls it before reading any sequence rows; callers can also run it alone.
    """
    lengths = (0, 1, 5, 6, 7, 11, 12, 13, 1023, 1024, 2046, 2047, 2048,
               12000, 12281, 12282, 12283, 12288, 12289)
    rna_examples = ["A" * n for n in lengths]
    rna_examples += ["N", "NN", "AAAAANAAAAAA", "NACGTACNACGTN", "ACGT" * 9]
    alphabet = _protein_alphabet(protein_tokenizer)
    protein_examples = ["M" * n for n in (0, 1, 1022, 1023, 1024, 1199, 1200,
                                         2045, 2046, 2047, 2048, 2049)]
    protein_examples += sorted(alphabet)
    checked = {"RNA": 0, "Protein": 0}
    for label, tokenizer, examples in (
        ("RNA", rna_tokenizer, rna_examples),
        ("Protein", protein_tokenizer, protein_examples),
    ):
        for sequence in examples:
            fast = (rna_fast_token_count(sequence) if label == "RNA"
                    else protein_fast_token_count(sequence, alphabet))
            actual = _actual_count(tokenizer, sequence)
            if fast is None or fast != actual:
                raise ValueError(f"{label} tokenizer count contract failed for a synthetic "
                                 f"string of length {len(sequence)}: fast={fast}, actual={actual}")
            checked[label] += 1
    return checked


def _new_scope():
    return {"rows": 0, "counts": {}, "stats": {}, "_ids": set()}


def _increment(scope, key, amount=1):
    scope["counts"][key] = scope["counts"].get(key, 0) + amount


def _measure(scope, key, value, limits):
    stats = scope["stats"].setdefault(key, {"n": 0, "min": None, "max": None, "sum": 0})
    stats["n"] += 1
    stats["sum"] += value
    stats["min"] = value if stats["min"] is None else min(stats["min"], value)
    stats["max"] = value if stats["max"] is None else max(stats["max"], value)
    for limit in limits:
        _increment(scope, f"{key}_gt_{limit}", int(value > limit))


def _boundary_exclusion(row, chromosome_sizes):
    """Apply the example's sequential 196,608-nt boundary checks, without DNA reads.

    The extra filter checks [anchor - 196608, anchor), not a symmetric second
    window. A high boundary equal to chromosome size is excluded by the release.
    Reasons are disjoint: chrMT first, main boundary next, then the extra filter.
    """
    chromosome = row["chr"]
    if chromosome == "chrMT":
        return "chrMT"
    if chromosome not in chromosome_sizes:
        raise ValueError(f"No chromosome size supplied for {chromosome!r}")
    size = chromosome_sizes[chromosome]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"Invalid chromosome size for {chromosome!r}")
    start, end = int(row["start"]), int(row["end"])
    if start < 1 or end < start or row["strand"] not in ("+", "-"):
        raise ValueError("Invalid released coordinate or strand")
    anchor = (start if row["strand"] == "+" else end) - 1
    if anchor - 98304 < 0 or anchor + 98304 >= size:
        return "main_boundary"
    if anchor - 196608 < 0 or anchor >= size:
        return "extra_shifted_boundary"
    return None


@contextmanager
def _csv_field_capacity(source_bytes):
    """Allow long transcripts while restoring csv's process-wide parser setting."""
    previous = csv.field_size_limit()
    try:
        csv.field_size_limit(max(previous, source_bytes))
        yield
    finally:
        csv.field_size_limit(previous)


def audit_csv(path, rna_tokenizer, protein_tokenizer, *, expected_sha256=None,
              expected_bytes=None, chromosome_sizes=None, include_data_checks=False):
    """Measure one CSV without writing, selecting a cohort, or altering sequences.

    Hash/byte expectations are checked before parsing. Raw split names are preserved
    (including ``eval``); they are not mapped to the released loader's split aliases.
    Character lengths include unsupported symbols; only present sequences contribute
    length/token statistics. Suffix scenarios count an in-memory 3-prime substring,
    never replace the source sequence, and do not authorize a training crop.
    Optional data checks add annotation, pairing, target, overlap and distribution
    summaries in this same pass. They do not interpret annotation labels as coding
    classes or apply the displayed hypothetical cohort restrictions.
    """
    if type(include_data_checks) is not bool:
        raise TypeError("include_data_checks must be an explicit boolean")
    path = Path(path)
    data_checks = None
    scopes = {"all": _new_scope()}
    alphabets = {"RNA": {}, "Protein": {}}
    validation = {"interval_rows": CHECK_EVERY, "actual_row_checks": 0,
                  "actual_sequence_checks": 0}
    boundary_counts = {"chrMT": 0, "main_boundary": 0, "extra_shifted_boundary": 0}
    boundary_required = ("chr", "start", "end", "strand") if chromosome_sizes is not None else ()
    if chromosome_sizes is not None:
        scopes["example_boundary_policy"] = _new_scope()

    # Hash and parse the same open file; changed file metadata invalidates the report.
    with path.open("rb") as binary:
        before = os.fstat(binary.fileno())
        digest = hashlib.sha256()
        for chunk in iter(lambda: binary.read(1024 * 1024), b""):
            digest.update(chunk)
        source_sha256 = digest.hexdigest()
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise ValueError(f"CSV size mismatch: {before.st_size} != {expected_bytes}")
        if expected_sha256 is not None and source_sha256 != expected_sha256.lower():
            raise ValueError("CSV SHA-256 does not match the expected source")
        validation["synthetic_checks"] = validate_tokenizers(rna_tokenizer, protein_tokenizer)
        protein_alphabet = _protein_alphabet(protein_tokenizer)
        binary.seek(0)
        # csv handles quoted commas/newlines; utf-8-sig also accepts a plain UTF-8 CSV.
        with _csv_field_capacity(before.st_size), \
                io.TextIOWrapper(binary, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                raise ValueError("Duplicate CSV columns make the audit ambiguous")
            missing = set(_REQUIRED + boundary_required).difference(headers)
            if missing:
                raise ValueError(f"Missing CSV columns: {sorted(missing)}")
            if include_data_checks:
                data_checks = GTExDataChecks(headers, has_boundary=chromosome_sizes is not None,
                                             is_missing=is_missing_sequence)
            for index, row in enumerate(reader, start=1):
                if None in row or any(row[name] is None for name in headers):
                    raise ValueError(f"Malformed CSV row {index}")
                split = row["split"]
                row_scopes = [scopes["all"], scopes.setdefault(f"split:{split}", _new_scope())]
                reason = None
                if chromosome_sizes is not None:
                    reason = _boundary_exclusion(row, chromosome_sizes)
                    if reason:
                        boundary_counts[reason] += 1
                    else:
                        row_scopes += [
                            scopes.setdefault("example_boundary_policy", _new_scope()),
                            scopes.setdefault(f"example_boundary_policy:{split}", _new_scope()),
                        ]
                check = index == 1 or index % CHECK_EVERY == 0
                if check:
                    validation["actual_row_checks"] += 1
                measurements = []
                missing_labels = []
                fallback_labels = []
                for label, tokenizer in (("RNA", rna_tokenizer), ("Protein", protein_tokenizer)):
                    sequence = row[label]
                    if is_missing_sequence(sequence):
                        missing_labels.append(label)
                        continue
                    for character in set(sequence):
                        alphabets[label][character] = alphabets[label].get(character, 0) + sequence.count(character)
                    limits = RNA_LENGTH_LIMITS if label == "RNA" else PROTEIN_LENGTH_LIMITS
                    measurements.append((f"{label}_length", len(sequence), limits))
                    fast = (rna_fast_token_count(sequence) if label == "RNA"
                            else protein_fast_token_count(sequence, protein_alphabet))
                    if fast is None:
                        fallback_labels.append(label)
                    actual = _actual_count(tokenizer, sequence) if check or fast is None else None
                    if check:
                        validation["actual_sequence_checks"] += 1
                    if fast is not None and actual is not None and fast != actual:
                        raise ValueError(f"{label} count mismatch at CSV row {index}: {fast} != {actual}")
                    measurements.append((f"{label}_tokens", fast if fast is not None else actual, TOKEN_LIMITS))
                    if label == "RNA":
                        for width in RNA_LENGTH_LIMITS:
                            suffix = sequence[-width:]
                            count = rna_fast_token_count(suffix)
                            actual = _actual_count(tokenizer, suffix) if check or count is None else None
                            if check:
                                validation["actual_sequence_checks"] += 1
                            if count is not None and actual is not None and count != actual:
                                raise ValueError(f"RNA suffix count mismatch at CSV row {index}, width {width}")
                            measurements.append((f"RNA_after_3prime_{width}_tokens",
                                                 count if count is not None else actual, TOKEN_LIMITS))
                if data_checks is not None:
                    data_checks.add(row, measurements, boundary_exclusion=reason)
                for scope in row_scopes:
                    scope["rows"] += 1
                    transcript = row["transcript_id_gtex"]
                    if transcript:
                        _increment(scope, "duplicate_transcript_ids", int(transcript in scope["_ids"]))
                        scope["_ids"].add(transcript)
                    else:
                        _increment(scope, "missing_transcript_ids")
                    for label in ("RNA", "Protein"):
                        _increment(scope, f"missing_{label}", int(label in missing_labels))
                        _increment(scope, f"{label}_tokenizer_fallback", int(label in fallback_labels))
                    for key, value, limits in measurements:
                        _measure(scope, key, value, limits)
            after = os.fstat(binary.fileno())
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("CSV changed during the audit; no report is returned")

    for scope in scopes.values():
        scope["unique_transcript_ids"] = len(scope.pop("_ids"))
    all_rows = scopes["all"]
    result = {
        "schema_version": 1,
        "scope": "released_csv_measurement_not_historical_training_cohort",
        "source_sha256": source_sha256, "source_bytes": before.st_size,
        "row_count": all_rows["rows"],
        "unique_transcript_ids": all_rows["unique_transcript_ids"],
        "duplicate_transcript_ids": all_rows["counts"].get("duplicate_transcript_ids", 0),
        "missing_sequences": {label: all_rows["counts"].get(f"missing_{label}", 0)
                              for label in ("RNA", "Protein")},
        "fallback_rows": {label: all_rows["counts"].get(f"{label}_tokenizer_fallback", 0)
                          for label in ("RNA", "Protein")},
        "splits": {key.removeprefix("split:"): scope["rows"] for key, scope in scopes.items()
                   if key.startswith("split:")},
        "alphabets": {label: dict(sorted(counts.items())) for label, counts in alphabets.items()},
        "scopes": scopes, "token_count_validation": validation,
    }
    if chromosome_sizes is not None:
        result["example_boundary_exclusions"] = {
            "chrMT": boundary_counts["chrMT"],
            "main_window": boundary_counts["main_boundary"],
            "additional_shifted_filter": boundary_counts["extra_shifted_boundary"],
        }
        result["example_boundary_policy"] = {
            "exclusions": boundary_counts,
            "retained_rows": all_rows["rows"] - sum(boundary_counts.values()),
            "purpose": "Historical inference-example boundary scenario; not a recovered training cohort",
        }
    if data_checks is not None:
        result["data_checks"] = data_checks.finish()
    return result


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"Audit {name} must be a nonnegative integer")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Audit {name} must be a lowercase SHA-256 digest")


def _validate_audit(report, *, allow_legacy):
    """Reject incomplete or internally inconsistent saved counts without changing them."""
    if not isinstance(report, dict):
        raise ValueError("Audit JSON must contain an object")
    # Check the whole payload, including future measurement fields, for JSON validity.
    try:
        json.dumps(report, allow_nan=False)
        _sha256(report["source_sha256"], "source_sha256")
        if _count(report["source_bytes"], "source_bytes") == 0:
            raise ValueError("Audit source_bytes must be positive")
        times = [datetime.fromisoformat(report[name]) for name in ("started_utc", "finished_utc")]
        if any(t.tzinfo is None or t.utcoffset() != timezone.utc.utcoffset(t) for t in times):
            raise ValueError("Audit completion timestamps must be timezone-aware UTC")
        if times[1] < times[0]:
            raise ValueError("Audit finished_utc precedes started_utc")
        identity = report.get("audit_identity")
        if identity is None:
            if not allow_legacy:
                raise ValueError("Historical audit lacks audit_identity; explicit allow_legacy is required")
        else:
            if not isinstance(identity, dict) or set(_IDENTITY_FIELDS).difference(identity):
                raise ValueError("Audit identity is missing source, method, or tokenizer fields")
            if not isinstance(identity["method_version"], str) or not identity["method_version"]:
                raise ValueError("Audit method_version must be declared")
            for name in _IDENTITY_FIELDS:
                if name != "method_version" and identity[name] != report.get(name):
                    raise ValueError(f"Audit identity disagrees with report field {name}")
            for name in ("dataset_revision", "model_revision", "transformers"):
                if not isinstance(identity[name], str) or not identity[name]:
                    raise ValueError(f"Audit identity {name} must be declared")
            _sha256(identity["chromosome_sizes_sha256"], "chromosome_sizes_sha256")
            vocabularies = identity["vocabulary_sha256"]
            if not isinstance(vocabularies, dict) or set(vocabularies) != {"rna", "protein"}:
                raise ValueError("Audit identity needs both tokenizer vocabulary hashes")
            for name, digest in vocabularies.items():
                _sha256(digest, f"{name} vocabulary")

        rows = _count(report["row_count"], "row_count")
        unique = _count(report["unique_transcript_ids"], "unique_transcript_ids")
        duplicate = _count(report["duplicate_transcript_ids"], "duplicate_transcript_ids")
        scopes, splits = report["scopes"], report["splits"]
        if not isinstance(scopes, dict) or not isinstance(splits, dict):
            raise ValueError("Audit scopes and splits must be mappings")
        if scopes["all"]["rows"] != rows:
            raise ValueError("Audit all scope disagrees with row_count")
        missing_ids = scopes["all"]["counts"].get("missing_transcript_ids", 0)
        if unique + duplicate + _count(missing_ids, "missing_transcript_ids") != rows:
            raise ValueError("Audit transcript identity counts do not reconcile")
        if sum(_count(n, f"split:{name}") for name, n in splits.items()) != rows:
            raise ValueError("Audit split counts do not sum to row_count")
        for name, n in splits.items():
            if scopes[f"split:{name}"]["rows"] != n:
                raise ValueError(f"Audit split:{name} scope disagrees with its row count")
        for scope_name, scope in scopes.items():
            scope_rows = _count(scope["rows"], f"{scope_name} rows")
            if scope_rows > rows or not isinstance(scope["stats"], dict):
                raise ValueError(f"Audit {scope_name} has invalid rows/statistics")
            for key, value in scope["counts"].items():
                if _count(value, f"{scope_name}/{key}") > scope_rows:
                    raise ValueError(f"Audit {scope_name}/{key} exceeds scope rows")
            for key, stat in scope["stats"].items():
                n = _count(stat["n"], f"{scope_name}/{key}/n")
                total = _count(stat["sum"], f"{scope_name}/{key}/sum")
                if n > scope_rows:
                    raise ValueError(f"Audit {scope_name}/{key}/n exceeds scope rows")
                if n:
                    minimum = _count(stat["min"], f"{scope_name}/{key}/min")
                    maximum = _count(stat["max"], f"{scope_name}/{key}/max")
                    if minimum > maximum or not n * minimum <= total <= n * maximum:
                        raise ValueError(f"Audit {scope_name}/{key} statistics do not reconcile")
                elif total or stat["min"] is not None or stat["max"] is not None:
                    raise ValueError(f"Audit empty {scope_name}/{key} has nonempty statistics")
            for label in ("RNA", "Protein"):
                missing = scope["counts"].get(f"missing_{label}")
                if scope_name == "all":
                    root_missing = _count(report["missing_sequences"].get(label, 0), f"missing {label}")
                    if missing is not None and missing != root_missing:
                        raise ValueError(f"Audit missing {label} counts disagree")
                    missing = root_missing
                present = scope["stats"].get(f"{label}_length", {}).get("n", 0)
                tokens = scope["stats"].get(f"{label}_tokens", {}).get("n", 0)
                if present != tokens or (missing is not None and present + missing != scope_rows):
                    raise ValueError(f"Audit {scope_name} {label} availability does not reconcile")
            if identity is not None:
                statistic_keys = ("RNA_length", "Protein_length", "RNA_tokens", "Protein_tokens")
                statistic_keys += tuple(f"RNA_after_3prime_{width}_tokens" for width in RNA_LENGTH_LIMITS)
                for key in statistic_keys:
                    present = scope["stats"].get(key, {}).get("n", 0)
                    if key.startswith("RNA_after_") and present != scope["stats"].get("RNA_length", {}).get("n", 0):
                        raise ValueError(f"Audit {scope_name} lacks complete suffix statistics")
                    for limit in (RNA_LENGTH_LIMITS if key == "RNA_length" else
                                  PROTEIN_LENGTH_LIMITS if key == "Protein_length" else TOKEN_LIMITS):
                        count_key = f"{key}_gt_{limit}"
                        if present and count_key not in scope["counts"]:
                            raise ValueError(f"Audit {scope_name} lacks {count_key}")
                        if scope["counts"].get(count_key, 0) > present:
                            raise ValueError(f"Audit {scope_name}/{count_key} exceeds measured sequences")
        for key, stat in scopes["all"]["stats"].items():
            for metric in ("n", "sum"):
                split_total = sum(scopes[f"split:{name}"]["stats"].get(key, {}).get(metric, 0)
                                  for name in splits)
                if split_total != stat[metric]:
                    raise ValueError(f"Audit {key}/{metric} does not reconcile across raw splits")
        if not isinstance(report["fallback_rows"], dict) or not isinstance(report["alphabets"], dict):
            raise ValueError("Audit fallback rows and alphabets must be mappings")
        for label in ("RNA", "Protein"):
            present = scopes["all"]["stats"].get(f"{label}_length", {}).get("n", 0)
            fallback = _count(report["fallback_rows"].get(label, 0), f"{label} fallback rows")
            if fallback > present:
                raise ValueError(f"Audit {label} fallback rows exceed present sequences")
            scope_fallback = scopes["all"]["counts"].get(f"{label}_tokenizer_fallback")
            if scope_fallback is not None and scope_fallback != fallback:
                raise ValueError(f"Audit {label} fallback counts disagree")
            alphabet = report["alphabets"][label]
            if not isinstance(alphabet, dict) or any(not isinstance(c, str) or len(c) != 1 for c in alphabet):
                raise ValueError(f"Audit {label} alphabet must count individual characters")
            total = sum(_count(n, f"{label} alphabet") for n in alphabet.values())
            if total != scopes["all"]["stats"].get(f"{label}_length", {}).get("sum", 0):
                raise ValueError(f"Audit {label} alphabet does not match total character length")
        if "example_boundary_exclusions" in report:
            exclusions = report["example_boundary_exclusions"]
            excluded = sum(_count(n, f"boundary {key}") for key, n in exclusions.items())
            retained = scopes["example_boundary_policy"]["rows"]
            if excluded + retained != rows:
                raise ValueError("Audit genomic-boundary counts do not reconcile")
        if "data_checks" in report or (identity is not None and "data_checks_version" in identity):
            checks = report["data_checks"]
            if identity is None or identity.get("data_checks_version") != checks["version"]:
                raise ValueError("Extended audit requires a matching data_checks_version in audit_identity")
            validate_data_checks(checks, report)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"Incomplete or invalid audit JSON: {error}") from error
    return report


def read_audit(path, expected_identity=None, allow_legacy=False):
    """Read a completed report, optionally matching a declared identity subset.

    A subset lets a reader verify source/method without importing the producer's
    tokenizer environment. Legacy records require explicit opt-in and compare the
    requested subset against their original top-level fields; they are not upgraded.
    """
    report = _validate_audit(json.loads(Path(path).read_text(encoding="utf-8")),
                             allow_legacy=allow_legacy)
    if expected_identity is not None:
        if not isinstance(expected_identity, dict):
            raise ValueError("expected_identity must be a mapping")
        actual = report.get("audit_identity", report)
        if any(key not in actual or actual[key] != value for key, value in expected_identity.items()):
            raise ValueError("Saved audit does not match the requested identity")
    return report


def _retain_existing(path, report):
    try:
        existing = read_audit(path, allow_legacy=True)
    except (ValueError, OSError) as error:
        raise FileExistsError(f"Existing audit cannot be reused: {path}. Choose an explicit new RESULT_PATH.") from error
    # Unknown fields remain measurements for comparison; only named producer fields
    # may differ. A new timestamp never justifies replacing a completed observation.
    comparable = lambda value: {key: item for key, item in value.items() if key not in _PRODUCER_FIELDS}
    if comparable(existing) != comparable(report):
        raise FileExistsError(f"Audit identity or measurements differ at {path}. Choose an explicit new RESULT_PATH.")
    return False


def save_audit(path, report):
    """Publish a new completed JSON once; retain equivalent existing bytes unchanged.

    Returns True when saved and False when already retained. A conflicting legacy,
    identity, measurement or invalid destination raises without replacing that file.
    Same-directory hard-link publication is atomic and refuses destination races;
    an unsupported filesystem fails rather than falling back to overwrite behavior.
    """
    _validate_audit(report, allow_legacy=False)
    path = Path(path)
    if path.exists():
        return _retain_existing(path, report)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return _retain_existing(path, report)
        return True
    finally:
        if temporary is not None:
            failure = sys.exception()
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Could not remove temporary audit file {temporary}: {cleanup_error}")
