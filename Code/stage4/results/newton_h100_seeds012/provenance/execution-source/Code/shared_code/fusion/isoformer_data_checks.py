"""Streaming GTEx metadata checks; no I/O, cohort selection, or target transforms.

The CSV audit supplies each unchanged row and its already measured token lengths.
Sequence identities are SHA-256 digests of exact UTF-8 strings, so the accumulator
does not retain full sequences. Shared proteins are reported as observations, not
automatically classified as evaluation leakage.
"""

from bisect import bisect_left
import hashlib
from itertools import combinations
import math


DATA_CHECKS_VERSION = "gtex-data-checks-v1"
# Official released tissue order, also used by shared_code.datasets.GTEX_TISSUES.
# Keep this stdlib audit independent of that data-loading module's dependencies.
GTEX_TISSUES = (
    "Adipose Tissue", "Adrenal Gland", "Bladder", "Blood", "Blood Vessel", "Brain",
    "Breast", "Cervix Uteri", "Colon", "Esophagus", "Fallopian Tube", "Heart",
    "Kidney", "Liver", "Lung", "Muscle", "Nerve", "Ovary", "Pancreas", "Pituitary",
    "Prostate", "Salivary Gland", "Skin", "Small Intestine", "Spleen", "Stomach",
    "Testis", "Thyroid", "Uterus", "Vagina",
)
REQUIRED_METADATA = (
    "RNA_type_gff", "gene_id_gtex", "transcript_id_ensembl_protein", "CDS",
    "chr", "strand", "start", "end",
)
LENGTH_BOUNDS = {
    "RNA": (200, 500, 1000, 2000, 5000, 12000, 12282, 12288, 20000, 50000, 100000),
    "Protein": (100, 300, 600, 1022, 1024, 1200, 2046, 2047, 2048, 5000, 10000),
}
POLICY_DEFINITIONS = {
    "rna_missing": "No RNA string supplied under the exact missing-marker rule.",
    "protein_missing": "No protein string supplied; this does not establish a noncoding annotation.",
    "protein_id_missing": "No associated Ensembl protein transcript ID supplied.",
    "cds_missing": "No CDS string supplied.",
    "protein_length_gt_1200": "Supplied protein exceeds 1,200 characters.",
    "protein_length_gt_2048": "Supplied protein exceeds 2,048 characters.",
    "rna_tokens_gt_2048": "Unchanged supplied RNA exceeds 2,048 unpadded tokens.",
    "rna_after_3prime_12000_tokens_gt_2048": "Hypothetical 12,000-character RNA suffix exceeds 2,048 tokens.",
    "rna_after_3prime_12282_tokens_gt_2048": "Hypothetical 12,282-character RNA suffix exceeds 2,048 tokens.",
    "rna_after_3prime_12288_tokens_gt_2048": "Hypothetical 12,288-character RNA suffix exceeds 2,048 tokens.",
    "dna_boundary_excluded": "Coordinates fail the inference example's chromosome-window checks.",
}
EXAMPLE_LIMIT = 5
_CONFLICT_FIELDS = (
    "transcripts_with_multiple_gene_ids", "transcripts_with_multiple_rna_sequences",
    "transcripts_with_multiple_protein_sequences", "genes_with_multiple_chromosomes",
    "genes_with_multiple_strands", "protein_ids_with_multiple_sequences",
)


def _append_example(examples, identifier, missing):
    if len(examples) < EXAMPLE_LIMIT and not missing(identifier) and identifier not in examples:
        examples.append(identifier)


class GTExDataChecks:
    """Accumulate one CSV pass; retain compact identity maps and bounded examples."""

    def __init__(self, headers, *, has_boundary, is_missing):
        absent = sorted(set(REQUIRED_METADATA).difference(headers))
        if absent:
            raise ValueError(f"Extended GTEx audit requires metadata columns: {absent}")
        self.missing = is_missing
        self.has_boundary = has_boundary
        self.rows = 0
        self.splits = set()
        self.annotations = {}
        self.policies = {}
        self.identities = {key: {} for key in ("gene_id", "transcript_id", "RNA", "Protein")}
        self.identity_missing = dict.fromkeys(self.identities, 0)
        self.first_transcript_mapping = {}
        self.first_gene_mapping = {}
        self.first_protein_mapping = {}
        self.conflicts = {key: set() for key in _CONFLICT_FIELDS}
        self.missing_gene_ids = 0
        self.missing_transcript_ids = 0
        self.targets = {
            name: {"present": name in headers, "rows": 0, "missing": 0,
                   "nonnumeric": 0, "nonfinite": 0, "finite": 0, "negative": 0,
                   "zero": 0, "min": None, "max": None, "mean": None,
                   "variance": None, "_m2": 0.0}
            for name in GTEX_TISSUES
        }
        self.observed_order = [name for name in headers if name in GTEX_TISSUES]
        self.distributions = {
            label: {"upper_bounds": [*bounds, None], "counts": [0] * (len(bounds) + 1),
                    "present": 0, "missing": 0}
            for label, bounds in LENGTH_BOUNDS.items()
        }
        self.representatives = {"first_rows": [], "longest_RNA": None,
                                "longest_Protein": None, "missing_protein_examples": []}

    def _record_mapping(self, key, values, first, conflict_names):
        if self.missing(key):
            return
        previous = first.setdefault(key, [None] * len(values))
        for index, value in enumerate(values):
            if value is None:
                continue
            if previous[index] is None:
                previous[index] = value
            elif previous[index] != value:
                self.conflicts[conflict_names[index]].add(key)

    def _record_identity(self, kind, value, split, transcript):
        if value is None:
            self.identity_missing[kind] += 1
            return
        memberships = self.identities[kind].setdefault(value, {})
        membership = memberships.setdefault(split, [0, transcript])
        membership[0] += 1
        if self.missing(membership[1]) and not self.missing(transcript):
            membership[1] = transcript

    def _record_targets(self, row):
        for name, stats in self.targets.items():
            stats["rows"] += 1
            text = row.get(name, "")
            if not text.strip():
                stats["missing"] += 1
                continue
            try:
                value = float(text)
            except (ValueError, TypeError):
                stats["nonnumeric"] += 1
                continue
            if not math.isfinite(value):
                stats["nonfinite"] += 1
                continue
            stats["finite"] += 1
            stats["negative"] += int(value < 0)
            stats["zero"] += int(value == 0)
            stats["min"] = value if stats["min"] is None else min(stats["min"], value)
            stats["max"] = value if stats["max"] is None else max(stats["max"], value)
            if stats["mean"] is None:
                stats["mean"] = value
            else:
                delta = value - stats["mean"]
                stats["mean"] += delta / stats["finite"]
                stats["_m2"] += delta * (value - stats["mean"])
            if not math.isfinite(stats["mean"]) or not math.isfinite(stats["_m2"]):
                raise ValueError(f"Target {name!r} exceeds the audit's finite numeric range")

    def add(self, row, measurements, *, boundary_exclusion):
        """Record one original row and already calculated length/token measurements."""
        self.rows += 1
        split, annotation = row["split"], row["RNA_type_gff"]
        self.splits.add(split)
        transcript, gene = row["transcript_id_gtex"], row["gene_id_gtex"]
        protein_id = row["transcript_id_ensembl_protein"]
        present = {
            "rna": not self.missing(row["RNA"]), "protein": not self.missing(row["Protein"]),
            "protein_id": not self.missing(protein_id), "cds": not self.missing(row["CDS"]),
        }
        counts = self.annotations.setdefault(annotation, {
            "rows": 0, "rna_present": 0, "rna_missing": 0, "protein_present": 0,
            "protein_missing": 0, "protein_id_present": 0, "protein_id_missing": 0,
            "cds_present": 0, "cds_missing": 0, "missing_protein_with_id": 0,
            "missing_protein_with_cds": 0,
        })
        counts["rows"] += 1
        for label, available in present.items():
            counts[f"{label}_present"] += int(available)
            counts[f"{label}_missing"] += int(not available)
        counts["missing_protein_with_id"] += int(not present["protein"] and present["protein_id"])
        counts["missing_protein_with_cds"] += int(not present["protein"] and present["cds"])

        gene_value = None if self.missing(gene) else gene
        transcript_value = None if self.missing(transcript) else transcript
        sequence_hashes = {
            label: hashlib.sha256(row[label].encode("utf-8")).hexdigest() if present[label.lower()] else None
            for label in ("RNA", "Protein")
        }
        self.missing_gene_ids += int(gene_value is None)
        self.missing_transcript_ids += int(transcript_value is None)
        self._record_mapping(transcript, [gene_value, sequence_hashes["RNA"], sequence_hashes["Protein"]],
                             self.first_transcript_mapping, _CONFLICT_FIELDS[:3])
        self._record_mapping(gene, [None if self.missing(row[key]) else row[key] for key in ("chr", "strand")],
                             self.first_gene_mapping, _CONFLICT_FIELDS[3:5])
        self._record_mapping(protein_id, [sequence_hashes["Protein"]], self.first_protein_mapping,
                             ("protein_ids_with_multiple_sequences",))
        for kind, value in (("gene_id", gene_value), ("transcript_id", transcript_value), *sequence_hashes.items()):
            self._record_identity(kind, value, split, transcript)
        self._record_targets(row)

        numeric = {key: value for key, value, _ in measurements}
        summary = {"row_number": self.rows, "transcript_id": transcript, "gene_id": gene,
                   "split": split, "annotation": annotation,
                   **{key: numeric.get(key) for key in ("RNA_length", "Protein_length", "RNA_tokens", "Protein_tokens")},
                   "protein_id_present": present["protein_id"], "cds_present": present["cds"]}
        if len(self.representatives["first_rows"]) < EXAMPLE_LIMIT:
            self.representatives["first_rows"].append(summary)
        if not present["protein"] and len(self.representatives["missing_protein_examples"]) < EXAMPLE_LIMIT:
            self.representatives["missing_protein_examples"].append(summary)
        for label in ("RNA", "Protein"):
            length = numeric.get(f"{label}_length")
            distribution = self.distributions[label]
            if length is None:
                distribution["missing"] += 1
            else:
                distribution["present"] += 1
                distribution["counts"][bisect_left(LENGTH_BOUNDS[label], length)] += 1
                longest = self.representatives[f"longest_{label}"]
                if longest is None or length > longest[f"{label}_length"]:
                    self.representatives[f"longest_{label}"] = summary

        reasons = [f"{label}_missing" for label, available in present.items() if not available]
        for key, threshold in (("Protein_length", 1200), ("Protein_length", 2048), ("RNA_tokens", 2048),
                               *((f"RNA_after_3prime_{width}_tokens", 2048) for width in (12000, 12282, 12288))):
            if numeric.get(key) is not None and numeric[key] > threshold:
                reason_key = key.replace("RNA", "rna", 1).replace("Protein", "protein", 1)
                reasons.append(f"{reason_key}_gt_{threshold}")
        if self.has_boundary and boundary_exclusion is not None:
            reasons.append("dna_boundary_excluded")
        definitions = {key: value for key, value in POLICY_DEFINITIONS.items()
                       if self.has_boundary or key != "dna_boundary_excluded"}
        group = self.policies.setdefault((split, annotation), {
            "split": split, "annotation": annotation, "rows": 0, "rna_present": 0,
            "protein_present": 0, "rna_and_protein_present": 0,
            "dna_eligible": 0 if self.has_boundary else None,
            "all_three_present_after_boundary": 0 if self.has_boundary else None,
            "reasons": dict.fromkeys(definitions, 0), "overlapping_reason_pairs": {},
            "example_ids": {key: [] for key in definitions},
        })
        group["rows"] += 1
        group["rna_present"] += int(present["rna"])
        group["protein_present"] += int(present["protein"])
        paired = present["rna"] and present["protein"]
        group["rna_and_protein_present"] += int(paired)
        if self.has_boundary:
            group["dna_eligible"] += int(boundary_exclusion is None)
            group["all_three_present_after_boundary"] += int(paired and boundary_exclusion is None)
        for reason in reasons:
            group["reasons"][reason] += 1
            _append_example(group["example_ids"][reason], transcript, self.missing)
        for first, second in combinations(sorted(reasons), 2):
            pair = f"{first} & {second}"
            group["overlapping_reason_pairs"][pair] = group["overlapping_reason_pairs"].get(pair, 0) + 1

    def finish(self):
        """Return JSON-compatible summaries without selecting or changing any rows."""
        overlaps = {}
        for kind, values in self.identities.items():
            pairs = []
            for split_a, split_b in combinations(sorted(self.splits), 2):
                pair = {"split_a": split_a, "split_b": split_b, "shared_unique_values": 0,
                        "rows_a": 0, "rows_b": 0, "example_transcript_ids_a": [], "example_transcript_ids_b": []}
                for membership in values.values():
                    if split_a in membership and split_b in membership:
                        pair["shared_unique_values"] += 1
                        for side, split in (("a", split_a), ("b", split_b)):
                            pair[f"rows_{side}"] += membership[split][0]
                            _append_example(pair[f"example_transcript_ids_{side}"], membership[split][1], self.missing)
                pairs.append(pair)
            overlaps[kind] = {"unique_values": len(values), "missing_rows": self.identity_missing[kind], "pairs": pairs}
        targets = {}
        for name, stats in self.targets.items():
            targets[name] = {key: value for key, value in stats.items() if key != "_m2"}
            if stats["finite"]:
                targets[name]["variance"] = max(0.0, stats["_m2"] / stats["finite"])
        return {
            "version": DATA_CHECKS_VERSION, "row_count": self.rows,
            "annotations": {"field": "RNA_type_gff", "by_type": dict(sorted(self.annotations.items()))},
            "mappings": {"missing_gene_ids": self.missing_gene_ids, "missing_transcript_ids": self.missing_transcript_ids,
                         **{key: len(values) for key, values in self.conflicts.items()},
                         "examples": {key: sorted(values)[:EXAMPLE_LIMIT] for key, values in self.conflicts.items()}},
            "overlaps": overlaps,
            "targets": {"expected_order": list(GTEX_TISSUES), "observed_order": self.observed_order,
                        "missing_columns": [name for name in GTEX_TISSUES if not self.targets[name]["present"]],
                        "order_matches": self.observed_order == list(GTEX_TISSUES), "columns": targets},
            "policy_impacts": {"definitions": {key: value for key, value in POLICY_DEFINITIONS.items()
                                                if self.has_boundary or key != "dna_boundary_excluded"},
                               "groups": [self.policies[key] for key in sorted(self.policies)]},
            "length_distributions": self.distributions,
            "representative_rows": self.representatives,
        }


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError("Extended audit counts must be nonnegative integers")
    return value


def _examples(values, maximum=EXAMPLE_LIMIT):
    if not isinstance(values, list) or len(values) > maximum or any(not isinstance(v, str) for v in values):
        raise ValueError("Extended audit examples must be bounded identifier lists")


def validate_data_checks(checks, report):
    """Validate completed extended summaries against their common row denominators."""
    rows = report["row_count"]
    if checks["version"] != DATA_CHECKS_VERSION or checks["row_count"] != rows:
        raise ValueError("Extended audit version or row_count does not match the report")
    annotations = checks["annotations"]
    if annotations["field"] != "RNA_type_gff":
        raise ValueError("Extended audit annotation field must retain RNA_type_gff")
    types = annotations["by_type"]
    if sum(_count(group["rows"]) for group in types.values()) != rows:
        raise ValueError("Extended annotation rows do not reconcile")
    for annotation, group in types.items():
        if not isinstance(annotation, str):
            raise ValueError("Extended annotation labels must retain source strings")
        for label in ("rna", "protein", "protein_id", "cds"):
            if _count(group[f"{label}_present"]) + _count(group[f"{label}_missing"]) != group["rows"]:
                raise ValueError("Extended annotation availability does not reconcile")
        for label in ("id", "cds"):
            supplied = group["protein_id_present" if label == "id" else "cds_present"]
            if _count(group[f"missing_protein_with_{label}"]) > min(group["protein_missing"], supplied):
                raise ValueError("Extended annotation missing-protein intersection is invalid")

    mappings = checks["mappings"]
    for key in ("missing_gene_ids", "missing_transcript_ids", *_CONFLICT_FIELDS):
        if _count(mappings[key]) > rows:
            raise ValueError("Extended mapping count exceeds rows")
    for key in _CONFLICT_FIELDS:
        _examples(mappings["examples"][key], min(EXAMPLE_LIMIT, mappings[key]))
    expected_pairs = set(combinations(sorted(report["splits"]), 2))
    for kind in ("gene_id", "transcript_id", "RNA", "Protein"):
        overlap = checks["overlaps"][kind]
        if _count(overlap["unique_values"]) + _count(overlap["missing_rows"]) > rows:
            raise ValueError("Extended overlap availability does not reconcile")
        pairs = set()
        for pair in overlap["pairs"]:
            names = pair["split_a"], pair["split_b"]
            if names not in expected_pairs or names in pairs:
                raise ValueError("Extended overlap has an invalid or duplicate split pair")
            pairs.add(names)
            shared = _count(pair["shared_unique_values"])
            for side, split in zip(("a", "b"), names):
                affected = _count(pair[f"rows_{side}"])
                if not shared <= affected <= report["splits"][split] or (shared == 0 and affected != 0):
                    raise ValueError("Extended overlap affected rows do not reconcile")
                _examples(pair[f"example_transcript_ids_{side}"], min(EXAMPLE_LIMIT, shared))
        if pairs != expected_pairs:
            raise ValueError("Extended overlap is missing raw split pairs")

    targets = checks["targets"]
    if targets["expected_order"] != list(GTEX_TISSUES) or set(targets["columns"]) != set(GTEX_TISSUES):
        raise ValueError("Extended targets must describe the canonical 30 tissue columns")
    observed = targets["observed_order"]
    if len(observed) != len(set(observed)) or not set(observed).issubset(GTEX_TISSUES):
        raise ValueError("Extended target observed order is invalid")
    missing_columns = [name for name in GTEX_TISSUES if not targets["columns"][name]["present"]]
    if targets["missing_columns"] != missing_columns or set(observed) != set(GTEX_TISSUES).difference(missing_columns):
        raise ValueError("Extended target column availability does not reconcile")
    if type(targets["order_matches"]) is not bool or targets["order_matches"] != (observed == list(GTEX_TISSUES)):
        raise ValueError("Extended target order verdict is inconsistent")
    for name, stats in targets["columns"].items():
        if type(stats["present"]) is not bool or stats["rows"] != rows:
            raise ValueError(f"Extended target {name} rows/column status is invalid")
        if sum(_count(stats[key]) for key in ("missing", "nonnumeric", "nonfinite", "finite")) != rows:
            raise ValueError(f"Extended target {name} value categories do not reconcile")
        if _count(stats["negative"]) + _count(stats["zero"]) > stats["finite"]:
            raise ValueError(f"Extended target {name} signed counts exceed finite values")
        if not stats["present"] and stats["missing"] != rows:
            raise ValueError(f"Extended absent target {name} has measured values")
        if stats["finite"]:
            numbers = [stats[key] for key in ("min", "max", "mean", "variance")]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in numbers):
                raise ValueError(f"Extended target {name} has invalid finite statistics")
            if not stats["min"] <= stats["mean"] <= stats["max"] or stats["variance"] < 0:
                raise ValueError(f"Extended target {name} statistics do not reconcile")
        elif any(stats[key] is not None for key in ("min", "max", "mean", "variance")):
            raise ValueError(f"Extended empty target {name} has finite statistics")

    policies = checks["policy_impacts"]
    has_boundary = "example_boundary_policy" in report["scopes"]
    reason_keys = set(POLICY_DEFINITIONS) - (set() if has_boundary else {"dna_boundary_excluded"})
    if set(policies["definitions"]) != reason_keys:
        raise ValueError("Extended policy definitions are incomplete")
    split_totals = dict.fromkeys(report["splits"], 0)
    type_totals = dict.fromkeys(types, 0)
    combinations_seen = set()
    for group in policies["groups"]:
        key = group["split"], group["annotation"]
        n = _count(group["rows"])
        if key in combinations_seen or key[0] not in split_totals or key[1] not in type_totals:
            raise ValueError("Extended policy grouping is invalid")
        combinations_seen.add(key)
        split_totals[key[0]] += n
        type_totals[key[1]] += n
        if set(group["reasons"]) != reason_keys or set(group["example_ids"]) != reason_keys:
            raise ValueError("Extended policy reasons/examples are incomplete")
        for reason, count in group["reasons"].items():
            if _count(count) > n:
                raise ValueError("Extended policy count exceeds its common denominator")
            _examples(group["example_ids"][reason], min(EXAMPLE_LIMIT, count))
        for label in ("rna", "protein"):
            if _count(group[f"{label}_present"]) + group["reasons"][f"{label}_missing"] != n:
                raise ValueError("Extended policy availability does not reconcile")
        paired = _count(group["rna_and_protein_present"])
        if not max(0, group["rna_present"] + group["protein_present"] - n) <= paired <= min(group["rna_present"], group["protein_present"]):
            raise ValueError("Extended policy paired denominator is invalid")
        if has_boundary:
            dna = _count(group["dna_eligible"])
            all_three = _count(group["all_three_present_after_boundary"])
            if dna + group["reasons"]["dna_boundary_excluded"] != n or not max(0, paired + dna - n) <= all_three <= min(paired, dna):
                raise ValueError("Extended policy genomic-boundary denominator is invalid")
        elif group["dna_eligible"] is not None or group["all_three_present_after_boundary"] is not None:
            raise ValueError("Extended policy cannot claim an unmeasured genomic window")
        for pair, count in group["overlapping_reason_pairs"].items():
            first, second = pair.split(" & ")
            if first >= second or first not in reason_keys or second not in reason_keys or _count(count) > min(group["reasons"][first], group["reasons"][second]):
                raise ValueError("Extended policy overlap count is invalid")
    if split_totals != report["splits"] or type_totals != {key: group["rows"] for key, group in types.items()}:
        raise ValueError("Extended policy group denominators do not reconcile")

    for label, bounds in LENGTH_BOUNDS.items():
        distribution = checks["length_distributions"][label]
        if distribution["upper_bounds"] != [*bounds, None] or len(distribution["counts"]) != len(bounds) + 1:
            raise ValueError("Extended length distribution has inconsistent fixed bins")
        present, missing = _count(distribution["present"]), _count(distribution["missing"])
        source = report["scopes"]["all"]["stats"].get(f"{label}_length", {})
        if sum(_count(n) for n in distribution["counts"]) != present or present + missing != rows or present != source.get("n", 0):
            raise ValueError("Extended length distribution counts do not reconcile")
        if sum(group[f"{label.lower()}_present"] for group in types.values()) != present:
            raise ValueError("Extended annotation counts disagree with sequence measurements")
        if checks["overlaps"][label]["missing_rows"] != missing:
            raise ValueError("Extended overlap counts disagree with sequence availability")
    representatives = checks["representative_rows"]
    summaries = []
    for key in ("first_rows", "missing_protein_examples"):
        if not isinstance(representatives[key], list) or len(representatives[key]) > EXAMPLE_LIMIT:
            raise ValueError("Extended representative rows exceed the example limit")
        summaries.extend(representatives[key])
    for label in ("RNA", "Protein"):
        summary = representatives[f"longest_{label}"]
        if summary is not None:
            if summary[f"{label}_length"] != report["scopes"]["all"]["stats"][f"{label}_length"]["max"]:
                raise ValueError("Extended representative longest sequence disagrees with statistics")
            summaries.append(summary)
    for summary in summaries:
        if not 1 <= _count(summary["row_number"]) <= rows:
            raise ValueError("Extended representative row number is invalid")
        if any(key in summary for key in ("RNA", "Protein", "CDS", "DNA")):
            raise ValueError("Extended representative rows must not store full sequences")
