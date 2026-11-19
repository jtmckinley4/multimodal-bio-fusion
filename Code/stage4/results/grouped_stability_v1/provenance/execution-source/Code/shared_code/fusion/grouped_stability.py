"""Exact-sequence stability partitions and the bounded BLF comparison panel.

The notebook owns the experiment and execution switches. These helpers keep group
assignment, saved evidence and comparison eligibility consistent without changing
the supplied dataset, cache order, historical partitions or completed fits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import run_identity, swap_results


PROTOCOL = "stability-exact-sequence-v1"
SPLIT_COLUMN = "GroupedSplit"
SPLITS = ("train", "val", "test")
TRAINING_SEEDS = (0, 1, 2)
BUNDLES = {
    "original": {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"},
    "mrnafm": {"dna": "ntv2_100m", "rna": "mrnafm", "prot": "esm2_8m"},
}


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def canonical_sequence(text):
    """Normalize identity only; retain the original input strings for cache reuse."""
    if not isinstance(text, str):
        raise ValueError("A sequence group requires a nucleotide string.")
    sequence = text.strip().upper().replace("U", "T")
    if not sequence or set(sequence) - set("ACGT"):
        raise ValueError("Sequence groups require nonempty A/C/G/T/U strings.")
    return sequence


def _assignment(group, seed):
    # A stable hash avoids changing partitions when library versions or row order change.
    bucket = int(hashlib.sha256(f"{seed}:stability:{group}".encode()).hexdigest(), 16) % 1000
    return "train" if bucket < 700 else "val" if bucket < 850 else "test"


def _valid_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and not set(value) - set("0123456789abcdef"))


def build_split(rows, *, source_sha256, split_seed=42):
    """Assign whole sequence groups; do not reorder, aggregate, crop or relabel rows."""
    if type(split_seed) is not int or split_seed < 0 or not _valid_sha(source_sha256):
        raise ValueError("Provide a nonnegative split seed and the source CSV SHA-256.")
    required = {"sequence", "length", "Value", "Split"}
    if not required <= set(rows.columns) or rows.empty or not rows.index.is_unique:
        raise ValueError("Retained rows need sequence/length/Value/Split and unique source indices.")
    if any(type(index) not in (int, np.int32, np.int64) for index in rows.index):
        raise ValueError("Preserve the CSV's integer source indices.")
    sequences = [canonical_sequence(text) for text in rows["sequence"]]
    lengths = rows["length"].to_numpy()
    if any(len(sequence) != length or not 0 < length <= 999
           for sequence, length in zip(sequences, lengths)):
        raise ValueError("Use the existing retained cohort with unchanged lengths <=999 nt.")
    if not set(rows["Split"]) <= set(SPLITS):
        raise ValueError("Unexpected supplied split labels.")
    labels = rows["Value"].to_numpy(dtype=np.float32)
    if not np.isfinite(labels).all():
        raise ValueError("Targets must be finite at training-loader precision.")
    groups = [hashlib.sha256(text.encode()).hexdigest() for text in sequences]
    assignments = [_assignment(group, split_seed) for group in groups]
    positions = {name: np.flatnonzero(np.array(assignments) == name) for name in SPLITS}
    if any(not len(indices) for indices in positions.values()):
        raise ValueError("A partition received no rows; do not silently choose another seed.")
    if any(len(np.unique(labels[positions[name]])) < 2 for name in ("val", "test")):
        raise ValueError("Validation/test targets cannot be constant for Spearman selection.")
    prepared = rows.copy()
    prepared[SPLIT_COLUMN] = assignments
    row_values = rows[["sequence", "length"]].astype(str).to_numpy().tolist()
    record = {
        "schema": "stability.grouped-split.v1", "protocol": PROTOCOL,
        "split_seed": split_seed, "fractions": [0.70, 0.15, 0.15],
        "assignment": "sha256(seed:stability:sha256(canonical_sequence)) mod 1000; <700/<850",
        "canonicalization": "strip, uppercase, U->T for identity only",
        "source_sha256": source_sha256,
        "source_indices": [int(index) for index in rows.index],
        "groups": groups, "partitions": assignments,
        "rows": {"columns": ["sequence", "length"], "count": len(rows),
                 "sha256": _digest(row_values)},
        "supplied_targets": run_identity.array_record(rows["Value"].to_numpy(dtype=np.float64)),
        "targets": {**run_identity.array_record(labels), "columns": ["Value"],
                    "task": "regression", "transform": "as_supplied"},
        "splits": {name: run_identity.array_record(indices, np.int64)
                   for name, indices in positions.items()},
        "split_targets": {name: run_identity.array_record(labels[indices])
                          for name, indices in positions.items()},
        "test_truth": labels[positions["test"]].tolist(),
        "counts": {name: {"rows": len(indices),
                          "groups": len({groups[i] for i in indices})}
                   for name, indices in positions.items()},
    }
    record["fingerprint"] = _digest(record)
    return prepared, record


def validate_split(record, *, split_seed=42):
    """Validate a saved assignment without opening the CSV or choosing new groups."""
    if not isinstance(record, dict):
        raise ValueError("The split record must be an object.")
    payload = {key: value for key, value in record.items() if key != "fingerprint"}
    if record.get("fingerprint") != _digest(payload):
        raise ValueError("Split-record fingerprint differs; preserve the existing record.")
    if (record.get("schema") != "stability.grouped-split.v1"
            or record.get("protocol") != PROTOCOL or record.get("split_seed") != split_seed
            or record.get("fractions") != [0.70, 0.15, 0.15]
            or not _valid_sha(record.get("source_sha256"))):
        raise ValueError("Split record belongs to a different protocol or source identity.")
    groups, partitions = record["groups"], record["partitions"]
    count = record["rows"]["count"]
    if (not count or len(groups) != count or len(partitions) != count
            or len(record["source_indices"]) != count
            or len(set(record["source_indices"])) != count
            or any(not _valid_sha(group) for group in groups)
            or partitions != [_assignment(group, split_seed) for group in groups]):
        raise ValueError("Saved row/group assignments are inconsistent.")
    for name in SPLITS:
        positions = np.flatnonzero(np.asarray(partitions) == name)
        expected = {"rows": len(positions), "groups": len({groups[i] for i in positions})}
        if (not len(positions) or record["counts"][name] != expected
                or record["splits"][name] != run_identity.array_record(positions, np.int64)):
            raise ValueError("Saved split counts or positions differ.")
    if run_identity.array_record(np.asarray(record["test_truth"], dtype=np.float32)) != record["split_targets"]["test"]:
        raise ValueError("Saved test targets differ from their recorded identity.")
    return record


def read_split(path, *, split_seed=42):
    return validate_split(json.loads(Path(path).read_text(encoding="utf-8")), split_seed=split_seed)


def publish_split(path, record):
    """Publish once; concurrent seed jobs may reuse only the exact same assignment."""
    validate_split(record, split_seed=record["split_seed"])
    path = Path(path)
    if path.exists():
        if read_split(path, split_seed=record["split_seed"]) != record:
            raise ValueError("Existing split differs; preserve it and use a new experiment root.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".grouped-split-", suffix=".partial",
                                         delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
        try:
            os.link(temporary, path)
        except FileExistsError:
            if read_split(path, split_seed=record["split_seed"]) != record:
                raise ValueError("Another writer published a different split; preserve both inputs.")
            return False
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_result_destination(root, cache_root):
    """Refuse a cache path or a directory containing unrelated historical evidence."""
    root, cache_root = Path(root).resolve(), Path(cache_root).resolve()
    if root.is_relative_to(cache_root) or cache_root.is_relative_to(root):
        raise ValueError("Grouped results and embedding caches must not overlap.")
    if root.exists():
        entries = set()
        for path in root.iterdir():
            # Another seed may be publishing the same split. Only our temporary
            # files are allowed; unrelated partial files/directories still fail.
            publishing = path.name.startswith(".grouped-split-") and path.name.endswith(".partial")
            if publishing and not path.is_symlink() and (path.is_file() or not path.exists()):
                continue
            entries.add(path.name)
        unexpected = entries - {"split.json", "fits", "reports"}
        if unexpected or (not (root / "split.json").exists() and entries):
            raise ValueError("Use a new grouped-experiment directory, not historical results.")


@dataclass(frozen=True)
class FitSpec:
    key: str
    variant: str
    bundle: str

    @property
    def encoders(self):
        return dict(BUNDLES[self.bundle])

    @property
    def modalities(self):
        return ((self.variant.split(":", 1)[1],) if self.variant.startswith("unimodal")
                else ("dna", "rna", "prot"))


def fit_panel():
    """Twelve fits per seed; shared single-stream controls appear exactly once."""
    singles = (
        FitSpec("ntv2_native", "unimodal:dna", "original"),
        FitSpec("ntv2_grid", "unimodal-grid:dna", "original"),
        FitSpec("rnafm_native", "unimodal:rna", "original"),
        FitSpec("rnafm_grid", "unimodal-grid:rna", "original"),
        FitSpec("mrnafm_native", "unimodal:rna", "mrnafm"),
        FitSpec("esm2_native", "unimodal:prot", "original"),
    )
    return singles + tuple(FitSpec(f"{bundle}_{variant}", variant, bundle)
                           for bundle in BUNDLES for variant in ("concat", "mil", "cross"))


def config_for(spec, base_config, seed):
    if type(seed) is not int or seed not in TRAINING_SEEDS:
        raise ValueError("This panel declares training seeds 0, 1 and 2 only.")
    return replace(base_config, seed=seed, entropy_lambda=0.01 if spec.variant == "mil" else 0.0)


def fit_path(root, spec, seed):
    return Path(root) / "fits" / spec.key / f"seed{seed}.json"


def make_identity(rows, split_record, spec, config, cache_records):
    validate_split(split_record, split_seed=split_record["split_seed"])
    identity = run_identity.blf_identity(
        rows=rows, benchmark="mrna_stability", variant=spec.variant, config=config,
        encoders=spec.encoders, caches={m: cache_records[k] for m, k in spec.encoders.items()},
        split_column=SPLIT_COLUMN,
    )
    for field in ("rows", "targets", "splits", "split_targets"):
        if identity["inputs"][field] != split_record[field]:
            raise ValueError(f"Prepared {field} differ from the saved grouped split.")
    identity["inputs"]["grouped_stability"] = {
        "protocol": PROTOCOL, "split_fingerprint": split_record["fingerprint"],
        "source_sha256": split_record["source_sha256"], "fit_key": spec.key,
    }
    return run_identity.validate_identity(identity)


def validate_result(data, split_record, spec, config):
    """Require the new computation identity and actual aligned saved predictions."""
    if (data.get("variant") != spec.variant or data.get("benchmark") != "mrna_stability"
            or data.get("config") != asdict(config)):
        raise ValueError("Saved variant, benchmark or full training configuration differs.")
    if (type(data.get("epochs")) is not int or data["epochs"] < 1
            or type(data.get("best_epoch")) is not int
            or not 0 <= data["best_epoch"] < data["epochs"]):
        raise ValueError("Missing or invalid completed-fit epochs/best_epoch.")
    for field in ("test_score", "val_score"):
        value = data.get(field)
        # Training can record an undefined validation score; such a single cannot
        # win validation selection. Missing score fields are malformed records.
        if field == "val_score" and field in data and value is None:
            continue
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError(f"Missing or invalid completed-fit {field}.")
    identity = run_identity.validate_identity(
        data.get("identity"), config=asdict(config), variant=spec.variant, benchmark="mrna_stability",
    )
    run_identity.validate_model_request(identity)
    if (identity["loader"] != run_identity.blf_loader_spec(config, spec.modalities)
            or identity["model"].get("contract") != "blf-frozen-codon-grid-v1"
            or identity["model"].get("modalities") != list(spec.modalities)
            or identity["model"].get("n_outputs") != 1
            or identity["model"].get("textcnn") != {"kernel_sizes": [3, 4, 5], "dropout": 0.2}
            or identity["model"].get("numerics") != {
                "inputs": "float32", "parameters": "float32", "amp": False}):
        raise ValueError("Saved model or loader contract differs from the declared panel.")
    inputs = run_identity.recorded_blf_inputs(data)
    if inputs["encoders"] != spec.encoders:
        raise ValueError("Saved encoder mapping differs.")
    expected = {"protocol": PROTOCOL, "split_fingerprint": split_record["fingerprint"],
                "source_sha256": split_record["source_sha256"], "fit_key": spec.key}
    if identity["inputs"].get("grouped_stability") != expected:
        raise ValueError("Saved result belongs to another grouped experiment.")
    for field in ("rows", "targets", "splits", "split_targets"):
        if inputs[field] != split_record[field]:
            raise ValueError(f"Saved {field} do not match the grouped split.")
    issue = swap_results.prediction_issue(data, split_record["test_truth"], task="regression")
    if issue:
        raise ValueError(issue)
    return data


def read_results(root, split_record, base_config, seeds=TRAINING_SEEDS, validator=None):
    """Read expected paths only; missing/incompatible records never trigger a fit."""
    validate_split(split_record, split_seed=split_record["split_seed"])
    records, issues = {}, []
    for seed in seeds:
        for spec in fit_panel():
            path = fit_path(root, spec, seed)
            if not path.is_file():
                issues.append({"fit": spec.key, "seed": seed, "issue": "missing"})
                continue
            try:
                payload = path.read_bytes()
                data = json.loads(payload)
                validate_result(data, split_record, spec, config_for(spec, base_config, seed))
                if validator is not None:
                    validator(data)
                records[spec.key, seed] = {
                    "data": data, "path": str(path.resolve()),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            except (ValueError, TypeError, KeyError, OSError) as error:
                issues.append({"fit": spec.key, "seed": seed, "issue": str(error)})
    return records, issues


def selection_and_comparisons(records, selection_seed=0):
    """Select on the new seed-0 validation scores, never on a test score or old winner."""
    if selection_seed != 0:
        raise ValueError("The agreed baseline-selection seed is zero.")
    candidates = {
        "original": ("ntv2_native", "ntv2_grid", "rnafm_native", "rnafm_grid", "esm2_native"),
        "mrnafm": ("ntv2_native", "ntv2_grid", "mrnafm_native", "esm2_native"),
        "all": tuple(spec.key for spec in fit_panel() if spec.variant.startswith("unimodal")),
    }
    for key in candidates["all"]:
        entry = records.get((key, selection_seed))
        value = None if entry is None else entry["data"].get("val_score")
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError("Complete, finite seed-0 single-input validation scores are required.")
    selected = {bundle: max(keys, key=lambda key: records[key, selection_seed]["data"]["val_score"])
                for bundle, keys in candidates.items()}
    comparisons = []
    for bundle in BUNDLES:
        for variant in ("concat", "mil", "cross"):
            for baseline in (bundle, "all"):
                comparisons.append({
                    "encoder": bundle, "kind": "complementarity",
                    "comparison": f"{variant} vs selected single ({baseline})",
                    "first": f"{bundle}_{variant}", "second": selected[baseline],
                })
        for variant in ("mil", "cross"):
            comparisons.append({
                "encoder": bundle, "kind": "method", "comparison": f"{variant} vs concat",
                "first": f"{bundle}_{variant}", "second": f"{bundle}_concat",
            })
    comparisons.append({"encoder": "mrnafm", "kind": "representation",
                        "comparison": "mRNA-FM single vs RNA-FM single",
                        "first": "mrnafm_native", "second": "rnafm_native"})
    return selected, comparisons


def report_bundle(records, issues, base_config, seeds=TRAINING_SEEDS):
    """Adapt validated saved records to the existing paired-report calculation."""
    expected_keys = {(spec.key, seed) for spec in fit_panel() for seed in TRAINING_SEEDS}
    if issues or tuple(seeds) != TRAINING_SEEDS or set(records) != expected_keys:
        raise ValueError("All 36 compatible fits are required before the three-seed report.")
    selected, comparisons = selection_and_comparisons(records)
    models = {spec.key: {seed: records[spec.key, seed] for seed in seeds} for spec in fit_panel()}
    return {
        "benchmark": "mrna_stability", "config": asdict(base_config), "seeds": tuple(seeds),
        "models": models, "best_singles": selected, "mil_lambda": 0.01,
        "comparisons": comparisons, "issues": [],
        "sources": {entry["path"]: entry["sha256"] for entry in records.values()},
    }


def report_context(split_record):
    return {**{field: split_record[field] for field in ("rows", "targets", "splits", "split_targets")},
            "grouped_stability": {"protocol": PROTOCOL,
                                  "split_fingerprint": split_record["fingerprint"],
                                  "selection_policy": "fixed-new-validation-seed0"}}


def read_report(root, records, split_record, base_config, *, draws=2000, bootstrap_seed=0):
    """Read a matching report, validating returned fit hashes without rescoring."""
    expected_keys = {(spec.key, seed) for spec in fit_panel() for seed in TRAINING_SEEDS}
    if set(records) != expected_keys:
        return None, None
    selected, comparisons = selection_and_comparisons(records)
    test_positions = np.flatnonzero(np.asarray(split_record["partitions"]) == "test")
    groups = np.asarray(split_record["groups"])[test_positions].tolist()
    expected = {"schema": "swap.paired-report.v1", "benchmark": "mrna_stability",
                "config": asdict(base_config), "seeds": list(TRAINING_SEEDS),
                "selection_seed": 0, "best_singles": selected, "mil_lambda": 0.01,
                "comparisons": comparisons, "context": report_context(split_record),
                "draws": draws, "bootstrap_seed": bootstrap_seed,
                "truth": run_identity.array_record(np.asarray(split_record["test_truth"], dtype=np.float32)),
                "groups_sha256": _digest(groups), "task": "regression",
                "scoring_target_precision": "float32", "n_classes": None}
    eligible = []
    for path in sorted((Path(root) / "reports").glob("mrna_stability__*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        identity = saved.get("identity", {})
        if any(identity.get(field) != value for field, value in expected.items()):
            continue
        if (not isinstance(saved.get("analysis"), dict)
                or saved.get("analysis_sha256") != _digest(saved["analysis"])):
            raise ValueError(f"Saved report is corrupt; preserve {path}.")
        actual_sources = {}
        for source, digest in identity.get("sources", {}).items():
            normalized = source.replace("\\", "/")
            matches = [(key, seed) for key, seed in records
                       if normalized.endswith(f"/fits/{key}/seed{seed}.json")]
            if len(matches) != 1 or matches[0] in actual_sources:
                raise ValueError("Report has an unexpected or ambiguous fit source.")
            actual_sources[matches[0]] = digest
        if actual_sources != {key: entry["sha256"] for key, entry in records.items()}:
            raise ValueError("Report fit hashes differ from the returned records.")
        eligible.append((saved, path))
    if len(eligible) > 1:
        raise ValueError("Several reports match; preserve them and select a distinct analysis directory.")
    return eligible[0] if eligible else (None, None)
