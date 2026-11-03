"""Inspect the historical BioLangFusion Phase A grid and supply reader helpers.

Preview reads completed JSON records against an explicit historical reference. It
does not open token caches or establish execution reuse. The current shared trainer
does not capture the full input/source/environment identity required by this runner's
historical protocol, so --execute fails before cache or model access. Restoring that
producer contract is a separate integration task; do not bypass it with the trainer's
smaller config/variant/benchmark identity.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd  # Keep the notebook environment's NumPy/pandas-before-PyTorch import order.

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
from shared_code.fusion import blf, train
from shared_code.fusion.token_embeddings import file_fingerprint

EXPECTED_WIDTHS = {"dna": 512, "rna": 640, "prot": 320}
EXPECTED_SPLITS = {"ecoli_proteins": {"train": 3010, "val": 728, "test": 712},
                   "fungal_expression": {"train": 2153, "val": 555, "test": 430}}
CANDIDATES = (
    ("unimodal:dna", 0.0), ("unimodal:rna", 0.0), ("unimodal:prot", 0.0),
    ("concat", 0.0), ("mil", 0.01), ("mil", 0.5), ("mil", 1.0),
    ("mil", 0.0), ("cross", 0.0),
)


def planned_runs(config, seeds=(0, 1, 2)):
    """The nine declared candidates, repeated in the same order for each seed."""
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers.")
    return [(variant, replace(config, seed=seed, entropy_lambda=weight))
            for seed in seeds for variant, weight in CANDIDATES]


def result_path(directory, benchmark, variant, config):
    """Use the shared trainer's readable filename; eligibility still requires full identity."""
    tag = (f"{benchmark}__{variant.replace(':', '-')}__lambda{config.entropy_lambda}"
           f"__ch{config.channels}__{config.tconv}__seed{config.seed}")
    return Path(directory) / f"{tag}.json"


def recorded_identity(result):
    """Read preserved producer metadata without inventing absent historical fields."""
    extra = getattr(result, "extra_fields", None)
    identity = extra.get("identity") if isinstance(extra, dict) else None
    if not isinstance(identity, dict):
        raise ValueError("Missing or invalid recorded producer identity; preserve this legacy result.")
    return identity


def select_device(requested):
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    device = next(name for name in ("cuda", "mps", "cpu") if available[name]) if requested == "auto" else requested
    if not available[device]:
        raise RuntimeError(f"Requested device {device} is unavailable; no backend fallback was made.")
    return device


def validate_streams(streams):
    """Reject invalid labels before GridDataset can silently cast them to integers."""
    expected_task = "classification" if streams.key == "ecoli_proteins" else "regression"
    if streams.task != expected_task or streams.widths != EXPECTED_WIDTHS:
        raise ValueError("Benchmark task or encoder widths differ from the Phase A protocol.")
    splits = streams.rows[streams.split_column]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Exactly the nonempty train, val, and test splits are required.")
    if splits.value_counts().to_dict() != EXPECTED_SPLITS[streams.key]:
        raise ValueError("Retained splits differ from the declared benchmark counts.")
    labels = np.asarray(streams.labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError("Labels must be finite numeric scalars.")
    if streams.task == "classification":
        if not np.equal(labels, np.floor(labels)).all() or set(labels) != {0, 1, 2}:
            raise ValueError("E. coli requires exactly the integer class IDs 0, 1, and 2.")
        if set(labels[streams.split_rows("train")]) != {0, 1, 2} or streams.n_outputs != 3:
            raise ValueError("The training split must contain all three classes, with three outputs.")
    elif streams.n_outputs != 1:
        raise ValueError("Fungal expression requires one regression output.")


def validate_prediction_arrays(result, task, expected_truth):
    """Check stored test arrays against independently identified, ordered test targets."""
    if task not in ("classification", "regression"):
        raise ValueError("Prediction validation supports scalar regression and E. coli classification.")
    if result.test_truth is None or result.test_prediction is None:
        raise ValueError("Per-row truth and predictions are required; preserve score-only results.")
    truth, predicted = np.asarray(result.test_truth), np.asarray(result.test_prediction)
    # GridDataset casts regression labels to float32 before the trainer sees them.
    dtype = np.int64 if task == "classification" else np.float32
    expected = np.asarray(expected_truth, dtype=dtype)
    if expected.size == 0 or truth.shape != expected.shape or predicted.shape != expected.shape:
        raise ValueError("Stored arrays must match the nonempty ordered test-label shape.")
    if any(not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all()
           for values in (truth, predicted)):
        raise ValueError("Stored test arrays must contain finite numeric values.")
    if not np.array_equal(truth, expected):
        raise ValueError("Stored truth differs from the current ordered test labels.")
    if task == "classification" and (
            not np.equal(predicted, np.floor(predicted)).all() or not set(predicted) <= {0, 1, 2}):
        raise ValueError("Stored classification predictions must be class IDs 0, 1, or 2.")
    measured = train.score_rows(task, truth, predicted)
    if not np.isfinite(measured) or not np.isclose(measured, result.test_score, rtol=0, atol=1e-12):
        raise ValueError("Stored test score disagrees with the per-row predictions.")


def validate_predictions(result, streams):
    """Use the same validation rule with the runner's verified stream row order."""
    validate_prediction_arrays(result, streams.task, streams.labels[streams.split_rows("test")])


def validate_completion(result, config, task):
    """Check completion and first-best validation selection without refitting."""
    if type(result.epochs) is not int or not 1 <= result.epochs <= config.max_epochs:
        raise ValueError("Invalid completed epoch count.")
    if len(result.history) != result.epochs or [row["epoch"] for row in result.history] != list(range(result.epochs)):
        raise ValueError("Completed history is inconsistent with its epoch count.")
    scores = [row["val_score"] for row in result.history]
    lower = 0.0 if task == "classification" else -1.0
    if any(value is not None and (type(value) not in (float, int) or not np.isfinite(value)
                                 or not lower <= value <= 1.0) for value in scores):
        raise ValueError("Invalid validation history.")
    comparable = [-np.inf if value is None else value for value in scores]
    if (type(result.best_epoch) is not int or result.best_epoch != int(np.argmax(comparable))
            or result.val_score != comparable[result.best_epoch] or not np.isfinite(result.val_score)):
        raise ValueError("Saved validation selection does not match the first best epoch.")


COHORT_FIELDS = (
    "version", "model_kwargs", "widths", "n_outputs", "inputs", "environment", "code_sha256",
)
EXECUTION_BLOCKED = (
    "Phase A execution is disabled: the current trainer does not capture this runner's "
    "required full input/source/environment identity. Keep existing results and checkpoints; "
    "integrate that producer contract before starting or resuming a fit. Metadata preview "
    "remains available without --execute."
)


def _historical_cohort(result, benchmark, variant, config):
    """Require the declared candidate and preserved full-cohort fields for inspection."""
    if (result.benchmark != benchmark or result.variant != variant
            or result.config != asdict(config)):
        raise ValueError("Saved benchmark, variant or full training recipe differs.")
    identity = recorded_identity(result)
    if identity.get("version") != 1 or any(key not in identity for key in COHORT_FIELDS):
        raise ValueError("Incomplete or unsupported recorded producer identity.")
    if identity.get("variant") != variant or identity.get("config") != asdict(config):
        raise ValueError("Saved result and producer identity disagree.")
    if (identity["widths"] != EXPECTED_WIDTHS or identity["model_kwargs"] != {}
            or identity["n_outputs"] != (3 if benchmark == "ecoli_proteins" else 1)):
        raise ValueError("Recorded model dimensions or arguments differ from the protocol.")
    if any(not isinstance(identity[key], dict) or not identity[key]
           for key in ("inputs", "environment", "code_sha256")):
        raise ValueError("Recorded input, environment or source identity is missing.")
    json.dumps(identity, allow_nan=False)
    task = "classification" if benchmark == "ecoli_proteins" else "regression"
    validate_completion(result, config, task)
    lower = 0.0 if task == "classification" else -1.0
    if (type(result.test_score) not in (int, float) or not np.isfinite(result.test_score)
            or not lower <= result.test_score <= 1.0):
        raise ValueError("Invalid saved test score.")
    return {key: identity[key] for key in COHORT_FIELDS}


def preview_runs(benchmark, runs, directory, reference_path=None):
    """Inventory historical records; no cache reads, fits or resume eligibility claims.

    The default reference is the first declared native-DNA candidate/seed. Passing
    another reference is explicit. Matching recorded fields is an inspection of the
    producer's account, not independent verification of current input/cache bytes.
    Checkpoint presence remains unresolved instead of parsing it as a current resume.
    """
    reference_path = Path(reference_path) if reference_path else result_path(
        directory, benchmark, *runs[0])
    reference = None
    reference_issue = ""
    try:
        record = train.Result.load(reference_path)
        matching = [(variant, config) for variant, config in runs
                    if variant == record.variant and asdict(config) == record.config]
        if len(matching) != 1:
            raise ValueError("Reference is outside the declared candidate/seed grid.")
        reference = _historical_cohort(record, benchmark, *matching[0])
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        reference_issue = str(error)

    prepared = []
    for variant, config in runs:
        path = result_path(directory, benchmark, variant, config)
        item = {"variant": variant, "config": config, "path": path,
                "status": "not_found_locally", "detail": "No completed result or checkpoint here."}
        if path.is_file():
            try:
                result = train.Result.load(path)
                cohort = _historical_cohort(result, benchmark, variant, config)
                if reference is None:
                    item.update(status="reference_pending", detail=reference_issue,
                                val_score=result.val_score, test_score=result.test_score,
                                best_epoch=result.best_epoch)
                elif cohort != reference:
                    raise ValueError("Recorded producer cohort differs from the explicit reference.")
                else:
                    item.update(
                        status="recorded_evidence_available",
                        val_score=result.val_score, test_score=result.test_score,
                        best_epoch=result.best_epoch,
                        detail="Stored recipe/cohort/history match; input/cache bytes and paired arrays "
                               "are not independently verified by this metadata preview.")
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                item.update(status="blocked_preserve", detail=str(error))
        elif path.with_suffix(".checkpoint").is_file():
            item.update(status="checkpoint_requires_review",
                        detail="Checkpoint preserved; this metadata preview does not establish resumability.")
        prepared.append(item)
    return prepared


def execute_runs(*args, **kwargs):
    """Retain a clear refusal for callers of the old producer entry point."""
    raise RuntimeError(EXECUTION_BLOCKED)


def plan_record(benchmark, prepared, cache_root, results_dir, reference_path):
    """Record inspection separately from producer identity and execution approval."""
    return {
        "format_version": 2, "benchmark": benchmark,
        "kind": "historical Phase A metadata inspection", "execution_enabled": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": str(REPO_ROOT), "reference_path": str(reference_path),
        "runner_sha256": file_fingerprint(__file__),
        "cache_root": str(cache_root), "cache_bytes_verified": False,
        "results_dir": str(results_dir),
        "verification": "Stored candidate/cohort metadata and history only; no fit or resume implied.",
        "runs": [{"variant": item["variant"], "config": asdict(item["config"]),
                  "path": str(item["path"]), "status": item["status"], "detail": item["detail"]}
                 for item in prepared],
    }


def write_plan(path, record):
    """Publish an explicit plan output; retain a different existing plan for its history."""
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise FileExistsError(f"{path} contains a different plan; choose a new output path.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("ecoli_proteins", "fungal_expression"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True,
                        help="Recorded location only; this preview does not open caches.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto",
                        help="Retained CLI option; metadata inspection does not select a device.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--reference-result", type=Path,
                        help="Explicit historical reference; defaults to the first declared candidate.")
    parser.add_argument("--plan-output", type=Path, help="Optional new JSON inspection record.")
    parser.add_argument("--execute", action="store_true",
                        help="Currently refused until the full producer-identity contract is integrated.")
    args = parser.parse_args(argv)
    # Refuse before streams, devices, checkpoints or output destinations are touched.
    if args.execute:
        raise RuntimeError(EXECUTION_BLOCKED)
    runs = planned_runs(train.TrainConfig(), tuple(args.seeds))
    cache_root, results_dir = args.cache_root.resolve(), args.results_dir.resolve()
    reference = (args.reference_result.resolve() if args.reference_result else
                 result_path(results_dir, args.benchmark, *runs[0]))
    print(f"METADATA INSPECTION ONLY: {args.benchmark}, {len(runs)} candidates.")
    print("Caches are not opened; this preview does not authorize or establish execution reuse.")
    prepared = preview_runs(args.benchmark, runs, results_dir, reference)
    for item in prepared:
        print(f"{item['status']}: {item['path'].name} | {item['detail']}")
    if args.plan_output:
        write_plan(args.plan_output.resolve(), plan_record(
            args.benchmark, prepared, cache_root, results_dir, reference))
    return 1 if any(item["status"] in {"blocked_preserve", "reference_pending"}
                    for item in prepared) else 0


if __name__ == "__main__":
    raise SystemExit(main())
