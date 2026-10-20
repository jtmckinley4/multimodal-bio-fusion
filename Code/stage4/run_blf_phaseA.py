"""Preview or explicitly execute the fixed BioLangFusion Phase A comparison.

Preview verifies existing token-cache bytes and saved identities; it never builds a model.
Execution is serial and uses the shared trainer's result reuse and checkpoint recovery.
Keep one writer per results directory and leave its source files and caches unchanged.
"""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
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


def preview_runs(streams, runs, device, directory):
    """Classify every request before any fit; mismatched artifacts remain untouched."""
    validate_streams(streams)
    prepared = []
    common_identity = None
    for variant, config in runs:
        identity = train.run_identity(streams, variant, config, device, {})
        common = {key: value for key, value in identity.items() if key not in ("variant", "config")}
        if common_identity is not None and common != common_identity:
            raise RuntimeError("Inputs, code, or environment changed while preparing the grid.")
        common_identity = common
        path = result_path(directory, streams.key, variant, config)
        checkpoint = path.with_suffix(".checkpoint")
        item = {"variant": variant, "config": config, "path": path, "identity": identity,
                "status": "fit_required", "detail": "No completed result or checkpoint."}
        try:
            if path.exists():
                result = train.Result.load(path)
                if (result.identity != identity or result.variant != variant
                        or result.benchmark != streams.key or result.config != asdict(config)):
                    raise ValueError("Completed result differs from the requested computation.")
                validate_completion(result, config, streams.task)
                validate_predictions(result, streams)
                item.update(status="reuse", detail="Matching identity, history, and test predictions.")
            elif checkpoint.exists():
                # Only point this runner at trusted local checkpoints: loading uses pickle.
                saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                required = {"epoch", "model", "optimiser", "schedule", "random", "best", "history", "identity"}
                if not isinstance(saved, dict) or not required <= saved.keys() or saved["identity"] != identity:
                    raise ValueError("Checkpoint is incomplete or belongs to a different computation.")
                random_fields = {"python", "numpy", "torch", "loader"}
                if device in ("cuda", "mps"):
                    random_fields.add(device)
                if not isinstance(saved["random"], dict) or not random_fields <= saved["random"].keys():
                    raise ValueError("Checkpoint random-generator state is incomplete.")
                if (type(saved["epoch"]) is not int or not 0 <= saved["epoch"] < config.max_epochs
                        or len(saved["history"]) != saved["epoch"] + 1):
                    raise ValueError("Checkpoint epoch and history disagree.")
                item.update(status="resume", detail=f"Matching checkpoint through epoch {saved['epoch']} (zero-based).")
        except Exception as error:
            item.update(status="blocked_preserve", detail=str(error))
        prepared.append(item)
    return prepared


def execute_runs(streams, prepared, device, directory):
    """Use the existing trainer sequentially; refuse the whole grid if any item is blocked."""
    if any(item["status"] == "blocked_preserve" for item in prepared):
        raise RuntimeError("Execution blocked; preserve conflicting artifacts and review their destination.")
    for number, item in enumerate(prepared, start=1):
        variant, config = item["variant"], item["config"]
        if train.run_identity(streams, variant, config, device, {}) != item["identity"]:
            raise RuntimeError("The verified computation changed; stop and review before continuing.")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{timestamp}] {number}/{len(prepared)} {item['status']}: {item['path'].name}", flush=True)
        result = train.run(streams, variant, config, device, directory, require_predictions=True,
                           log=lambda message: print(message, flush=True))
        validate_completion(result, config, streams.task)
        validate_predictions(result, streams)


def plan_record(streams, prepared, device, cache_root, results_dir):
    """Record one preview, keeping common identity once and individual settings beside paths."""
    git = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "-C", str(REPO_ROOT),
                          "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", "-C", str(REPO_ROOT),
                             "status", "--short"], capture_output=True, text=True, check=True)
    common = {key: value for key, value in prepared[0]["identity"].items() if key not in ("variant", "config")}
    return {
        "format_version": 1, "benchmark": streams.key, "device": device,
        "repository": str(REPO_ROOT), "git_commit": git.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()), "git_status": status.stdout.splitlines(),
        "source_note": "Git HEAD is checkout context; recorded file hashes identify the source used.",
        "runner_sha256": file_fingerprint(__file__), "identity_common": common,
        "cache_root": str(cache_root), "results_dir": str(results_dir),
        "verification": "Preview of existing cache bytes and local artifacts; no fit implied.",
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
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--plan-output", type=Path, help="Optional new JSON preview record; existing different files are preserved.")
    parser.add_argument("--execute", action="store_true", help="Explicitly run the verified grid; otherwise preview only.")
    args = parser.parse_args(argv)
    device = select_device(args.device)
    runs = planned_runs(train.TrainConfig(), tuple(args.seeds))
    cache_root, results_dir = args.cache_root.resolve(), args.results_dir.resolve()
    print(f"{'EXECUTION REQUESTED' if args.execute else 'PREVIEW ONLY'}: {args.benchmark}, {len(runs)} candidates, {device}.")
    print("Verifying cache bytes may take time; it does not load encoder models or extract embeddings.", flush=True)
    streams = blf.BenchmarkStreams.open(args.benchmark, cache_root, allow_legacy_cache=False)
    try:
        prepared = preview_runs(streams, runs, device, results_dir)
        for item in prepared:
            print(f"{item['status']}: {item['path'].name} | {item['detail']}")
        if args.plan_output:
            write_plan(args.plan_output.resolve(), plan_record(streams, prepared, device, cache_root, results_dir))
        if args.execute:
            execute_runs(streams, prepared, device, results_dir)
        return 1 if any(item["status"] == "blocked_preserve" for item in prepared) else 0
    finally:
        streams.close()


if __name__ == "__main__":
    raise SystemExit(main())
