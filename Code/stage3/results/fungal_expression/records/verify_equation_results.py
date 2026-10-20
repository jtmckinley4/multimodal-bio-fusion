"""Verify the twelve retained fungal Equation 1 results without fitting a model.

Run from any directory: python -B path/to/verify_equation_results.py
Each invocation writes a new report beside this script. Original results stay unchanged.
This reproduces a historical analysis; it does not certify execution provenance or
make legacy records eligible for the trainer's strict resume path.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd  # Preserve the project's NumPy/pandas-before-PyTorch import order.
import scipy

REPO_ROOT = Path(__file__).resolve().parents[5]
CODE_DIR = REPO_ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "stage4"))

from shared_code.datasets import blf_benchmark_rows
from shared_code.fusion import train
from run_blf_phaseA import result_path, validate_completion, validate_prediction_arrays

RESULT_DIR = Path(__file__).resolve().parent.parent
HISTORICAL_COMMIT = "8b8dd12cc24ee9a59257c7910921f92d24fa120e"
HISTORICAL_NOTEBOOK = "Code/stage3/03_blf_build_check.ipynb"
BENCHMARK = "fungal_expression"
SEEDS = (0, 1, 2)
VARIANTS = ("unimodal:dna", "concat", "mil", "cross")
COMPARISONS = (
    ("concat", "unimodal:dna"), ("mil", "unimodal:dna"), ("cross", "unimodal:dna"),
    ("cross", "mil"), ("cross", "concat"), ("mil", "concat"),
)
# Rounded values in the original stage3-seed-sweep run record, not expected new results.
REPORTED_COMPARISONS = {
    ("concat", "unimodal:dna"): (-0.015, -0.049, 0.017),
    ("mil", "unimodal:dna"): (-0.005, -0.041, 0.031),
    ("cross", "unimodal:dna"): (0.016, -0.014, 0.048),
    ("cross", "mil"): (0.022, 0.009, 0.036),
    ("cross", "concat"): (0.031, 0.012, 0.053),
}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_output(*arguments):
    """Read local Git evidence; never fetch, stage, or commit."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}", *arguments],
        cwd=REPO_ROOT, text=True, encoding="utf-8",
    ).strip()


def verify_record(path, variant, config, expected_truth):
    """Require the named configuration, valid history, and matching ordered targets."""
    if not isinstance(json.loads(Path(path).read_text(encoding="utf-8")), dict):
        raise ValueError("A result must be a JSON object.")
    result = train.Result.load(path)
    if (result.benchmark != BENCHMARK or result.variant != variant
            or result.config != asdict(config)):
        raise ValueError("Result metadata differs from the explicitly requested configuration.")
    if np.asarray(expected_truth).ndim != 1:
        raise ValueError("This historical comparison requires one scalar target per test row.")
    validate_completion(result, config, "regression")
    validate_prediction_arrays(result, "regression", expected_truth)
    return result


def seed_average_bootstrap(truth, predictions, draws=4000, seed=0):
    """Pair rows across all models; average three Spearman scores inside each draw.

    This is the historical noise4 cell's method. Seeds remain fixed fitted models,
    not independent test observations. No predictions are averaged before scoring.
    """
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be a positive integer.")
    truth = np.asarray(truth)
    if truth.ndim != 1 or not len(truth) or not np.isfinite(truth).all():
        raise ValueError("Test targets must be a nonempty finite vector.")
    if set(predictions) != set(VARIANTS):
        raise ValueError("Exactly the four declared variants are required.")
    for variant in VARIANTS:
        if len(predictions[variant]) != len(SEEDS):
            raise ValueError(f"{variant}: all three seed predictions are required.")
        if any(np.asarray(values).shape != truth.shape or not np.isfinite(values).all()
               for values in predictions[variant]):
            raise ValueError(f"{variant}: predictions must match the finite test-target vector.")

    def averaged_scores(indices):
        scores = np.array([
            np.mean([train.score_rows("regression", truth[indices], values[indices])
                     for values in predictions[variant]])
            for variant in VARIANTS
        ])
        if not np.isfinite(scores).all():
            raise ValueError("An observed or resampled Spearman score is undefined; no intervals.")
        return scores

    observed = averaged_scores(np.arange(len(truth)))
    rng = np.random.default_rng(seed)
    sampled_scores = np.empty((draws, len(VARIANTS)))
    for draw in range(draws):
        indices = rng.integers(0, len(truth), len(truth))
        try:
            sampled_scores[draw] = averaged_scores(indices)
        except ValueError as error:
            raise ValueError(f"Bootstrap draw {draw}: {error}") from error

    comparisons = []
    for first, second in COMPARISONS:
        a, b = VARIANTS.index(first), VARIANTS.index(second)
        differences = sampled_scores[:, a] - sampled_scores[:, b]
        low, high = np.percentile(differences, [2.5, 97.5])
        comparisons.append({
            "first": first, "second": second, "difference": float(observed[a] - observed[b]),
            "ci_low": float(low), "ci_high": float(high),
            "fraction_above_zero": float((differences > 0).mean()),
        })
    return comparisons


def main():
    report = {
        "purpose": "Verify saved fungal Equation 1 numbers; no training or encoder inference.",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "incomplete", "errors": [], "inputs": [], "comparisons": [],
        "strict_execution_resume_qualified": False,
        "limits": [
            "Matching target values does not prove sequence-row identity when labels repeat.",
            "Dataset and ordered-row hashes are measured now, not historical producer identities.",
            "Saved Git source is not proof of the execution-time code or environment.",
            "History consistency does not prove predictions came from the best checkpoint.",
            "These records lack execution identities; encoder/cache/producer provenance stays unknown.",
            "Intervals condition on this split, fixed configurations and three fitted seeds.",
            "Rows, not seeds, are resampled; training uncertainty is not eliminated or fully estimated.",
            "The historical row bootstrap does not account for biological relatedness among rows.",
            "Six comparisons use unadjusted percentile intervals; the positive fraction is not a p-value.",
            "Comparison is against DNA; this verification does not certify a best-baseline selection rule.",
        ],
    }
    source_paths = [
        Path(__file__), CODE_DIR / "stage4/run_blf_phaseA.py",
        CODE_DIR / "shared_code/datasets.py", CODE_DIR / "shared_code/fusion/train.py",
    ]
    protected = {path: fingerprint(path) for path in source_paths}
    try:
        report["analysis_code"] = {
            "head": git_output("rev-parse", "HEAD"),
            "working_tree_changes": git_output("status", "--short"),
            "sources": {path.relative_to(REPO_ROOT).as_posix(): value
                        for path, value in protected.items()},
        }
        report["versions"] = {
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "torch_imported_for_existing_validators": train.torch.__version__,
        }
        # Inspect historical source only. In particular, never execute the training cell.
        notebook = json.loads(git_output("show", f"{HISTORICAL_COMMIT}:{HISTORICAL_NOTEBOOK}"))
        report["historical_method"] = {
            "commit": HISTORICAL_COMMIT, "path": HISTORICAL_NOTEBOOK,
            "cells": [{"id": cell["id"], "source_sha256": hashlib.sha256(
                "".join(cell["source"]).encode("utf-8")).hexdigest(),
                "saved_outputs_sha256": hashlib.sha256(json.dumps(
                    cell.get("outputs", []), sort_keys=True).encode("utf-8")).hexdigest()}
                for cell in notebook["cells"]
                if cell.get("id") in ("tconv4", "noise2", "noise3", "noise4")],
            "draws": 4000, "random_seed": 0, "percentiles": [2.5, 97.5],
            "seed_order": list(SEEDS),
            "statistic": "Mean of per-seed Spearman scores, then difference between models.",
            "source_output_mismatch": (
                "noise4 source averages scores across seeds, but its saved interval table is per-seed. "
                "This report executes the saved method independently; it does not refresh that notebook."
            ),
        }
        if len(report["historical_method"]["cells"]) != 4:
            raise ValueError("The four expected historical method/selection cells were not found.")
        report["selection_evidence"] = {
            "mil": (
                "tconv4 source selects the highest validation score among positive weights, and its "
                "saved output reports Equation lambda=0.01. noise3 fixes that weight across seeds. "
                "Prediction reruns replaced selected seed-0 files; the original lambda=0.01 "
                "validation value is absent from the inspected 86c942e and 8b8dd12 snapshots."
            ),
            "baseline": (
                "noise3 hard-codes DNA. Original native validation scores rank DNA above protein "
                "and RNA, but the earlier verdict uses test scores for best_single. A validation-only "
                "selection procedure is not established."
            ),
            "reading_route": "docs/run-records.md#fungal-equation-1-saved-result-verification",
        }

        dataset_path = REPO_ROOT / "datasets/Fungal_expression.csv"
        protected[dataset_path] = fingerprint(dataset_path)
        rows = blf_benchmark_rows(BENCHMARK, split="test")
        if len(rows) != 430:
            raise ValueError(f"Expected 430 retained fungal test rows, found {len(rows)}.")
        expected_truth = rows["Value"].to_numpy(dtype=np.float32)
        # This new fingerprint identifies today's ordered input, without backfilling old JSON.
        ordered_rows = [
            [int(index), row["sequence"], float(row["Value"]), row["Split"]]
            for index, row in rows.iterrows()
        ]
        report["dataset"] = {
            "path": dataset_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": protected[dataset_path], "test_rows": len(rows),
            "rules": "Registry file order; nonmissing, stripped uppercase sequence; <=999 nt; Split=test.",
            "target_dtype": "float32",
            "ordered_rows_sha256": hashlib.sha256(json.dumps(
                ordered_rows, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest(),
            "ordered_rows_fields": ["original_csv_dataframe_index", "sequence", "Value", "Split"],
        }
        results = {variant: [] for variant in VARIANTS}
        for variant in VARIANTS:
            for seed in SEEDS:
                config = train.TrainConfig(
                    lr=3e-5, weight_decay=1e-5, batch_size=32, max_epochs=500,
                    early_stopping_patience=20, plateau_patience=5,
                    entropy_lambda=0.01 if variant == "mil" else 0.0,
                    channels=100, tconv="equation", seed=seed,
                )
                path = result_path(RESULT_DIR, BENCHMARK, variant, config)
                entry = {"path": path.relative_to(REPO_ROOT).as_posix(),
                         "variant": variant, "config": asdict(config)}
                report["inputs"].append(entry)
                try:
                    protected[path] = fingerprint(path)
                    entry["sha256"] = protected[path]
                    result = verify_record(path, variant, config, expected_truth)
                    results[variant].append(result)
                    measured = train.score_rows("regression", np.asarray(result.test_truth),
                                                np.asarray(result.test_prediction))
                    entry.update(status="passed", rows=len(result.test_truth),
                                 stored_test_score=result.test_score, recomputed_test_score=measured,
                                 score_absolute_error=abs(measured - result.test_score),
                                 epochs=result.epochs, best_epoch=result.best_epoch,
                                 execution_identity_present=result.identity is not None)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                    report["errors"].append(f"{path.name}: {entry['error']}")
        if report["errors"]:
            raise ValueError("At least one expected result failed verification; no intervals produced.")

        report["seed_scores"] = {
            variant: {"scores": [result.test_score for result in runs],
                      "mean": float(np.mean([result.test_score for result in runs])),
                      "sample_sd": float(np.std([result.test_score for result in runs], ddof=1))}
            for variant, runs in results.items()
        }
        predictions = {variant: [np.asarray(result.test_prediction) for result in runs]
                       for variant, runs in results.items()}
        print("All 12 records passed. Recomputing 4,000 paired row resamples; no models run.", flush=True)
        report["comparisons"] = seed_average_bootstrap(expected_truth, predictions)
        for comparison in report["comparisons"]:
            old = REPORTED_COMPARISONS.get((comparison["first"], comparison["second"]))
            if old is not None:
                new = tuple(round(comparison[key], 3) for key in ("difference", "ci_low", "ci_high"))
                comparison["historically_reported_rounded"] = list(old)
                comparison["matches_reported_at_three_decimals"] = new == old
        report["status"] = "passed_internal_consistency_only"
    except (OSError, ValueError, TypeError, KeyError, subprocess.CalledProcessError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
        report["status"] = "failed"
    finally:
        changed = [path.relative_to(REPO_ROOT).as_posix() for path, old_hash in protected.items()
                   if not path.exists() or fingerprint(path) != old_hash]
        report["consulted_files_unchanged"] = not changed
        if changed:
            report["errors"].append(f"Consulted files changed during verification: {changed}")
            report["status"] = "failed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        destination = Path(__file__).parent / f"equation-verification-{stamp}.json"
        # Exclusive creation preserves both the original results and earlier verification reports.
        with destination.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"{report['status']}: {destination}", flush=True)
    return 0 if report["status"] == "passed_internal_consistency_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
