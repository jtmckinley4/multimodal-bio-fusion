"""Matched CPU ridge comparisons from already prepared, row-aligned matrices.

This is a new evaluation route, not a replacement for the historical probes.
It tunes the ridge penalty on validation MSE, refits on train + validation,
and retains paired test predictions. No encoder, downloader or trainer is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCHEMA = "matched-ridge-v1"


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def notebook_source(path, cell_ids):
    """Capture selected producer code, excluding outputs, counts and metadata.

    Callers supply stable cell IDs for the preparation and comparison code, rather
    than hashing an entire notebook whose saved displays change during execution.
    """
    wanted = tuple(cell_ids)
    if not wanted or len(wanted) != len(set(wanted)):
        raise ValueError("Specify distinct producer code-cell IDs.")
    with Path(path).open(encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells = {cell.get("id"): cell for cell in notebook["cells"]}
    if any(cell_id not in cells or cells[cell_id].get("cell_type") != "code"
           for cell_id in wanted):
        raise ValueError("A requested producer code cell is absent or is not code.")
    sources = [{"id": cell_id, "source": "".join(cells[cell_id]["source"])}
               for cell_id in wanted]
    return {"notebook": Path(path).name, "sha256": _digest(sources), "cells": sources}


def _matrix_digest(matrix):
    """Hash shape, dtype and row order without one extra whole-matrix copy."""
    digest = hashlib.sha256(_json_bytes({"shape": list(matrix.shape),
                                       "dtype": matrix.dtype.str}))
    for start in range(0, len(matrix), 1024):
        digest.update(np.ascontiguousarray(matrix[start:start + 1024]).tobytes())
    return digest.hexdigest()


def _labels(values, length, label, *, unique=False):
    values = list(values)
    if len(values) != length or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label} must contain one nonempty string per row.")
    if unique and len(set(values)) != length:
        raise ValueError(f"{label} must identify unique rows, including duplicate sequences.")
    return values


def _checked_inputs(matrices, y, row_ids, groups, splits, contrasts, alphas,
                    metric, draws, seed, require_disjoint_groups):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Targets must be a finite, nonempty one-dimensional array.")
    row_ids = _labels(row_ids, len(y), "row_ids", unique=True)
    groups = _labels(groups, len(y), "groups")
    if metric not in {"pearson", "spearman"}:
        raise ValueError("metric must be pearson or spearman.")
    if type(draws) is not int or draws < 0 or (0 < draws < 20):
        raise ValueError("draws must be zero or an integer of at least 20.")
    if type(seed) is not int or seed < 0:
        raise ValueError("The bootstrap seed must be a nonnegative integer.")
    if type(require_disjoint_groups) is not bool:
        raise ValueError("require_disjoint_groups must be an explicit boolean.")
    alphas = tuple(float(a) for a in alphas)
    if (not alphas or len(alphas) != len(set(alphas))
            or any(not math.isfinite(a) or a <= 0 for a in alphas)):
        raise ValueError("alphas must be distinct, positive finite penalties.")
    if not matrices or any(not isinstance(k, str) or not k for k in matrices):
        raise ValueError("Every matrix needs a nonempty representation name.")
    checked_matrices = {}
    for label in sorted(matrices):
        matrix = np.asarray(matrices[label])
        if (matrix.ndim != 2 or matrix.shape[0] != len(y) or matrix.shape[1] < 1
                or matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all()):
            raise ValueError(f"{label}: expected a finite numeric matrix with one row per target.")
        checked_matrices[label] = matrix
    if not splits:
        raise ValueError("At least one explicit split assignment is required.")
    checked_splits, overlap = {}, {}
    group_array = np.asarray(groups)
    for assignment, split in splits.items():
        if not isinstance(assignment, str) or not assignment or len(split) != 3:
            raise ValueError("Name each assignment and supply train, validation, test indices.")
        indices = tuple(np.asarray(part) for part in split)
        if any(a.ndim != 1 or a.dtype.kind not in "iu" or len(a) < 2 for a in indices):
            raise ValueError(f"{assignment}: every partition needs at least two integer row indices.")
        joined = np.concatenate(indices)
        if (len(joined) != len(y) or not np.array_equal(np.sort(joined), np.arange(len(y)))):
            raise ValueError(f"{assignment}: partitions must cover every row exactly once.")
        sets = [set(group_array[a]) for a in indices]
        found = {"train_validation": len(sets[0] & sets[1]),
                 "train_test": len(sets[0] & sets[2]),
                 "validation_test": len(sets[1] & sets[2])}
        if require_disjoint_groups and any(found.values()):
            raise ValueError(f"{assignment}: groups overlap across partitions: {found}")
        checked_splits[assignment] = indices
        overlap[assignment] = found
    checked_contrasts = []
    for contrast in contrasts:
        if len(contrast) != 3:
            raise ValueError("Each contrast needs a label, first representation, second representation.")
        label, first, second = contrast
        if (not isinstance(label, str) or not label or first not in matrices
                or second not in matrices or first == second):
            raise ValueError(f"Invalid fixed contrast: {contrast!r}")
        checked_contrasts.append((label, first, second))
    labels = [c[0] for c in checked_contrasts]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Supply at least one uniquely named fixed contrast.")
    return (checked_matrices, y, row_ids, groups, checked_splits,
            checked_contrasts, alphas, overlap)


def _correlation(y, prediction, metric):
    # A constant target or predictor has no defined correlation, not a score of zero.
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(prediction) == 0:
        return None
    function = pearsonr if metric == "pearson" else spearmanr
    answer = float(function(y, prediction).statistic)
    return answer if math.isfinite(answer) else None


def _fit_prediction(matrix, y, split, alphas):
    train, validation, test = split
    candidates = []
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(matrix[train], y[train])
        predicted = model.predict(matrix[validation])
        candidates.append((float(np.mean((y[validation] - predicted) ** 2)), alpha))
    if any(not math.isfinite(mse) for mse, _ in candidates):
        raise ValueError("Nonfinite validation predictions; no result was published.")
    # Stable tie handling follows the declared alpha order, without consulting test scores.
    selected = min(range(len(candidates)), key=lambda i: candidates[i][0])
    validation_mse, alpha = candidates[selected]
    final_rows = np.concatenate((train, validation))
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(matrix[final_rows], y[final_rows])
    prediction = np.asarray(model.predict(matrix[test]), dtype=float)
    if prediction.shape != y[test].shape or not np.isfinite(prediction).all():
        raise ValueError("Nonfinite or misaligned test predictions; no result was published.")
    return prediction, {"alpha": alpha, "validation_mse": validation_mse,
                        "candidate_validation_mse": [
                            {"alpha": a, "mse": loss} for loss, a in candidates]}


def paired_interval(y, first, second, groups, *, metric="spearman", draws=2000, seed=0):
    """Conditional test-group bootstrap; undefined draws remain explicitly counted."""
    y, first, second = (np.asarray(a, dtype=float) for a in (y, first, second))
    if (y.ndim != 1 or first.shape != y.shape or second.shape != y.shape
            or not all(np.isfinite(a).all() for a in (y, first, second))):
        raise ValueError("Paired arrays must be finite and have the same one-dimensional shape.")
    groups = _labels(groups, len(y), "test groups")
    if metric not in {"pearson", "spearman"}:
        raise ValueError("Unknown correlation metric.")
    if type(draws) is not int or draws < 0 or (0 < draws < 20):
        raise ValueError("draws must be zero or an integer of at least 20.")
    if type(seed) is not int or seed < 0:
        raise ValueError("The bootstrap seed must be a nonnegative integer.")
    a, b = _correlation(y, first, metric), _correlation(y, second, metric)
    unique = list(dict.fromkeys(groups))
    answer = {"difference": None if a is None or b is None else a - b,
              "ci_low": None, "ci_high": None, "positive_fraction": None,
              "rows": len(y), "groups": len(unique), "draws": draws, "valid_draws": 0,
              "status": "unavailable: undefined observed correlation"}
    if answer["difference"] is None:
        return answer
    if draws == 0:
        return {**answer, "status": "interval not requested"}
    if len(unique) < 2:
        return {**answer, "status": "unavailable: fewer than two test groups"}
    positions = {g: [] for g in unique}
    for i, group in enumerate(groups):
        positions[group].append(i)
    members = [np.asarray(positions[g], dtype=int) for g in unique]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        sampled = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        first_score = _correlation(y[sampled], first[sampled], metric)
        second_score = _correlation(y[sampled], second[sampled], metric)
        if first_score is not None and second_score is not None:
            differences.append(first_score - second_score)
    answer["valid_draws"] = len(differences)
    if len(differences) != draws:
        return {**answer, "status": "unavailable: undefined bootstrap draws; interval withheld"}
    low, high = np.quantile(differences, [0.025, 0.975])
    return {**answer, "ci_low": float(low), "ci_high": float(high),
            "positive_fraction": float(np.mean(np.asarray(differences) > 0)),
            "status": "conditional paired interval"}


def read(path):
    """Read a completed analysis without fitting, repairing or changing it."""
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported matched-probe record.")
    content = {k: v for k, v in document.items() if k != "content_sha256"}
    if _digest(content) != document.get("content_sha256"):
        raise ValueError(f"{path}: result content checksum mismatch; preserve and inspect it.")
    if _digest(document.get("identity")) != document.get("identity_sha256"):
        raise ValueError(f"{path}: computation identity checksum mismatch.")
    for key in ("scores", "comparisons", "summary", "predictions"):
        if not isinstance(document.get(key), list) or not document[key]:
            raise ValueError(f"{path}: missing {key}.")
    return document


def run(output_path, *, name, matrices, y, row_ids, groups, splits, contrasts,
        evidence, metric="spearman", alphas=(1e-3, 1e-2, 1e-1, 1, 10),
        draws=2000, seed=0, require_disjoint_groups=True):
    """Compute or reuse one immutable matched analysis; return (document, reused).

    ``splits`` maps assignment names to (train, validation, test) integer indices.
    ``contrasts`` contains (label, first, second) fixed before test evaluation.
    Evidence describes the caller's cache association and its historical limits;
    hashing today's matrices does not reconstruct how an old cache was produced.
    """
    if not isinstance(name, str) or not name or not isinstance(evidence, dict) or not evidence:
        raise ValueError("Supply a name and nonempty input/provenance evidence.")
    (matrices, y, row_ids, groups, splits, contrasts, alphas, overlap) = _checked_inputs(
        matrices, y, row_ids, groups, splits, contrasts, alphas, metric, draws, seed,
        require_disjoint_groups)
    source = Path(__file__).read_text(encoding="utf-8")
    packages = {p: version(p) for p in ("numpy", "scipy", "scikit-learn")}
    identity = {
        "schema": SCHEMA, "name": name, "metric": metric, "alphas": list(alphas),
        "selection": "minimum validation MSE; declared alpha order breaks ties",
        "final_fit": "scaler and ridge refitted on train plus validation",
        "bootstrap": {"draws": draws, "seed": seed, "unit": "test group",
                      "confidence": 0.95, "multiplicity_adjustment": "none"},
        "contrasts": [list(c) for c in contrasts], "row_ids": row_ids,
        "targets": y.tolist(), "groups": groups,
        "require_disjoint_groups": require_disjoint_groups,
        "group_overlap_counts": overlap,
        "splits": {key: {label: rows.tolist() for label, rows in
                          zip(("train", "validation", "test"), split)}
                   for key, split in splits.items()},
        "matrices": {label: {"shape": list(matrix.shape), "dtype": matrix.dtype.str,
                             "sha256": _matrix_digest(matrix)}
                     for label, matrix in matrices.items()},
        "input_evidence": evidence, "packages": packages,
        "helper_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }
    identity_hash = _digest(identity)
    output_path = Path(output_path)
    if output_path.exists():
        document = read(output_path)
        if document["identity_sha256"] != identity_hash:
            raise ValueError(f"{output_path}: different computation; choose a new destination.")
        return document, True
    started = time.perf_counter()
    utc = datetime.now(timezone.utc).isoformat()
    scores, comparisons, predictions = [], [], []
    for assignment, split in splits.items():
        test = split[2]
        predicted = {}
        for label, matrix in matrices.items():
            prediction, selection = _fit_prediction(matrix, y, split, alphas)
            predicted[label] = prediction
            primary = _correlation(y[test], prediction, metric)
            scores.append({"assignment": assignment, "representation": label,
                           "metric": metric, "score": primary,
                           "status": "defined" if primary is not None else "undefined correlation",
                           "pearson": _correlation(y[test], prediction, "pearson"),
                           "spearman": _correlation(y[test], prediction, "spearman"),
                           "test_mse": float(np.mean((y[test] - prediction) ** 2)),
                           "train_rows": len(split[0]), "validation_rows": len(split[1]),
                           "test_rows": len(test), **selection})
            predictions.append({"assignment": assignment, "representation": label,
                                "row_ids": [row_ids[i] for i in test],
                                "groups": [groups[i] for i in test], "truth": y[test].tolist(),
                                "prediction": prediction.tolist()})
        for label, first, second in contrasts:
            interval = paired_interval(y[test], predicted[first], predicted[second],
                                       [groups[i] for i in test], metric=metric,
                                       draws=draws, seed=seed)
            comparisons.append({"assignment": assignment, "comparison": label,
                                "first": first, "second": second, "metric": metric,
                                **interval})
    summary = []
    for label, _, _ in contrasts:
        selected = [c for c in comparisons if c["comparison"] == label]
        values = [c["difference"] for c in selected]
        complete = all(v is not None for v in values)
        summary.append({"comparison": label, "metric": metric,
                        "assignments": list(splits), "count": len(values),
                        "mean_difference": float(np.mean(values)) if complete else None,
                        "assignment_sd": (float(np.std(values, ddof=1))
                                          if complete and len(values) > 1 else None),
                        "status": "complete" if complete else "undefined assignment; no subset mean"})
    document = {
        "schema": SCHEMA, "identity": identity, "identity_sha256": identity_hash,
        "scores": scores, "comparisons": comparisons, "summary": summary,
        "predictions": predictions,
        "interpretation": [
            "Each interval conditions on one fitted assignment and resamples its test groups.",
            "Assignments may overlap; their spread is descriptive, not independent-replicate uncertainty.",
            "Positive bootstrap fractions are not p-values; intervals are unadjusted for multiplicity.",
            "Input fingerprints describe current supplied matrices, not an unknown historical producer.",
        ],
        "producer": {"started_utc": utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                     "seconds": time.perf_counter() - started, "host": platform.node(),
                     "python": platform.python_version(), "packages": packages,
                     "helper_source": source},
    }
    document["content_sha256"] = _digest(document)
    # Publish a fully serialized file without replacing an existing result, even in a race.
    payload = _json_bytes(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f"{output_path.name}.{uuid4().hex}.partial")
    created = False
    try:
        with temporary.open("xb") as handle:
            created = True
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "nt":
            # Windows rename fails if the destination exists; POSIX rename replaces it.
            os.rename(temporary, output_path)
        else:
            os.link(temporary, output_path)
    finally:
        if created:
            temporary.unlink(missing_ok=True)
    return document, False
