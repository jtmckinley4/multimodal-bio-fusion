"""Validate supplied out-of-fold evidence and apply explicit correction arithmetic.

These helpers do not choose folds, fit predictors, read datasets or publish results.
The caller supplies every row assignment and prediction. Checks establish consistency
of that declared evidence, not whether an external trainer actually excluded labels
from preprocessing, hyperparameter selection or early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _labels(values, name, *, unique=False):
    labels = tuple(values)
    if not labels or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (str, Integral))
        or (isinstance(value, str) and not value.strip())
        for value in labels
    ):
        raise ValueError(f"{name} requires nonempty string or integer identifiers.")
    if unique and len(set(labels)) != len(labels):
        raise ValueError(f"{name} contains duplicate identifiers.")
    return labels


def _positions(values, name, *, allow_empty=False):
    positions = tuple(values)
    if (not positions and not allow_empty) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral) or value < 0
        for value in positions
    ):
        raise ValueError(f"{name} requires nonnegative integer row positions.")
    if len(set(positions)) != len(positions):
        raise ValueError(f"{name} contains duplicate row positions.")
    return tuple(int(value) for value in positions)


def _finite_values(values, name):
    array = np.asarray(values)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} requires finite real numeric values, excluding booleans.")
    # Own the array: assembly/arithmetic must never modify a caller's predictions.
    return np.array(array, dtype=np.float64, copy=True)


@dataclass(frozen=True)
class FoldRows:
    """Explicit positions in the caller's full row table, not fold-local offsets.

    ``stop`` contains rows used for checkpoint/scheduler selection. An explicit empty
    tuple means no such rows are supplied; it does not choose an epoch schedule.
    """

    fit: tuple[int, ...]
    stop: tuple[int, ...]
    predict: tuple[int, ...]

    def __post_init__(self):
        for name in ("fit", "stop", "predict"):
            object.__setattr__(self, name, _positions(
                getattr(self, name), name, allow_empty=name == "stop",
            ))


@dataclass(frozen=True)
class Predictions:
    """Values with an explicit row axis and, for classification, full logit axis.

    Regression values have shape (rows,). Classification values have shape
    (rows, classes); class IDs and probabilities are not substituted for logits.
    Their origin as logits remains the producer's declared contract.
    """

    task: str
    row_ids: tuple
    values: np.ndarray
    class_order: tuple = ()

    def __post_init__(self):
        rows = _labels(self.row_ids, "prediction row_ids", unique=True)
        values = _finite_values(self.values, "predictions")
        classes = tuple(self.class_order)
        if self.task == "regression":
            if classes or values.shape != (len(rows),):
                raise ValueError("Regression requires one value per row and no class axis.")
        elif self.task == "classification":
            classes = _labels(classes, "class_order", unique=True)
            if len(classes) < 2 or values.shape != (len(rows), len(classes)):
                raise ValueError("Classification requires full logits in the stated class order.")
        else:
            raise ValueError("task must explicitly be regression or classification.")
        values.setflags(write=False)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "class_order", classes)


def validate_plan(plans, *, row_ids, groups, train_positions,
                  validation_positions, test_positions):
    """Validate supplied group-disjoint fold roles and exact OOF prediction coverage.

    Only declared training rows may enter a fold, including its stopping set. Outer
    partitions must be row- and group-disjoint. Unused source-table rows are allowed;
    this helper does not choose or change the cohort. Every training group must be
    predicted together in exactly one fold.
    """
    rows = _labels(row_ids, "row_ids", unique=True)
    group_ids = _labels(groups, "groups")
    if len(group_ids) != len(rows):
        raise ValueError("groups and row_ids must describe the same row axis.")
    partitions = {
        "train": _positions(train_positions, "train_positions"),
        "validation": _positions(validation_positions, "validation_positions", allow_empty=True),
        "test": _positions(test_positions, "test_positions", allow_empty=True),
    }
    if any(position >= len(rows) for values in partitions.values() for position in values):
        raise ValueError("A partition position is outside the supplied row table.")

    def group_set(positions):
        return {group_ids[position] for position in positions}

    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if group_set(partitions[first]) & group_set(partitions[second]):
            raise ValueError(f"Outer {first}/{second} partitions share a group.")

    allowed = set(partitions["train"])
    prediction_rows, prediction_groups = set(), set()
    plans = tuple(plans)
    if not plans or any(not isinstance(plan, FoldRows) for plan in plans):
        raise ValueError("Provide explicit FoldRows for every OOF fold.")
    for fold, plan in enumerate(plans):
        roles = {name: set(getattr(plan, name)) for name in ("fit", "stop", "predict")}
        if any(not positions <= allowed for positions in roles.values()):
            raise ValueError(f"Fold {fold} uses a row outside the permitted training partition.")
        for first, second in (("fit", "stop"), ("fit", "predict"), ("stop", "predict")):
            if group_set(roles[first]) & group_set(roles[second]):
                raise ValueError(f"Fold {fold} {first}/{second} roles share a group.")
        if prediction_rows & roles["predict"]:
            raise ValueError("A training row has more than one OOF prediction assignment.")
        if prediction_groups & group_set(roles["predict"]):
            raise ValueError("A training group is predicted across multiple OOF folds.")
        prediction_rows.update(roles["predict"])
        prediction_groups.update(group_set(roles["predict"]))
    if prediction_rows != allowed:
        raise ValueError("OOF folds must predict every permitted training row exactly once.")


def assemble_oof(plans, predictions, *, row_ids, groups, train_positions,
                 validation_positions, test_positions, task, class_order=()):
    """Restore supplied fold predictions to the explicit training-row order.

    A fold's prediction row IDs must match its ``predict`` positions in order.
    Nothing is fitted and no historical model predictions are recovered here.
    """
    plans, predictions = tuple(plans), tuple(predictions)
    rows, group_ids = tuple(row_ids), tuple(groups)
    train = tuple(train_positions)
    validate_plan(
        plans, row_ids=rows, groups=group_ids, train_positions=train,
        validation_positions=validation_positions, test_positions=test_positions,
    )
    if len(predictions) != len(plans):
        raise ValueError("Supply exactly one Predictions record for each fold.")
    classes = tuple(class_order)
    output_rows = tuple(rows[position] for position in train)
    # Construct through Predictions to validate the requested task and class axis too.
    shape = (len(train), len(classes)) if task == "classification" else (len(train),)
    requested = Predictions(task, output_rows, np.zeros(shape), classes)
    assembled = np.empty_like(requested.values)
    destination = {position: index for index, position in enumerate(train)}
    for plan, prediction in zip(plans, predictions):
        expected_rows = tuple(rows[position] for position in plan.predict)
        if not isinstance(prediction, Predictions) or (
            prediction.task != task or prediction.row_ids != expected_rows
            or prediction.class_order != classes
        ):
            raise ValueError("Fold predictions differ from the declared row, task or class order.")
        assembled[[destination[position] for position in plan.predict]] = prediction.values
    return Predictions(task, output_rows, assembled, classes)


def regression_residuals(targets, *, target_row_ids, oof):
    """Return y - OOF prediction; supplied target order must match the OOF rows."""
    if not isinstance(oof, Predictions) or oof.task != "regression":
        raise ValueError("Residual targets require regression OOF predictions, not class IDs.")
    target = Predictions("regression", tuple(target_row_ids), targets)
    if target.row_ids != oof.row_ids:
        raise ValueError("Target rows must match OOF prediction rows in order.")
    return Predictions("regression", oof.row_ids, target.values - oof.values)


def _add(anchor, correction, task):
    if not isinstance(anchor, Predictions) or not isinstance(correction, Predictions):
        raise ValueError("Supply Predictions for both the anchor and its correction.")
    if anchor.task != task or correction.task != task:
        raise ValueError(f"This correction requires {task} values.")
    if anchor.row_ids != correction.row_ids or anchor.class_order != correction.class_order:
        raise ValueError("Anchor and correction must have identical row and class order.")
    return Predictions(task, anchor.row_ids, anchor.values + correction.values, anchor.class_order)


def corrected_regression(anchor, correction):
    """Add a fitted regression correction to a final anchor on the same rows."""
    return _add(anchor, correction, "regression")


def corrected_logits(anchor, correction):
    """Add class-wise logit updates; do not apply softmax, argmax or residual labels."""
    return _add(anchor, correction, "classification")


def fold_diagnostics(plans, *, targets, lengths, **context):
    """Describe an existing fold plan without choosing a better-looking assignment."""
    plans = tuple(plans)
    validate_plan(plans, **context)
    targets = _finite_values(targets, "targets")
    lengths = _finite_values(lengths, "lengths")
    count = len(context["row_ids"])
    if targets.shape != (count,) or lengths.shape != (count,) or (lengths <= 0).any():
        raise ValueError("Fold inspection needs one finite target and positive length per source row.")
    groups = np.asarray(context["groups"])
    rows = []
    for fold, plan in enumerate(plans):
        for role in ("fit", "stop", "predict"):
            indices = np.asarray(getattr(plan, role), dtype=int)
            _, counts = np.unique(groups[indices], return_counts=True)
            values = targets[indices]
            rows.append({
                "fold": fold, "role": role, "rows": len(indices), "groups": len(counts),
                "largest_group_row_fraction": float(counts.max() / len(indices)) if len(indices) else None,
                "target_unique": len(np.unique(values)),
                "target_min": float(values.min()) if len(values) else None,
                "target_max": float(values.max()) if len(values) else None,
                "target_sd": float(values.std()) if len(values) else None,
                "median_length": float(np.median(lengths[indices])) if len(indices) else None,
                "constant_target": bool(len(values) and np.unique(values).size == 1),
            })
    return rows


def residual_summary(entry):
    """Summarize supplied regression predictions; no fitting or test selection."""
    predicted = Predictions("regression", tuple(entry["row_ids"]), entry["values"])
    truth = Predictions("regression", tuple(entry["row_ids"]), entry["truth"])
    residual = truth.values - predicted.values
    return {"rows": len(residual), "residual_mean": float(residual.mean()),
            "residual_sd": float(residual.std()), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "residual_q10": float(np.quantile(residual, .1)),
            "residual_median": float(np.median(residual)),
            "residual_q90": float(np.quantile(residual, .9))}
