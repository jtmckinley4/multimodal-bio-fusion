"""Invented OOF evidence and arithmetic only; no dataset, model or file producers."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import anchor_correction as correction


def fixture():
    """The first two rows are repeated observations of one invented sequence group."""
    context = {
        "row_ids": ("a-first", "a-second", "b", "c", "d", "e", "validation", "test"),
        "groups": ("A", "A", "B", "C", "D", "E", "V", "T"),
        "train_positions": (0, 1, 2, 3, 4, 5),
        "validation_positions": (6,),
        "test_positions": (7,),
    }
    plans = (
        correction.FoldRows(fit=(2, 3), stop=(4, 5), predict=(1, 0)),
        correction.FoldRows(fit=(0, 1), stop=(4, 5), predict=(3, 2)),
        correction.FoldRows(fit=(0, 1), stop=(2, 3), predict=(5, 4)),
    )
    return context, plans


def fold_predictions(context, plans, *, task="regression", class_order=()):
    predictions = []
    for plan in plans:
        rows = tuple(context["row_ids"][index] for index in plan.predict)
        values = np.asarray(plan.predict, dtype=float) + .25
        if task == "classification":
            values = np.column_stack([values, -2 * values])
        predictions.append(correction.Predictions(task, rows, values, class_order))
    return predictions


class DecisionEvidenceTests(unittest.TestCase):
    def test_fold_summary_exposes_constant_stopping_targets_and_group_concentration(self):
        context, plans = fixture()
        targets = np.array([2., 2., 3., 4., 0., 0., 9., 10.])
        before = targets.copy()
        rows = correction.fold_diagnostics(plans, targets=targets, lengths=np.arange(8) + 10, **context)
        first_stop = next(row for row in rows if row["fold"] == 0 and row["role"] == "stop")
        self.assertTrue(first_stop["constant_target"])
        self.assertEqual(first_stop["target_sd"], 0)
        predicted = next(row for row in rows if row["fold"] == 0 and row["role"] == "predict")
        self.assertEqual((predicted["rows"], predicted["groups"]), (2, 1))
        self.assertEqual(predicted["largest_group_row_fraction"], 1.)
        np.testing.assert_array_equal(targets, before)

    def test_fold_summary_does_not_accept_leakage_or_misaligned_targets(self):
        context, plans = fixture()
        for targets in ([1., 2.], [float("nan")] * 8):
            with self.assertRaises(ValueError):
                correction.fold_diagnostics(plans, targets=targets, lengths=[10] * 8, **context)
        with self.assertRaisesRegex(ValueError, "outside"):
            correction.fold_diagnostics((replace(plans[0], fit=(6,)), *plans[1:]),
                                        targets=np.arange(8.), lengths=[10] * 8, **context)

    def test_residual_summary_uses_truth_minus_prediction_without_mutating_inputs(self):
        entry = {"row_ids": ["a", "b"], "truth": [2., 0.], "values": [1., 1.]}
        summary = correction.residual_summary(entry)
        self.assertEqual(summary["residual_mean"], 0.)
        self.assertEqual(summary["residual_sd"], 1.)
        self.assertEqual(summary["rmse"], 1.)
        self.assertEqual(entry["values"], [1., 1.])
        with self.assertRaises(ValueError):
            correction.residual_summary({**entry, "values": [0.]})


class FoldPlanTests(unittest.TestCase):
    def test_explicit_group_disjoint_plan_and_empty_stopping_set_are_valid(self):
        context, plans = fixture()
        correction.validate_plan(plans, **context)
        without_stopping = [replace(plan, stop=()) for plan in plans]
        correction.validate_plan(without_stopping, **context)

    def test_fold_positions_are_copied_from_mutable_inputs(self):
        fit, stop, predict = [2, 3], [4, 5], [1, 0]
        plan = correction.FoldRows(fit, stop, predict)
        fit.clear()
        stop.reverse()
        predict.clear()
        self.assertEqual(plan, correction.FoldRows((2, 3), (4, 5), (1, 0)))

    def test_invalid_positions_and_repeated_rows_are_rejected(self):
        for bad in ((0, 0), (-1,), (True,), (np.bool_(False),), (1.0,), ()):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                correction.FoldRows(fit=bad, stop=(), predict=(1,))

    def test_heldout_group_cannot_appear_in_fit_even_at_another_row(self):
        context, plans = fixture()
        plans = (replace(plans[0], fit=(1, 2), predict=(0,)),) + plans[1:]
        with self.assertRaisesRegex(ValueError, "fit/predict roles share a group"):
            correction.validate_plan(plans, **context)

    def test_heldout_group_cannot_select_the_checkpoint_indirectly(self):
        context, plans = fixture()
        plans = (replace(plans[0], stop=(1, 4), predict=(0,)),) + plans[1:]
        with self.assertRaisesRegex(ValueError, "stop/predict roles share a group"):
            correction.validate_plan(plans, **context)

    def test_inner_stopping_groups_are_separate_from_fit_groups(self):
        context, plans = fixture()
        plans = (plans[0], replace(plans[1], fit=(0,), stop=(1, 4)), plans[2])
        with self.assertRaisesRegex(ValueError, "fit/stop roles share a group"):
            correction.validate_plan(plans, **context)

    def test_outer_rows_cannot_enter_any_fold_role(self):
        context, plans = fixture()
        for role in ("fit", "stop", "predict"):
            for outer in (6, 7):
                changed = replace(plans[0], **{role: (outer,)})
                with self.subTest(role=role, outer=outer), self.assertRaisesRegex(
                    ValueError, "outside the permitted training partition",
                ):
                    correction.validate_plan((changed,) + plans[1:], **context)

    def test_outer_partitions_cannot_share_groups_or_rows(self):
        context, plans = fixture()
        changes = (
            {"groups": context["groups"][:6] + ("A", "T")},
            {"groups": context["groups"][:6] + ("V", "V")},
            {"validation_positions": (0,)},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "partitions share a group"):
                correction.validate_plan(plans, **{**context, **change})

    def test_missing_and_duplicate_oof_predictions_are_rejected(self):
        context, plans = fixture()
        for changed in (plans[:-1], plans + (plans[0],)):
            with self.subTest(count=len(changed)), self.assertRaises(ValueError):
                correction.validate_plan(changed, **context)

    def test_same_group_cannot_be_predicted_in_two_separate_folds(self):
        context, plans = fixture()
        separated = (replace(plans[0], predict=(0,)), replace(plans[0], predict=(1,)))
        with self.assertRaisesRegex(ValueError, "group is predicted across multiple"):
            correction.validate_plan(separated + plans[1:], **context)

    def test_invalid_row_axis_and_identifiers_are_rejected(self):
        context, plans = fixture()
        changes = (
            {"groups": context["groups"][:-1]},
            {"groups": (None,) + context["groups"][1:]},
            {"row_ids": ("a-first",) + context["row_ids"][:-1]},
            {"train_positions": ()},
            {"test_positions": (99,)},
        )
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                correction.validate_plan(plans, **{**context, **change})
        for invalid in ((), ({"fit": (1,), "stop": (), "predict": (0,)},)):
            with self.assertRaises(ValueError):
                correction.validate_plan(invalid, **context)


class AssemblyTests(unittest.TestCase):
    def test_fold_and_prediction_row_order_is_restored_to_training_order(self):
        context, plans = fixture()
        predictions = fold_predictions(context, plans)
        actual = correction.assemble_oof(plans[::-1], predictions[::-1], task="regression", **context)
        self.assertEqual(actual.row_ids, context["row_ids"][:6])
        np.testing.assert_array_equal(actual.values, np.arange(6) + .25)
        context["train_positions"] = (4, 0, 2, 5, 1, 3)
        reordered = correction.assemble_oof(plans, predictions, task="regression", **context)
        np.testing.assert_array_equal(reordered.values, np.array(context["train_positions"]) + .25)

    def test_classification_retains_full_logits_and_declared_class_order(self):
        context, plans = fixture()
        classes = ("negative", "positive")
        predictions = fold_predictions(context, plans, task="classification", class_order=classes)
        actual = correction.assemble_oof(
            plans, predictions, task="classification", class_order=classes, **context,
        )
        expected = np.arange(6) + .25
        np.testing.assert_array_equal(actual.values, np.column_stack((expected, -2 * expected)))
        self.assertEqual(actual.class_order, classes)
        self.assertEqual(actual.values.shape, (6, 2))

    def test_mismatched_row_task_or_class_axes_are_rejected(self):
        context, plans = fixture()
        predictions = fold_predictions(context, plans)
        mutations = (
            replace(predictions[0], row_ids=predictions[0].row_ids[::-1]),
            correction.Predictions("classification", predictions[0].row_ids, [[1, 2], [3, 4]], (0, 1)),
        )
        for changed in mutations:
            with self.subTest(task=changed.task), self.assertRaises(ValueError):
                correction.assemble_oof(plans, [changed, *predictions[1:]], task="regression", **context)
        logits = fold_predictions(context, plans, task="classification", class_order=(0, 1))
        logits[0] = replace(logits[0], class_order=(1, 0))
        with self.assertRaises(ValueError):
            correction.assemble_oof(plans, logits, task="classification", class_order=(0, 1), **context)

    def test_missing_supplied_fold_predictions_are_rejected(self):
        context, plans = fixture()
        predictions = fold_predictions(context, plans)
        with self.assertRaisesRegex(ValueError, "exactly one Predictions"):
            correction.assemble_oof(plans, predictions[:-1], task="regression", **context)

    def test_assembly_does_not_mutate_input_records(self):
        context, plans = fixture()
        predictions = fold_predictions(context, plans)
        before = [prediction.values.copy() for prediction in predictions]
        actual = correction.assemble_oof(plans, predictions, task="regression", **context)
        for expected, prediction in zip(before, predictions):
            np.testing.assert_array_equal(prediction.values, expected)
            self.assertFalse(np.shares_memory(actual.values, prediction.values))


class PredictionAndArithmeticTests(unittest.TestCase):
    def test_regression_residuals_and_final_correction_have_independent_roles(self):
        oof = correction.Predictions("regression", ("a", "b"), [1., 4.])
        residual = correction.regression_residuals([3., 3.], target_row_ids=("a", "b"), oof=oof)
        np.testing.assert_array_equal(residual.values, [2., -1.])
        final_anchor = correction.Predictions("regression", ("new-a", "new-b"), [10., 20.])
        learned_update = correction.Predictions("regression", ("new-a", "new-b"), [.5, -2.])
        actual = correction.corrected_regression(final_anchor, learned_update)
        np.testing.assert_array_equal(actual.values, [10.5, 18.])

    def test_zero_correction_recovers_regression_anchor_exactly(self):
        anchor = correction.Predictions("regression", (10, 20), [1.25, -4.5])
        zero = correction.Predictions("regression", (10, 20), [0., 0.])
        np.testing.assert_array_equal(correction.corrected_regression(anchor, zero).values, anchor.values)

    def test_additive_logits_are_not_probabilities_or_class_ids(self):
        anchor = correction.Predictions("classification", ("a",), [[2., -1.]], ("yes", "no"))
        update = correction.Predictions("classification", ("a",), [[-.5, 3.]], ("yes", "no"))
        np.testing.assert_array_equal(correction.corrected_logits(anchor, update).values, [[1.5, 2.]])
        zero = replace(update, values=np.zeros((1, 2)))
        np.testing.assert_array_equal(correction.corrected_logits(anchor, zero).values, anchor.values)
        with self.assertRaises(ValueError):
            correction.regression_residuals([1], target_row_ids=("a",), oof=anchor)

    def test_arithmetic_rejects_row_class_and_task_mismatches(self):
        anchor = correction.Predictions("regression", ("a", "b"), [1, 2])
        swapped = correction.Predictions("regression", ("b", "a"), [1, 2])
        with self.assertRaises(ValueError):
            correction.corrected_regression(anchor, swapped)
        with self.assertRaises(ValueError):
            correction.regression_residuals([1, 2], target_row_ids=("b", "a"), oof=anchor)
        logits = correction.Predictions("classification", ("a", "b"), [[1, 2], [3, 4]], (0, 1))
        with self.assertRaises(ValueError):
            correction.corrected_logits(logits, replace(logits, class_order=(1, 0)))
        with self.assertRaises(ValueError):
            correction.corrected_regression(logits, logits)
        with self.assertRaises(ValueError):
            correction.corrected_logits(anchor, anchor)

    def test_invalid_shapes_nonfinite_and_nonnumeric_predictions_are_rejected(self):
        for values in ([np.nan], [np.inf], [-np.inf], [True], ["2"], [1 + 2j], [[1]], []):
            with self.subTest(values=values), self.assertRaises(ValueError):
                correction.Predictions("regression", ("a",), values)
        for values, classes in (([0, 1], (0, 1)), ([[0], [1]], (0,)), ([[0, 1], [1, 0]], (0, 0))):
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                correction.Predictions("classification", ("a", "b"), values, classes)
        with self.assertRaises(ValueError):
            correction.Predictions("expression", ("a",), [1])

    def test_prediction_values_and_identifiers_are_copied(self):
        values, rows = np.array([1., 2.]), ["a", "b"]
        prediction = correction.Predictions("regression", rows, values)
        values[:] = 99
        rows.reverse()
        np.testing.assert_array_equal(prediction.values, [1., 2.])
        self.assertEqual(prediction.row_ids, ("a", "b"))
        with self.assertRaises(ValueError):
            prediction.values[0] = 3


if __name__ == "__main__":
    unittest.main()
