"""Synthetic ridge, paired-evaluation and immutable-result checks; no research data."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code import matched_probes as probes  # noqa: E402


def fixture():
    x = np.arange(12, dtype=float)
    return dict(
        name="synthetic constituent comparison",
        matrices={"gene": x[:, None], "both": np.column_stack((x, x ** 2))},
        y=x.copy(), row_ids=[f"row-{i}" for i in range(12)],
        groups=[f"group-{i}" for i in range(12)],
        splits={"assignment-1": (np.arange(6), np.arange(6, 9), np.arange(9, 12))},
        contrasts=[("both minus gene", "both", "gene")],
        evidence={"source": "synthetic", "historical_producer": "not applicable"},
        metric="pearson", alphas=(0.001, 1000), draws=40,
    )


class RidgeSelectionTests(unittest.TestCase):
    def test_validation_targets_choose_alpha_but_test_targets_do_not(self):
        data = fixture()
        split = data["splits"]["assignment-1"]
        matrix, y = data["matrices"]["gene"], data["y"]
        predicted, first = probes._fit_prediction(matrix, y, split, data["alphas"])
        self.assertEqual(first["alpha"], 0.001)
        altered_test = y.copy()
        altered_test[split[2]] = [-10000, 90000, 8]
        unchanged, second = probes._fit_prediction(matrix, altered_test, split, data["alphas"])
        self.assertEqual(first, second)
        np.testing.assert_array_equal(predicted, unchanged)
        altered_validation = y.copy()
        altered_validation[split[1]] = 2.5
        _, third = probes._fit_prediction(matrix, altered_validation, split, data["alphas"])
        self.assertEqual(third["alpha"], 1000)

    def test_scaler_never_sees_test_rows_or_validation_during_selection(self):
        data = fixture()
        split = data["splits"]["assignment-1"]
        matrix = data["matrices"]["gene"]
        observed = []
        original = probes.StandardScaler.fit

        def observe(estimator, values, *args, **kwargs):
            observed.append(values.copy())
            return original(estimator, values, *args, **kwargs)

        with patch.object(probes.StandardScaler, "fit", observe):
            probes._fit_prediction(matrix, data["y"], split, data["alphas"])
        self.assertEqual(len(observed), 3)
        for values in observed[:-1]:
            np.testing.assert_array_equal(values, matrix[split[0]])
        np.testing.assert_array_equal(observed[-1], matrix[np.r_[split[0], split[1]]])

    def test_linear_relation_has_near_perfect_out_of_sample_correlation(self):
        data = fixture()
        predicted, _ = probes._fit_prediction(data["matrices"]["gene"], data["y"],
                                              data["splits"]["assignment-1"], (0.001,))
        self.assertAlmostEqual(probes._correlation(data["y"][9:], predicted, "pearson"), 1.0)


class SourceIdentityTests(unittest.TestCase):
    def test_saving_outputs_does_not_change_producer_code_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "study.ipynb"
            notebook = {"cells": [{"id": "prepare", "cell_type": "code", "source": ["x = 1\n"],
                                   "outputs": [], "execution_count": None, "metadata": {}}]}
            path.write_text(json.dumps(notebook), encoding="utf-8")
            original = probes.notebook_source(path, ["prepare"])
            notebook["cells"][0].update(outputs=[{"text": "saved output"}], execution_count=7,
                                         metadata={"timing": 10})
            path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertEqual(original, probes.notebook_source(path, ["prepare"]))
            notebook["cells"][0]["source"] = ["x = 2\n"]
            path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertNotEqual(original, probes.notebook_source(path, ["prepare"]))


class PairingTests(unittest.TestCase):
    def test_known_opposite_ordering_has_difference_two(self):
        result = probes.paired_interval(
            [0, 1, 2, 3], [0, 1, 2, 3], [3, 2, 1, 0], ["a", "a", "b", "b"],
            metric="spearman", draws=40,
        )
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["difference"], 2.0)
        self.assertEqual(result["valid_draws"], 40)
        self.assertEqual(result["ci_low"], 2.0)
        self.assertEqual(result["ci_high"], 2.0)

    def test_identical_predictions_have_zero_paired_difference(self):
        y = np.arange(30, dtype=float)
        result = probes.paired_interval(y, y ** 2, y ** 2, [str(i) for i in range(30)],
                                       metric="pearson", draws=40)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertEqual(result["ci_high"], 0.0)

    def test_resampling_keeps_every_member_of_a_selected_group(self):
        original = probes._correlation
        observed = []

        def observe(y, prediction, metric):
            observed.append(np.asarray(y).tolist())
            return original(y, prediction, metric)

        class FirstGroupTwice:
            def integers(self, high, size):
                return np.zeros(size, dtype=int)

        with patch.object(probes.np.random, "default_rng", return_value=FirstGroupTwice()), \
                patch.object(probes, "_correlation", observe):
            probes.paired_interval([0, 1, 2, 8, 9], [0, 1, 2, 8, 9], [9, 8, 2, 1, 0],
                                   ["a", "a", "a", "b", "b"], draws=20)
        # The two observed-score calls precede the paired bootstrap calls.
        self.assertEqual(observed[2:], [[0, 1, 2, 0, 1, 2]] * 40)

    def test_constant_predictions_are_unavailable_not_zero(self):
        result = probes.paired_interval([0, 1, 2], [1, 1, 1], [0, 1, 2],
                                       ["a", "b", "c"], draws=40)
        self.assertIsNone(result["difference"])
        self.assertIsNone(result["ci_low"])
        self.assertIn("undefined", result["status"])

    def test_single_group_cannot_support_group_interval(self):
        result = probes.paired_interval([0, 1, 2], [0, 1, 2], [2, 1, 0],
                                       ["a", "a", "a"], draws=40)
        self.assertEqual(result["difference"], 2.0)
        self.assertIsNone(result["ci_low"])
        self.assertIn("fewer than two", result["status"])

    def test_too_many_degenerate_resamples_do_not_publish_an_interval(self):
        result = probes.paired_interval([0, 1], [0, 1], [1, 0], ["a", "b"],
                                       draws=100, seed=0)
        self.assertIsNone(result["ci_low"])
        self.assertLess(result["valid_draws"], 80)
        self.assertIn("undefined bootstrap", result["status"])

    def test_one_undefined_draw_withholds_interval_instead_of_conditioning_on_valid_draws(self):
        class OneDegenerateDraw:
            def __init__(self):
                self.calls = 0

            def integers(self, high, size):
                self.calls += 1
                return np.array([0, 0] if self.calls == 1 else [0, 1])

        with patch.object(probes.np.random, "default_rng", return_value=OneDegenerateDraw()):
            result = probes.paired_interval([0, 1], [0, 1], [1, 0], ["a", "b"], draws=20)
        self.assertAlmostEqual(result["difference"], 2.0)
        self.assertEqual(result["valid_draws"], 19)
        self.assertIsNone(result["ci_low"])
        self.assertIsNone(result["ci_high"])
        self.assertIsNone(result["positive_fraction"])
        self.assertIn("interval withheld", result["status"])

    def test_nonfinite_or_different_length_arrays_fail(self):
        for first in ([0, 1], [0, np.nan, 2]):
            with self.subTest(first=first), self.assertRaises(ValueError):
                probes.paired_interval([0, 1, 2], first, [0, 1, 2], ["a", "b", "c"])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "result.json"

    def test_one_file_retains_actual_rows_splits_predictions_and_source(self):
        data = fixture()
        document, reused = probes.run(self.path, **data)
        self.assertFalse(reused)
        self.assertEqual(probes.read(self.path), document)
        self.assertEqual(document["identity"]["splits"]["assignment-1"]["test"], [9, 10, 11])
        self.assertEqual(document["predictions"][0]["row_ids"], ["row-9", "row-10", "row-11"])
        self.assertEqual(document["predictions"][0]["truth"], [9, 10, 11])
        self.assertIn("def run(", document["producer"]["helper_source"])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.path])

    def test_reuse_returns_without_any_refit_or_republication(self):
        first, _ = probes.run(self.path, **fixture())
        before, modified = self.path.read_bytes(), self.path.stat().st_mtime_ns
        with patch.object(probes, "_fit_prediction", side_effect=AssertionError("must not fit")):
            second, reused = probes.run(self.path, **fixture())
        self.assertTrue(reused)
        self.assertEqual(first, second)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.path.stat().st_mtime_ns, modified)

    def test_changed_inputs_splits_or_evidence_refuse_overwrite_before_fit(self):
        probes.run(self.path, **fixture())
        before = self.path.read_bytes()
        for mutation in ("matrix", "targets", "row order", "groups", "split", "evidence", "alphas"):
            data = fixture()
            if mutation == "matrix": data["matrices"]["gene"][0, 0] += 0.1
            elif mutation == "targets": data["y"][0] += 0.1
            elif mutation == "row order": data["row_ids"][0:2] = reversed(data["row_ids"][0:2])
            elif mutation == "groups": data["groups"][0] = "another group"
            elif mutation == "split":
                data["splits"]["assignment-1"] = (np.arange(5), np.arange(5, 9), np.arange(9, 12))
            elif mutation == "evidence": data["evidence"]["source"] = "different"
            elif mutation == "alphas": data["alphas"] = (1.0, 10.0)
            with self.subTest(mutation=mutation), patch.object(
                    probes, "_fit_prediction", side_effect=AssertionError("must not fit")):
                with self.assertRaisesRegex(ValueError, "different computation"):
                    probes.run(self.path, **data)
            self.assertEqual(self.path.read_bytes(), before)

    def test_malformed_existing_result_is_preserved(self):
        self.path.write_text('{"interrupted":', encoding="utf-8")
        before = self.path.read_bytes()
        with patch.object(probes, "_fit_prediction", side_effect=AssertionError("must not fit")):
            with self.assertRaises(ValueError):
                probes.run(self.path, **fixture())
        self.assertEqual(self.path.read_bytes(), before)

    def test_modified_prediction_detected_on_read(self):
        probes.run(self.path, **fixture())
        document = json.loads(self.path.read_text())
        document["predictions"][0]["prediction"][0] += 1
        self.path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum"):
            probes.read(self.path)

    def test_new_publication_race_preserves_competing_file(self):
        operation = "rename" if probes.os.name == "nt" else "link"
        original_publish = getattr(probes.os, operation)

        def occupied(source, destination):
            Path(destination).write_text("earlier result", encoding="utf-8")
            original_publish(source, destination)

        with patch.object(probes.os, operation, occupied), self.assertRaises(FileExistsError):
            probes.run(self.path, **fixture())
        self.assertEqual(self.path.read_text(), "earlier result")
        self.assertEqual(list(Path(self.temp.name).iterdir()), [self.path])

    def test_group_overlap_rejected_or_explicitly_recorded(self):
        data = fixture()
        data["groups"][9] = data["groups"][0]
        with self.assertRaisesRegex(ValueError, "groups overlap"):
            probes.run(self.path, **data)
        self.assertFalse(self.path.exists())
        document, _ = probes.run(self.path, require_disjoint_groups=False, **data)
        self.assertEqual(document["identity"]["group_overlap_counts"]["assignment-1"]["train_test"], 1)

    def test_invalid_alignment_fails_before_fitting(self):
        for mutation in ("duplicate id", "nonfinite target", "bad matrix", "duplicate index", "omitted row"):
            data = fixture()
            if mutation == "duplicate id": data["row_ids"][1] = data["row_ids"][0]
            elif mutation == "nonfinite target": data["y"][0] = np.nan
            elif mutation == "bad matrix": data["matrices"]["gene"] = np.ones((11, 2))
            elif mutation == "duplicate index":
                data["splits"]["assignment-1"] = (np.array([0, 0, 2, 3, 4, 5]), np.arange(6, 9), np.arange(9, 12))
            elif mutation == "omitted row":
                data["splits"]["assignment-1"] = (np.arange(5), np.arange(6, 9), np.arange(9, 12))
            with self.subTest(mutation=mutation), patch.object(
                    probes, "_fit_prediction", side_effect=AssertionError("must not fit")):
                with self.assertRaises(ValueError): probes.run(self.path, **data)
            self.assertFalse(self.path.exists())

    def test_all_assignments_share_rows_and_no_subset_average_is_reported(self):
        data = fixture()
        data["splits"]["assignment-2"] = (np.arange(6), np.arange(9, 12), np.arange(6, 9))
        original = probes._fit_prediction

        def constant_second(matrix, y, split, alphas):
            prediction, chosen = original(matrix, y, split, alphas)
            if split[2][0] == 6:
                prediction[:] = 1
            return prediction, chosen

        with patch.object(probes, "_fit_prediction", constant_second):
            document, _ = probes.run(self.path, **data)
        self.assertEqual(len(document["predictions"]), 4)
        self.assertEqual(document["summary"][0]["count"], 2)
        self.assertIsNone(document["summary"][0]["mean_difference"])
        self.assertIn("no subset mean", document["summary"][0]["status"])

    def test_missing_path_read_does_not_create_anything(self):
        with self.assertRaises(FileNotFoundError): probes.read(self.path)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
