"""Check concatenation against the original probe loops on synthetic data only."""

import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import numpy as np
import pandas as pd

from mbf import analysis


class ConcatenationTests(unittest.TestCase):
    def setUp(self):
        # Nonalphabetical order and unequal widths expose accidental reordering.
        self.keys = ["zeta", "alpha", "mu"]
        self.embeddings = {
            "alpha": np.arange(12, dtype=float).reshape(4, 3),
            "mu": np.arange(4, dtype=float).reshape(4, 1) + 50,
            "zeta": np.arange(8, dtype=float).reshape(4, 2) + 100,
        }

    def test_concatenation_preserves_selected_rows_columns_and_inputs(self):
        originals = {key: value.copy() for key, value in self.embeddings.items()}
        for keys in (self.keys, ["mu"], ["mu", "zeta"]):
            with self.subTest(keys=keys):
                actual = analysis.concatenate_embeddings(self.embeddings, keys)
                np.testing.assert_array_equal(
                    actual, np.concatenate([originals[key] for key in keys], axis=1)
                )
        for key, original in originals.items():
            np.testing.assert_array_equal(self.embeddings[key], original)
        self.assertEqual(self.keys, ["zeta", "alpha", "mu"])

    def test_pair_scores_gain_and_pinned_probe_call_contract(self):
        y = np.arange(120, dtype=float).reshape(4, 30)
        groups = np.array(["third", "first", "third", "second"])
        singles = {"zeta": 0.1, "alpha": 0.3, "mu": 0.2, "unused": np.nan}
        originals = {key: value.copy() for key, value in self.embeddings.items()}
        y_before, groups_before = y.copy(), groups.copy()
        for array in [*self.embeddings.values(), y, groups]:
            array.flags.writeable = False
        fold_scores = [np.full(5, value) for value in (0.6, 0.9, 0.5)]

        with patch.object(analysis, "probe_scores", side_effect=fold_scores) as probe:
            scores, gains = analysis.pairwise_concatenation_scores(
                self.embeddings, self.keys, y, singles, groups=groups
            )

        # One call per pair: provided single-model results are not recomputed.
        self.assertEqual(probe.call_count, 3)
        for call, (a, b) in zip(probe.call_args_list, itertools.combinations(self.keys, 2)):
            self.assertEqual(len(call.args), 2)
            np.testing.assert_array_equal(
                call.args[0], np.concatenate([originals[a], originals[b]], axis=1)
            )
            self.assertIs(call.args[1], y)
            self.assertEqual(set(call.kwargs), {"groups", "seed"})
            self.assertIs(call.kwargs["groups"], groups)
            self.assertEqual(call.kwargs["seed"], 42)
        expected_scores = [[np.nan, 0.6, 0.9], [0.6, np.nan, 0.5], [0.9, 0.5, np.nan]]
        expected_gains = [[np.nan, 0.3, 0.7], [0.3, np.nan, 0.2], [0.7, 0.2, np.nan]]
        for actual, values in ((scores, expected_scores), (gains, expected_gains)):
            pd.testing.assert_frame_equal(
                actual, pd.DataFrame(values, index=self.keys, columns=self.keys)
            )
        for key, original in originals.items():
            np.testing.assert_array_equal(self.embeddings[key], original)
        np.testing.assert_array_equal(y, y_before)
        np.testing.assert_array_equal(groups, groups_before)
        self.assertEqual([singles[key] for key in self.keys], [0.1, 0.3, 0.2])

    def test_matches_original_loops_for_scalar_and_thirty_output_targets(self):
        rng = np.random.default_rng(1729)
        embeddings = {key: rng.normal(size=(80, width))
                      for key, width in zip(self.keys, (2, 3, 1))}
        design = np.concatenate([embeddings[key] for key in self.keys], axis=1)
        grouped = np.repeat(np.array([f"sequence-{i}" for i in range(40)]), 2)
        for outputs, groups in ((1, grouped), (30, grouped), (1, None)):
            y = design @ rng.normal(size=(design.shape[1], outputs))
            y += rng.normal(scale=0.3, size=y.shape)
            if outputs == 1:
                y = y[:, 0]
            with self.subTest(outputs=outputs, grouped=groups is not None):
                singles = pd.Series({key: analysis.probe_scores(X, y, groups=groups).mean()
                                     for key, X in embeddings.items()})
                expected = pd.DataFrame(np.nan, index=self.keys, columns=self.keys)
                expected_gain = expected.copy()
                # Retain the original computation as the independent reference.
                for a, b in itertools.combinations(self.keys, 2):
                    fused = np.concatenate([embeddings[a], embeddings[b]], axis=1)
                    score = analysis.probe_scores(fused, y, groups=groups).mean()
                    expected.loc[a, b] = expected.loc[b, a] = score
                    gain = score - max(singles[a], singles[b])
                    expected_gain.loc[a, b] = expected_gain.loc[b, a] = gain
                scores, gains = analysis.pairwise_concatenation_scores(
                    embeddings, self.keys, y, singles, groups=groups
                )
                pd.testing.assert_frame_equal(scores, expected, rtol=1e-12, atol=1e-12)
                pd.testing.assert_frame_equal(gains, expected_gain, rtol=1e-12, atol=1e-12)

    def test_invalid_embedding_selection_is_rejected(self):
        cases = [
            (self.embeddings, [], ValueError),
            (self.embeddings, ["zeta", "zeta"], ValueError),
            (self.embeddings, ["missing"], KeyError),
            ({"a": np.zeros(4)}, ["a"], ValueError),
            ({"a": np.zeros((4, 2, 1))}, ["a"], ValueError),
            ({"a": np.zeros((4, 2)), "b": np.zeros((3, 1))}, ["a", "b"], ValueError),
        ]
        for embeddings, keys, error in cases:
            with self.subTest(keys=keys, shapes=[x.shape for x in embeddings.values()]):
                with self.assertRaises(error):
                    analysis.concatenate_embeddings(embeddings, keys)

    def test_invalid_pair_inputs_fail_before_any_probe_runs(self):
        valid = dict(embeddings=self.embeddings, keys=self.keys, y=np.arange(4.),
                     single_r2={"zeta": 0.1, "alpha": 0.3, "mu": 0.2}, groups=None)
        invalid_values = [
            {"keys": []}, {"keys": ["zeta"]}, {"keys": ["zeta", "zeta"]},
            {"embeddings": {**self.embeddings, "mu": np.zeros(4)}},
            {"embeddings": {**self.embeddings, "mu": np.zeros((3, 1))}},
            {"y": np.array(1.)}, {"y": np.zeros((4, 2, 1))}, {"y": np.zeros(3)},
            {"groups": np.zeros((4, 1))}, {"groups": np.zeros(3)},
            {"single_r2": {"zeta": 0.1, "alpha": 0.3, "mu": [0.2]}},
            {"single_r2": {"zeta": 0.1, "alpha": 0.3, "mu": np.nan}},
            {"single_r2": {"zeta": 0.1, "alpha": 0.3, "mu": np.inf}},
            {"single_r2": {"zeta": 0.1, "alpha": 0.3, "mu": "not a score"}},
            {"single_r2": {"zeta": 0.1, "alpha": 0.3, "mu": 0.2 + 1j}},
        ]
        missing_keys = [
            {"keys": ["zeta", "missing"]},
            {"single_r2": {"zeta": 0.1, "alpha": 0.3}},
        ]
        cases = [(change, ValueError) for change in invalid_values]
        cases += [(change, KeyError) for change in missing_keys]
        for changed, error in cases:
            with self.subTest(changed=changed):
                with patch.object(analysis, "probe_scores") as probe:
                    with self.assertRaises(error):
                        analysis.pairwise_concatenation_scores(**(valid | changed))
                    probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
