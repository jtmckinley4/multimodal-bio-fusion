"""Check repeated-assignment gains on mocks and small synthetic datasets only."""

import inspect
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import numpy as np

from shared_code import analysis


class AssignmentGainTests(unittest.TestCase):
    def setUp(self):
        self.keys = ["zeta", "alpha", "mu"]
        self.embeddings = {
            "alpha": np.arange(12, dtype=float).reshape(4, 3),
            "mu": np.arange(4, dtype=float).reshape(4, 1) + 50,
            "zeta": np.arange(8, dtype=float).reshape(4, 2) + 100,
        }

    def test_pair_order_assignment_baselines_and_probe_reuse(self):
        y = np.arange(120, dtype=float).reshape(4, 30)
        groups = np.array(["third", "first", "third", "second"])
        originals = {key: value.copy() for key, value in self.embeddings.items()}
        y_before, groups_before = y.copy(), groups.copy()
        for array in [*self.embeddings.values(), y, groups]:
            array.flags.writeable = False
        # The better single encoder alternates, so max(mean(single)) is wrong.
        singles = [np.array([0.5, 0.1, 0.3]), np.array([0.2, 0.6, 0.3]),
                   np.array([0.4, 0.2, 0.7])]
        pairs = [np.array([0.7, 0.6, 0.1]), np.array([0.5, 0.3, 0.6]),
                 np.array([0.1, 0.8, 0.7])]
        signature = inspect.signature(analysis.probe_assignment_means)
        with patch.object(analysis, "probe_assignment_means", side_effect=singles + pairs) as probe:
            actual = analysis.pairwise_assignment_gains(
                self.embeddings, self.keys, y, groups, n_assignments=3
            )

        expected_pairs = list(itertools.combinations(self.keys, 2))
        self.assertEqual(list(actual), expected_pairs)
        expected_gains = ([0.2, 0., -0.2], [0., 0.1, -0.1], [-0.3, 0.2, 0.])
        for pair, expected in zip(expected_pairs, expected_gains):
            self.assertIsInstance(actual[pair], np.ndarray)
            np.testing.assert_allclose(actual[pair], expected, atol=1e-14)
        # Each selected single is fit once and reused by every relevant pair.
        self.assertEqual(probe.call_count, len(self.keys) + len(expected_pairs))
        expected_inputs = [originals[key] for key in self.keys]
        expected_inputs += [np.concatenate([originals[a], originals[b]], axis=1)
                            for a, b in expected_pairs]
        for call, expected_X in zip(probe.call_args_list, expected_inputs):
            bound = signature.bind(*call.args, **call.kwargs)
            bound.apply_defaults()
            np.testing.assert_array_equal(bound.arguments["X"], expected_X)
            self.assertIs(bound.arguments["y"], y)
            self.assertIs(bound.arguments["groups"], groups)
            self.assertEqual(bound.arguments["n_assignments"], 3)
            self.assertIs(bound.arguments["continuous"], True)
        for key, original in originals.items():
            np.testing.assert_array_equal(self.embeddings[key], original)
        np.testing.assert_array_equal(y, y_before)
        np.testing.assert_array_equal(groups, groups_before)
        self.assertEqual(self.keys, ["zeta", "alpha", "mu"])

    def test_original_grouped_loops_for_scalar_and_thirty_output_targets(self):
        rng = np.random.default_rng(817)
        embeddings = {key: rng.normal(size=(80, width))
                      for key, width in zip(self.keys, (2, 3, 1))}
        design = np.concatenate([embeddings[key] for key in self.keys], axis=1)
        groups = np.repeat(np.array([f"sequence-{i}" for i in range(40)]), 2)
        for outputs in (1, 30):
            y = design @ rng.normal(size=(design.shape[1], outputs))
            y += rng.normal(scale=0.3, size=y.shape)
            if outputs == 1:
                y = y[:, 0]
            with self.subTest(outputs=outputs):
                # The former notebook loop remains the independent reference.
                singles = {key: analysis.probe_assignment_means(X, y, groups, 2)
                           for key, X in embeddings.items()}
                expected = {}
                for a, b in itertools.combinations(self.keys, 2):
                    fused = np.concatenate([embeddings[a], embeddings[b]], axis=1)
                    combined = analysis.probe_assignment_means(fused, y, groups, 2)
                    expected[a, b] = combined - np.maximum(singles[a], singles[b])
                actual = analysis.pairwise_assignment_gains(
                    embeddings, self.keys, y, groups, n_assignments=2
                )
                self.assertEqual(list(actual), list(expected))
                for pair in expected:
                    np.testing.assert_allclose(actual[pair], expected[pair], rtol=1e-12, atol=1e-12)

    def test_assignment_helper_keeps_seed_sequence_and_target_identity(self):
        X = self.embeddings["zeta"]
        y, groups = np.arange(4.), np.array(["a", "b", "a", "c"])
        fold_scores = [np.arange(5.) + offset for offset in (0., 10., 20.)]
        with patch.object(analysis, "probe_scores", side_effect=fold_scores) as probe:
            result = analysis.probe_assignment_means(X, y, groups, n_assignments=3)
        np.testing.assert_array_equal(result, [2., 12., 22.])
        self.assertEqual(probe.call_count, 3)
        for seed, call in enumerate(probe.call_args_list):
            self.assertIs(call.args[0], X)
            self.assertIs(call.args[1], y)
            self.assertIs(call.kwargs["groups"], groups)
            self.assertIs(call.kwargs["continuous"], True)
            self.assertEqual(call.kwargs["seed"], seed)

    def test_default_and_numpy_integer_assignment_counts(self):
        for requested, expected in ((None, 10), (np.int64(1), 1)):
            with self.subTest(n_assignments=requested):
                signature = inspect.signature(analysis.probe_assignment_means)
                with patch.object(analysis, "probe_assignment_means",
                                  return_value=np.zeros(expected)) as probe:
                    kwargs = {} if requested is None else {"n_assignments": requested}
                    result = analysis.pairwise_assignment_gains(
                        self.embeddings, self.keys[:2], np.arange(4.), np.arange(4), **kwargs
                    )
                self.assertEqual(list(result), [("zeta", "alpha")])
                self.assertEqual(result["zeta", "alpha"].shape, (expected,))
                for call in probe.call_args_list:
                    bound = signature.bind(*call.args, **call.kwargs)
                    bound.apply_defaults()
                    self.assertEqual(bound.arguments["n_assignments"], expected)

    def test_invalid_inputs_fail_before_any_probe_runs(self):
        valid = dict(embeddings=self.embeddings, keys=self.keys, y=np.arange(4.),
                     groups=np.arange(4), n_assignments=2)
        invalid = [
            {"keys": []}, {"keys": ["zeta"]}, {"keys": ["zeta", "zeta"]},
            {"embeddings": {**self.embeddings, "mu": np.zeros(4)}},
            {"embeddings": {**self.embeddings, "mu": np.zeros((4, 2, 1))}},
            {"embeddings": {**self.embeddings, "mu": np.zeros((3, 1))}},
            {"y": np.array(1.)}, {"y": np.zeros((4, 2, 1))}, {"y": np.zeros(3)},
            {"groups": None}, {"groups": np.array(1)},
            {"groups": np.zeros((4, 1))}, {"groups": np.zeros(3)},
        ]
        invalid += [{"n_assignments": value} for value in
                    (0, -1, np.int64(-2), 2.5, 2., True, np.bool_(True), "2", None)]
        cases = [(change, ValueError) for change in invalid]
        cases.append(({"keys": ["zeta", "missing"]}, KeyError))
        for change, error in cases:
            with self.subTest(change=change):
                with patch.object(analysis, "probe_assignment_means") as probe:
                    with self.assertRaises(error):
                        analysis.pairwise_assignment_gains(**(valid | change))
                    probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
