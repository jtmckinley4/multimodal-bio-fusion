"""Check pinned probe seeds and pair orientation using synthetic examples only."""

import ast
import inspect
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from mbf import analysis
from mbf.splits import grouped_folds


def _reference_scores(X, y, groups, seed=42):
    """Use the pre-refactor estimator directly, independently of probe wrappers."""
    pipeline = Pipeline([("scale", StandardScaler()),
                         ("model", RidgeCV(alphas=np.logspace(-3, 5, 20)))])
    folds = 5 if groups is None else grouped_folds(groups, seed=seed)
    return cross_val_score(pipeline, X, y, cv=folds, scoring="r2")


class ProbeSeedTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(920)
        self.keys = ["zeta", "alpha", "mu"]
        self.embeddings = {key: rng.normal(size=(80, width))
                           for key, width in zip(self.keys, (2, 3, 1))}
        design = np.concatenate(list(self.embeddings.values()), axis=1)
        self.scalar = design @ rng.normal(size=design.shape[1]) + rng.normal(size=80)
        self.multi = design @ rng.normal(size=(design.shape[1], 30))
        self.multi += rng.normal(size=self.multi.shape)
        self.groups = np.repeat([f"sequence-{i}" for i in range(40)], 2)

    def test_default_tables_and_pairs_match_prior_scalar_and_multioutput(self):
        for y in (self.scalar, self.multi):
            for groups in (self.groups, None):
                with self.subTest(shape=y.shape, grouped=groups is not None):
                    rows = []
                    for key, X in self.embeddings.items():
                        scores = _reference_scores(X, y, groups)
                        means = np.array([_reference_scores(X, y, groups, seed).mean()
                                          for seed in range(2)])
                        rows.append({"encoder": key, "target R2": scores.mean(),
                                     "target SD": scores.std(),
                                     "target R2, 2 assignments": means.mean(),
                                     "target SD, 2 assignments": means.std()})
                    expected = pd.DataFrame(rows).set_index("encoder")
                    actual = analysis.probe_table(
                        self.embeddings, {"target": y}, groups, n_assignments=2
                    )
                    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
                    expected_pairs = pd.DataFrame(np.nan, index=self.keys, columns=self.keys)
                    expected_gains = expected_pairs.copy()
                    for a, b in itertools.combinations(self.keys, 2):
                        fused = np.concatenate([self.embeddings[a], self.embeddings[b]], axis=1)
                        score = _reference_scores(fused, y, groups).mean()
                        expected_pairs.loc[a, b] = expected_pairs.loc[b, a] = score
                        gain = score - max(expected.loc[a, "target R2"], expected.loc[b, "target R2"])
                        expected_gains.loc[a, b] = expected_gains.loc[b, a] = gain
                    pairs, gains = analysis.pairwise_concatenation_scores(
                        self.embeddings, self.keys, y, actual["target R2"], groups
                    )
                    pd.testing.assert_frame_equal(pairs, expected_pairs, check_exact=True)
                    pd.testing.assert_frame_equal(gains, expected_gains, check_exact=True)

    def test_nondefault_seed_uses_same_partition_for_singles_and_pairs(self):
        seed = 17
        expected_folds = grouped_folds(self.groups, seed=seed)
        default_folds = grouped_folds(self.groups)
        self.assertTrue(any(not np.array_equal(a[1], b[1])
                            for a, b in zip(expected_folds, default_folds)))
        for y in (self.scalar, self.multi):
            with self.subTest(shape=y.shape), patch.object(
                    analysis, "cross_val_score", wraps=analysis.cross_val_score) as score_call:
                singles = analysis.probe_table(
                    self.embeddings, {"target": y}, self.groups, n_assignments=0, seed=seed
                )
                pairs, gains = analysis.pairwise_concatenation_scores(
                    self.embeddings, self.keys, y, singles["target R2"], self.groups, seed=seed
                )
                self.assertEqual(score_call.call_count, 6)
                for call in score_call.call_args_list:
                    for actual, expected in zip(call.kwargs["cv"], expected_folds):
                        np.testing.assert_array_equal(actual[0], expected[0])
                        np.testing.assert_array_equal(actual[1], expected[1])
            for key, X in self.embeddings.items():
                scores = _reference_scores(X, y, self.groups, seed)
                self.assertEqual(singles.loc[key, "target R2"], scores.mean())
                self.assertEqual(singles.loc[key, "target SD"], scores.std())
            for a, b in itertools.combinations(self.keys, 2):
                fused = np.concatenate([self.embeddings[a], self.embeddings[b]], axis=1)
                expected = _reference_scores(fused, y, self.groups, seed).mean()
                self.assertEqual(pairs.loc[a, b], expected)
                self.assertEqual(gains.loc[a, b], expected - max(
                    singles.loc[a, "target R2"], singles.loc[b, "target R2"]))

    def test_pinned_seed_does_not_shift_repeated_assignment_seeds(self):
        targets = {"scalar": self.scalar, "multi": self.multi}
        signature = inspect.signature(analysis.probe_scores)
        for options, count in (({}, 10), ({"n_assignments": 3}, 3)):
            with self.subTest(n_assignments=count), patch.object(
                    analysis, "probe_scores", return_value=np.arange(5.)) as probe:
                analysis.probe_table(self.embeddings, targets, self.groups, seed=77, **options)
            seeds = []
            for call in probe.call_args_list:
                bound = signature.bind(*call.args, **call.kwargs)
                bound.apply_defaults()
                seeds.append(bound.arguments["seed"])
                self.assertIs(bound.arguments["groups"], self.groups)
            self.assertEqual(seeds, [77, *range(count)] * len(self.embeddings) * len(targets))

    def test_new_seed_arguments_are_keyword_only(self):
        for function in (analysis.probe_table, analysis.pairwise_concatenation_scores):
            with self.subTest(function=function.__name__):
                seed = inspect.signature(function).parameters["seed"]
                self.assertEqual(seed.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertEqual(seed.default, 42)

    def test_notebooks_route_pinned_probes_through_the_setup_seed(self):
        # Reuse the existing Setup boundary parser without executing notebook code.
        from test_notebook_setup import NOTEBOOKS, read_notebook, setup_cells, source

        functions = {"probe_table", "probe_scores", "pairwise_concatenation_scores"}
        for name in NOTEBOOKS:
            with self.subTest(notebook=name):
                cells = read_notebook(name)
                assignments = [node.value for _, cell in setup_cells(cells)
                               for node in ast.walk(ast.parse(source(cell)))
                               if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "PROBE_SEED"
                                       for target in node.targets)]
                self.assertEqual(len(assignments), 1)
                self.assertIsInstance(assignments[0], ast.Constant)
                self.assertEqual(assignments[0].value, 42)
                calls = [node for cell in cells if cell["cell_type"] == "code"
                         for node in ast.walk(ast.parse(source(cell)))
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                         and isinstance(node.func.value, ast.Name)
                         and node.func.value.id == "analysis" and node.func.attr in functions]
                self.assertTrue(calls)
                for call in calls:
                    seeds = [keyword.value for keyword in call.keywords if keyword.arg == "seed"]
                    self.assertEqual(len(seeds), 1, f"{name}: {ast.unparse(call)}")
                    self.assertIsInstance(seeds[0], ast.Name)
                    self.assertEqual(seeds[0].id, "PROBE_SEED")

    def test_pairwise_matrix_mirrors_one_ordered_asymmetric_evaluation(self):
        values = {"zeta": 1, "alpha": 3, "mu": 8}
        calls = []

        def asymmetric(a, b):
            calls.append((a, b))
            return 10 * values[a] + values[b]

        actual = analysis.pairwise_matrix(self.keys, asymmetric)
        self.assertEqual(calls, [("zeta", "alpha"), ("zeta", "mu"), ("alpha", "mu")])
        expected = pd.DataFrame([[np.nan, 13., 18.], [13., np.nan, 38.], [18., 38., np.nan]],
                                index=self.keys, columns=self.keys)
        pd.testing.assert_frame_equal(actual, expected)
        self.assertEqual(self.keys, ["zeta", "alpha", "mu"])


if __name__ == "__main__":
    unittest.main()
