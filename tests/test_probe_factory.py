"""Synthetic checks of the promised shared estimator and independent fits."""

from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code import analysis
from shared_code.splits import grouped_folds


def _prior_pipeline(continuous=True):
    """Retain the original construction as an independent numerical reference."""
    model = (RidgeCV(alphas=np.logspace(-3, 5, 20)) if continuous
             else LogisticRegression(max_iter=2000))
    return Pipeline([("scale", StandardScaler()), ("model", model)])


def _prior_scores(X, y, groups=None, continuous=True, seed=42):
    cv = 5 if groups is None else grouped_folds(groups, seed=seed)
    return cross_val_score(_prior_pipeline(continuous), X, y, cv=cv,
                           scoring="r2" if continuous else "accuracy")


class ProbeFactoryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(145)
        self.X = rng.normal(size=(100, 5)) * [1, 100, 0.2, 5, 30]
        self.y = self.X @ rng.normal(size=5) + rng.normal(size=100)
        self.multi = self.X @ rng.normal(size=(5, 30)) + rng.normal(size=(100, 30))
        self.groups = np.repeat([f"group-{i}" for i in range(50)], 2)

    def test_regression_scores_equal_prior_behavior(self):
        for y in (self.y, self.multi):
            for options in ({}, {"seed": 0}, {"seed": 42},
                            {"groups": self.groups},
                            {"groups": self.groups, "seed": 0}):
                with self.subTest(shape=y.shape, grouped="groups" in options,
                                  seed=options.get("seed", "default")):
                    np.testing.assert_array_equal(
                        analysis.probe_scores(self.X, y, **options),
                        _prior_scores(self.X, y, **options),
                    )

    def test_classification_scores_equal_prior_behavior(self):
        labels = (self.y > np.median(self.y)).astype(int)
        for groups in (self.groups, None):
            with self.subTest(grouped=groups is not None):
                np.testing.assert_array_equal(
                    analysis.probe_scores(self.X, labels, groups=groups, continuous=False),
                    _prior_scores(self.X, labels, groups=groups, continuous=False),
                )

    def test_fixed_split_predictions_equal_prior_behavior(self):
        for y in (self.y, self.multi):
            with self.subTest(shape=y.shape):
                np.testing.assert_array_equal(
                    analysis.fit_predict(self.X[:70], y[:70], self.X[70:]),
                    _prior_pipeline().fit(self.X[:70], y[:70]).predict(self.X[70:]),
                )

    def test_entry_points_agree_on_each_held_out_fold(self):
        for y in (self.y, self.multi):
            with self.subTest(shape=y.shape):
                independent = np.array([
                    r2_score(y[test], analysis.fit_predict(self.X[train], y[train], self.X[test]))
                    for train, test in grouped_folds(self.groups)
                ])
                np.testing.assert_array_equal(
                    analysis.probe_scores(self.X, y, groups=self.groups), independent
                )

    def test_estimators_own_independent_fitted_state(self):
        first = analysis._ridge_pipeline()
        second = analysis._ridge_pipeline()
        first.fit(self.X[:70], self.y[:70])
        expected = first.predict(self.X[70:])
        second.fit(self.X[:70] + 500, self.y[:70] * -3)
        np.testing.assert_array_equal(first.predict(self.X[70:]), expected)
        self.assertFalse(hasattr(analysis._ridge_pipeline(), "n_features_in_"))


if __name__ == "__main__":
    unittest.main()
