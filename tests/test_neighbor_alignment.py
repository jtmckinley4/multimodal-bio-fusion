"""Check neighbor overlap on invented arrays, without reading research results."""

from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

import numpy as np

from shared_code import analysis


class NeighborAlignmentTests(unittest.TestCase):
    def test_duplicate_vectors_exclude_the_query_by_identity(self):
        # X has three duplicate pairs. Y shares three of the six directed neighbors:
        # 0 -> 1, 1 -> 0 and 5 -> 4. Identical vectors still represent distinct rows.
        X = np.array([[0.], [0.], [10.], [10.], [30.], [30.]])
        Y = np.array([[0.], [1.], [20.], [60.], [61.], [90.]])
        self.assertEqual(analysis.mutual_knn_alignment(X, Y, k=1), 0.5)

    def overlap_for_tied_orders(self, x_neighbors, y_neighbors):
        # All coordinates coincide, so these library orders are valid distance ties.
        # Inject the orders to make self-later and self-absent cases reproducible
        # across sklearn versions without imposing a new nonself tie-break rule.
        X = np.zeros((len(x_neighbors), 2))
        Y = np.zeros((len(y_neighbors), 3))
        x_model, y_model = Mock(), Mock()
        for model, neighbors in ((x_model, x_neighbors), (y_model, y_neighbors)):
            model.fit.return_value = model
            model.kneighbors.return_value = np.array(neighbors)
        with patch.object(analysis, "NearestNeighbors", side_effect=[x_model, y_model]):
            return analysis.mutual_knn_alignment(X, Y, k=1)

    def test_self_can_appear_after_another_tied_neighbor(self):
        # Each query is second in both lists. Removing it gives disjoint sets;
        # dropping the first entry instead would spuriously give overlap 1.
        actual = self.overlap_for_tied_orders(
            [[1, 0], [0, 1], [1, 2]],
            [[2, 0], [2, 1], [0, 2]],
        )
        self.assertEqual(actual, 0.0)

    def test_self_can_be_absent_from_the_returned_tied_neighbors(self):
        # More than k+1 rows tie at distance zero. If self is absent, keep the
        # first k returned neighbors; do not drop a valid row unconditionally.
        actual = self.overlap_for_tied_orders(
            [[1, 2], [0, 2], [0, 1], [0, 1]],
            [[2, 3], [2, 3], [0, 3], [1, 2]],
        )
        self.assertEqual(actual, 0.25)

    def test_no_tie_result_matches_previous_calculation(self):
        rng = np.random.default_rng(157)
        X, Y = rng.normal(size=(20, 3)), rng.normal(size=(20, 5))
        for k in (1, 3, 10, 19):
            with self.subTest(k=k):
                neighbors_x = analysis.NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(
                    X, return_distance=False
                )
                neighbors_y = analysis.NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(
                    Y, return_distance=False
                )
                np.testing.assert_array_equal(neighbors_x[:, 0], np.arange(len(X)))
                np.testing.assert_array_equal(neighbors_y[:, 0], np.arange(len(Y)))
                previous = np.mean([
                    len(set(neighbors_x[i, 1:]) & set(neighbors_y[i, 1:])) / k
                    for i in range(len(X))
                ])
                self.assertEqual(analysis.mutual_knn_alignment(X, Y, k), previous)

    def test_matched_row_order_symmetry_and_inputs_are_preserved(self):
        rng = np.random.default_rng(905)
        X, Y = rng.normal(size=(12, 2)), rng.normal(size=(12, 4))
        original_x, original_y = X.copy(), Y.copy()
        X.flags.writeable = Y.flags.writeable = False
        value = analysis.mutual_knn_alignment(X, Y, k=3)
        order = np.array([5, 3, 8, 1, 10, 7, 2, 11, 9, 0, 4, 6])
        self.assertGreaterEqual(value, 0.)
        self.assertLessEqual(value, 1.)
        self.assertEqual(value, analysis.mutual_knn_alignment(Y, X, k=3))
        self.assertAlmostEqual(value, analysis.mutual_knn_alignment(X[order], Y[order], k=3))
        self.assertEqual(analysis.mutual_knn_alignment(X, X, k=3), 1.)
        self.assertEqual(analysis.mutual_knn_alignment(X, Y, k=np.int64(11)), 1.)
        np.testing.assert_array_equal(X, original_x)
        np.testing.assert_array_equal(Y, original_y)

    def test_invalid_shapes_or_neighbor_count_fail_before_search(self):
        X = np.arange(8.).reshape(4, 2)
        invalid = [
            (X.ravel(), X, 1),
            (X, X[:3], 1),
            (X[:1], X[:1], 1),
            (X[:0], X[:0], 1),
            *[(X, X, k) for k in (0, -1, 4, 1.5, True)],
        ]
        for left, right, k in invalid:
            with self.subTest(left=left.shape, right=right.shape, k=k):
                with patch.object(analysis, "NearestNeighbors") as search:
                    with self.assertRaises(ValueError):
                        analysis.mutual_knn_alignment(left, right, k)
                    search.assert_not_called()


if __name__ == "__main__":
    unittest.main()
