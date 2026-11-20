"""Check the pair-task splits, controls and binary scoring on small tables.

The properties asserted are the ones a binding result depends on: a held-out entity group
never appears in training, a dual-disjoint split reports what it discarded, reassignment
builds pairs nobody measured and labels them as such, the degree baseline reads no sequence,
and the two metrics behave on cases whose value is known by hand.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import pair_tasks as PT, train  # noqa: E402


def frame(n_left=12, n_right=12, seed=0, groups=True, density=1.0):
    """A stub table of pairs; density below one leaves some partners unrecorded."""
    rng = np.random.default_rng(seed)
    lefts = [f"R{i:02d}" for i in range(n_left)]
    rights = [f"P{i:02d}" for i in range(n_right)]
    rows = [{"left": a, "right": b,
             "label": int(rng.random() < 0.3),
             "provenance": "measured_positive"} for a in lefts for b in rights
            if density >= 1.0 or rng.random() < density]
    for row in rows:
        row["provenance"] = "measured_positive" if row["label"] else "measured_negative"
    return PT.PairFrame(
        "stub", pd.DataFrame(rows),
        {a: "ACGU" * 10 for a in lefts}, {b: "MKV" * 10 for b in rights},
        # Pairs of entities share a group, so a split cannot separate relatives.
        {a: f"lg{i // 2}" for i, a in enumerate(lefts)} if groups else {},
        {b: f"rg{i // 2}" for i, b in enumerate(rights)} if groups else {})


class FrameTests(unittest.TestCase):
    def test_a_repeated_pair_is_refused(self):
        table = frame().pairs
        doubled = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            PT.PairFrame("stub", doubled, {}, {})

    def test_an_unknown_provenance_is_refused(self):
        table = frame().pairs.copy()
        table.loc[0, "provenance"] = "assumed"
        with self.assertRaises(ValueError):
            PT.PairFrame("stub", table, {a: "" for a in table["left"]},
                         {b: "" for b in table["right"]})

    def test_ungrouped_entities_are_counted_rather_than_assumed_independent(self):
        self.assertEqual(frame(groups=False).grouping_coverage()["left"]["grouped"], 0)
        self.assertEqual(frame().grouping_coverage()["left"]["grouped"], 12)


class SplitTests(unittest.TestCase):
    def entities(self, pairs, side):
        return set(pairs[side])

    def test_a_held_out_left_group_never_appears_in_training(self):
        pairs = frame()
        split = PT.disjoint_split(pairs, held_out="left", seed=1)
        groups = {name: {pairs.group_of("left", e)
                         for e in self.entities(pairs.pairs.iloc[rows], "left")}
                  for name, rows in split["indices"].items()}
        self.assertEqual(groups["train"] & groups["test"], set())
        self.assertEqual(groups["train"] & groups["val"], set())
        self.assertEqual(split["dropped"], 0)

    def test_a_right_disjoint_split_holds_out_the_other_side(self):
        pairs = frame()
        split = PT.disjoint_split(pairs, held_out="right", seed=1)
        train_right = self.entities(pairs.pairs.iloc[split["indices"]["train"]], "right")
        test_right = self.entities(pairs.pairs.iloc[split["indices"]["test"]], "right")
        self.assertEqual(train_right & test_right, set())
        # The other side is deliberately shared, which is what makes this a different question.
        self.assertTrue(self.entities(pairs.pairs.iloc[split["indices"]["train"]], "left")
                        & self.entities(pairs.pairs.iloc[split["indices"]["test"]], "left"))

    def test_a_dual_disjoint_split_separates_both_sides_and_reports_the_discards(self):
        pairs = frame(n_left=20, n_right=20)
        split = PT.disjoint_split(pairs, held_out="both", seed=3)
        kept = sum(len(rows) for rows in split["indices"].values())
        self.assertEqual(kept + split["dropped"], len(pairs.pairs))
        self.assertGreater(split["dropped"], 0)
        for side in ("left", "right"):
            train = self.entities(pairs.pairs.iloc[split["indices"]["train"]], side)
            test = self.entities(pairs.pairs.iloc[split["indices"]["test"]], side)
            self.assertEqual(train & test, set())

    def test_the_split_is_reproducible_and_seed_dependent(self):
        pairs = frame()
        first = PT.disjoint_split(pairs, seed=5)["indices"]
        again = PT.disjoint_split(pairs, seed=5)["indices"]
        other = PT.disjoint_split(pairs, seed=6)["indices"]
        for name in PT.SPLITS:
            np.testing.assert_array_equal(first[name], again[name])
        self.assertTrue(any(len(first[n]) != len(other[n])
                            or not np.array_equal(first[n], other[n]) for n in PT.SPLITS))

    def test_the_partitions_land_near_the_requested_ratios(self):
        pairs = frame(n_left=40, n_right=10)
        split = PT.disjoint_split(pairs, held_out="left", ratios=(0.7, 0.15, 0.15), seed=2)
        total = len(pairs.pairs)
        for name, wanted in zip(PT.SPLITS, (0.7, 0.15, 0.15)):
            self.assertAlmostEqual(len(split["indices"][name]) / total, wanted, delta=0.08)

    def test_an_empty_partition_is_an_error_rather_than_a_silent_split(self):
        pairs = frame(n_left=2, n_right=2)
        with self.assertRaises(ValueError):
            PT.disjoint_split(pairs, held_out="left", seed=0)

    def test_the_split_record_counts_label_provenance_per_partition(self):
        split = PT.disjoint_split(frame(), held_out="left", seed=1)
        for name in PT.SPLITS:
            self.assertEqual(sum(split["provenance"][name].values()),
                             len(split["indices"][name]))


class ControlTests(unittest.TestCase):
    def test_reassignment_builds_unmeasured_pairs_and_says_so(self):
        pairs = frame(seed=4, density=0.5)
        rows = np.arange(len(pairs.pairs))
        control = PT.partner_reassignment(pairs, rows, seed=0)
        self.assertTrue((control.pairs["label"] == 1).all())
        self.assertTrue((control.pairs["provenance"] == "artificial_reassigned").all())
        measured = set(zip(pairs.pairs["left"], pairs.pairs["right"]))
        self.assertEqual([p for p in zip(control.pairs["left"], control.pairs["right"])
                          if p in measured], [])

    def test_the_degree_baseline_uses_partner_frequency_and_no_sequence(self):
        """Labels driven entirely by which entities are popular; the baseline should see it."""
        rng = np.random.default_rng(0)
        lefts = [f"R{i}" for i in range(20)]
        rights = [f"P{i}" for i in range(10)]
        strong = set(lefts[:5])
        rows = [{"left": a, "right": b,
                 "label": int(rng.random() < (0.9 if a in strong else 0.1)),
                 "provenance": "measured_positive"} for a in lefts for b in rights]
        pairs = PT.PairFrame("popularity", pd.DataFrame(rows),
                             {a: "" for a in lefts}, {b: "" for b in rights})
        order = rng.permutation(len(pairs.pairs))
        train, test = order[:150], order[150:]
        scores = PT.degree_baseline(pairs, train, test)
        truth = pairs.pairs.iloc[test]["label"].to_numpy()
        self.assertGreater(PT.auroc(truth, scores), 0.8)

    def test_reassignment_is_refused_where_every_partner_is_already_recorded(self):
        pairs = frame(seed=4)  # dense: every left entity is recorded against every right
        with self.assertRaises(ValueError):
            PT.partner_reassignment(pairs, np.arange(len(pairs.pairs)), seed=0)

    def test_the_degree_baseline_falls_back_for_an_entity_unseen_in_training(self):
        pairs = frame()
        split = PT.disjoint_split(pairs, held_out="left", seed=1)
        scores = PT.degree_baseline(pairs, split["indices"]["train"], split["indices"]["test"])
        self.assertEqual(len(scores), len(split["indices"]["test"]))
        self.assertTrue(np.isfinite(scores).all())


class MetricTests(unittest.TestCase):
    def test_known_values(self):
        truth = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(PT.auroc(truth, np.array([0.1, 0.2, 0.3, 0.4])), 1.0)
        self.assertAlmostEqual(PT.auroc(truth, np.array([0.4, 0.3, 0.2, 0.1])), 0.0)
        self.assertAlmostEqual(PT.auroc(truth, np.ones(4)), 0.5)
        self.assertAlmostEqual(PT.average_precision(truth, np.array([0.1, 0.2, 0.3, 0.4])), 1.0)
        self.assertAlmostEqual(PT.average_precision(truth, np.ones(4)), 0.5)

    def test_average_precision_matches_a_hand_computed_ranking(self):
        # Ranked 1, 0, 1, 0: precision 1.0 at the first hit and 2/3 at the second.
        truth = np.array([1, 0, 1, 0])
        value = PT.average_precision(truth, np.array([0.9, 0.8, 0.7, 0.6]))
        self.assertAlmostEqual(value, (1.0 + 2 / 3) / 2)

    def test_one_class_is_reported_as_undefined_rather_than_guessed(self):
        self.assertTrue(np.isnan(PT.auroc(np.ones(4), np.arange(4.0))))
        self.assertTrue(np.isnan(PT.average_precision(np.zeros(4), np.arange(4.0))))

    def test_the_metrics_agree_with_scikit_learn(self):
        from sklearn.metrics import average_precision_score, roc_auc_score
        rng = np.random.default_rng(0)
        for _ in range(5):
            truth = (rng.random(60) < 0.4).astype(int)
            scores = rng.random(60).round(2)  # rounded, so ties occur
            self.assertAlmostEqual(PT.auroc(truth, scores), roc_auc_score(truth, scores), places=9)
            self.assertAlmostEqual(PT.average_precision(truth, scores),
                                   average_precision_score(truth, scores), places=9)

    def test_the_shared_bootstrap_runs_on_a_pair_metric(self):
        """Pair comparisons reuse the project's grouped paired bootstrap, metric apart."""
        rng = np.random.default_rng(0)
        truth = (rng.random(80) < 0.5).astype(int)
        better = truth + rng.normal(scale=0.5, size=80)
        worse = rng.normal(size=80)
        groups = np.repeat(np.arange(40), 2)  # each entity measured twice
        result = train.seed_averaged_bootstrap(
            "auroc", truth, {"fusion": [better], "single": [worse]},
            [("fusion vs single", "fusion", "single")], groups=groups, draws=200,
            score_fn=PT.score_rows)[0]
        self.assertGreater(result["difference"], 0)
        self.assertLess(result["ci_low"], result["ci_high"])

    def test_binding_scores_report_the_positive_rate_the_curve_is_read_against(self):
        scores = PT.binding_scores(np.array([1, 0, 0, 0]), np.array([0.9, 0.1, 0.2, 0.3]))
        self.assertEqual(scores["positives"], 1)
        self.assertAlmostEqual(scores["positive_rate"], 0.25)
        self.assertAlmostEqual(scores["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
