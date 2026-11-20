"""Check the pair fitting loop on a small synthetic task with a known answer.

The task is built so that the label depends on the pair and not on either partner alone: a
pair is positive when its left and right entities carry matching hidden tags. A scorer that
reads both sides can learn it; the single-partner baselines cannot do better than chance. That
makes the loop's own properties checkable without any real binding data: selection reads
validation only, the test rows are scored once with the selected weights, and the controls
behave as a result with real evidence behind it should.

Nothing here says anything about binding. It says the machinery measures what it claims to.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import pair_tasks as PT, pair_training as PTR  # noqa: E402

WIDTH, TOKENS, TAGS = 8, 5, 4


def matched_task(n_left=48, n_right=24, seed=0):
    """Pairs are positive when both entities carry the same hidden tag."""
    rng = np.random.default_rng(seed)
    lefts = [f"R{i:03d}" for i in range(n_left)]
    rights = [f"P{i:03d}" for i in range(n_right)]
    tag = {name: int(rng.integers(TAGS)) for name in [*lefts, *rights]}
    vectors = {}
    for name in [*lefts, *rights]:
        tokens = rng.normal(scale=0.3, size=(TOKENS, WIDTH))
        tokens[:, tag[name]] += 2.0        # the tag is visible in the embeddings
        vectors[name] = tokens.astype(np.float32)
    rows = [{"left": a, "right": b, "label": int(tag[a] == tag[b]),
             "provenance": "measured_positive" if tag[a] == tag[b] else "measured_negative"}
            for a in lefts for b in rights if rng.random() < 0.5]
    frame = PT.PairFrame("matched", pd.DataFrame(rows),
                         {a: "" for a in lefts}, {b: "" for b in rights})
    embeddings = {"left": PTR.DictEmbeddings({a: vectors[a] for a in lefts}),
                  "right": PTR.DictEmbeddings({b: vectors[b] for b in rights})}
    return frame, embeddings


def config(**kwargs):
    return PTR.PairConfig(**{"max_epochs": 6, "batch_size": 16, "early_stopping_patience": 3,
                             "lr": 5e-3, **kwargs})


class FittingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.frame, cls.embeddings = matched_task()
        cls.split = PT.disjoint_split(cls.frame, held_out="left", seed=11)
        cls.results = {}
        for arm, kwargs in (("left_only", {"hidden": 32}), ("right_only", {"hidden": 32}),
                            ("concat", {"hidden": 32}), ("bilinear", {"hidden": 32})):
            cls.results[arm] = PTR.fit(arm, cls.frame, cls.split, cls.embeddings, config(),
                                       model_kwargs=kwargs, log=None)

    def test_a_pair_reading_arm_learns_the_task_and_the_baselines_cannot(self):
        """Only the arms that read both sides can see a label that lives in the pair.

        A single-partner arm is not pinned to chance: an entity whose tag is common among
        its partners is positive more often, which it can learn. What it cannot do is
        approach an arm that reads the pair.
        """
        paired = self.results["concat"]["test"]["auroc"]
        self.assertGreater(paired, 0.9)
        for arm in ("left_only", "right_only"):
            self.assertGreater(paired - self.results[arm]["test"]["auroc"], 0.25, arm)

    def test_every_arm_scores_the_same_test_rows_in_the_same_order(self):
        rows = self.results["concat"]["test_rows"]
        for arm, result in self.results.items():
            self.assertEqual(result["test_rows"], rows, arm)
            self.assertEqual(len(result["test_score"]), len(rows), arm)

    def test_the_selected_epoch_comes_from_validation(self):
        result = self.results["concat"]
        best = max(result["history"], key=lambda row: row["val_auprc"])
        self.assertEqual(result["best_epoch"], best["epoch"])

    def test_the_recipe_and_the_split_travel_with_the_result(self):
        result = self.results["concat"]
        self.assertEqual(result["selection_metric"], "auprc")
        self.assertEqual(result["split"]["held_out"], "left")
        self.assertEqual(sum(result["split"]["provenance"]["test"].values()),
                         len(result["test_rows"]))

    def test_the_positive_weight_comes_from_the_training_rows_only(self):
        labels = self.frame.pairs.iloc[self.split["indices"]["train"]]["label"].to_numpy()
        expected = (len(labels) - labels.sum()) / labels.sum()
        self.assertAlmostEqual(self.results["concat"]["positive_weight"], expected, places=5)

    def test_an_affinity_table_runs_through_the_same_loop(self):
        """Step 8's endpoint is continuous; the arms, splits and intervals do not change."""
        rng = np.random.default_rng(0)
        frame, embeddings = matched_task(seed=2)
        table = frame.pairs.copy()
        table["label"] = table["label"] * 2.0 + rng.normal(scale=0.5, size=len(table))
        table["endpoint"] = "pKd"
        affinity = PT.PairFrame("affinity", table, frame.left_sequences, frame.right_sequences,
                                kind="affinity")
        split = PT.disjoint_split(affinity, held_out="left", seed=11)
        result = PTR.fit("concat", affinity, split, embeddings, config(max_epochs=6),
                         model_kwargs={"hidden": 32}, log=None)
        self.assertEqual(result["selection_metric"], "spearman")
        self.assertIsNone(result["positive_weight"])
        self.assertGreater(result["test"]["spearman"], 0.5)
        self.assertGreater(result["test"]["concordance"], 0.6)
        self.assertTrue(np.isfinite(result["test"]["rmse"]))
        controls = PTR.control_scores(result, affinity, split, embeddings, config())
        self.assertNotIn("partner_reassigned", controls)  # nothing measured to compare with
        comparison = PTR.comparison_rows([result, result | {"arm": "left_only"}], affinity,
                                         split, draws=50)
        self.assertEqual(comparison["metric"], "spearman")

    def test_an_affinity_table_mixing_assay_endpoints_is_refused(self):
        frame, _ = matched_task(seed=3)
        table = frame.pairs.copy()
        table["label"] = table["label"].astype(float)
        table["endpoint"] = ["pKd" if i % 2 else "IC50" for i in range(len(table))]
        with self.assertRaises(ValueError):
            PT.PairFrame("mixed", table, frame.left_sequences, frame.right_sequences,
                         kind="affinity")

    def test_turning_the_weight_off_is_recorded(self):
        result = PTR.fit("concat", self.frame, self.split, self.embeddings,
                         config(max_epochs=1, positive_weight="none"),
                         model_kwargs={"hidden": 16}, log=None)
        self.assertIsNone(result["positive_weight"])

    def test_a_fit_is_reproducible_at_one_seed_and_moves_with_another(self):
        settings = dict(model_kwargs={"hidden": 16}, log=None)
        first = PTR.fit("concat", self.frame, self.split, self.embeddings,
                        config(max_epochs=2, seed=3), **settings)["test_score"]
        again = PTR.fit("concat", self.frame, self.split, self.embeddings,
                        config(max_epochs=2, seed=3), **settings)["test_score"]
        other = PTR.fit("concat", self.frame, self.split, self.embeddings,
                        config(max_epochs=2, seed=4), **settings)["test_score"]
        np.testing.assert_allclose(first, again, rtol=1e-6, atol=1e-6)
        self.assertGreater(np.abs(np.asarray(first) - np.asarray(other)).max(), 1e-6)

    def test_predictions_follow_the_rows_they_were_asked_for(self):
        """Batching must not reorder a row's score, whatever order the rows arrive in."""
        rows = self.split["indices"]["test"]
        model = self.results["concat"]["model"]
        forward = PTR.predict(model, PTR.PairBatches(self.frame, rows, self.embeddings,
                                                     config()))
        backward = PTR.predict(model, PTR.PairBatches(self.frame, rows[::-1], self.embeddings,
                                                      config()))
        np.testing.assert_allclose(forward, backward[::-1], rtol=1e-5, atol=1e-5)

    def test_long_entities_are_truncated_at_the_recorded_limit(self):
        batches = PTR.PairBatches(self.frame, self.split["indices"]["test"][:4],
                                  self.embeddings, config(max_tokens=2))
        left, _, left_mask, _, _, _ = next(iter(batches))
        self.assertEqual(left.shape[1], 2)
        self.assertTrue(bool(left_mask.all()))


class ControlTests(FittingTests):
    def test_the_degree_baseline_and_the_reassignment_set_are_reported(self):
        controls = PTR.control_scores(self.results["concat"], self.frame, self.split,
                                      self.embeddings, config(), seed=1)
        self.assertFalse(controls["degree"]["reads_sequences"])
        self.assertTrue(np.isfinite(controls["degree"]["auroc"]))
        self.assertGreater(controls["partner_reassigned"]["rows"], 0)
        # A scorer that reads the pair scores its real positives above invented partners.
        self.assertGreater(controls["partner_reassigned"]["mean_score_real"],
                           controls["partner_reassigned"]["mean_score_reassigned"])

    def test_comparisons_are_paired_against_the_stronger_baseline(self):
        comparison = PTR.comparison_rows(list(self.results.values()), self.frame, self.split,
                                         draws=100, seed=0)
        self.assertIn(comparison["reference"], ("left_only", "right_only"))
        names = {row["comparison"] for row in comparison["rows"]}
        self.assertIn(f"concat vs {comparison['reference']}", names)
        concat = next(row for row in comparison["rows"]
                      if row["comparison"].startswith("concat vs"))
        self.assertGreater(concat["difference"], 0)
        self.assertLessEqual(concat["ci_low"], concat["difference"])

    def test_arms_scored_on_different_rows_are_refused(self):
        other = dict(self.results["concat"])
        other["test_rows"] = other["test_rows"][:-1]
        with self.assertRaises(ValueError):
            PTR.comparison_rows([self.results["left_only"], other], self.frame, self.split)


class MicrobatchTests(unittest.TestCase):
    """Slicing a batch is a memory setting, so it must not change the fit.

    Cross-attention between two long streams holds an attention matrix per head per direction,
    which is what exhausted a laptop's GPU memory on the first real run. The fix puts the batch
    through in slices and accumulates, which is only correct if the step it takes is the step
    the whole batch would have taken.
    """

    def setUp(self):
        self.frame, self.embeddings = matched_task(seed=3)
        self.split = PT.disjoint_split(self.frame, held_out="left", seed=1)

    def config(self, **changes):
        return PTR.PairConfig(max_epochs=3, batch_size=8, early_stopping_patience=5,
                              lr=5e-3, seed=0, **changes)

    def test_a_sliced_batch_takes_the_same_step_as_the_whole_batch(self):
        whole = PTR.fit("concat", self.frame, self.split, self.embeddings,
                        self.config(), log=None)
        sliced = PTR.fit("concat", self.frame, self.split, self.embeddings,
                         self.config(microbatch=3), log=None)
        np.testing.assert_allclose(sliced["test_score"], whole["test_score"],
                                   rtol=1e-4, atol=1e-4)
        self.assertEqual(sliced["best_epoch"], whole["best_epoch"])
        for a, b in zip(sliced["history"], whole["history"]):
            self.assertAlmostEqual(a["train_loss"], b["train_loss"], places=4)

    def test_a_microbatch_at_or_above_the_batch_size_changes_nothing(self):
        whole = PTR.fit("concat", self.frame, self.split, self.embeddings,
                        self.config(), log=None)
        same = PTR.fit("concat", self.frame, self.split, self.embeddings,
                       self.config(microbatch=64), log=None)
        np.testing.assert_allclose(same["test_score"], whole["test_score"],
                                   rtol=1e-5, atol=1e-5)

    def test_a_ragged_last_slice_is_weighted_by_its_own_size(self):
        """8 pairs in slices of 3 is 3, 3, 2; an unweighted mean would overweight the last."""
        whole = PTR.fit("bilinear", self.frame, self.split, self.embeddings,
                        self.config(), log=None)
        sliced = PTR.fit("bilinear", self.frame, self.split, self.embeddings,
                         self.config(microbatch=3), log=None)
        np.testing.assert_allclose(sliced["test_score"], whole["test_score"],
                                   rtol=1e-4, atol=1e-4)

    def test_the_slices_cover_the_batch_exactly_once(self):
        config = self.config(microbatch=3)
        self.assertEqual(config.slices(8), [(0, 3), (3, 6), (6, 8)])
        self.assertEqual(self.config().slices(8), [(0, 8)])
        covered = [index for start, stop in config.slices(8) for index in range(start, stop)]
        self.assertEqual(covered, list(range(8)))

    def test_a_nonsense_microbatch_is_refused(self):
        with self.assertRaises(ValueError):
            self.config(microbatch=0).slices(8)

    def test_an_unset_microbatch_is_left_out_of_the_identity(self):
        """A memory setting that does nothing must not invalidate fits saved before it existed."""
        self.assertNotIn("microbatch", self.config().as_record())
        self.assertEqual(self.config(microbatch=4).as_record()["microbatch"], 4)


if __name__ == "__main__":
    unittest.main()


class BucketTests(unittest.TestCase):
    """Grouping a batch by length is a memory setting, and it is recorded because it is not free.

    A batch is padded to its longest member. When a table holds a few entities at the token cap
    and many short ones, a batch drawn at random is nearly always padded to the cap, and the
    model spends most of its arithmetic and nearly all of its memory on padding. Grouping rows
    of similar length removes that waste, but it changes which rows share a batch, so unlike
    microbatching it changes the fit and has to appear in the identity.
    """

    def ragged_task(self):
        """Four short RNAs and one long one, so the lengths actually differ."""
        rng = np.random.default_rng(0)
        lengths = {"R000": 4, "R001": 4, "R002": 4, "R003": 4, "R004": 64}
        lefts = {name: rng.normal(size=(length, WIDTH)).astype(np.float32)
                 for name, length in lengths.items()}
        rights = {f"P{i:03d}": rng.normal(size=(4, WIDTH)).astype(np.float32) for i in range(2)}
        rows = [{"left": a, "right": b, "label": 1, "provenance": "measured_positive"}
                for a in lefts for b in rights]
        frame = PT.PairFrame("ragged", pd.DataFrame(rows),
                             {a: "" for a in lefts}, {b: "" for b in rights})
        return frame, {"left": PTR.DictEmbeddings(lefts),
                       "right": PTR.DictEmbeddings(rights)}

    def batches(self, **changes):
        frame, embeddings = self.ragged_task()
        rows = np.arange(len(frame.pairs))
        return frame, rows, PTR.PairBatches(frame, rows, embeddings,
                                            config(batch_size=2, **changes), shuffle=True)

    def test_an_unset_setting_is_absent_from_the_identity(self):
        """A fit saved before the setting existed must still match its own identity."""
        self.assertNotIn("bucket_by_length", config().as_record())

    def test_the_setting_is_recorded_once_it_is_on(self):
        self.assertIs(config(bucket_by_length=True).as_record()["bucket_by_length"], True)

    def test_every_row_is_batched_exactly_once(self):
        _, rows, batches = self.batches(bucket_by_length=True)
        seen = np.concatenate(batches._batches())
        self.assertEqual(sorted(seen.tolist()), sorted(rows.tolist()))

    def test_the_long_rows_are_batched_together(self):
        _, rows, batches = self.batches(bucket_by_length=True)
        widths = batches._lengths()
        for batch in batches._batches():
            span = widths[np.isin(rows, batch)]
            self.assertEqual(span.min(), span.max())

    def test_a_random_batch_mixes_the_lengths_instead(self):
        """The behaviour the setting replaces, so the test fails if the default changes."""
        _, rows, batches = self.batches()
        widths = batches._lengths()
        spans = [widths[np.isin(rows, batch)] for batch in batches._batches()]
        self.assertTrue(any(span.min() != span.max() for span in spans))

    def test_the_batch_order_still_moves_between_epochs(self):
        """Bucketing fixes what is in a batch, not the order the batches arrive in."""
        _, _, batches = self.batches(bucket_by_length=True)
        first = [b.tolist() for b in batches._batches()]
        batches.epoch += 1
        self.assertNotEqual(first, [b.tolist() for b in batches._batches()])

    def test_a_padded_batch_is_no_wider_than_its_own_rows(self):
        frame, _, batches = self.batches(bucket_by_length=True)
        for left, right, _, _, _, picked in batches:
            wanted = max(len(batches.embeddings["left"].tokens(name))
                         for name in frame.pairs.iloc[picked]["left"])
            self.assertEqual(left.shape[1], wanted)
            self.assertEqual(right.shape[1], 4)
