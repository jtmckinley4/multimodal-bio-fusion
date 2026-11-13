"""Check IsoFormer's aggregator on small tensors, and one end-to-end fit on a stub cache.

The properties asserted here come from the paper: each modality keeps its own width, the
module is robust to a missing modality, and the head produces one value per tissue. No
checkpoints, no embeddings, no network.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import isoformer, train  # noqa: E402
from shared_code.fusion import token_embeddings as TE  # noqa: E402

# Real widths are 512 (NT v2 100M) and 640 (ESM-2 150M); these are the same
# shape of number, divisible by the eight cross-attention heads.
WIDTHS = {"dna": 16, "rna": 16, "prot": 24}
TISSUES = tuple(f"T{i}" for i in range(5))


def batch_of(lengths=(7, 5), widths=WIDTHS):
    streams, masks = {}, {}
    for m, w in widths.items():
        longest = max(lengths)
        block = torch.zeros(len(lengths), longest, w)
        mask = torch.zeros(len(lengths), longest, dtype=torch.bool)
        for i, n in enumerate(lengths):
            block[i, :n] = torch.randn(n, w)
            mask[i, :n] = True
        streams[m], masks[m] = block, mask
    return isoformer.ExpressionBatch(streams, masks, torch.randn(len(lengths), len(TISSUES)))


class AggregatorTests(unittest.TestCase):
    def test_every_modality_subset_predicts_one_value_per_tissue(self):
        for variant in ("dna", "rna", "prot", "dna+prot", "dna+rna", "dna+rna+prot"):
            model = isoformer.build(WIDTHS, len(TISSUES), variant)
            out, _ = model(batch_of())
            self.assertEqual(out.shape, (2, len(TISSUES)), variant)
            self.assertTrue(bool(torch.isfinite(out).all()), variant)

    def test_a_single_modality_model_has_no_cross_attention(self):
        """With nothing to attend to, the paper's module reduces to pool and predict."""
        self.assertEqual(len(isoformer.build(WIDTHS, 5, "rna").layers), 0)
        self.assertGreater(len(isoformer.build(WIDTHS, 5, "dna+rna").layers), 0)

    def test_a_model_never_touches_a_modality_it_was_not_given(self):
        # eval(), or the two passes differ by dropout rather than by the modality.
        model = isoformer.build(WIDTHS, len(TISSUES), "dna+rna").eval()
        batch = batch_of()
        with torch.no_grad():
            before, _ = model(batch)
        batch.streams["prot"] = torch.randn_like(batch.streams["prot"]) * 50
        with torch.no_grad():
            after, _ = model(batch)
        torch.testing.assert_close(before, after)

    def test_padding_does_not_change_a_row(self):
        torch.manual_seed(0)
        model = isoformer.build(WIDTHS, len(TISSUES), "dna+rna+prot").eval()
        tight = batch_of(lengths=(5,))
        padded = isoformer.ExpressionBatch(
            {m: torch.cat([s, torch.randn(1, 4, s.shape[2])], dim=1)
             for m, s in tight.streams.items()},
            {m: torch.cat([v, torch.zeros(1, 4, dtype=torch.bool)], dim=1)
             for m, v in tight.masks.items()},
            tight.labels)
        with torch.no_grad():
            torch.testing.assert_close(model(tight)[0], model(padded)[0])

    def test_each_modality_keeps_its_own_width_through_aggregation(self):
        layer = isoformer.CrossAttentionAggregator(WIDTHS)
        batch = batch_of()
        out = layer(batch.streams, batch.masks)
        for m, w in WIDTHS.items():
            self.assertEqual(out[m].shape[2], w, m)

    def test_a_width_that_does_not_divide_the_heads_is_refused_by_name(self):
        with self.assertRaises(ValueError) as raised:
            isoformer.build({"dna": 16, "rna": 15, "prot": 24}, 5, "dna+rna")
        self.assertIn("rna", str(raised.exception))

    def test_gradients_reach_every_parameter(self):
        model = isoformer.build(WIDTHS, len(TISSUES), "dna+rna+prot")
        model(batch_of())[0].sum().backward()
        missing = [n for n, p in model.named_parameters()
                   if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
        self.assertEqual(missing, [])


class ExpressionScoringTests(unittest.TestCase):
    def test_the_panel_score_is_the_mean_spearman_over_tissues(self):
        truth = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0]])
        exact = train.score("expression", truth, truth.copy())
        self.assertAlmostEqual(exact, 1.0, places=6)
        flipped = train.score("expression", truth, -truth)
        self.assertAlmostEqual(flipped, -1.0, places=6)

    def test_r_squared_is_one_on_a_perfect_fit_and_zero_on_the_mean(self):
        truth = np.array([[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0]])
        self.assertAlmostEqual(train.r_squared(truth, truth.copy()), 1.0, places=6)
        mean_only = np.tile(truth.mean(axis=0), (len(truth), 1))
        self.assertAlmostEqual(train.r_squared(truth, mean_only), 0.0, places=6)


class PipelineTests(unittest.TestCase):
    """One fit over a cache written by the real shard writer."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        rng = np.random.default_rng(0)
        n = 12
        lengths = {"dna": [6] * n, "rna": [4 + i % 3 for i in range(n)], "prot": [3] * n}
        self.caches = {}
        for m, counts in lengths.items():
            directory = root / f"{m}_stub"
            directory.mkdir()
            TE._write_shard(directory / "shard_00000.npz",
                            [rng.normal(size=(c, WIDTHS[m])).astype(np.float16) for c in counts])
            (directory / "meta.json").write_text('{"n_shards": 1, "dtype": "float16"}')
            self.caches[m] = TE.summarize(directory)
        self.rows = pd.DataFrame({"split": ["train"] * 8 + ["test"] * 4,
                                  **{t: rng.normal(size=n) for t in TISSUES}})

    def tearDown(self):
        # Shard readers keep files open; release them before removing the fixture on Windows.
        for cache in self.caches.values():
            cache.close()
        self.dir.cleanup()

    def streams(self):
        return isoformer.GTExStreams("stub", self.rows, self.caches, TISSUES)

    def test_the_cache_check_catches_a_row_count_mismatch(self):
        streams = self.streams()
        streams.check()
        streams.rows = self.rows.iloc[:5]
        with self.assertRaises(ValueError):
            streams.check()

    def test_loaders_hold_out_a_validation_slice_of_train(self):
        made = isoformer.loaders_for(self.streams(), batch_size=4, val_fraction=0.25)
        sizes = {k: len(v.dataset) for k, v in made.items()}
        self.assertEqual(sizes["train"] + sizes["val"], 8)
        self.assertEqual(sizes["val"], 2)
        self.assertEqual(sizes["test"], 4)

    def test_batches_stay_under_the_attention_budget(self):
        """A long row must shrink its batch, not blow up the memory of a full-size one."""
        rng = np.random.default_rng(0)
        lengths = {"dna": np.full(200, 1000), "rna": rng.integers(80, 2000, 200),
                   "prot": rng.integers(1, 1022, 200)}
        budget, ceiling = 300_000_000, 16
        sampler = isoformer.BudgetedBatches(lengths, budget, ceiling, shuffle=False, seed=0)
        sizes = []
        for batch in sampler:
            longest = {m: lengths[m][batch].max() for m in isoformer.MODALITIES}
            cost = len(batch) * isoformer.N_HEADS * sum(
                longest[a] * longest[b] for a in longest for b in longest if a != b)
            self.assertTrue(len(batch) == 1 or cost <= budget,
                            f"a batch of {len(batch)} cost {cost / 1e6:.0f}M cells")
            self.assertLessEqual(len(batch), ceiling)
            sizes.append(len(batch))
        self.assertEqual(sum(sizes), 200, "every row must appear exactly once")
        self.assertLess(min(sizes), max(sizes), "batch size should vary with row length")

    def test_shuffling_reorders_batches_without_changing_their_membership(self):
        lengths = {m: np.full(40, 100) for m in isoformer.MODALITIES}
        sampler = isoformer.BudgetedBatches(lengths, 10 ** 12, 4, shuffle=True, seed=0)
        first = [tuple(b) for b in sampler]
        second = [tuple(b) for b in sampler]
        self.assertEqual(sorted(first), sorted(second))
        self.assertNotEqual(first, second)

    def test_every_variant_trains_and_scores(self):
        streams = self.streams()
        made = isoformer.loaders_for(streams, batch_size=4, val_fraction=0.25)
        config = train.TrainConfig(max_epochs=2, early_stopping_patience=1)
        for variant in ("rna", "dna+rna", "dna+rna+prot"):
            model = isoformer.builder(streams, variant, config)
            result = train.fit(model, streams, config, "cpu", log=None, loaders=made)
            self.assertEqual(result.epochs, 2, variant)
            self.assertIsNotNone(result.test_prediction, variant)
            self.assertEqual(np.asarray(result.test_prediction).shape, (4, len(TISSUES)))


class TrackingCache:
    """Synthetic token rows that fail immediately if an unused stream is read."""

    def __init__(self, modality, lengths, readable=True):
        self.lengths = np.asarray(lengths)
        self.width = WIDTHS[modality]
        self.readable = readable
        self.accessed = []

    def __getitem__(self, index):
        if not self.readable:
            raise AssertionError("an unused token cache was indexed")
        self.accessed.append(index)
        shape = (int(self.lengths[index]), self.width)
        return (np.arange(np.prod(shape)).reshape(shape) + index).astype(np.float16)


class SelectedStreamLoadingTests(unittest.TestCase):
    def streams(self, selected=isoformer.MODALITIES):
        lengths = {"dna": [6] * 20, "rna": [2 + i % 8 for i in range(20)],
                   "prot": [1 + i % 6 for i in range(20)]}
        rows = pd.DataFrame({"split": ["train"] * 14 + ["test"] * 6,
                             **{t: np.arange(20) + j for j, t in enumerate(TISSUES)}})
        caches = {m: TrackingCache(m, lengths[m], readable=m in selected)
                  for m in isoformer.MODALITIES}
        return isoformer.GTExStreams("stub", rows, caches, TISSUES)

    def test_selected_streams_preserve_splits_batches_labels_and_tensors(self):
        # Distinct lengths make the all-three budget consequential, even for single fits.
        settings = dict(batch_size=4, val_fraction=0.25, seed=7, budget=2_000)
        for selected in (("dna",), ("rna",), ("prot",), ("dna", "rna"),
                         ("dna", "prot"), ("rna", "prot"), isoformer.MODALITIES):
            with self.subTest(modalities=selected):
                original = isoformer.loaders_for(self.streams(), **settings)
                streams = self.streams(selected)
                narrowed = isoformer.loaders_for(streams, modalities=selected, **settings)
                for split in ("train", "val", "test"):
                    left, right = original[split], narrowed[split]
                    np.testing.assert_array_equal(left.dataset.indices, right.dataset.indices)
                    self.assertEqual(left.batch_sampler.batches, right.batch_sampler.batches)
                    self.assertEqual(left.dataset.modalities, isoformer.MODALITIES)
                    self.assertEqual(right.dataset.modalities, selected)
                    for epoch in range(3):
                        # Iterating loaders consumes a PyTorch base seed even without workers.
                        torch.manual_seed(epoch)
                        full_batches = list(left)
                        full_rng = torch.get_rng_state()
                        torch.manual_seed(epoch)
                        selected_batches = list(right)
                        self.assertTrue(torch.equal(full_rng, torch.get_rng_state()))
                        self.assertEqual(len(full_batches), len(selected_batches))
                        for full, only in zip(full_batches, selected_batches):
                            self.assertEqual(tuple(only.streams), selected)
                            self.assertEqual(tuple(only.masks), selected)
                            torch.testing.assert_close(full.labels, only.labels, rtol=0, atol=0)
                            for modality in selected:
                                torch.testing.assert_close(full.streams[modality], only.streams[modality],
                                                           rtol=0, atol=0)
                                self.assertTrue(torch.equal(full.masks[modality], only.masks[modality]))
                            moved = only.to("cpu")
                            self.assertEqual(tuple(moved.streams), selected)
                            torch.testing.assert_close(moved.labels, only.labels, rtol=0, atol=0)
                for modality, cache in streams.caches.items():
                    self.assertEqual(bool(cache.accessed), modality in selected)

    def test_direct_dataset_reads_only_selected_streams(self):
        streams = self.streams(("rna",))
        dataset = isoformer.ExpressionDataset(streams, np.array([4, 1]), modalities=("rna",))
        tokens, label = dataset[0]
        self.assertEqual(tuple(tokens), ("rna",))
        self.assertEqual(streams.caches["rna"].accessed, [4])
        np.testing.assert_array_equal(label, streams.labels[4])

    def test_invalid_selections_are_rejected_before_cache_access(self):
        streams = self.streams(())
        for selection in ((), ("rna", "rna"), ("protein",), "rna", 7, (["rna"],), (None,)):
            with self.subTest(selection=selection):
                with self.assertRaisesRegex(ValueError, "modalities"):
                    isoformer.loaders_for(streams, modalities=selection)
                with self.assertRaisesRegex(ValueError, "modalities"):
                    isoformer.ExpressionDataset(streams, np.array([0]), modalities=selection)
        self.assertTrue(all(not cache.accessed for cache in streams.caches.values()))

    def test_missing_selected_cache_is_rejected(self):
        streams = self.streams()
        del streams.caches["rna"]
        with self.assertRaisesRegex(ValueError, "no caches.*rna"):
            isoformer.ExpressionDataset(streams, np.array([0]), modalities=("rna",))

    def test_collation_refuses_empty_unknown_and_mixed_stream_selections(self):
        label = np.zeros(len(TISSUES), dtype=np.float32)
        tokens = np.ones((2, WIDTHS["rna"]), dtype=np.float32)
        cases = [[], [({}, label)], [({"unknown": tokens}, label)],
                 [({"rna": tokens}, label), ({"dna": tokens}, label)], [(tokens, label)]]
        for batch in cases:
            with self.subTest(batch_size=len(batch)):
                with self.assertRaises(ValueError):
                    isoformer.collate(batch)


class GTExStreamCleanupTests(unittest.TestCase):
    def setUp(self):
        self.rows = pd.DataFrame({"split": ["train"], **{
            tissue: [0.0] for tissue in isoformer.datasets_module.GTEX_TISSUES}})
        self.caches = [MagicMock() for _ in isoformer.MODALITIES]
        for cache in self.caches:
            cache.__len__.return_value = 1

    def test_partial_open_failure_closes_only_opened_caches(self):
        failure = OSError("third cache failed to open")
        with patch("pandas.read_csv", return_value=self.rows), \
                patch.object(isoformer, "summarize", side_effect=[*self.caches[:2], failure]):
            with self.assertRaises(OSError) as raised:
                isoformer.GTExStreams.open("unused")
        self.assertIs(raised.exception, failure)
        for cache in self.caches[:2]:
            cache.close.assert_called_once_with()
        self.caches[2].close.assert_not_called()

    def test_check_failure_preserves_original_and_reports_all_cleanup_failures(self):
        failure = ValueError("wrong row set")
        self.caches[0].close.side_effect = OSError("first cleanup failed")
        self.caches[1].close.side_effect = OSError("second cleanup failed")
        with patch("pandas.read_csv", return_value=self.rows), \
                patch.object(isoformer, "summarize", side_effect=self.caches), \
                patch.object(isoformer.GTExStreams, "check", side_effect=failure):
            with self.assertRaises(ValueError) as raised:
                isoformer.GTExStreams.open("unused")
        self.assertIs(raised.exception, failure)
        for cache in self.caches:
            cache.close.assert_called_once_with()
        if hasattr(failure, "add_note"):
            notes = " ".join(failure.__notes__)
            self.assertIn("first cleanup failed", notes)
            self.assertIn("second cleanup failed", notes)

    def test_keyboard_interrupt_during_open_also_releases_owned_caches(self):
        failure = KeyboardInterrupt()
        with patch("pandas.read_csv", return_value=self.rows), \
                patch.object(isoformer, "summarize", side_effect=[self.caches[0], failure]):
            with self.assertRaises(KeyboardInterrupt) as raised:
                isoformer.GTExStreams.open("unused")
        self.assertIs(raised.exception, failure)
        self.caches[0].close.assert_called_once_with()

    def test_successful_open_keeps_caches_until_explicit_close(self):
        with patch("pandas.read_csv", return_value=self.rows), \
                patch.object(isoformer, "summarize", side_effect=self.caches):
            streams = isoformer.GTExStreams.open("unused")
        for cache in self.caches:
            cache.close.assert_not_called()
        streams.close()
        for cache in self.caches:
            cache.close.assert_called_once_with()

    def test_close_attempts_every_cache_and_preserves_first_error(self):
        first = OSError("first failure")
        self.caches[0].close.side_effect = first
        self.caches[1].close.side_effect = RuntimeError("later failure")
        streams = isoformer.GTExStreams("stub", self.rows,
                                       dict(zip(isoformer.MODALITIES, self.caches)), ())
        with self.assertRaises(OSError) as raised:
            streams.close()
        self.assertIs(raised.exception, first)
        for cache in self.caches:
            cache.close.assert_called_once_with()
        if hasattr(first, "add_note"):
            self.assertIn("later failure", " ".join(first.__notes__))


if __name__ == "__main__":
    unittest.main()
