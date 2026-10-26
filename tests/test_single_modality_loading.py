"""Selective loading preserves the existing experiment on tiny CPU fixtures.

The retained model constructors/forwards below are the prior all-stream behavior.
No encoder, scientific dataset, GPU, or existing research artifact is accessed.
"""

from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import blf, codon_grid, textcnn, train  # noqa: E402


WIDTHS = {"dna": 4, "rna": 3, "prot": 2}
RESOLUTIONS = (
    {"dna": "kmer", "rna": "nucleotide", "prot": "codon"},
    {"dna": "nucleotide", "rna": "codon", "prot": "codon"},
)


class RecordingCache:
    def __init__(self, rows, width):
        self.rows, self.width = rows, width
        self.reads = []
        self.forbidden = False

    def __getitem__(self, index):
        if self.forbidden:
            raise AssertionError("Unused modality cache was read")
        self.reads.append(index)
        return self.rows[index]


def fixture(resolutions=None):
    resolutions = resolutions or RESOLUTIONS[0]
    # Include unequal lengths and 6-mer remainder tokens.
    lengths = [30, 33, 42, 51, 36, 45, 48, 39, 54, 57, 60, 63]
    rows = pd.DataFrame({
        "length": lengths,
        "Split": ["train"] * 6 + ["val"] * 3 + ["test"] * 3,
        "Value": np.arange(12, dtype=np.float32) / 12,
    })
    rng = np.random.default_rng(413)
    caches = {}
    for modality, width in WIDTHS.items():
        resolution = resolutions[modality]
        counts = [n // 6 + n % 6 if resolution == "kmer"
                  else n if resolution == "nucleotide" else n // 3 for n in lengths]
        values = [rng.normal(size=(n, width)).astype(np.float16) for n in counts]
        caches[modality] = RecordingCache(values, width)
    encoders = {
        "dna": "ntv2_100m" if resolutions["dna"] == "kmer" else "hyenadna",
        "rna": "rnafm" if resolutions["rna"] == "nucleotide" else "mrnafm",
        "prot": "esm2_8m",
    }
    return blf.BenchmarkStreams("selective-fixture", rows, caches, "regression",
                                encoders=encoders)


class OriginalUnimodal(nn.Module):
    """Pre-change construction and forward, without selective-loader metadata."""

    def __init__(self, modality, tconv, resolutions):
        super().__init__()
        self.modality = modality
        self.fusion_name = f"unimodal:{modality}"
        self.head = textcnn.TextCNN(WIDTHS[modality], 1, channels=2)

    def forward(self, batch):
        stream = batch.streams[self.modality]
        return self.head(
            stream, codon_grid.mask_of(batch.tokens[self.modality], stream.shape[1])), {}


class OriginalGriddedUnimodal(nn.Module):
    """Keep the original grid constructor, including the unused DNA parameters."""

    def __init__(self, modality, tconv, resolutions):
        super().__init__()
        self.modality = modality
        self.fusion_name = f"unimodal-grid:{modality}"
        self.grid = codon_grid.CodonGrid(WIDTHS["dna"], tconv=tconv,
                                         resolutions=resolutions)
        self.head = textcnn.TextCNN(WIDTHS[modality], 1, channels=2)

    def forward(self, batch):
        grid, mask = self.grid(batch.streams["dna"], batch.streams["rna"],
                               batch.streams["prot"], batch.codons, batch.tokens["dna"])
        return self.head(grid[self.modality], mask), {}


def model(gridded, modality, *, original=False, tconv="equation", resolutions=None):
    resolutions = resolutions or RESOLUTIONS[0]
    torch.manual_seed(101)
    if original:
        cls = OriginalGriddedUnimodal if gridded else OriginalUnimodal
        return cls(modality, tconv, resolutions)
    if gridded:
        return blf.GriddedUnimodal(WIDTHS, modality, 1, channels=2,
                                   tconv=tconv, resolutions=resolutions)
    return blf.Unimodal(WIDTHS, modality, 1, channels=2)


class Interrupted(Exception):
    pass


class SingleModalityLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def assert_nested_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, np.ndarray):
            np.testing.assert_array_equal(first, second)
        elif isinstance(first, dict):
            self.assertEqual(list(first), list(second))
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def forbid_unused(self, streams, modality):
        for name, cache in streams.caches.items():
            cache.forbidden = name != modality

    def test_selected_batches_never_read_pad_or_transfer_unused_streams(self):
        for gridded in (False, True):
            for modality in blf.MODALITIES:
                with self.subTest(gridded=gridded, modality=modality):
                    streams = fixture()
                    self.forbid_unused(streams, modality)
                    selected = model(gridded, modality)
                    self.assertEqual(selected.required_modalities, (modality,))
                    batch = next(iter(blf.loader(
                        streams, "train", batch_size=3, shuffle=False,
                        modalities=selected.required_modalities))).to("cpu")
                    self.assertEqual(tuple(batch.streams), (modality,))
                    self.assertEqual(tuple(batch.tokens), (modality,))
                    self.assertEqual(batch.streams[modality].dtype, torch.float32)
                    self.assertEqual(streams.caches[modality].reads, [0, 1, 2])
                    for other in set(blf.MODALITIES) - {modality}:
                        self.assertEqual(streams.caches[other].reads, [])

    def test_selected_loader_preserves_padding_labels_shuffle_and_random_state(self):
        for modality in blf.MODALITIES:
            with self.subTest(modality=modality):
                full = blf.loader(fixture(), "train", batch_size=2, seed=41)
                selected = blf.loader(fixture(), "train", batch_size=2, seed=41,
                                      modalities=(modality,))
                # Consecutive epochs test evolving sampler state, not just first batches.
                for _ in range(2):
                    torch.manual_seed(79)
                    full_batches = list(full)
                    full_rng = torch.get_rng_state().clone()
                    torch.manual_seed(79)
                    selected_batches = list(selected)
                    self.assertTrue(torch.equal(full_rng, torch.get_rng_state()))
                    self.assertTrue(torch.equal(full.generator.get_state(),
                                                selected.generator.get_state()))
                    for all_batch, one_batch in zip(full_batches, selected_batches):
                        self.assertEqual(tuple(all_batch.streams), blf.MODALITIES)
                        for attr in ("streams", "tokens"):
                            self.assert_nested_equal(getattr(all_batch, attr)[modality],
                                                     getattr(one_batch, attr)[modality])
                        self.assert_nested_equal(all_batch.codons, one_batch.codons)
                        self.assert_nested_equal(all_batch.labels, one_batch.labels)

    def test_construction_retains_parameter_order_initial_values_and_rng(self):
        for gridded in (False, True):
            for modality in blf.MODALITIES:
                for resolutions in RESOLUTIONS:
                    for setting in codon_grid.TCONV_SETTINGS:
                        with self.subTest(gridded=gridded, modality=modality,
                                          resolutions=resolutions, setting=setting):
                            old = model(gridded, modality, original=True, tconv=setting,
                                        resolutions=resolutions)
                            old_rng = torch.get_rng_state().clone()
                            new = model(gridded, modality, tconv=setting,
                                        resolutions=resolutions)
                            self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
                            self.assertEqual([n for n, _ in old.named_parameters()],
                                             [n for n, _ in new.named_parameters()])
                            self.assert_nested_equal(old.state_dict(), new.state_dict())

    def test_logits_gradients_and_optimizer_step_match_original_full_stream_path(self):
        for gridded in (False, True):
            for modality in blf.MODALITIES:
                for resolutions in RESOLUTIONS:
                    for setting in codon_grid.TCONV_SETTINGS:
                        with self.subTest(gridded=gridded, modality=modality,
                                          resolutions=resolutions, setting=setting):
                            streams = fixture(resolutions)
                            all_batch = next(iter(blf.loader(
                                streams, "train", batch_size=3, shuffle=False)))
                            self.forbid_unused(streams, modality)
                            one_batch = next(iter(blf.loader(
                                streams, "train", batch_size=3, shuffle=False,
                                modalities=(modality,))))
                            old = model(gridded, modality, original=True, tconv=setting,
                                        resolutions=resolutions)
                            new = model(gridded, modality, tconv=setting,
                                        resolutions=resolutions)
                            old_opt = torch.optim.Adam(old.parameters(), lr=0.01)
                            new_opt = torch.optim.Adam(new.parameters(), lr=0.01)
                            outputs, rngs = [], []
                            for predictor, batch, optimiser in (
                                    (old, all_batch, old_opt), (new, one_batch, new_opt)):
                                torch.manual_seed(37)  # Same training-mode dropout mask.
                                output, aux = predictor(batch)
                                self.assertEqual(aux, {})
                                ((output.squeeze(-1) - batch.labels) ** 2).mean().backward()
                                outputs.append(output.detach().clone())
                                rngs.append(torch.get_rng_state().clone())
                                optimiser.step()
                            self.assert_nested_equal(outputs[0], outputs[1])
                            self.assert_nested_equal(rngs[0], rngs[1])
                            self.assert_nested_equal(
                                {n: p.grad for n, p in old.named_parameters()},
                                {n: p.grad for n, p in new.named_parameters()})
                            self.assert_nested_equal(old.state_dict(), new.state_dict())
                            self.assert_nested_equal(old_opt.state_dict(), new_opt.state_dict())

    def config(self):
        return train.TrainConfig(lr=0.02, max_epochs=5, batch_size=3, channels=2,
                                 plateau_patience=0, early_stopping_patience=99,
                                 seed=17, tconv="equation")

    def test_default_fit_keeps_the_original_dropout_shuffle_and_scheduler_trajectory(self):
        for gridded, modality in ((False, "rna"), (True, "dna"),
                                  (True, "rna"), (True, "prot")):
            with self.subTest(gridded=gridded, modality=modality):
                old = model(gridded, modality, original=True)
                new = model(gridded, modality)
                selected_streams = fixture()
                self.forbid_unused(selected_streams, modality)
                # Constant score exercises scheduler decay without stopping after one epoch.
                with patch.object(train, "score", return_value=0.5):
                    full = train.fit(old, fixture(), self.config(), "cpu", log=None)
                    selected = train.fit(new, selected_streams, self.config(), "cpu", log=None)
                self.assertEqual(full.history, selected.history)
                self.assertEqual(full.test_truth, selected.test_truth)
                self.assertEqual(full.test_prediction, selected.test_prediction)
                self.assertEqual(full.best_epoch, selected.best_epoch)
                self.assertLess(selected.history[-1]["lr"], selected.history[0]["lr"])
                self.assert_nested_equal(old.state_dict(), new.state_dict())

    def test_original_all_stream_checkpoint_resumes_with_selective_default_loaders(self):
        for gridded, modality in ((False, "rna"), (True, "rna"), (True, "dna")):
            with self.subTest(gridded=gridded, modality=modality):
                with tempfile.TemporaryDirectory() as directory:
                    full_path = Path(directory) / "full.checkpoint"
                    resumed_path = Path(directory) / "resumed.checkpoint"
                    save = train._save_checkpoint

                    def stop_after_saved_epoch(*args, **kwargs):
                        save(*args, **kwargs)
                        if args[1] == 1:
                            raise Interrupted()

                    with patch.object(train, "score", return_value=0.5):
                        full = train.fit(
                            model(gridded, modality, original=True), fixture(), self.config(),
                            "cpu", log=None, checkpoint=full_path, checkpoint_every=1,
                            keep_checkpoint=True)
                        with patch.object(train, "_save_checkpoint",
                                          side_effect=stop_after_saved_epoch):
                            with self.assertRaises(Interrupted):
                                train.fit(
                                    model(gridded, modality, original=True), fixture(),
                                    self.config(), "cpu", log=None, checkpoint=resumed_path,
                                    checkpoint_every=1, keep_checkpoint=True)
                        torch.rand(7)
                        np.random.normal(size=7)
                        random.random()
                        streams = fixture()
                        self.forbid_unused(streams, modality)
                        resumed = train.fit(
                            model(gridded, modality), streams, self.config(), "cpu",
                            log=None, checkpoint=resumed_path, checkpoint_every=1,
                            keep_checkpoint=True)
                    self.assertEqual(full.history, resumed.history)
                    self.assertEqual(full.test_prediction, resumed.test_prediction)
                    full_saved = torch.load(full_path, map_location="cpu", weights_only=False)
                    resumed_saved = torch.load(resumed_path, map_location="cpu", weights_only=False)
                    for key in ("model", "optimiser", "scheduler", "loaders", "identity", "rng"):
                        self.assert_nested_equal(full_saved[key], resumed_saved[key])

    def test_custom_loaders_remain_authoritative(self):
        streams = fixture()
        supplied = {split: blf.loader(streams, split, batch_size=3, seed=17)
                    for split in ("train", "val", "test")}
        selected = model(False, "rna")
        selected.required_modalities = ("must-not-be-used-by-default-loader",)
        with patch.object(train, "loader", side_effect=AssertionError("Custom loaders ignored")):
            result = train.fit(selected, streams, self.config(), "cpu", log=None,
                               loaders=supplied)
        self.assertEqual(result.epochs, self.config().max_epochs)
        self.assertTrue(all(cache.reads for cache in streams.caches.values()))


if __name__ == "__main__":
    unittest.main()
