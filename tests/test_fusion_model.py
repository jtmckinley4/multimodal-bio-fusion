"""Check the BioLangFusion model on small tensors: no checkpoints, no cached embeddings.

The shapes and lengths asserted here come from the paper rather than from the code: the
codon frame is T/3, the DNA stream is upsampled from T/6 tokens, the RNA stream is pooled
three to one, and the gated head produces one weight per modality per sequence.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion import blf, codon_grid, heads, textcnn, train  # noqa: E402
from shared_code.fusion import token_embeddings as TE  # noqa: E402

WIDTHS = {"dna": 12, "rna": 9, "prot": 6}


def streams_for(nucleotides, batch=2, extra=0):
    """One padded batch of three streams for rows of the given coding-sequence lengths."""
    lengths = [nucleotides] * batch
    dna = torch.randn(batch, max(n // 6 + n % 6 for n in lengths) + extra, WIDTHS["dna"])
    rna = torch.randn(batch, max(lengths) + extra, WIDTHS["rna"])
    prot = torch.randn(batch, max(n // 3 for n in lengths) + extra, WIDTHS["prot"])
    codons = torch.tensor([n // 3 for n in lengths])
    dna_tokens = torch.tensor([n // 6 + n % 6 for n in lengths])
    return dna, rna, prot, codons, dna_tokens


class CodonGridTests(unittest.TestCase):
    def test_every_stream_lands_on_the_protein_frame(self):
        for nucleotides in (996, 999, 81, 150):
            grid = codon_grid.CodonGrid(WIDTHS["dna"])
            streams, mask = grid(*streams_for(nucleotides))
            for modality, stream in streams.items():
                self.assertEqual(stream.shape[1], nucleotides // 3, modality)
            self.assertTrue(bool(mask.all()))

    def test_both_published_convolution_settings_reach_the_frame(self):
        # Equation 1 gives kernel 2 stride 2; Appendix List 1 gives kernel 3 stride 2
        # padding 2. They differ by three positions before the stream is fitted to T'.
        for setting in codon_grid.TCONV_SETTINGS:
            grid = codon_grid.CodonGrid(WIDTHS["dna"], tconv=setting)
            streams, _ = grid(*streams_for(999))
            self.assertEqual(streams["dna"].shape[1], 333, setting)

    def test_a_row_is_resampled_the_same_alone_as_in_a_padded_batch(self):
        """Padding a batch must not change any row: no valid codon is built from padding."""
        torch.manual_seed(0)
        grid = codon_grid.CodonGrid(WIDTHS["dna"]).eval()
        short, long = 300, 999
        dna = torch.zeros(2, long // 6 + long % 6, WIDTHS["dna"])
        rna = torch.zeros(2, long, WIDTHS["rna"])
        prot = torch.zeros(2, long // 3, WIDTHS["prot"])
        for row, n in enumerate((short, long)):
            dna[row, :n // 6 + n % 6] = torch.randn(n // 6 + n % 6, WIDTHS["dna"])
            rna[row, :n] = torch.randn(n, WIDTHS["rna"])
            prot[row, :n // 3] = torch.randn(n // 3, WIDTHS["prot"])
        codons = torch.tensor([short // 3, long // 3])
        dna_tokens = torch.tensor([short // 6 + short % 6, long // 6 + long % 6])
        with torch.no_grad():
            batched, _ = grid(dna, rna, prot, codons, dna_tokens)
            alone, _ = grid(dna[:1, :short // 6 + short % 6], rna[:1, :short],
                            prot[:1, :short // 3], codons[:1], dna_tokens[:1])
        for modality in alone:
            torch.testing.assert_close(batched[modality][:1, :short // 3], alone[modality],
                                       msg=f"{modality} changed when batched with a longer row")

    def test_fit_length_pads_and_trims(self):
        x = torch.ones(2, 5, 3)
        self.assertEqual(codon_grid.fit_length(x, 8).shape[1], 8)
        self.assertEqual(float(codon_grid.fit_length(x, 8)[:, 5:].abs().sum()), 0.0)
        self.assertEqual(codon_grid.fit_length(x, 3).shape[1], 3)


class TextCNNTests(unittest.TestCase):
    def test_padding_does_not_change_a_row(self):
        torch.manual_seed(0)
        head = textcnn.TextCNN(WIDTHS["rna"], 1, channels=4).eval()
        real = torch.randn(1, 40, WIDTHS["rna"])
        padded = torch.cat([real, torch.randn(1, 15, WIDTHS["rna"])], dim=1)
        mask = torch.zeros(1, 55, dtype=torch.bool)
        mask[:, :40] = True
        with torch.no_grad():
            torch.testing.assert_close(head(real, torch.ones(1, 40, dtype=torch.bool)),
                                       head(padded, mask))

    def test_a_row_shorter_than_the_widest_kernel_still_predicts(self):
        head = textcnn.TextCNN(WIDTHS["rna"], 1, channels=4).eval()
        out = head(torch.randn(1, 3, WIDTHS["rna"]), torch.ones(1, 3, dtype=torch.bool))
        self.assertTrue(bool(torch.isfinite(out).all()))


class FusionHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.grid = codon_grid.CodonGrid(WIDTHS["dna"])
        self.streams, self.mask = self.grid(*streams_for(600, batch=3))

    def test_each_head_returns_the_codon_frame_at_the_shared_width(self):
        for name, factory in heads.FUSIONS.items():
            fused, _ = factory(WIDTHS)(self.streams, self.mask)
            self.assertEqual(fused.shape[:2], (3, 200), name)
            self.assertEqual(fused.shape[2], heads.SHARED_WIDTH, name)

    def test_gradients_reach_every_parameter_of_every_head(self):
        # Detached, so each head is backward through its own graph and not the grid's.
        streams = {m: s.detach() for m, s in self.streams.items()}
        for name, factory in heads.FUSIONS.items():
            model = factory(WIDTHS)
            fused, _ = model(streams, self.mask)
            fused.sum().backward()
            missing = [n for n, p in model.named_parameters()
                       if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            self.assertEqual(missing, [], f"{name} has parameters without a finite gradient")

    def test_gated_attention_gives_one_weight_per_modality_per_sequence(self):
        _, aux = heads.GatedMILFusion(WIDTHS)(self.streams, self.mask)
        self.assertEqual(aux["alpha"].shape, (3, 3))
        torch.testing.assert_close(aux["alpha"].sum(dim=1), torch.ones(3))

    def test_the_attention_temperature_stays_inside_the_published_range(self):
        model = heads.GatedMILFusion(WIDTHS)
        low, high = heads.TEMPERATURE_RANGE
        with torch.no_grad():
            model.log_temperature.fill_(50.0)
            self.assertAlmostEqual(float(model.temperature()), high, places=4)
            model.log_temperature.fill_(-50.0)
            self.assertAlmostEqual(float(model.temperature()), low, places=4)

    def test_the_summary_ignores_padded_codons(self):
        """A mean over the padded frame would make a row depend on its batch mates."""
        x = torch.ones(1, 4, 2)
        x[:, 2:] = 7.0
        mask = torch.tensor([[True, True, False, False]])
        torch.testing.assert_close(heads.masked_mean(x, mask), torch.ones(1, 2))


class PipelineTests(unittest.TestCase):
    """The whole model over a cache built here, with the real shard writer and loader."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        rng = np.random.default_rng(0)
        self.lengths = [99, 150, 201, 102, 96, 309, 153, 198, 105, 249, 300, 93]
        splits = ["train"] * 6 + ["val"] * 3 + ["test"] * 3
        self.rows = pd.DataFrame({
            "sequence": ["A" * n for n in self.lengths],
            "length": self.lengths,
            "Value": rng.normal(size=len(self.lengths)),
            "Split": splits,
        })
        tokens_per_row = {
            "dna": [n // 6 + n % 6 for n in self.lengths],
            "rna": list(self.lengths),
            "prot": [n // 3 for n in self.lengths],
        }
        self.caches = {}
        for modality, counts in tokens_per_row.items():
            cache_dir = root / modality
            cache_dir.mkdir()
            TE._write_shard(cache_dir / "shard_00000.npz",
                            [rng.normal(size=(c, WIDTHS[modality])).astype(np.float16)
                             for c in counts])
            (cache_dir / "meta.json").write_text(
                '{"n_shards": 1, "dtype": "float16", "encoder": "%s"}' % modality)
            self.caches[modality] = TE.summarize(cache_dir)

    def tearDown(self):
        self.dir.cleanup()

    def streams(self):
        return blf.BenchmarkStreams("stub", self.rows, self.caches, "regression")

    def test_the_cache_checks_catch_a_stream_off_the_codon_grid(self):
        streams = self.streams()
        streams.check()
        streams.rows = self.rows.assign(length=[n + 3 for n in self.lengths])
        with self.assertRaises(ValueError):
            streams.check()

    def test_a_batch_carries_padded_streams_and_their_real_token_counts(self):
        batch = next(iter(blf.loader(self.streams(), "train", batch_size=4, shuffle=False)))
        self.assertEqual(batch.streams["rna"].shape[0], 4)
        self.assertEqual(list(batch.tokens["rna"].numpy()), [99, 150, 201, 102])
        self.assertEqual(list(batch.tokens["dna"].numpy()), [19, 25, 36, 17])
        self.assertEqual(list(batch.codons.numpy()), [33, 50, 67, 34])

    def test_every_variant_trains_and_scores(self):
        streams = self.streams()
        config = train.TrainConfig(max_epochs=2, batch_size=2, channels=3,
                                   early_stopping_patience=1, entropy_lambda=0.5)
        for variant in list(heads.FUSIONS) + ["unimodal:rna"]:
            model = blf.build(streams, variant, channels=config.channels)
            result = train.fit(model, streams, config, "cpu", log=None)
            self.assertEqual(result.epochs, 2, variant)
            self.assertTrue(np.isfinite(result.test_score), variant)
        self.assertIsNotNone(train.fit(blf.build(streams, "mil", channels=3), streams,
                                       config, "cpu", log=None).alpha)

    def test_a_paired_bootstrap_resamples_the_difference_between_two_runs(self):
        """Two models are scored on the same rows, so the difference is what to resample."""
        truth = [float(i) for i in range(30)]

        def stored(name, prediction):
            return train.Result(variant=name, benchmark="stub", config={}, epochs=1,
                                best_epoch=0, val_score=0.0, test_score=0.0, seconds=0.0,
                                test_truth=truth, test_prediction=prediction)

        out = train.paired_bootstrap(stored("rising", truth), stored("falling", truth[::-1]),
                                     "regression", draws=200)
        self.assertAlmostEqual(out["difference"], 2.0, places=6)
        self.assertEqual(out["first_wins"], 1.0)
        self.assertGreater(out["ci_low"], 1.9)

    def test_a_paired_bootstrap_refuses_runs_it_cannot_compare(self):
        truth = [float(i) for i in range(30)]
        with_rows = train.Result(variant="a", benchmark="stub", config={}, epochs=1,
                                 best_epoch=0, val_score=0.0, test_score=0.0, seconds=0.0,
                                 test_truth=truth, test_prediction=truth)
        without = train.Result(variant="b", benchmark="stub", config={}, epochs=1,
                               best_epoch=0, val_score=0.0, test_score=0.0, seconds=0.0)
        other_rows = train.Result(variant="c", benchmark="stub", config={}, epochs=1,
                                  best_epoch=0, val_score=0.0, test_score=0.0, seconds=0.0,
                                  test_truth=truth[:20], test_prediction=truth[:20])
        with self.assertRaises(ValueError):
            train.paired_bootstrap(with_rows, without, "regression", draws=10)
        with self.assertRaises(ValueError):
            train.paired_bootstrap(with_rows, other_rows, "regression", draws=10)

    def test_require_predictions_retrains_a_result_that_predates_them(self):
        streams = self.streams()
        with tempfile.TemporaryDirectory() as directory:
            config = train.TrainConfig(max_epochs=1, batch_size=2, channels=3,
                                       early_stopping_patience=1)
            self.assertIsNotNone(
                train.run(streams, "concat", config, "cpu", directory, log=None).test_prediction)
            path = next(Path(directory).glob("*.json"))
            old = json.loads(path.read_text())
            old["test_truth"] = old["test_prediction"] = None
            path.write_text(json.dumps(old))
            # Without the flag the incomplete record is handed back untouched.
            self.assertIsNone(
                train.run(streams, "concat", config, "cpu", directory, log=None).test_prediction)
            self.assertIsNotNone(
                train.run(streams, "concat", config, "cpu", directory, log=None,
                          require_predictions=True).test_prediction)

    def test_an_interrupted_run_resumes_from_its_checkpoint(self):
        """Stopping after a few epochs and restarting must continue, not start over."""
        streams = self.streams()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.checkpoint"
            short = train.TrainConfig(max_epochs=3, batch_size=2, channels=3,
                                      early_stopping_patience=99)
            train.fit(blf.build(streams, "concat", channels=3), streams, short, "cpu",
                      log=None, checkpoint=path, checkpoint_every=1)
            self.assertFalse(path.exists(), "a finished run should clean up its checkpoint")

            # Interrupt after two epochs by letting a short run write the checkpoint, then
            # hand that checkpoint to a longer run and check it does not redo those epochs.
            train._save_checkpoint(path, 1, blf.build(streams, "concat", channels=3),
                                   torch.optim.Adam(blf.build(streams, "concat",
                                                              channels=3).parameters()),
                                   {"score": 0.5, "epoch": 1, "state": None},
                                   [{"epoch": 0}, {"epoch": 1}], 12.5)
            longer = train.TrainConfig(max_epochs=4, batch_size=2, channels=3,
                                       early_stopping_patience=99)
            result = train.fit(blf.build(streams, "concat", channels=3), streams, longer,
                               "cpu", log=None, checkpoint=path, checkpoint_every=1)
            self.assertEqual([h["epoch"] for h in result.history], [0, 1, 2, 3])
            self.assertGreater(result.seconds, 12.5, "resumed time should include the first run")
            self.assertFalse(path.exists())

    def test_an_undefined_validation_score_still_leaves_a_model_to_restore(self):
        """Spearman is undefined when nothing varies, which the first epochs can produce."""
        streams = self.streams()
        constant = self.rows.copy()
        constant.loc[constant["Split"] == "val", "Value"] = 1.0
        streams.rows = constant
        config = train.TrainConfig(max_epochs=3, batch_size=2, channels=3,
                                   early_stopping_patience=1)
        result = train.fit(blf.build(streams, "concat", channels=3), streams, config, "cpu",
                           log=None)
        self.assertIsNone(result.val_score)
        self.assertTrue(np.isfinite(result.test_score))


class ShardReaderTests(unittest.TestCase):
    def test_a_mapped_shard_holds_what_numpy_would_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard_00000.npz"
            rows = [np.arange(6, dtype=np.float16).reshape(2, 3), np.ones((1, 3), np.float16)]
            TE._write_shard(path, rows)
            mapped = TE.open_shard(path)
            with np.load(path) as reference:
                for name in reference.files:
                    np.testing.assert_array_equal(np.asarray(mapped[name]), reference[name])


if __name__ == "__main__":
    unittest.main()
