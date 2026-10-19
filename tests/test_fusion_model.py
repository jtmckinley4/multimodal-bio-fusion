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
from types import SimpleNamespace
from unittest import mock

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
        for cache in self.caches.values():
            cache.close()
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

    def test_an_interrupted_run_resumes_from_its_checkpoint(self):
        """Resume preserves dropout, row order, scheduler state, and the final model on CPU."""
        streams = self.streams()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.checkpoint"
            config = train.TrainConfig(max_epochs=5, batch_size=2, channels=3,
                                       early_stopping_patience=99, plateau_patience=0, seed=17)
            identity = {"fixture": "same CPU computation", "seed": config.seed}
            torch.manual_seed(config.seed)
            uninterrupted = blf.build(streams, "unimodal:rna", channels=3)
            # A constant score forces LR reductions, so a missing scheduler state matters.
            with mock.patch.object(train, "score", return_value=0.5):
                expected = train.fit(uninterrupted, streams, config, "cpu", log=None)
                torch.manual_seed(config.seed)
                interrupted = blf.build(streams, "unimodal:rna", channels=3)
                save = train._save_checkpoint

                def stop_after_second_epoch(*args, **kwargs):
                    save(*args, **kwargs)
                    if args[1] == 1:
                        raise KeyboardInterrupt("controlled interruption")

                with mock.patch.object(train, "_save_checkpoint", side_effect=stop_after_second_epoch):
                    with self.assertRaises(KeyboardInterrupt):
                        train.fit(interrupted, streams, config, "cpu", log=None,
                                  checkpoint=path, checkpoint_every=1, identity=identity)
                torch.rand(50)  # Simulate unrelated work before resuming.
                resumed = blf.build(streams, "unimodal:rna", channels=3)
                actual = train.fit(resumed, streams, config, "cpu", log=None,
                                   checkpoint=path, checkpoint_every=1, identity=identity)
            self.assertEqual(actual.history, expected.history)
            self.assertEqual(actual.best_epoch, expected.best_epoch)
            self.assertEqual(actual.test_score, expected.test_score)
            for name, weights in uninterrupted.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], weights, rtol=0, atol=0)
            self.assertFalse(path.exists())

    def test_run_seeds_before_build_and_rejects_mismatched_completed_results(self):
        streams = self.streams()
        streams.allow_legacy_cache = True  # Fixtures intentionally have no producer input record.
        config = train.TrainConfig(max_epochs=1, channels=3, seed=31)
        initial_states = []
        real_build = blf.build

        def observe_build(*args, **kwargs):
            model = real_build(*args, **kwargs)
            initial_states.append({name: value.detach().clone()
                                   for name, value in model.state_dict().items()})
            return model

        def no_training(model, streams, config, device, **kwargs):
            return train.Result(model.fusion_name, streams.key, vars(config), 1, 0, 0.5, 0.5,
                                0.0, identity=kwargs["identity"])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(train, "build", side_effect=observe_build), \
                    mock.patch.object(train, "fit", side_effect=no_training) as fitting:
                train.run(streams, "unimodal:rna", config, "cpu", root / "first", log=None)
                torch.rand(50)
                train.run(streams, "unimodal:rna", config, "cpu", root / "second", log=None)
                for name in initial_states[0]:
                    torch.testing.assert_close(initial_states[0][name], initial_states[1][name],
                                               rtol=0, atol=0)
                # Identical identity reuses the result without constructing or fitting again.
                train.run(streams, "unimodal:rna", config, "cpu", root / "first", log=None)
                self.assertEqual(fitting.call_count, 2)
                config.lr *= 2  # Learning rate is absent from the readable filename.
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    train.run(streams, "unimodal:rna", config, "cpu", root / "first", log=None)

    def test_equal_length_reordering_is_not_evidence_of_input_identity(self):
        streams = self.streams()
        for modality, cache in streams.caches.items():
            encoder = blf.ENCODERS[blf.ENCODER_OF_MODALITY[modality]]
            cache.meta["encoder"] = encoder.key
            cache.meta["input_sha256"] = TE.input_fingerprint(
                TE.inputs_for_rows(encoder, streams.rows["sequence"]))
        self.assertEqual(streams.check_input_identity(), "recorded_inputs")
        streams.rows = streams.rows.copy()
        streams.rows.loc[0, "sequence"] = "C" * self.lengths[0]
        streams.check()  # Counts still pass.
        with self.assertRaisesRegex(ValueError, "cached inputs differ"):
            streams.check_input_identity()

    def test_legacy_input_identity_requires_an_explicit_choice(self):
        streams = self.streams()
        with self.assertRaisesRegex(ValueError, "legacy cache"):
            streams.check_input_identity()
        streams.allow_legacy_cache = True
        self.assertEqual(streams.identity()["input_identity_status"], "unverified_legacy")

    def test_legacy_checkpoint_is_rejected_before_loading_its_weights(self):
        config = train.TrainConfig(max_epochs=2, channels=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.checkpoint"
            torch.save({"epoch": 0, "model": {}}, path)
            with self.assertRaisesRegex(ValueError, "legacy or incompatible"):
                train.fit(blf.build(self.streams(), "unimodal:rna", channels=3),
                          self.streams(), config, "cpu", log=None, checkpoint=path,
                          identity={"fixture": "current computation"})
            self.assertTrue(path.exists())

    def test_cache_mutation_requires_reopening(self):
        streams = self.streams()
        streams.allow_legacy_cache = True
        streams.identity()
        manifest = streams.caches["dna"].cache_dir / "meta.json"
        manifest.write_text(manifest.read_text() + " ")
        with self.assertRaisesRegex(RuntimeError, "close and reopen"):
            streams.identity()

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


class CacheOpeningTests(unittest.TestCase):
    """Opening caches distinguishes missing files from incompatible biological rows."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.key = "fixture"
        self.cache_dir = self.root / self.key / "ntv2_100m"
        self.metadata_path = self.cache_dir / "meta.json"
        rows = pd.DataFrame({"sequence": ["AAAAAA"], "length": [6],
                             "Value": [1.0], "Split": ["train"]})
        spec = SimpleNamespace(task="regression", label_column="Value", split_column="Split")
        registry = mock.patch.dict(blf.datasets_module.DATASETS, {self.key: spec})
        loader = mock.patch.object(blf.datasets_module, "blf_benchmark_rows", return_value=rows)
        registry.start()
        loader.start()
        self.addCleanup(registry.stop)
        self.addCleanup(loader.stop)

    def write_complete_caches(self):
        counts = {"dna": 1, "rna": 6, "prot": 2}
        for modality, encoder in blf.ENCODER_OF_MODALITY.items():
            cache_dir = self.root / self.key / encoder
            cache_dir.mkdir(parents=True)
            TE._write_shard(cache_dir / "shard_00000.npz",
                            [np.ones((counts[modality], 2), dtype=np.float16)])
            (cache_dir / "meta.json").write_text(json.dumps(
                {"n_shards": 1, "dtype": "float16", "encoder": encoder}), encoding="utf-8")

    def test_missing_directory_or_manifest_names_the_required_file(self):
        for directory_exists in (False, True):
            with self.subTest(directory_exists=directory_exists):
                if directory_exists:
                    self.cache_dir.mkdir(parents=True)
                with self.assertRaises(FileNotFoundError) as raised:
                    blf.BenchmarkStreams.open(self.key, self.root)
                self.assertIn(str(self.metadata_path.resolve()), str(raised.exception))
                self.assertIn("Step 2", str(raised.exception))
                self.assertEqual(self.cache_dir.exists(), directory_exists)

    def test_invalid_manifest_reports_its_path(self):
        self.cache_dir.mkdir(parents=True)
        for contents in ("{", "[]", "null", '"not an object"'):
            with self.subTest(contents=contents):
                self.metadata_path.write_text(contents, encoding="utf-8")
                with self.assertRaises(ValueError) as raised:
                    blf.BenchmarkStreams.open(self.key, self.root)
                self.assertIn(str(self.metadata_path.resolve()), str(raised.exception))
                self.assertEqual(self.metadata_path.read_text(encoding="utf-8"), contents)

    def test_missing_or_invalid_shard_count_is_not_an_empty_cache(self):
        self.cache_dir.mkdir(parents=True)
        manifests = [{}] + [{"n_shards": value} for value in (None, 0, -1, True, 1.0, "1")]
        for manifest in manifests:
            with self.subTest(manifest=manifest):
                self.metadata_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "positive integer n_shards") as raised:
                    blf.BenchmarkStreams.open(self.key, self.root)
                self.assertIn(str(self.metadata_path.resolve()), str(raised.exception))

    def test_missing_declared_shard_keeps_the_incomplete_cache_error(self):
        self.cache_dir.mkdir(parents=True)
        self.metadata_path.write_text('{"n_shards": 1}', encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "shard_00000.npz is missing"):
            blf.BenchmarkStreams.open(self.key, self.root)

    def test_complete_legacy_cache_still_requires_explicit_acceptance(self):
        self.write_complete_caches()
        with self.assertRaisesRegex(ValueError, "legacy cache"):
            blf.BenchmarkStreams.open(self.key, self.root)
        with self.assertWarnsRegex(RuntimeWarning, "Legacy token caches accepted explicitly"):
            streams = blf.BenchmarkStreams.open(self.key, self.root, allow_legacy_cache=True)
        try:
            self.assertEqual(streams.widths, {"dna": 2, "rna": 2, "prot": 2})
            for modality, count in {"dna": 1, "rna": 6, "prot": 2}.items():
                self.assertEqual(len(streams.caches[modality]), 1)
                self.assertEqual(streams.caches[modality][0].shape, (count, 2))
        finally:
            streams.close()

    def test_a_later_missing_manifest_closes_already_opened_caches(self):
        self.write_complete_caches()
        (self.root / self.key / "rnafm" / "meta.json").unlink()
        first_cache = TE.summarize(self.cache_dir)
        with mock.patch.object(blf, "summarize", return_value=first_cache), \
                mock.patch.object(first_cache, "close", wraps=first_cache.close) as closing:
            with self.assertRaises(FileNotFoundError):
                blf.BenchmarkStreams.open(self.key, self.root)
            closing.assert_called_once()


class ShardReaderTests(unittest.TestCase):
    def test_input_fingerprint_keeps_row_boundaries_and_order(self):
        self.assertNotEqual(TE.input_fingerprint(["AA", "C"]),
                            TE.input_fingerprint(["A", "AC"]))
        self.assertNotEqual(TE.input_fingerprint(["AAA", "CCC"]),
                            TE.input_fingerprint(["CCC", "AAA"]))

    def test_interrupted_extraction_records_inputs_before_writing_shards(self):
        encoder = SimpleNamespace(key="fixture", checkpoint="fixture", revision="fixed", max_len=20)
        model = SimpleNamespace(eval=lambda: None)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with mock.patch.object(TE, "load_encoder", return_value=(object(), model)), \
                    mock.patch.object(TE, "token_states", return_value=np.ones((3, 2), np.float16)), \
                    mock.patch.object(TE, "_write_shard", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    TE.extract(encoder, ["AAA", "CCC"], cache, "cpu", progress=None)
            self.assertTrue((cache / "meta.json").exists())
            before = (cache / "meta.json").read_bytes()
            self.assertEqual(json.loads(before)["input_sha256"],
                             TE.input_fingerprint(["AAA", "CCC"]))
            with mock.patch.object(TE, "load_encoder") as loading:
                with self.assertRaisesRegex(ValueError, "input_sha256"):
                    TE.extract(encoder, ["CCC", "AAA"], cache, "cpu", progress=None)
                loading.assert_not_called()
            self.assertEqual(before, (cache / "meta.json").read_bytes())

    def test_unmanifested_shards_cannot_be_given_invented_provenance(self):
        encoder = SimpleNamespace(key="fixture", checkpoint="fixture", revision="fixed", max_len=20)
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            shard = cache / "shard_00000.npz"
            TE._write_shard(shard, [np.ones((3, 2), np.float16)])
            before = shard.read_bytes()
            with self.assertRaisesRegex(ValueError, "without an input manifest"):
                TE.extract(encoder, ["AAA"], cache, "cpu", progress=None)
            self.assertEqual(before, shard.read_bytes())
            self.assertFalse((cache / "meta.json").exists())

    def test_a_mapped_shard_holds_what_numpy_would_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard_00000.npz"
            rows = [np.arange(6, dtype=np.float16).reshape(2, 3), np.ones((1, 3), np.float16)]
            TE._write_shard(path, rows)
            mapped = TE.open_shard(path)
            try:
                with np.load(path) as reference:
                    for name in reference.files:
                        np.testing.assert_array_equal(np.asarray(mapped[name]), reference[name])
            finally:
                # Windows cannot remove a shard while its memory maps remain open.
                for array in mapped.values():
                    memory_map = getattr(array, "_mmap", None)
                    if memory_map is not None:
                        memory_map.close()
                mapped.clear()


class ResultPersistenceTests(unittest.TestCase):
    def test_legacy_result_remains_readable_without_inventing_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.json"
            path.write_text(json.dumps({"variant": "concat", "benchmark": "fixture",
                "config": {}, "epochs": 1, "best_epoch": 0, "val_score": 0.5,
                "test_score": 0.4, "seconds": 2.0}))
            before = path.read_bytes()
            result = train.Result.load(path)
            self.assertIsNone(result.identity)
            self.assertEqual(result.test_score, 0.4)
            self.assertEqual(before, path.read_bytes())

    def test_failed_publication_preserves_the_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            result = train.Result("concat", "fixture", {}, 1, 0, 0.5, 0.4, 2.0)
            result.save(path)
            before = path.read_bytes()
            result.test_score = 0.6
            with mock.patch.object(train.os, "replace", side_effect=OSError("fixture failure")):
                with self.assertRaises(OSError):
                    result.save(path)
            self.assertEqual(before, path.read_bytes())


if __name__ == "__main__":
    unittest.main()
