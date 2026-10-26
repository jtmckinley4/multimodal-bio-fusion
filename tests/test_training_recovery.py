"""Tiny CPU recovery checks; no encoders, scientific data, or existing run artifacts."""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import isoformer, train  # noqa: E402


@dataclass
class TinyBatch:
    inputs: torch.Tensor
    labels: torch.Tensor

    def to(self, device):
        return TinyBatch(self.inputs.to(device), self.labels.to(device))


class TinyRows(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        # Exercise the Python and NumPy RNGs as well as dropout's torch RNG.
        noise = float(np.random.normal(scale=0.01)) + random.random() * 0.01
        return torch.tensor([index / 12, (index % 3) / 3, noise]), index / 12


def tiny_collate(rows):
    inputs, labels = zip(*rows)
    return TinyBatch(torch.stack(inputs), torch.tensor(labels))


class EpochBatches:
    """The epoch-counter shuffle contract used by the IsoFormer batch sampler."""

    def __init__(self):
        self.epoch = 0

    def __iter__(self):
        batches = np.arange(12).reshape(4, 3).tolist()
        np.random.default_rng(51 + self.epoch).shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self):
        return 4


class TinyModel(nn.Module):
    fusion_name = "tiny"

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 7), nn.ReLU(), nn.Dropout(0.4),
                                    nn.Linear(7, 1))

    def forward(self, batch):
        return self.layers(batch.inputs), {}


class Interrupted(Exception):
    pass


class TrainingRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.streams = SimpleNamespace(key="tiny-benchmark", task="regression")
        self.config = train.TrainConfig(lr=0.03, batch_size=3, max_epochs=7,
                                        plateau_patience=0, early_stopping_patience=99,
                                        seed=19)

    def model(self):
        torch.manual_seed(101)
        return TinyModel()

    def loaders(self, epoch_sampler=False):
        made = {}
        for index, name in enumerate(("train", "val", "test")):
            if name == "train" and epoch_sampler:
                made[name] = DataLoader(TinyRows(), batch_sampler=EpochBatches(),
                                        collate_fn=tiny_collate)
            else:
                made[name] = DataLoader(
                    TinyRows(), batch_size=3, shuffle=(name == "train"),
                    generator=torch.Generator().manual_seed(71 + index),
                    collate_fn=tiny_collate)
        return made

    def interrupt(self, path, epoch_sampler=False):
        save = train._save_checkpoint

        def save_then_interrupt(*args, **kwargs):
            save(*args, **kwargs)
            if args[1] == 2:
                raise Interrupted()

        with patch.object(train, "_save_checkpoint", side_effect=save_then_interrupt):
            with self.assertRaises(Interrupted):
                train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                          loaders=self.loaders(epoch_sampler), checkpoint=path,
                          checkpoint_every=1, keep_checkpoint=True)

    def test_resume_matches_uninterrupted_dropout_shuffle_and_scheduler_decay(self):
        for epoch_sampler in (False, True):
            with self.subTest(epoch_sampler=epoch_sampler), tempfile.TemporaryDirectory() as d:
                full_path, resumed_path = Path(d) / "full.pt", Path(d) / "resumed.pt"
                # Constant validation makes every epoch after the first decay the LR.
                with patch.object(train, "score", return_value=0.5):
                    full = train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                                     loaders=self.loaders(epoch_sampler), checkpoint=full_path,
                                     checkpoint_every=1, keep_checkpoint=True)
                    self.interrupt(resumed_path, epoch_sampler)
                    # Simulate unrelated notebook work and fresh loader/model construction.
                    torch.rand(29)
                    np.random.normal(size=31)
                    [random.random() for _ in range(23)]
                    resumed = train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                                        loaders=self.loaders(epoch_sampler),
                                        checkpoint=resumed_path, checkpoint_every=1,
                                        keep_checkpoint=True)
                self.assertEqual(full.history, resumed.history)
                self.assertEqual(full.test_prediction, resumed.test_prediction)
                self.assertLess(full.history[-1]["lr"], full.history[0]["lr"])
                full_saved = torch.load(full_path, map_location="cpu", weights_only=False)
                resumed_saved = torch.load(resumed_path, map_location="cpu", weights_only=False)
                self.assertEqual(full_saved["scheduler"], resumed_saved["scheduler"])
                for name, value in full_saved["model"].items():
                    self.assertTrue(torch.equal(value, resumed_saved["model"][name]), name)

    def test_old_checkpoint_can_resume_but_warns_that_recovery_is_best_effort(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "old.pt"
            model = self.model()
            optimiser = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            # The positional helper is also a supported historical caller.
            train._save_checkpoint(
                path, 0, model, optimiser,
                {"score": 0.5, "epoch": 0, "state": model.state_dict()},
                [{"epoch": 0}], 12.5)
            old = torch.load(path, map_location="cpu", weights_only=False)
            old.pop("rng")  # A checkpoint written before RNG recovery was introduced.
            torch.save(old, path)
            with self.assertWarnsRegex(RuntimeWarning, "legacy checkpoint.*best-effort"):
                with patch.object(train, "score", return_value=0.5):
                    result = train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                                       loaders=self.loaders(), checkpoint=path)
            self.assertEqual([row["epoch"] for row in result.history], list(range(7)))
            self.assertGreater(result.seconds, 12.5)
            self.assertFalse(path.exists())

    def test_checkpoint_identity_mismatch_is_rejected_without_rewriting(self):
        for field in ("config", "variant", "benchmark"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                path = Path(d) / "resume.pt"
                with patch.object(train, "score", return_value=0.5):
                    self.interrupt(path)
                before = path.read_bytes()
                config, streams, model = self.config, self.streams, self.model()
                if field == "config":
                    config = replace(config, weight_decay=0.02)
                elif field == "variant":
                    model.fusion_name = "other"
                else:
                    streams = SimpleNamespace(key="other", task="regression")
                with self.assertRaisesRegex(ValueError, "saved.*do not match"):
                    train.fit(model, streams, config, "cpu", log=None,
                              loaders=self.loaders(), checkpoint=path)
                self.assertEqual(path.read_bytes(), before)

    def stored_result(self, directory):
        path = Path(directory) / "same.json"
        train.Result(variant="tiny", benchmark=self.streams.key, config=asdict(self.config),
                     epochs=1, best_epoch=0, val_score=0.5, test_score=0.6,
                     seconds=1.0).save(path)
        return path

    def test_result_identity_mismatch_preserves_result_and_weights(self):
        for field in ("config", "variant", "benchmark"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as d:
                path = self.stored_result(d)
                weights = Path(d) / "same.weights.pt"
                weights.write_bytes(b"existing weights must stay untouched")
                before = path.read_bytes(), weights.read_bytes()
                config, streams, variant = self.config, self.streams, "tiny"
                if field == "config":
                    # LR does not appear in the default filename, but is part of identity.
                    config = replace(config, lr=0.1)
                elif field == "variant":
                    variant = "other"
                else:
                    streams = SimpleNamespace(key="other", task="regression")
                with patch.object(train, "build", side_effect=AssertionError("must not train")):
                    with self.assertRaisesRegex(ValueError, "saved.*do not match"):
                        train.run(streams, variant, config, "cpu", d, tag="same", log=None)
                self.assertEqual((path.read_bytes(), weights.read_bytes()), before)

    def test_matching_historical_result_is_reused_without_retraining_for_weights(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.stored_result(d)
            before = path.read_bytes()
            with patch.object(train, "build", side_effect=AssertionError("must not train")):
                result = train.run(self.streams, "tiny", self.config, "cpu", d,
                                   tag="same", log=None)
            self.assertIsNone(result.test_prediction)
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse((Path(d) / "same.weights.pt").exists())

    def test_requested_variant_is_retained_for_a_custom_builder_model(self):
        # run() supplies this identity when a custom builder names its model differently.
        identity = {"config": asdict(self.config), "benchmark": self.streams.key,
                    "variant": "requested-custom-variant"}
        with patch.object(train, "score", return_value=0.5):
            result = train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                               loaders=self.loaders(), run_identity=identity)
        self.assertEqual(result.variant, "requested-custom-variant")


class LegacyGTExResultTests(unittest.TestCase):
    """Reuse small saved-record fixtures without building models or fitting."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name)
        self.config = train.TrainConfig(batch_size=16, seed=0)
        self.streams = SimpleNamespace(key="gtex_pilot")
        for module, name in ((train, "build"), (train, "fit"), (train, "_save_weights"),
                             (isoformer, "build")):
            self.enterContext(patch.object(module, name,
                                          side_effect=AssertionError("must only read results")))

    def store_result(self, variant="dna", tag=None, benchmark="gtex_pilot",
                     saved_variant="IsoFormer", predictions=True):
        tag = tag or f"gtex_pilot__{variant}__heads8__budget300M__seed0"
        path = self.folder / f"{tag}.json"
        train.Result(
            variant=saved_variant, benchmark=benchmark, config=asdict(self.config),
            epochs=1, best_epoch=0, val_score=0.5, test_score=0.6, seconds=1.0,
            test_truth=[1.0, 2.0] if predictions else None,
            test_prediction=[1.1, 1.9] if predictions else None).save(path)
        # Reading results must not inspect, replace or deserialize existing weights.
        path.with_suffix(".weights.pt").write_bytes(b"preserved weights fixture")
        return path

    def read_result(self, path, **overrides):
        arguments = dict(streams=self.streams, variant="dna", config=self.config,
                         device="cpu", results_dir=self.folder, log=None,
                         builder=isoformer.builder, tag=path.stem, n_heads=8)
        arguments.update(overrides)
        before = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        try:
            with patch.object(train.Result, "save", side_effect=AssertionError("must not save")):
                return train.run(**arguments)
        finally:
            self.assertEqual({p.name: p.read_bytes() for p in self.folder.iterdir()}, before)

    def test_six_historical_modalities_reuse_raw_record_without_training_or_writes(self):
        for variant in ("dna", "rna", "prot", "dna+prot", "dna+rna", "dna+rna+prot"):
            with self.subTest(variant=variant):
                path = self.store_result(variant)
                with self.assertWarnsRegex(RuntimeWarning, "interpreting legacy GTEx"):
                    result = self.read_result(path, variant=variant, require_predictions=True)
                self.assertEqual(result.variant, "IsoFormer")
                self.assertEqual(result.test_prediction, [1.1, 1.9])

    def test_changed_configuration_benchmark_or_builder_is_rejected(self):
        path = self.store_result()
        changes = [dict(config=replace(self.config, lr=0.1)),
                   dict(config=replace(self.config, seed=1)),
                   dict(streams=SimpleNamespace(key="another-benchmark")),
                   dict(builder=None), dict(builder=lambda *args, **kwargs: None)]
        for change in changes:
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "saved.*do not match"):
                    self.read_result(path, **change)

    def test_changed_or_unknown_model_settings_are_rejected(self):
        path = self.store_result()
        for change in (dict(n_heads=4), dict(n_layers=2), dict(dropout=0.2),
                       dict(unknown_option=True)):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
                    self.read_result(path, **change)

    def test_explicit_unchanged_model_defaults_are_accepted(self):
        path = self.store_result()
        with self.assertWarnsRegex(RuntimeWarning, "interpreting legacy GTEx"):
            self.read_result(path, n_layers=1, dropout=0.1)

    def test_changed_constructor_default_is_not_assumed_to_match_history(self):
        path = self.store_result()
        defaults = isoformer.IsoFormer.__init__.__defaults__
        # Dropout is the final default; an omitted keyword would now mean 0.2.
        with patch.object(isoformer.IsoFormer.__init__, "__defaults__", defaults[:-1] + (0.2,)):
            with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
                self.read_result(path)

    def test_changed_modality_or_unsupported_tag_is_rejected(self):
        path = self.store_result()
        with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
            self.read_result(path, variant="rna")
        for tag in ("gtex_pilot__dna__heads8__seed0",
                    "gtex_pilot__dna__heads8__budget200M__seed0",
                    "gtex_pilot__dna__heads8__budget300M__seed1", "arbitrary-name"):
            with self.subTest(tag=tag):
                path = self.store_result(tag=tag)
                with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
                    self.read_result(path)
        path = self.store_result(variant="rna+prot")
        with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
            self.read_result(path, variant="rna+prot")

    def test_non_gtex_record_does_not_gain_the_legacy_alias(self):
        path = self.store_result(benchmark="another-benchmark")
        with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
            self.read_result(path, streams=SimpleNamespace(key="another-benchmark"))

    def test_missing_predictions_do_not_trigger_replacement_training(self):
        path = self.store_result(predictions=False)
        with self.assertWarnsRegex(RuntimeWarning, "interpreting legacy GTEx"):
            self.assertIsNone(self.read_result(path).test_prediction)
        with self.assertWarnsRegex(RuntimeWarning, "interpreting legacy GTEx"):
            with self.assertRaisesRegex(ValueError, "legacy GTEx result has no per-row predictions"):
                self.read_result(path, require_predictions=True)

    def test_current_format_reuse_stays_strict_without_legacy_warning(self):
        path = self.store_result(saved_variant="dna")
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            self.assertEqual(self.read_result(path).variant, "dna")
        self.assertEqual(seen, [])
        with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
            self.read_result(path, variant="rna")

    def test_checkpoint_identity_does_not_accept_the_alias(self):
        saved = dict(config=asdict(self.config), benchmark="gtex_pilot", variant="IsoFormer")
        requested = dict(saved, variant="dna")
        with self.assertRaisesRegex(ValueError, "saved variant.*do not match"):
            train._check_identity(saved, requested, "legacy.checkpoint")


if __name__ == "__main__":
    unittest.main()
