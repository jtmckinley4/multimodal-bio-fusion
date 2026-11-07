"""Tiny CPU recovery checks; no encoders, scientific data, or existing run artifacts."""

from contextlib import ExitStack
from copy import deepcopy
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

    def identity(self, epoch_sampler):
        generator = {"type": "torch.utils.data.dataloader.DataLoader", "fields": ["generator"]}
        sampler = {"type": "torch.utils.data.sampler.RandomSampler", "fields": ["generator"]}
        recovery = {name: {"loader": generator} for name in ("train", "val", "test")}
        if epoch_sampler:
            recovery["train"] = {"batch_sampler": {
                "type": f"{EpochBatches.__module__}.EpochBatches", "fields": ["epoch"]}}
        else:
            recovery["train"].update({"sampler": sampler, "batch_sampler.sampler": sampler})
        return {
            "schema": "fusion.run.v1", "config": asdict(self.config),
            "variant": "tiny", "benchmark": self.streams.key,
            "inputs": {"encoders": {}}, "model": {"contract": "synthetic-v1"},
            "loader": {"kind": "custom", "recovery": recovery},
        }

    def interrupt(self, path, epoch_sampler=False, identity=None):
        save = train._save_checkpoint

        def save_then_interrupt(*args, **kwargs):
            save(*args, **kwargs)
            if args[1] == 2:
                raise Interrupted()

        with patch.object(train, "_save_checkpoint", side_effect=save_then_interrupt):
            with self.assertRaises(Interrupted):
                train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                          loaders=self.loaders(epoch_sampler), checkpoint=path,
                          checkpoint_every=1, keep_checkpoint=True, run_identity=identity)

    def test_resume_matches_uninterrupted_dropout_shuffle_and_scheduler_decay(self):
        for epoch_sampler, versioned in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(epoch_sampler=epoch_sampler, versioned=versioned), tempfile.TemporaryDirectory() as d:
                full_path, resumed_path = Path(d) / "full.pt", Path(d) / "resumed.pt"
                identity = self.identity(epoch_sampler) if versioned else None
                # Constant validation makes every epoch after the first decay the LR.
                with patch.object(train, "score", return_value=0.5):
                    full = train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                                     loaders=self.loaders(epoch_sampler), checkpoint=full_path,
                                     checkpoint_every=1, keep_checkpoint=True, run_identity=identity)
                    self.interrupt(resumed_path, epoch_sampler, identity)
                    # Simulate unrelated notebook work and fresh loader/model construction.
                    torch.rand(29)
                    np.random.normal(size=31)
                    [random.random() for _ in range(23)]
                    resumed = train.fit(self.model(), self.streams, self.config, "cpu", log=None,
                                        loaders=self.loaders(epoch_sampler),
                                        checkpoint=resumed_path, checkpoint_every=1,
                                        keep_checkpoint=True, run_identity=identity)
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


class CheckpointPreflightTests(unittest.TestCase):
    """Inspect tiny real-format files without training, state restoration or writes."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "fixture.checkpoint"
        self.config = train.TrainConfig(max_epochs=5, seed=1)
        self.identity = {"config": asdict(self.config), "benchmark": "fixture", "variant": "concat"}
        model = nn.Linear(2, 1)
        optimiser = self.fixture_optimiser(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", patience=self.config.plateau_patience)
        scheduler.step(.5)
        splits = {
            name: SimpleNamespace(generator=torch.Generator().manual_seed(index))
            for index, name in enumerate(("train", "val", "test"))
        }
        # Serialize through the real producer but do not optimize or iterate data.
        with patch.object(optimiser, "step", side_effect=AssertionError("no optimization")):
            train._save_checkpoint(
                self.path, 0, model, optimiser,
                {"score": .5, "epoch": 0, "state": model.state_dict()},
                [{"epoch": 0, "train_loss": .2, "val_score": .5, "lr": self.config.lr}],
                1.25, scheduler=scheduler, splits=splits, identity=self.identity,
            )
        self.payload = torch.load(self.path, map_location="cpu", weights_only=False)

    def fixture_optimiser(self, model):
        optimiser = torch.optim.Adam(model.parameters(), lr=self.config.lr,
                                     weight_decay=self.config.weight_decay)
        # Hand-written state exercises checkpoint validation without optimization.
        # The unused bias deliberately has no slots, as real models may have unused layers.
        parameter = next(model.parameters())
        optimiser.state[parameter] = {"step": torch.tensor(1.0),
                                      "exp_avg": torch.zeros_like(parameter),
                                      "exp_avg_sq": torch.zeros_like(parameter)}
        return optimiser

    def versioned_identity(self):
        # Explicit expected recovery state, independent of the producer implementation.
        recovery = {
            name: {"loader": {"type": "types.SimpleNamespace", "fields": ["generator"]}}
            for name in ("train", "val", "test")
        }
        return {
            **self.identity, "schema": "fusion.run.v1", "inputs": {"encoders": {}},
            "model": {"contract": "synthetic-v1"},
            "loader": {"kind": "custom", "recovery": recovery},
        }

    def test_versioned_checkpoint_requires_every_declared_loader_component(self):
        identity = self.versioned_identity()
        payload = deepcopy(self.payload)
        payload["identity"] = identity
        payload["producer"] = {"host": "synthetic-producer"}
        torch.save(payload, self.path)
        self.inspect(identity)
        for split in ("train", "val", "test"):
            missing = deepcopy(payload)
            missing["loaders"][split] = {}
            torch.save(missing, self.path)
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "recovery components"):
                self.inspect(identity)

    def test_versioned_checkpoint_checks_component_type_and_complete_state_fields(self):
        identity = self.versioned_identity()
        identity["loader"]["recovery"]["train"]["loader"]["fields"] = ["epoch", "generator"]
        payload = deepcopy(self.payload)
        payload["identity"] = identity
        payload["loaders"]["train"]["loader"]["epoch"] = 2
        torch.save(payload, self.path)
        self.inspect(identity)
        for change in ("type", "epoch", "generator", "extra_state"):
            altered = deepcopy(payload)
            component = altered["loaders"]["train"]["loader"]
            if change == "type":
                component["type"] = "another.Loader"
            elif change == "extra_state":
                component["state_dict"] = {"unrequested": 1}
            else:
                del component[change]
            torch.save(altered, self.path)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "recovery (type|fields)"):
                self.inspect(identity)

    def test_run_blocks_incomplete_legacy_checkpoint_before_preparation(self):
        payload = deepcopy(self.payload)
        del payload["scheduler"]
        torch.save(payload, self.path)
        before = self.path.read_bytes()
        with ExitStack() as stack:
            forbidden = AssertionError("invalid recovery must fail before preparation")
            for module, name in ((torch, "manual_seed"), (np.random, "seed"),
                                 (random, "seed"), (train, "build"), (train, "fit")):
                stack.enter_context(patch.object(module, name, side_effect=forbidden))
            with self.assertRaisesRegex(ValueError, "missing fields: scheduler"):
                train.run(SimpleNamespace(key="fixture"), "concat", self.config, "cpu",
                          self.path.parent, tag="fixture", log=None)
        self.assertEqual(before, self.path.read_bytes())
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_strict_fit_rechecks_payload_before_rng_model_or_optimizer_restoration(self):
        identity = self.versioned_identity()
        payload = deepcopy(self.payload)
        payload["identity"] = identity
        payload["loaders"]["train"] = {}
        torch.save(payload, self.path)
        model = nn.Linear(2, 1)
        before = self.path.read_bytes()
        with ExitStack() as stack:
            forbidden = AssertionError("strict restore must validate first")
            for module, name in ((torch, "manual_seed"), (torch.optim, "Adam"),
                                 (model, "to"), (model, "load_state_dict"),
                                 (train, "_restore_loaders"), (train, "_restore_rng")):
                stack.enter_context(patch.object(module, name, side_effect=forbidden))
            with self.assertRaisesRegex(ValueError, "recovery components"):
                train.fit(model, SimpleNamespace(key="fixture", task="regression"),
                          self.config, "cpu", checkpoint=self.path, run_identity=identity,
                          log=None)
        self.assertEqual(before, self.path.read_bytes())

    def test_direct_legacy_fit_cannot_weaken_a_versioned_checkpoint_contract(self):
        payload = deepcopy(self.payload)
        payload["identity"] = self.versioned_identity()
        torch.save(payload, self.path)
        model = nn.Linear(2, 1)
        model.fusion_name = "concat"
        before = self.path.read_bytes()
        with patch.object(torch, "manual_seed", side_effect=AssertionError("must not seed")):
            with self.assertRaisesRegex(ValueError, "versioned checkpoint requires its full"):
                train.fit(model, SimpleNamespace(key="fixture", task="regression"),
                          self.config, "cpu", checkpoint=self.path, log=None)
        self.assertEqual(before, self.path.read_bytes())

    def test_explicit_legacy_identity_rechecks_payload_in_actual_restore_path(self):
        payload = deepcopy(self.payload)
        del payload["scheduler"]
        torch.save(payload, self.path)
        model = nn.Linear(2, 1)
        before = self.path.read_bytes()
        with patch.object(torch, "manual_seed", side_effect=AssertionError("must not seed")):
            with self.assertRaisesRegex(ValueError, "missing fields: scheduler"):
                train.fit(model, SimpleNamespace(key="fixture", task="regression"),
                          self.config, "cpu", checkpoint=self.path, run_identity=self.identity,
                          log=None)
        self.assertEqual(before, self.path.read_bytes())

    def test_checkpoint_producer_is_stored_separately_from_matching_identity(self):
        identity = self.versioned_identity()
        model = nn.Linear(2, 1)
        optimiser = self.fixture_optimiser(model)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="max", patience=self.config.plateau_patience)
        scheduler.step(.5)
        splits = {name: SimpleNamespace(generator=torch.Generator())
                  for name in ("train", "val", "test")}
        producer = {"host": "synthetic-a", "device": "cpu"}
        train._save_checkpoint(
            self.path, 0, model, optimiser,
            {"score": .5, "epoch": 0, "state": model.state_dict()},
            [{"epoch": 0}], 1.25, scheduler=scheduler, splits=splits,
            identity=identity, producer=producer)
        payload = torch.load(self.path, map_location="cpu", weights_only=False)
        self.assertEqual(payload["identity"], identity)
        self.assertEqual(payload["producer"], producer)
        self.inspect(identity)

    def test_versioned_scheduler_requires_complete_matching_state(self):
        identity = self.versioned_identity()
        original = deepcopy(self.payload)
        original["identity"] = identity
        changes = (
            {"patience": self.config.plateau_patience + 1}, {"mode": "min"},
            {"last_epoch": 0}, {"num_bad_epochs": -1}, {"_last_lr": [999.0]},
            {"best": float("nan")}, {"min_lrs": [0.1]},
        )
        for change in changes:
            payload = deepcopy(original)
            payload["scheduler"].update(change)
            torch.save(payload, self.path)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "scheduler"):
                self.inspect(identity)
        for field in original["scheduler"]:
            if field == "default_min_lr":  # Optional across supported PyTorch versions.
                continue
            payload = deepcopy(original)
            del payload["scheduler"][field]
            torch.save(payload, self.path)
            with self.subTest(missing=field), self.assertRaisesRegex(ValueError, "incomplete scheduler"):
                self.inspect(identity)

    def test_versioned_rng_checks_validate_private_generators_without_mutation(self):
        identity = self.versioned_identity()
        original = deepcopy(self.payload)
        original["identity"] = identity
        changes = {
            "torch": torch.zeros(2, dtype=torch.uint8),
            "numpy": ("unknown-generator", np.zeros(624, dtype=np.uint32), 0, 0, 0.0),
            "python": (99, tuple(range(625)), None),
        }
        for field, value in changes.items():
            payload = deepcopy(original)
            payload["rng"][field] = value
            torch.save(payload, self.path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "invalid RNG recovery"):
                self.inspect(identity)
        original["loaders"]["train"]["loader"]["generator"] = torch.zeros(2, dtype=torch.uint8)
        torch.save(original, self.path)
        with self.assertRaisesRegex(ValueError, "invalid train loader generator"):
            self.inspect(identity)

    def test_versioned_adam_state_rejects_missing_or_malformed_slots_and_settings(self):
        identity = self.versioned_identity()
        original = deepcopy(self.payload)
        original["identity"] = identity
        for change in ("empty", "missing_step", "missing_moment", "bad_step", "shape", "dtype",
                       "unknown_id", "settings"):
            payload = deepcopy(original)
            optimiser = payload["optimiser"]
            slot = optimiser["state"][0]
            if change == "empty":
                optimiser["state"] = {}
            elif change == "missing_step":
                del slot["step"]
            elif change == "missing_moment":
                del slot["exp_avg_sq"]
            elif change == "bad_step":
                slot["step"] = torch.tensor(0.0)
            elif change == "shape":
                slot["exp_avg_sq"] = torch.zeros(99)
            elif change == "dtype":
                slot["exp_avg_sq"] = slot["exp_avg_sq"].double()
            elif change == "unknown_id":
                optimiser["state"][999] = optimiser["state"].pop(0)
            else:
                optimiser["param_groups"][0]["weight_decay"] = 99
            torch.save(payload, self.path)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Adam"):
                self.inspect(identity)

    def test_actual_optimizer_parameter_shapes_are_checked_without_restoration(self):
        state = deepcopy(self.payload["optimiser"])
        state["state"][0]["exp_avg"] = torch.zeros(99)
        state["state"][0]["exp_avg_sq"] = torch.zeros(99)
        model = nn.Linear(2, 1)
        optimiser = self.fixture_optimiser(model)
        with patch.object(optimiser, "load_state_dict", side_effect=AssertionError("must not restore")):
            with self.assertRaisesRegex(ValueError, "constructed model parameter"):
                train._validate_adam_state(state, asdict(self.config), self.path,
                                           parameter_groups=optimiser.param_groups)

    def inspect(self, identity=None):
        before = self.path.read_bytes() if self.path.exists() else None
        rng_before = torch.get_rng_state().clone(), np.random.get_state(), random.getstate()
        with ExitStack() as stack:
            forbidden = AssertionError("checkpoint inspection must only read")
            for module, name in ((train, "fit"), (train, "build"), (train, "_save_checkpoint"),
                                 (train, "_save_weights"), (train, "_restore_rng"),
                                 (train, "_restore_loaders"), (torch, "save")):
                stack.enter_context(patch.object(module, name, side_effect=forbidden))
            load = stack.enter_context(patch.object(torch, "load", wraps=torch.load))
            try:
                result = train.checkpoint_preflight(self.path, identity or self.identity)
                self.assertIsNone(result)
                load.assert_called_once_with(self.path, map_location="cpu", weights_only=False)
            finally:
                self.assertEqual(self.path.read_bytes() if self.path.exists() else None, before)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before[0]))
                current_numpy = np.random.get_state()
                self.assertEqual(current_numpy[0], rng_before[1][0])
                self.assertTrue(np.array_equal(current_numpy[1], rng_before[1][1]))
                self.assertEqual(current_numpy[2:], rng_before[1][2:])
                self.assertEqual(random.getstate(), rng_before[2])

    def test_current_checkpoint_is_read_on_cpu_without_mutation(self):
        self.inspect()

    def test_changed_identity_or_extra_recorded_identity_is_rejected(self):
        for field, value in (("benchmark", "other"), ("variant", "cross"),
                             ("config", {**asdict(self.config), "seed": 2})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity"):
                self.inspect({**self.identity, field: value})
        payload = deepcopy(self.payload)
        payload["identity"]["unknown"] = "another contract"
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, "identity"):
            self.inspect()

    def test_every_required_recovery_field_is_checked(self):
        for field in ("identity", "scheduler", "rng", "loaders", "model", "optimiser",
                      "best", "history", "epoch", "elapsed"):
            payload = deepcopy(self.payload)
            del payload[field]
            torch.save(payload, self.path)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing fields"):
                self.inspect()

    def test_invalid_epoch_history_elapsed_and_best_state_are_rejected(self):
        changes = (
            {"epoch": True}, {"epoch": -1}, {"epoch": self.config.max_epochs},
            {"history": []}, {"history": [{"epoch": 1}]}, {"elapsed": float("nan")},
            {"elapsed": -1}, {"best": {"score": .5, "epoch": 0, "state": {}}},
            {"best": {**self.payload["best"], "epoch": 1}},
            {"best": {**self.payload["best"], "score": float("nan")}},
            {"model": {"wrong_key": torch.zeros(1)}},
        )
        for change in changes:
            torch.save({**self.payload, **change}, self.path)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.inspect()

    def test_malformed_recovery_components_are_rejected(self):
        changes = (
            {"scheduler": {}}, {"optimiser": {}}, {"rng": {}}, {"loaders": {}},
            {"rng": {**self.payload["rng"], "torch": torch.zeros(2)}},
            {"rng": {**self.payload["rng"], "device_type": "cuda"}},
            {"loaders": {**self.payload["loaders"], "train": {"loader": {"type": "Loader", "generator": None}}}},
        )
        for change in changes:
            torch.save({**self.payload, **change}, self.path)
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                self.inspect()

    def test_undefined_best_score_is_valid_when_best_weights_exist(self):
        payload = deepcopy(self.payload)
        payload["best"]["score"] = -float("inf")
        torch.save(payload, self.path)
        self.inspect()

    def test_missing_and_unreadable_files_report_value_error(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "could not be read"):
            self.inspect()
        self.path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "could not be read"):
            self.inspect()


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
