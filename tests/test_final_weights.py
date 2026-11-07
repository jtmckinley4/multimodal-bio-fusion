"""Check final artifact publication with a fixed CPU model and no optimization."""
import hashlib
import json
from copy import deepcopy
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import train


class FinalWeightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.streams = SimpleNamespace(key="fixture")
        self.config = train.TrainConfig(tconv="equation")
        self.model = torch.nn.Linear(2, 1)
        self.tag = "fixture__concat__lambda0.0__ch100__equation__seed0"
        self.weights = self.folder / (self.tag + ".weights.pt")
        self.record = self.folder / (self.tag + ".json")
        self.checkpoint = self.folder / (self.tag + ".checkpoint")
        self.fit = self.enterContext(patch.object(train, "fit", side_effect=self.completed_fit))
        self.enterContext(patch.object(torch.optim, "Adam", side_effect=AssertionError("No optimization allowed")))

    def result(self):
        return train.Result(variant="concat", benchmark="fixture", config=asdict(self.config),
                            epochs=4, best_epoch=2, val_score=0.7, test_score=0.6, seconds=1.0,
                            test_truth=[1.0, 2.0], test_prediction=[1.1, 1.9])

    def completed_fit(self, model, streams, config, device, **kwargs):
        self.assertTrue(kwargs["keep_checkpoint"])
        self.assertEqual(device, "cpu")
        self.checkpoint.write_bytes(b"retained recovery fixture")
        # fit() returns with its best-validation state already restored.
        with torch.no_grad():
            model.weight.fill_(7)
            model.bias.fill_(4)
        return self.result()

    def run_model(self, **kwargs):
        return train.run(self.streams, "concat", self.config, "cpu", self.folder, log=None,
                         builder=lambda *args, **options: self.model, **kwargs)

    def identity(self):
        # A custom contract keeps this publication fixture independent of research inputs.
        return {
            "schema": "fusion.run.v1", "config": asdict(self.config),
            "variant": "concat", "benchmark": "fixture",
            "inputs": {"encoders": {}}, "model": {"contract": "synthetic-v1"},
            "loader": {"kind": "custom", "recovery": {
                "train": {}, "val": {}, "test": {}}},
        }

    def test_versioned_identity_and_separate_producer_are_retained(self):
        identity = self.identity()
        producer = {"host": "synthetic-a", "device": "cpu"}
        result = self.run_model(run_identity=identity, producer=producer)
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        record = json.loads(self.record.read_text())
        for saved in (payload, record, result.extra_fields):
            self.assertEqual(saved["identity"], identity)
            self.assertEqual(saved["producer"], producer)
        self.assertEqual(self.fit.call_args.kwargs["run_identity"], identity)
        self.assertEqual(self.fit.call_args.kwargs["producer"], producer)

        # Reuse does not relabel the original producer as the consuming machine.
        before = self.record.read_bytes(), self.weights.read_bytes()
        self.fit.reset_mock()
        reused = self.run_model(run_identity=identity, producer={"host": "synthetic-b"})
        self.assertEqual(reused.extra_fields["producer"], producer)
        self.fit.assert_not_called()
        self.assertEqual(before, (self.record.read_bytes(), self.weights.read_bytes()))

    def test_versioned_reuse_refuses_changed_inputs_model_or_loader(self):
        identity = self.identity()
        self.run_model(run_identity=identity)
        before = self.record.read_bytes(), self.weights.read_bytes()
        self.fit.reset_mock()
        for field in ("inputs", "model", "loader"):
            requested = deepcopy(identity)
            requested[field]["changed_fixture_setting"] = True
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "identity does not match"):
                self.run_model(run_identity=requested)
            self.fit.assert_not_called()
            self.assertEqual(before, (self.record.read_bytes(), self.weights.read_bytes()))

    def test_versioned_request_preserves_historical_completed_result(self):
        self.result().save(self.record)
        before = self.record.read_bytes()
        with self.assertRaisesRegex(ValueError, "no versioned computation identity"):
            self.run_model(run_identity=self.identity())
        self.fit.assert_not_called()
        self.assertEqual(before, self.record.read_bytes())
        self.assertFalse(self.weights.exists())

    def test_versioned_completed_result_requires_full_identity_for_run_reuse(self):
        for schema in ("fusion.run.v1", None, "", False, "future-version"):
            result = self.result()
            result.extra_fields["identity"] = {**self.identity(), "schema": schema}
            result.save(self.record)
            self.weights.write_bytes(b"preserved completed weights")
            self.checkpoint.write_bytes(b"preserved leftover checkpoint")
            before = {path.name: path.read_bytes() for path in self.folder.iterdir()}
            with self.subTest(schema=schema):
                with patch.object(torch, "manual_seed", side_effect=AssertionError("must not seed")):
                    with patch.object(torch, "load", side_effect=AssertionError("must not deserialize")):
                        with self.assertRaisesRegex(ValueError, "requires its full requested"):
                            self.run_model()
                self.fit.assert_not_called()
                self.assertEqual(before, {path.name: path.read_bytes() for path in self.folder.iterdir()})
                # Reading a record remains possible without claiming computation compatibility.
                self.assertEqual(train.Result.load(self.record).extra_fields["identity"],
                                 result.extra_fields["identity"])

    def test_malformed_identity_declarations_never_downgrade_to_legacy_reuse(self):
        for declared in ([], False, 0, "", "not-an-identity"):
            result = self.result()
            result.extra_fields["identity"] = declared
            result.save(self.record)
            before = self.record.read_bytes()
            with self.subTest(identity=declared), self.assertRaisesRegex(ValueError, "malformed computation identity"):
                self.run_model()
            self.fit.assert_not_called()
            self.assertEqual(before, self.record.read_bytes())
            self.assertFalse(self.weights.exists())

    def test_unversioned_identity_metadata_keeps_historical_read_compatibility(self):
        for declared in (None, {}, {"config": asdict(self.config)}):
            result = self.result()
            result.extra_fields["identity"] = declared
            result.save(self.record)
            before = self.record.read_bytes()
            with self.subTest(identity=declared):
                self.run_model()
                self.fit.assert_not_called()
                self.assertEqual(before, self.record.read_bytes())
                self.assertFalse(self.weights.exists())

    def test_completed_record_takes_precedence_over_leftover_checkpoint(self):
        self.result().save(self.record)
        self.checkpoint.write_bytes(b"leftover checkpoint must not be inspected")
        before = self.record.read_bytes(), self.checkpoint.read_bytes()
        with patch.object(torch, "load", side_effect=AssertionError("completed reuse must not load weights")):
            self.run_model()
        self.fit.assert_not_called()
        self.assertEqual(before, (self.record.read_bytes(), self.checkpoint.read_bytes()))

    def test_unreadable_checkpoint_blocks_before_model_construction_or_seeding(self):
        self.checkpoint.write_bytes(b"not a checkpoint")
        with patch.object(torch, "manual_seed", side_effect=AssertionError("must not seed")):
            with self.assertRaisesRegex(ValueError, "checkpoint could not be read"):
                train.run(self.streams, "concat", self.config, "cpu", self.folder, log=None,
                          builder=lambda *args, **kwargs: self.fail("must not construct model"))
        self.fit.assert_not_called()
        self.assertEqual(self.checkpoint.read_bytes(), b"not a checkpoint")
        self.assertFalse(self.record.exists())
        self.assertFalse(self.weights.exists())

    def test_invalid_identity_does_not_create_a_result_directory(self):
        folder = self.folder / "not-created"
        identity = self.identity()
        identity["schema"] = "unknown-version"
        with self.assertRaises(ValueError):
            train.run(self.streams, "concat", self.config, "cpu", folder,
                      run_identity=identity, log=None)
        self.assertFalse(folder.exists())
        self.fit.assert_not_called()

    def test_best_state_is_linked_and_reused_without_fitting(self):
        result = self.run_model()
        artifact = result.extra_fields["model_artifact"]
        self.assertEqual(artifact["sha256"], hashlib.sha256(self.weights.read_bytes()).hexdigest())
        self.assertEqual(artifact["size_bytes"], self.weights.stat().st_size)
        payload = torch.load(self.weights, map_location="cpu", weights_only=True)
        self.assertEqual(payload["best_epoch"], 2)
        self.assertEqual(payload["config"], asdict(self.config))
        self.assertTrue(torch.equal(payload["state_dict"]["weight"], torch.full((1, 2), 7.0)))
        self.assertTrue(torch.equal(payload["state_dict"]["bias"], torch.full((1,), 4.0)))
        self.assertEqual(json.loads(self.record.read_text())["model_artifact"], artifact)
        self.assertFalse(self.checkpoint.exists())
        before = self.record.read_bytes(), self.weights.read_bytes()
        self.fit.reset_mock()
        self.assertEqual(self.run_model(require_predictions=True).test_prediction, [1.1, 1.9])
        self.fit.assert_not_called()
        self.assertEqual(before, (self.record.read_bytes(), self.weights.read_bytes()))

    def test_historical_result_without_weights_does_not_trigger_training(self):
        self.result().save(self.record)
        before = self.record.read_bytes()
        self.run_model(require_predictions=True)
        self.fit.assert_not_called()
        self.assertEqual(self.record.read_bytes(), before)
        self.assertFalse(self.weights.exists())

    def test_existing_weights_block_training_without_a_reusable_record(self):
        self.weights.write_bytes(b"preserved previous artifact")
        with self.assertRaises(FileExistsError):
            self.run_model()
        self.fit.assert_not_called()
        self.assertEqual(self.weights.read_bytes(), b"preserved previous artifact")

    def test_failed_weight_write_keeps_checkpoint_without_publishing_result(self):
        with patch.object(train.torch, "save", side_effect=OSError("simulated disk error")):
            with self.assertRaises(OSError):
                self.run_model()
        self.assertTrue(self.checkpoint.exists())
        self.assertFalse(self.record.exists())
        self.assertFalse(self.weights.exists())
        self.assertFalse(list(self.folder.glob("*.partial")))

    def test_failed_result_publication_keeps_weights_and_checkpoint(self):
        original_link = train.os.link

        def fail_result(source, target):
            if Path(target) == self.record:
                raise OSError("simulated publication error")
            return original_link(source, target)

        with patch.object(train.os, "link", side_effect=fail_result):
            with self.assertRaises(OSError):
                self.run_model()
        self.assertTrue(self.checkpoint.exists())
        self.assertTrue(self.weights.exists())
        self.assertFalse(self.record.exists())
        self.assertFalse(list(self.folder.glob("*.partial")))

    def test_score_only_record_never_retrains_for_missing_predictions(self):
        result = self.result()
        result.test_prediction = result.test_truth = None
        result.save(self.record)
        before = self.record.read_bytes()
        self.assertIsNone(self.run_model().test_prediction)
        with self.assertRaisesRegex(ValueError, "completed result has no per-row"):
            self.run_model(require_predictions=True)
        self.fit.assert_not_called()
        self.assertEqual(self.record.read_bytes(), before)
        self.assertFalse(self.weights.exists())

    def test_concurrent_result_is_preserved_on_training_publication(self):
        original_fit = self.completed_fit
        concurrent = b'{"another_writer": "completed evidence"}'

        def another_writer(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            self.record.write_bytes(concurrent)
            return result

        self.fit.side_effect = another_writer
        with self.assertRaises(FileExistsError):
            self.run_model()
        self.assertEqual(self.record.read_bytes(), concurrent)
        self.assertTrue(self.checkpoint.exists())
        self.assertTrue(self.weights.exists())
        self.assertFalse(list(self.folder.glob("*.partial")))

    def test_failed_replacement_preserves_existing_json(self):
        self.record.write_bytes(b'{"historical": true}')
        with patch.object(train.os, "replace", side_effect=OSError("simulated publication error")):
            with self.assertRaises(OSError):
                self.result().save(self.record)
        self.assertEqual(self.record.read_bytes(), b'{"historical": true}')

    def test_weight_publication_never_overwrites_an_existing_destination(self):
        self.weights.write_bytes(b"another completed artifact")
        with self.assertRaises(FileExistsError):
            train._save_weights(self.weights, self.model, self.result())
        self.assertEqual(self.weights.read_bytes(), b"another completed artifact")
        self.assertFalse(list(self.folder.glob("*.partial")))


if __name__ == "__main__":
    unittest.main()
