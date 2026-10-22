"""Check final artifact publication with a fixed CPU model and no optimization."""
import hashlib
import json
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
        with patch.object(train.os, "replace", side_effect=OSError("simulated publication error")):
            with self.assertRaises(OSError):
                self.run_model()
        self.assertTrue(self.checkpoint.exists())
        self.assertTrue(self.weights.exists())
        self.assertFalse(self.record.exists())
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
