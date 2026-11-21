"""Tiny CPU checks of row slicing within an unchanged logical optimizer batch.

No research inputs, notebooks, pretrained encoders or existing run artifacts are used.
Dropout is disabled: these checks establish gradient weighting and row handling,
not identical random-number consumption in a stochastic GPU model.
"""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion import region_experiment, train  # noqa: E402


RegionBatch = region_experiment.RegionBatch


def region_batch(task="regression", rows=range(8)):
    """Invent distinct rows whose padding and final logical batch are unequal."""
    rows = tuple(rows)
    all_lengths = (7, 3, 1, 5, 2, 6, 4, 1)
    lengths = torch.tensor([all_lengths[i] for i in rows], dtype=torch.long)
    longest = int(lengths.max())
    rna = torch.zeros(len(rows), longest, 2, dtype=torch.float64)
    protein = torch.zeros(len(rows), longest, 1, dtype=torch.float64)
    valid = torch.arange(longest)[None, :] < lengths[:, None]
    region = torch.zeros_like(valid)
    labels = []
    for local, source in enumerate(rows):
        length = int(lengths[local])
        positions = torch.arange(length, dtype=torch.float64)
        rna[local, :length, 0] = (source + 1) / 9 + positions / 19
        rna[local, :length, 1] = (source % 3 - 1) / 4 - positions / 23
        protein[local, :length, 0] = (source % 5) / 7 + positions / 31
        region[local, 1:length:2] = True
        labels.append((source * 3 % 7 - 2) / 5)
    labels = torch.tensor(labels, dtype=torch.float64)
    if task == "classification":
        labels = torch.tensor([i % 3 for i in rows], dtype=torch.long)
    elif task == "expression":
        labels = torch.stack((labels, labels.flip(0) * 0.7 + 0.2), dim=1)
    return RegionBatch(
        streams={"rna": rna, "prot": protein},
        masks={"rna": valid.clone(), "prot": valid.clone()},
        region=region, lengths=lengths, labels=labels,
        row_ids=tuple(f"row-{i}" for i in rows),
        positions=tuple(tuple(range(all_lengths[i])) for i in rows),
    )


class LogicalBatches:
    """Five rows, then three: slicing by two leaves odd pieces in both batches."""

    def __init__(self, task="regression"):
        self.batches = (region_batch(task, range(5)), region_batch(task, range(5, 8)))
        self.dataset = range(8)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class TinyMaskedModel(nn.Module):
    fusion_name = "tiny-sliced-regression"

    def __init__(self, task="regression"):
        super().__init__()
        outputs = {"regression": 1, "expression": 2, "classification": 3}[task]
        self.predictor = nn.Linear(3, outputs, dtype=torch.float64)
        self.gate = nn.Linear(3, 3, dtype=torch.float64)

    def forward(self, batch):
        summaries = []
        for key in ("rna", "prot"):
            mask = batch.masks[key].unsqueeze(-1)
            summaries.append((batch.streams[key] * mask).sum(1) / mask.sum(1))
        features = torch.cat(summaries, dim=1)
        return self.predictor(features), {"alpha": self.gate(features).softmax(dim=-1)}


def tiny_model(task="regression"):
    torch.manual_seed(41)
    return TinyMaskedModel(task)


def strict_identity(config, slice_size):
    return {
        "schema": "fusion.run.v1", "benchmark": "synthetic-region-slices",
        "variant": TinyMaskedModel.fusion_name, "config": asdict(config),
        "inputs": {"encoders": {}}, "model": {"contract": "synthetic-slice-fixture-v1"},
        "loader": {
            "kind": "custom", "batch_slice_size": slice_size,
            "recovery": {name: {} for name in ("train", "val", "test")},
        },
    }


class RegionBatchSliceTests(unittest.TestCase):
    def test_slices_preserve_every_row_field_and_crop_only_padding(self):
        batch = region_batch()
        before = deepcopy(batch)
        pieces = list(batch.slices(3))
        self.assertEqual([len(piece.labels) for piece in pieces], [3, 3, 2])
        self.assertEqual([piece.streams["rna"].shape[1] for piece in pieces], [7, 6, 4])
        self.assertEqual(tuple(i for piece in pieces for i in piece.row_ids), batch.row_ids)
        self.assertEqual(tuple(i for piece in pieces for i in piece.positions), batch.positions)
        start = 0
        for piece in pieces:
            stop = start + len(piece.labels)
            longest = int(piece.lengths.max())
            self.assertTrue(torch.equal(piece.labels, batch.labels[start:stop]))
            self.assertTrue(torch.equal(piece.lengths, batch.lengths[start:stop]))
            self.assertTrue(torch.equal(piece.region, batch.region[start:stop, :longest]))
            for key in batch.streams:
                self.assertEqual(piece.streams[key].device.type, "cpu")
                self.assertTrue(torch.equal(piece.streams[key], batch.streams[key][start:stop, :longest]))
                self.assertTrue(torch.equal(piece.masks[key], batch.masks[key][start:stop, :longest]))
            start = stop
        for key in batch.streams:
            self.assertTrue(torch.equal(batch.streams[key], before.streams[key]))
            self.assertTrue(torch.equal(batch.masks[key], before.masks[key]))
        self.assertTrue(torch.equal(batch.region, before.region))
        self.assertTrue(torch.equal(batch.labels, before.labels))
        self.assertTrue(torch.equal(batch.lengths, before.lengths))

    def test_slice_size_must_be_a_positive_integer(self):
        for invalid in (0, -1, True, 1.5, "2"):
            with self.subTest(size=invalid), self.assertRaises((ValueError, TypeError)):
                list(region_batch().slices(invalid))


class TrainerBatchSliceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.config = train.TrainConfig(
            lr=0.02, weight_decay=0.03, batch_size=5, max_epochs=1,
            entropy_lambda=0.2, seed=23,
        )

    def fit_and_record_updates(self, task, slice_size):
        model = tiny_model(task)
        updates = []
        original_adam = torch.optim.Adam

        class RecordingAdam(original_adam):
            def step(self, closure=None):
                # Capture all accumulated gradients before exactly one logical update.
                updates.append([
                    parameter.grad.detach().clone()
                    for group in self.param_groups for parameter in group["params"]
                ])
                return super().step(closure)

        streams = SimpleNamespace(key="synthetic-region-slices", task=task)
        loaders = {name: LogicalBatches(task) for name in ("train", "val", "test")}
        with patch.object(torch.optim, "Adam", RecordingAdam):
            result = train.fit(
                model, streams, self.config, "cpu", log=None, loaders=loaders,
                batch_slice_size=slice_size,
                run_identity=strict_identity(self.config, slice_size),
            )
        return model, result, updates

    def test_unequal_slices_match_full_batch_gradients_updates_and_losses(self):
        dense, dense_result, dense_updates = self.fit_and_record_updates("regression", None)
        sliced, sliced_result, sliced_updates = self.fit_and_record_updates("regression", 2)
        self.assertEqual(len(dense_updates), 2)
        self.assertEqual(len(sliced_updates), 2)
        for dense_step, sliced_step in zip(dense_updates, sliced_updates):
            for whole, parts in zip(dense_step, sliced_step):
                torch.testing.assert_close(whole, parts, rtol=1e-10, atol=1e-12)
        for name, parameter in dense.state_dict().items():
            torch.testing.assert_close(parameter, sliced.state_dict()[name],
                                       rtol=1e-10, atol=1e-12)
        self.assertAlmostEqual(dense_result.history[0]["train_loss"],
                               sliced_result.history[0]["train_loss"], places=12)
        np.testing.assert_allclose(dense_result.test_prediction,
                                   sliced_result.test_prediction, rtol=1e-10, atol=1e-12)

    def test_other_tasks_reject_slicing_before_forward(self):
        for task in ("classification", "expression"):
            with self.subTest(task=task):
                model = tiny_model(task)
                streams = SimpleNamespace(key="synthetic-region-slices", task=task)
                with patch.object(model, "forward", side_effect=AssertionError("forward must not run")):
                    with self.assertRaisesRegex(ValueError, "scalar regression"):
                        train.fit(
                            model, streams, self.config, "cpu", log=None,
                            loaders={name: LogicalBatches(task) for name in ("train", "val", "test")},
                            run_identity=strict_identity(self.config, 2),
                            batch_slice_size=2,
                        )

    def test_predict_preserves_row_order_and_weights_rows_not_slice_means(self):
        model = tiny_model()
        batch = region_batch()
        with torch.no_grad():
            expected_prediction, expected_aux = model(batch)
        moved_rows = []
        original_to = RegionBatch.to

        def checked_to(piece, device):
            moved_rows.append(len(piece.labels))
            self.assertLessEqual(len(piece.labels), 2)
            return original_to(piece, device)

        with patch.object(RegionBatch, "to", checked_to):
            truth, prediction, alpha = train.predict(
                model, LogicalBatches(), "cpu", "regression", batch_slice_size=2,
            )
        self.assertEqual(moved_rows, [2, 2, 1, 2, 1])
        np.testing.assert_array_equal(truth, batch.labels.numpy())
        np.testing.assert_allclose(prediction, expected_prediction.squeeze(-1).numpy(),
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(alpha, expected_aux["alpha"].mean(0).numpy(),
                                   rtol=1e-12, atol=1e-12)

    def test_training_moves_slices_instead_of_the_full_logical_batch(self):
        original_to = RegionBatch.to
        moved_rows = []

        def checked_to(piece, device):
            moved_rows.append(len(piece.labels))
            self.assertLessEqual(len(piece.labels), 2)
            return original_to(piece, device)

        with patch.object(RegionBatch, "to", checked_to):
            self.fit_and_record_updates("regression", 2)
        # Training, validation and final test all respect the same slice bound.
        self.assertEqual(moved_rows, [2, 2, 1, 2, 1] * 3)

    def test_strict_fit_rejects_slice_identity_mismatch_before_forward(self):
        streams = SimpleNamespace(key="synthetic-region-slices", task="regression")
        for recorded, requested in ((None, 2), (2, None), (4, 2)):
            model = tiny_model()
            with self.subTest(recorded=recorded, requested=requested):
                with patch.object(model, "forward", side_effect=AssertionError("forward must not run")):
                    with self.assertRaisesRegex(ValueError, "slice"):
                        train.fit(
                            model, streams, self.config, "cpu", log=None,
                            loaders={name: LogicalBatches() for name in ("train", "val", "test")},
                            run_identity=strict_identity(self.config, recorded),
                            batch_slice_size=requested,
                        )

    def test_strict_run_rejects_slice_mismatch_before_model_or_output_creation(self):
        streams = SimpleNamespace(key="synthetic-region-slices", task="regression")
        for recorded, requested in ((None, 2), (2, None), (4, 2)):
            with self.subTest(recorded=recorded, requested=requested), tempfile.TemporaryDirectory() as folder:
                destination = Path(folder) / "not-created"
                builder = Mock(side_effect=AssertionError("builder must not run"))
                with self.assertRaisesRegex(ValueError, "slice"):
                    train.run(
                        streams, TinyMaskedModel.fusion_name, self.config, "cpu", destination,
                        log=None, builder=builder, batch_slice_size=requested,
                        run_identity=strict_identity(self.config, recorded),
                    )
                builder.assert_not_called()
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
