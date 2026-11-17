"""Synthetic tensor checks for proposed fusion interventions; no fits or research data."""

from pathlib import Path
import sys
import unittest

import numpy as np  # Import before PyTorch in the Windows project environment.
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code"))
from shared_code.fusion import intervention_ops as ops


class InterventionOperationsTests(unittest.TestCase):
    def test_masked_mean_max_have_known_answers_and_ignore_extra_padding(self):
        tokens = torch.tensor([[[2., -4.], [6., -2.], [100., 100.]]])
        mask = torch.tensor([[True, True, False]])
        torch.testing.assert_close(ops.masked_summary(tokens, mask, reduction="mean"),
                                   torch.tensor([[4., -3.]]))
        torch.testing.assert_close(ops.masked_summary(tokens, mask, reduction="max"),
                                   torch.tensor([[6., -2.]]))
        expected = torch.tensor([[4., -3., 6., -2.]])
        torch.testing.assert_close(ops.masked_summary(tokens, mask, reduction="mean_max"), expected)
        padded = torch.cat((tokens, torch.full((1, 4, 2), -1000.)), dim=1)
        padded_mask = torch.cat((mask, torch.zeros((1, 4), dtype=torch.bool)), dim=1)
        torch.testing.assert_close(ops.masked_summary(padded, padded_mask, reduction="mean_max"),
                                   expected)

    def test_summary_padding_has_zero_gradient(self):
        tokens = torch.tensor([[[1., 3.], [2., 4.], [99., 99.]]], requires_grad=True)
        ops.masked_summary(tokens, torch.tensor([[True, True, False]]),
                           reduction="mean_max").sum().backward()
        torch.testing.assert_close(tokens.grad[0, 2], torch.zeros(2))
        self.assertGreater(tokens.grad[0, :2].sum().item(), 0)

    def test_gate_collection_preserves_order_positions_and_independent_copies(self):
        first = torch.tensor([[[.1, .2], [.9, .9], [.3, .4]],
                              [[.5, .6], [.7, .8], [.9, .9]]], requires_grad=True)
        first_mask = torch.tensor([[True, False, True], [True, True, False]])
        second = torch.tensor([[[.2, .8]]])
        records = ops.collect_gate_records([
            (["row-b", "row-a"], first, first_mask),
            (["row-c"], second, torch.tensor([[True]])),
        ])
        self.assertEqual([r.row_id for r in records], ["row-b", "row-a", "row-c"])
        self.assertEqual(records[0].positions.tolist(), [0, 2])
        torch.testing.assert_close(records[0].weights, torch.tensor([[.1, .2], [.3, .4]]))
        self.assertFalse(records[0].weights.requires_grad)
        self.assertEqual(records[0].weights.device.type, "cpu")
        with torch.no_grad():
            first.fill_(0)
        torch.testing.assert_close(records[0].weights, torch.tensor([[.1, .2], [.3, .4]]))

    def test_sequence_gate_collection_does_not_average(self):
        weights = torch.tensor([[.2, .8], [.9, .1]])
        records = ops.collect_gate_records([(["a", "b"], weights, None)])
        torch.testing.assert_close(records[0].weights, weights[0])
        torch.testing.assert_close(records[1].weights, weights[1])
        self.assertIsNone(records[0].positions)

    def test_rejects_ambiguous_masks_empty_streams_and_nonfinite_values(self):
        tokens = torch.ones((1, 2, 1))
        for changed, mask in [
            (tokens, torch.zeros((1, 2), dtype=torch.bool)),
            (tokens, torch.ones((1, 2))),
            (tokens, torch.ones((1, 1), dtype=torch.bool)),
            (torch.empty((1, 0, 1)), torch.empty((1, 0), dtype=torch.bool)),
            (torch.tensor([[[float("nan")], [1.]]]), torch.tensor([[False, True]])),
        ]:
            with self.subTest(shape=changed.shape, mask=mask), self.assertRaises(ValueError):
                ops.masked_summary(changed, mask, reduction="mean")

    def test_rejects_duplicate_row_ids_and_mixed_diagnostic_layouts(self):
        sequence = torch.ones((1, 2))
        cases = [
            [],
            [(["a"], sequence, None), (["a"], sequence, None)],
            [([""], sequence, None)],
            [(["a", "b"], sequence, None)],
            [(["a"], sequence, torch.ones((1, 1), dtype=torch.bool))],
            [(["a"], sequence, None), (["b"], torch.ones((1, 3)), None)],
            [(["a"], torch.ones((1, 2, 2)), torch.zeros((1, 2), dtype=torch.bool))],
        ]
        for batches in cases:
            with self.subTest(batches=len(batches)), self.assertRaises(ValueError):
                ops.collect_gate_records(batches)


if __name__ == "__main__":
    unittest.main()
