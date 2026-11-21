"""Check the matched IsoFormer fusion head on invented tensors."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np  # Load NumPy before torch for the Windows MKL runtime.
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion.codon_grid import CodonGrid  # noqa: E402
from shared_code.fusion.heads import IsoFormerSequentialFusion  # noqa: E402
from shared_code.fusion.textcnn import TextCNN  # noqa: E402


class SequentialFusionTests(unittest.TestCase):
    def test_protein_reads_the_rna_updated_dna_query(self):
        model = IsoFormerSequentialFusion({"dna": 1, "rna": 1, "prot": 1},
                                          out_width=1, n_heads=1).double()
        values = {"dna": (0.2, -0.2), "rna": (0.0, 1.0), "prot": (-1.0, 1.0)}
        streams = {name: torch.tensor(tokens, dtype=torch.double).reshape(1, 2, 1)
                   for name, tokens in values.items()}
        mask = torch.ones(1, 2, dtype=torch.bool)

        # Identity linear maps reduce the block to scalar dot products and a residual.
        with torch.no_grad():
            for projection in model.project.values():
                projection[0].weight.fill_(1)
                projection[0].bias.zero_()
            for attention in (model.read_rna, model.read_protein):
                for linear in (attention.query, attention.key, attention.value, attention.dense):
                    linear.weight.fill_(1)
                    linear.bias.zero_()
            output, auxiliary = model(streams, mask)

        def weighted_context(query, context):
            weights = [math.exp(query * token) for token in context]
            return sum(weight * token for weight, token in zip(weights, context)) / sum(weights)

        dna, rna, protein = (tuple(map(math.tanh, values[name])) for name in values)
        after_rna = [query + weighted_context(query, rna) for query in dna]
        expected = [query + weighted_context(query, protein) for query in after_rna]
        parallel = [updated + weighted_context(original, protein)
                    for updated, original in zip(after_rna, dna)]
        torch.testing.assert_close(output.flatten(), torch.tensor(expected, dtype=torch.double))
        self.assertGreater(max(abs(sequence_score - parallel_score)
                               for sequence_score, parallel_score in zip(expected, parallel)),
                           0.1)
        self.assertEqual(auxiliary, {})

    def test_padding_does_not_change_real_tokens_or_gradients(self):
        widths = {"dna": 3, "rna": 2, "prot": 5}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = IsoFormerSequentialFusion(widths, out_width=8, n_heads=2)
            mask = torch.tensor([[True, True, False, False, False], [True] * 5])
            streams = {
                name: torch.randn(2, 5, width).masked_fill(~mask.unsqueeze(-1), float("nan"))
                .requires_grad_()
                for name, width in widths.items()
            }

        output, _ = model(streams, mask)
        with torch.no_grad():
            for row, length in enumerate((2, 5)):
                alone, _ = model({name: value[row:row + 1, :length]
                                  for name, value in streams.items()}, mask[row:row + 1, :length])
                torch.testing.assert_close(output[row:row + 1, :length], alone)
        torch.testing.assert_close(output[~mask], torch.zeros_like(output[~mask]))

        output.square().sum().backward()
        for name, value in streams.items():
            with self.subTest(modality=name):
                self.assertTrue(torch.isfinite(value.grad).all())
                torch.testing.assert_close(value.grad[~mask], torch.zeros_like(value.grad[~mask]))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_codon_grid_and_textcnn_keep_the_expected_shapes(self):
        widths = {"dna": 3, "rna": 2, "prot": 5}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            grid = CodonGrid(widths["dna"], tconv="equation")
            fusion = IsoFormerSequentialFusion(widths, out_width=8, n_heads=2)
            predictor = TextCNN(fusion.out_width, n_outputs=3, channels=4, dropout=0.0)
            streams, mask = grid(torch.randn(2, 3, widths["dna"]),
                                 torch.randn(2, 18, widths["rna"]),
                                 torch.randn(2, 6, widths["prot"]),
                                 torch.tensor([4, 6]), torch.tensor([2, 3]))

        fused, _ = fusion(streams, mask)
        prediction = predictor(fused, mask)
        self.assertEqual(fused.shape, (2, 6, 8))
        self.assertEqual(prediction.shape, (2, 3))
        self.assertTrue(torch.isfinite(prediction).all())

        prediction.square().sum().backward()
        for block in (grid, fusion.read_rna, fusion.read_protein, predictor):
            gradients = [p.grad for p in block.parameters()]
            self.assertTrue(all(grad is not None and torch.isfinite(grad).all()
                                for grad in gradients))
            self.assertTrue(any(grad.abs().sum() > 0 for grad in gradients))


if __name__ == "__main__":
    unittest.main()
