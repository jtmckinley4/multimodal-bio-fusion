"""Synthetic CPU checks of the released-path operations; no weights or fitting.

These tests do not instantiate real backbone widths or establish checkpoint compatibility.
One backward pass verifies that injected encoder adapters remain differentiable; there is
no optimizer, model download, research input, cache reader or artifact producer.
"""

import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np  # Load NumPy before torch for this repository's Windows MKL runtime.
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))

from shared_code.fusion.isoformer_reference import (  # noqa: E402
    LiveIsoFormer, ReferenceCrossAttention, ReferenceDimensions, ReferenceFusion,
)


TINY = ReferenceDimensions(dna=4, rna=6, protein=2, n_heads=2, n_tissues=3)


def fixture(batch=2):
    generator = torch.Generator().manual_seed(31)
    streams = [torch.randn(batch, length, width, generator=generator)
               for length, width in ((5, TINY.dna), (4, TINY.rna), (3, TINY.protein))]
    masks = {"rna_mask": torch.ones(batch, 4, dtype=torch.bool),
             "protein_mask": torch.ones(batch, 3, dtype=torch.bool)}
    return streams, masks


def make_fusion(**kwargs):
    options = {"pool_window": (1, 4), "protein_mask_policy": "mask_padding",
               "expected_dna_length": 5}
    options.update(kwargs)
    return ReferenceFusion(TINY, **options)


def identity_attention(layer, *, uniform=False):
    """Choose known linear maps for hand-computed attention examples."""
    with torch.no_grad():
        for name in ("query", "key", "value", "dense"):
            linear = getattr(layer, name)
            linear.weight.copy_(torch.eye(linear.out_features, linear.in_features))
            linear.bias.zero_()
        if uniform:
            layer.query.weight.zero_()
            layer.key.weight.zero_()


class AttentionTests(unittest.TestCase):
    def test_two_heads_match_independent_hand_calculation(self):
        layer = ReferenceCrossAttention(4, 4, 2)
        identity_attention(layer)
        query = torch.tensor([[[1., 0., 0., 1.]]])
        context = torch.tensor([[[1., 0., 0., 2.], [0., 1., 2., 0.]]])
        first = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
        second = math.exp(2 / math.sqrt(2)) / (math.exp(2 / math.sqrt(2)) + 1)
        expected = torch.tensor([[[1 + first, 1 - first, 2 * (1 - second), 1 + 2 * second]]])
        torch.testing.assert_close(layer(query, context), expected, atol=1e-6, rtol=1e-6)

    def test_uniform_attention_uses_valid_tokens_and_retains_residual(self):
        layer = ReferenceCrossAttention(2, 2, 1)
        identity_attention(layer, uniform=True)
        query = torch.tensor([[[3., 4.], [5., 6.]]])
        context = torch.tensor([[[2., 6.], [100., 200.], [4., 8.]]])
        mask = torch.tensor([[True, False, True]])
        expected = query + torch.tensor([3., 7.])
        torch.testing.assert_close(layer(query, context, context_mask=mask), expected)

    def test_invalid_masks_and_contexts_fail_clearly(self):
        layer = ReferenceCrossAttention(4, 6, 2)
        dna, context = torch.ones(2, 5, 4), torch.ones(2, 3, 6)
        for mask in (torch.ones(2, 3), torch.ones(2, 1, 3, dtype=torch.bool),
                     torch.zeros(2, 3, dtype=torch.bool)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    layer(dna, context, context_mask=mask)
        with self.assertRaisesRegex(ValueError, "batch size"):
            layer(dna, context[:1])
        with self.assertRaisesRegex(ValueError, "dtype"):
            layer(dna, context.double())


class ReferenceFusionTests(unittest.TestCase):
    def test_released_dimensions_are_metadata_without_loading_a_model(self):
        dims = ReferenceDimensions()
        self.assertEqual((dims.dna, dims.rna, dims.protein, dims.n_heads, dims.n_tissues),
                         (3072, 768, 640, 8, 30))
        with self.assertRaisesRegex(ValueError, "divisible"):
            ReferenceDimensions(dna=5, n_heads=2)

    def test_unresolved_pooling_and_mask_policy_are_not_defaults(self):
        for value in (None, (1, 1), (-1, 2), (1, 6), (1.5, 3), (False, 3)):
            with self.subTest(pool_window=value):
                with self.assertRaises(ValueError):
                    make_fusion(pool_window=value)
        for value in (None, "", "auto", "center"):
            with self.subTest(protein_mask_policy=value):
                with self.assertRaises(ValueError):
                    make_fusion(protein_mask_policy=value)
        with self.assertRaises(TypeError):
            ReferenceFusion(TINY)

    def test_expected_and_actual_pooling_lengths_are_checked(self):
        streams, masks = fixture()
        with self.assertRaisesRegex(ValueError, "recheck crop"):
            make_fusion(expected_dna_length=6)(*streams, **masks)
        with self.assertRaisesRegex(ValueError, "actual DNA"):
            make_fusion(pool_window=(4, 7), expected_dna_length=None)(*streams, **masks)

    def test_pooling_uses_only_declared_half_open_window_and_softplus_head(self):
        dims = ReferenceDimensions(dna=2, rna=2, protein=2, n_heads=1, n_tissues=1)
        model = ReferenceFusion(dims, pool_window=(1, 3), protein_mask_policy="mask_padding")
        with torch.no_grad():
            for layer in (model.cross_attention_layer_rna, model.cross_attention_layer_protein):
                layer.dense.weight.zero_()
                layer.dense.bias.zero_()
            model.head_layer_1.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]]))
            model.head_layer_1.bias.zero_()
            model.head_layer_2.weight.fill_(1.)
            model.head_layer_2.bias.zero_()
        dna = torch.tensor([[[100., 100.], [1., 2.], [3., 4.], [-100., -100.]]])
        context = torch.ones(1, 2, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        result = model(dna, context, context, rna_mask=mask, protein_mask=mask)
        expected = 2 * math.log1p(math.exp(2)) + 2 * math.log1p(math.exp(3))
        self.assertAlmostEqual(result["gene_expression_predictions"].item(), expected, places=5)
        torch.testing.assert_close(result["final_dna_embeddings"], dna)

    def test_rna_updates_dna_before_protein_reads_it(self):
        class AddThenMultiply(nn.Module):
            def __init__(self, factor):
                super().__init__()
                self.factor = factor

            def forward(self, dna, context, *, context_mask=None):
                return dna * self.factor + context.mean(dim=(1, 2), keepdim=True)

        model = make_fusion()
        model.cross_attention_layer_rna = AddThenMultiply(1)
        model.cross_attention_layer_protein = AddThenMultiply(2)
        streams, masks = fixture()
        dna, rna, protein = streams
        result = model(*streams, **masks)["final_dna_embeddings"]
        expected = 2 * (dna + rna.mean(dim=(1, 2), keepdim=True)) + protein.mean(dim=(1, 2), keepdim=True)
        torch.testing.assert_close(result, expected)

    def test_rna_padding_does_not_mix_batch_rows(self):
        model = make_fusion().eval()
        streams, masks = fixture()
        masks["rna_mask"][0, 2:] = False
        streams[1][0, 2:] = 500
        batched = model(*streams, **masks)
        single = model(streams[0][:1], streams[1][:1, :2], streams[2][:1],
                       rna_mask=masks["rna_mask"][:1, :2],
                       protein_mask=masks["protein_mask"][:1])
        torch.testing.assert_close(batched["gene_expression_predictions"][:1],
                                   single["gene_expression_predictions"], atol=1e-6, rtol=1e-6)

    def test_protein_mask_choice_changes_the_released_behavior_explicitly(self):
        dims = ReferenceDimensions(dna=2, rna=2, protein=2, n_heads=1, n_tissues=1)
        unmasked = ReferenceFusion(dims, pool_window=(0, 1), protein_mask_policy="released_unmasked")
        masked = ReferenceFusion(dims, pool_window=(0, 1), protein_mask_policy="mask_padding")
        for model in (unmasked, masked):
            for layer in (model.cross_attention_layer_rna, model.cross_attention_layer_protein):
                identity_attention(layer, uniform=True)
        dna, rna = torch.zeros(1, 1, 2), torch.zeros(1, 1, 2)
        protein = torch.tensor([[[2., 4.], [100., 200.]]])
        masks = {"rna_mask": torch.tensor([[True]]), "protein_mask": torch.tensor([[True, False]])}
        actual_unmasked = unmasked(dna, rna, protein, **masks)["final_dna_embeddings"]
        actual_masked = masked(dna, rna, protein, **masks)["final_dna_embeddings"]
        torch.testing.assert_close(actual_unmasked, torch.tensor([[[51., 102.]]]))
        torch.testing.assert_close(actual_masked, torch.tensor([[[2., 4.]]]))

    def test_released_fusion_has_no_added_normalization_or_dropout(self):
        model = make_fusion()
        self.assertFalse(any(isinstance(m, (nn.LayerNorm, nn.Dropout)) for m in model.modules()))
        streams, masks = fixture()
        result = model(*streams, **masks)
        self.assertEqual(result["gene_expression_predictions"].shape, (2, TINY.n_tissues))

    def test_explicit_all_present_preserves_existing_results_exactly(self):
        streams, masks = fixture()
        for policy in ("released_unmasked", "mask_padding"):
            with self.subTest(policy=policy):
                model = make_fusion(protein_mask_policy=policy).eval()
                original = model(*streams, **masks)
                explicit = model(*streams, **masks, protein_present=torch.ones(2, dtype=torch.bool))
                for key in original:
                    self.assertTrue(torch.equal(original[key], explicit[key]))

    def test_mixed_batch_bypasses_complete_projected_update_and_preserves_order(self):
        streams, masks = fixture(batch=4)
        dna, rna, all_proteins = streams
        present = torch.tensor([False, True, False, True])
        protein = all_proteins[present]
        protein_mask = masks["protein_mask"][present]
        for policy in ("released_unmasked", "mask_padding"):
            with self.subTest(policy=policy):
                model = make_fusion(protein_mask_policy=policy).eval()
                with torch.no_grad():
                    model.cross_attention_layer_protein.dense.bias.fill_(23)
                after_rna = model.cross_attention_layer_rna(dna, rna, context_mask=masks["rna_mask"])
                actual = model(dna, rna, protein, rna_mask=masks["rna_mask"],
                               protein_mask=protein_mask, protein_present=present)
                self.assertTrue(torch.equal(actual["final_dna_embeddings"][~present],
                                            after_rna[~present]))
                for compact_row, batch_row in enumerate((1, 3)):
                    selected_mask = (protein_mask[compact_row:compact_row + 1]
                                     if policy == "mask_padding" else None)
                    expected = model.cross_attention_layer_protein(
                        after_rna[batch_row:batch_row + 1],
                        protein[compact_row:compact_row + 1], context_mask=selected_mask)
                    torch.testing.assert_close(actual["final_dna_embeddings"][batch_row],
                                               expected[0])

    def test_all_absent_skips_protein_operator_and_retains_same_head(self):
        (dna, rna, _), masks = fixture()
        model = make_fusion(protein_mask_policy="released_unmasked")
        expected = model.cross_attention_layer_rna(dna, rna, context_mask=masks["rna_mask"])
        expected_predictions = model.head_layer_2(
            torch.nn.functional.softplus(model.head_layer_1(expected[:, 1:4].mean(dim=1))))
        with patch.object(model.cross_attention_layer_protein, "forward",
                          side_effect=AssertionError("Absent branch must not execute")):
            actual = model(dna, rna, None, rna_mask=masks["rna_mask"], protein_mask=None,
                           protein_present=torch.zeros(2, dtype=torch.bool))
        self.assertTrue(torch.equal(actual["final_dna_embeddings"], expected))
        self.assertTrue(torch.equal(actual["gene_expression_predictions"], expected_predictions))

    def test_absent_row_loss_has_no_gradient_to_present_protein_branch(self):
        streams, masks = fixture()
        dna, rna = (value.clone().requires_grad_() for value in streams[:2])
        protein = streams[2][1:].clone().requires_grad_()
        model = make_fusion(protein_mask_policy="released_unmasked")
        output = model(dna, rna, protein, rna_mask=masks["rna_mask"],
                       protein_mask=masks["protein_mask"][1:],
                       protein_present=torch.tensor([False, True]))["final_dna_embeddings"]
        output[0].sum().backward(retain_graph=True)
        self.assertGreater(dna.grad[0].abs().sum().item(), 0)
        self.assertGreater(rna.grad[0].abs().sum().item(), 0)
        self.assertEqual(dna.grad[1].abs().sum().item(), 0)
        self.assertEqual(rna.grad[1].abs().sum().item(), 0)
        self.assertEqual(protein.grad.abs().sum().item(), 0)
        for parameter in model.cross_attention_layer_protein.parameters():
            self.assertEqual(parameter.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        protein.grad = None
        output[1].sum().backward()
        self.assertGreater(protein.grad.abs().sum().item(), 0)
        self.assertGreater(model.cross_attention_layer_protein.dense.bias.grad.abs().sum().item(), 0)

    def test_availability_and_compact_rows_are_required_consistently(self):
        streams, masks = fixture()
        model = make_fusion()
        for present in (torch.ones(2), torch.ones(1, dtype=torch.bool),
                        torch.ones(2, 1, dtype=torch.bool)):
            with self.subTest(present=present):
                with self.assertRaises((TypeError, ValueError)):
                    model(*streams, **masks, protein_present=present)
        with self.assertRaisesRegex(ValueError, "present rows"):
            model(*streams, **masks, protein_present=torch.tensor([False, True]))
        with self.assertRaisesRegex(ValueError, "all-absent"):
            model(*streams, **masks, protein_present=torch.zeros(2, dtype=torch.bool))
        with self.assertRaises(TypeError):
            model(streams[0], streams[1], None, rna_mask=masks["rna_mask"],
                  protein_mask=None, protein_present=torch.tensor([False, True]))


class TinyEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.embedding = nn.Embedding(11, width)
        self.last_mask = None

    def forward(self, input_ids, attention_mask=None):
        self.last_mask = attention_mask
        return self.embedding(input_ids)


class LiveAdapterTests(unittest.TestCase):
    def model_and_inputs(self):
        encoders = {name: TinyEncoder(width) for name, width in
                    (("dna", TINY.dna), ("rna", TINY.rna), ("protein", TINY.protein))}
        model = LiveIsoFormer(encoders, make_fusion(protein_mask_policy="released_unmasked"))
        inputs = {name: {"input_ids": torch.arange(length).repeat(2, 1)}
                  for name, length in (("dna", 5), ("rna", 4), ("protein", 3))}
        masks = {"rna_mask": torch.ones(2, 4, dtype=torch.bool),
                 "protein_mask": torch.tensor([[True, True, False], [True, True, True]])}
        return model, inputs, masks

    def test_gradients_reach_all_encoders_and_head_without_an_optimizer(self):
        model, inputs, masks = self.model_and_inputs()
        with patch("urllib.request.urlopen", side_effect=AssertionError("No network")), \
             patch("torch.hub.load", side_effect=AssertionError("No model loading")):
            prediction = model(inputs, **masks)["gene_expression_predictions"]
            prediction.square().sum().backward()
        missing = [name for name, parameter in model.named_parameters()
                   if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())]
        self.assertEqual(missing, [])
        for name in ("dna", "rna", "protein"):
            self.assertGreater(model.encoders[name].embedding.weight.grad.abs().sum().item(), 0)

    def test_protein_encoder_receives_mask_even_with_unmasked_protein_fusion(self):
        model, inputs, masks = self.model_and_inputs()
        model(inputs, **masks)
        self.assertIs(model.encoders["rna"].last_mask, masks["rna_mask"])
        self.assertIs(model.encoders["protein"].last_mask, masks["protein_mask"])

    def test_invalid_input_contract_stops_before_calling_encoders(self):
        model, inputs, masks = self.model_and_inputs()
        with patch.object(model.encoders["dna"], "forward", side_effect=AssertionError("Too early")):
            with self.assertRaises(ValueError):
                model({"dna": inputs["dna"]}, **masks)
            inputs["rna"]["attention_mask"] = masks["rna_mask"]
            with self.assertRaisesRegex(ValueError, "explicit"):
                model(inputs, **masks)

    def test_all_absent_does_not_call_or_update_protein_encoder(self):
        model, inputs, masks = self.model_and_inputs()
        inputs["protein"] = {}
        with patch.object(model.encoders["protein"], "forward",
                          side_effect=AssertionError("Absent protein encoder must not execute")):
            prediction = model(inputs, rna_mask=masks["rna_mask"], protein_mask=None,
                               protein_present=torch.zeros(2, dtype=torch.bool))[
                                   "gene_expression_predictions"]
            prediction.square().sum().backward()
        self.assertIsNone(model.encoders["protein"].embedding.weight.grad)
        for parameter in model.fusion.cross_attention_layer_protein.parameters():
            self.assertIsNone(parameter.grad)
        for name in ("dna", "rna"):
            self.assertGreater(model.encoders[name].embedding.weight.grad.abs().sum().item(), 0)

    def test_mixed_batch_encodes_only_present_rows_and_retains_their_predictions(self):
        model, inputs, masks = self.model_and_inputs()
        present_inputs = {name: {key: value[1:] for key, value in values.items()}
                          for name, values in inputs.items()}
        expected = model(present_inputs, rna_mask=masks["rna_mask"][1:],
                         protein_mask=masks["protein_mask"][1:])["gene_expression_predictions"]
        inputs["protein"] = present_inputs["protein"]
        with patch.object(model.encoders["protein"], "forward",
                          wraps=model.encoders["protein"].forward) as protein_forward:
            actual = model(inputs, rna_mask=masks["rna_mask"],
                           protein_mask=masks["protein_mask"][1:],
                           protein_present=torch.tensor([False, True]))["gene_expression_predictions"]
        self.assertEqual(protein_forward.call_count, 1)
        self.assertEqual(protein_forward.call_args.kwargs["input_ids"].shape[0], 1)
        torch.testing.assert_close(actual[1:], expected, atol=1e-6, rtol=1e-6)

    def test_invalid_presence_or_compact_batch_fails_before_encoding(self):
        model, inputs, masks = self.model_and_inputs()
        with patch.object(model.encoders["dna"], "forward", side_effect=AssertionError("Too early")):
            with self.assertRaisesRegex(ValueError, "present rows"):
                model(inputs, **masks, protein_present=torch.tensor([False, True]))
            with self.assertRaisesRegex(ValueError, "protein.input_ids"):
                model(inputs, rna_mask=masks["rna_mask"], protein_mask=masks["protein_mask"][1:],
                      protein_present=torch.tensor([False, True]))
            with self.assertRaisesRegex(ValueError, "all-absent"):
                model(inputs, rna_mask=masks["rna_mask"], protein_mask=None,
                      protein_present=torch.zeros(2, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
