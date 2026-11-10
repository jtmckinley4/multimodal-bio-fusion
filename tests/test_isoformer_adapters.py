"""Tiny, offline checks of adapter calls; no pretrained assets or training."""

from pathlib import Path
import sys
import unittest

import numpy as np  # Import before torch on Windows; no unsafe OpenMP override.
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Code"))
from shared_code.fusion.isoformer_adapters import (
    EnformerAdapter, TokenEncoderAdapter, dna_base_ids, tokenize_sequence_batch,
)
from shared_code.fusion.isoformer_reference import (
    LiveIsoFormer, ReferenceDimensions, ReferenceFusion,
)


class SmallTokenizer:
    """Deliberately simple tokenizer exposing every invocation and retained token."""

    cls_token_id = 0
    pad_token_id = 1

    def __init__(self, *, protein=False):
        self.eos_token_id = 2 if protein else None
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = [0] + [3 + ord(c) % 7 for c in text]
        if self.eos_token_id is not None:
            ids.append(self.eos_token_id)
        padding = max(0, kwargs["max_length"] - len(ids))
        return {"input_ids": ids + [1] * padding,
                "attention_mask": [1] * len(ids) + [0] * padding}


class SmallEnformer(nn.Module):
    def __init__(self, positions=2, width=4):
        super().__init__()
        self.embeddings = nn.Embedding(5, width)
        self.positions = positions
        self.calls = []

    def forward(self, base_ids, **kwargs):
        self.calls.append((base_ids.clone(), kwargs))
        return self.embeddings(base_ids[:, :self.positions])


class SmallTokenModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.embeddings = nn.Embedding(12, width)
        self.calls = []
        self.output = None

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        value = self.embeddings(kwargs["input_ids"])
        self.output = {"hidden_states": (torch.zeros_like(value), value),
                       "logits": value + 999}
        return self.output


class DNAAdapterTests(unittest.TestCase):
    def test_base_mapping_has_no_special_token_and_preserves_N(self):
        value = dna_base_ids(["ACGTN", "NTGCA"], expected_length=5)
        self.assertEqual(value.tolist(), [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        self.assertEqual(value.dtype, torch.long)

    def test_no_silent_normalization_crop_or_padding(self):
        for values in (["acgt"], ["ACGU"], ["ACG."], ["ACG"], ["ACGTT"], ["AC G"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                dna_base_ids(values, expected_length=4)
        with self.assertRaises(TypeError):
            dna_base_ids("ACGT", expected_length=4)

    def test_embedding_request_preserves_backbone_state_and_gradients(self):
        base = SmallEnformer()
        base.eval()
        adapter = EnformerAdapter(base, input_length=5, output_length=2, width=4)
        self.assertIs(adapter.backbone, base)
        self.assertFalse(base.training)
        self.assertEqual(base.calls, [])
        ids = dna_base_ids(["ACGTN"], expected_length=5)
        value = adapter(ids)
        self.assertEqual(base.calls[0][1], {"return_only_embeddings": True})
        self.assertTrue(torch.equal(base.calls[0][0], ids))
        value.sum().backward()
        self.assertIsNotNone(base.embeddings.weight.grad)

    def test_shape_and_base_id_checks_fail_without_correction(self):
        adapter = EnformerAdapter(SmallEnformer(), input_length=5, output_length=2, width=4)
        for ids in (torch.zeros(1, 6, dtype=torch.long),
                    torch.full((1, 5), 5, dtype=torch.long),
                    torch.zeros(1, 5), torch.full((1, 5), -1, dtype=torch.long)):
            with self.subTest(shape=ids.shape), self.assertRaises(ValueError):
                adapter(ids)
        wrong_crop = EnformerAdapter(SmallEnformer(), input_length=5, output_length=3, width=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            wrong_crop(torch.zeros(1, 5, dtype=torch.long))


class TokenPreparationTests(unittest.TestCase):
    def test_explicit_nontruncating_call_and_separate_mask(self):
        tokenizer = SmallTokenizer()
        inputs, mask = tokenize_sequence_batch(tokenizer, ["AC", "A"], modality="rna",
                                              padding_length=4)
        self.assertEqual(tokenizer.calls[0], ("AC", {
            "padding": "max_length", "max_length": 4, "truncation": False,
            "add_special_tokens": True, "return_attention_mask": True,
        }))
        self.assertEqual(inputs["input_ids"].shape, (2, 4))
        self.assertEqual(mask.tolist(), [[True, True, True, False],
                                        [True, True, False, False]])
        self.assertEqual(set(inputs), {"input_ids"})

    def test_long_input_and_unsupported_symbol_are_not_silently_changed(self):
        tokenizer = SmallTokenizer(protein=True)
        sequence = "M*" * 513
        inputs, mask = tokenize_sequence_batch(tokenizer, [sequence], modality="protein",
                                              padding_length=1024)
        self.assertEqual(inputs["input_ids"].shape, (1, len(sequence) + 2))
        self.assertTrue(mask.all())
        self.assertEqual(tokenizer.calls[0][0], sequence)
        self.assertEqual(inputs["input_ids"][0, -1].item(), tokenizer.eos_token_id)

    def test_extra_batch_padding_is_refused(self):
        tokenizer = SmallTokenizer(protein=True)
        with self.assertRaisesRegex(ValueError, "Unequal per-example padded lengths"):
            tokenize_sequence_batch(tokenizer, ["A", "AAAA"], modality="protein",
                                    padding_length=4)

    def test_equal_long_sequences_can_be_batched_without_new_padding(self):
        inputs, mask = tokenize_sequence_batch(SmallTokenizer(protein=True), ["AAA", "MMM"],
                                              modality="protein", padding_length=4)
        self.assertEqual(inputs["input_ids"].shape, (2, 5))
        self.assertTrue(mask.all())

    def test_special_token_policy_is_required_not_mutated(self):
        rna = SmallTokenizer(protein=True)
        protein = SmallTokenizer()
        for tokenizer, modality in ((rna, "rna"), (protein, "protein")):
            with self.subTest(modality=modality), self.assertRaises(ValueError):
                tokenize_sequence_batch(tokenizer, ["AC"], modality=modality, padding_length=4)
            self.assertEqual(tokenizer.calls, [])
        self.assertEqual(rna.eos_token_id, 2)
        self.assertIsNone(protein.eos_token_id)

    def test_supplied_protein_cannot_tokenize_to_special_tokens_only(self):
        class NoContent(SmallTokenizer):
            def __call__(self, text, **kwargs):
                return {"input_ids": [self.cls_token_id, self.eos_token_id, 1, 1],
                        "attention_mask": [1, 1, 0, 0]}

        with self.assertRaisesRegex(ValueError, "retain CLS/content"):
            tokenize_sequence_batch(NoContent(protein=True), ["M"], modality="protein",
                                    padding_length=4)

    def test_bad_masks_or_ids_are_not_accepted_as_valid_tokenization(self):
        class Malformed(SmallTokenizer):
            def __init__(self, mutation):
                super().__init__()
                self.mutation = mutation

            def __call__(self, text, **kwargs):
                result = super().__call__(text, **kwargs)
                self.mutation(result)
                return result

        mutations = [lambda r: r["attention_mask"].__setitem__(0, 0),
                     lambda r: r["input_ids"].__setitem__(3, 9),
                     lambda r: r["input_ids"].__setitem__(0, 3),
                     lambda r: r["attention_mask"].__setitem__(2, 2),
                     lambda r: r["attention_mask"].__setitem__(2, 1.0)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                tokenize_sequence_batch(Malformed(mutation), ["AC"], modality="rna",
                                        padding_length=4)


class TokenAdapterTests(unittest.TestCase):
    def test_uses_final_hidden_state_and_both_masks_without_detaching(self):
        base = SmallTokenModel()
        adapter = TokenEncoderAdapter(base, width=4)
        ids = torch.tensor([[0, 3, 1]])
        mask = torch.tensor([[True, True, False]])
        value = adapter(ids, attention_mask=mask)
        self.assertIs(value, base.output["hidden_states"][-1])
        self.assertIs(base.calls[0]["attention_mask"], mask)
        self.assertIs(base.calls[0]["encoder_attention_mask"], mask)
        self.assertTrue(base.calls[0]["output_hidden_states"])
        self.assertTrue(base.calls[0]["return_dict"])
        value.sum().backward()
        self.assertIsNotNone(base.embeddings.weight.grad)

    def test_mask_and_hidden_state_shape_failures(self):
        ids = torch.tensor([[0, 3, 1]])
        adapter = TokenEncoderAdapter(SmallTokenModel(), width=4)
        for mask in (torch.ones(1, 3), torch.zeros(1, 3, dtype=torch.bool),
                     torch.ones(1, 2, dtype=torch.bool)):
            with self.subTest(shape=mask.shape), self.assertRaises(ValueError):
                adapter(ids, attention_mask=mask)
        wrong_width = TokenEncoderAdapter(SmallTokenModel(), width=3)
        with self.assertRaisesRegex(ValueError, "shape"):
            wrong_width(ids, attention_mask=torch.ones_like(ids, dtype=torch.bool))

    def test_supplied_adapters_connect_to_fusion_and_all_backbone_gradients(self):
        dna = SmallEnformer(width=4)
        rna = SmallTokenModel(width=3)
        protein = SmallTokenModel(width=2)
        fusion = ReferenceFusion(ReferenceDimensions(4, 3, 2, 2, 2), pool_window=(0, 2),
                                 protein_mask_policy="released_unmasked", expected_dna_length=2)
        model = LiveIsoFormer({
            "dna": EnformerAdapter(dna, input_length=5, output_length=2, width=4),
            "rna": TokenEncoderAdapter(rna, width=3),
            "protein": TokenEncoderAdapter(protein, width=2),
        }, fusion)
        rna_inputs, rna_mask = tokenize_sequence_batch(SmallTokenizer(), ["AC", "CA"],
                                                     modality="rna", padding_length=4)
        protein_inputs, protein_mask = tokenize_sequence_batch(
            SmallTokenizer(protein=True), ["M", "A"], modality="protein", padding_length=4)
        result = model({"dna": {"base_ids": dna_base_ids(["ACGTN", "NTGCA"], expected_length=5)},
                        "rna": rna_inputs, "protein": protein_inputs},
                       rna_mask=rna_mask, protein_mask=protein_mask)
        self.assertEqual(result["gene_expression_predictions"].shape, (2, 2))
        result["gene_expression_predictions"].sum().backward()
        for base in (dna, rna, protein):
            self.assertIsNotNone(base.embeddings.weight.grad)
            self.assertTrue(torch.isfinite(base.embeddings.weight.grad).all())

    def test_adapter_pipeline_handles_compact_and_all_absent_proteins(self):
        dna, rna, protein = SmallEnformer(width=4), SmallTokenModel(3), SmallTokenModel(2)
        fusion = ReferenceFusion(ReferenceDimensions(4, 3, 2, 2, 2), pool_window=(0, 2),
                                 protein_mask_policy="released_unmasked", expected_dna_length=2)
        model = LiveIsoFormer({
            "dna": EnformerAdapter(dna, input_length=5, output_length=2, width=4),
            "rna": TokenEncoderAdapter(rna, width=3),
            "protein": TokenEncoderAdapter(protein, width=2),
        }, fusion)
        dna_ids = dna_base_ids(["ACGTN", "NTGCA"], expected_length=5)
        rna_inputs, rna_mask = tokenize_sequence_batch(SmallTokenizer(), ["AC", "CA"],
                                                     modality="rna", padding_length=4)
        for present in (torch.tensor([False, True]), torch.tensor([False, False])):
            with self.subTest(present=present.tolist()):
                model.zero_grad(set_to_none=True)
                protein.calls.clear()
                if present.any():
                    protein_inputs, protein_mask = tokenize_sequence_batch(
                        SmallTokenizer(protein=True), ["M"], modality="protein", padding_length=4)
                else:
                    protein_inputs, protein_mask = {}, None
                result = model({"dna": {"base_ids": dna_ids}, "rna": rna_inputs,
                                "protein": protein_inputs}, rna_mask=rna_mask,
                               protein_mask=protein_mask, protein_present=present)
                rna_updated = fusion.cross_attention_layer_rna(
                    dna.embeddings(dna_ids[:, :2]), rna.embeddings(rna_inputs["input_ids"]),
                    context_mask=rna_mask)
                self.assertTrue(torch.equal(result["final_dna_embeddings"][~present],
                                            rna_updated[~present]))
                result["gene_expression_predictions"].sum().backward()
                self.assertIsNotNone(dna.embeddings.weight.grad)
                self.assertIsNotNone(rna.embeddings.weight.grad)
                if present.any():
                    self.assertEqual(protein.calls[0]["input_ids"].shape[0], 1)
                    self.assertIsNotNone(protein.embeddings.weight.grad)
                else:
                    self.assertEqual(protein.calls, [])
                    self.assertIsNone(protein.embeddings.weight.grad)


if __name__ == "__main__":
    unittest.main()
