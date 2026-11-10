"""In-memory adapters for IsoFormer's future pretrained encoder integration.

The caller supplies already constructed tokenizers and torch modules. This module
does not load configurations, weights, vocabularies or remote code. A working call
with a substitute module does not verify a pretrained checkpoint or its revision.

The released wrapper requests Enformer embeddings and the final NT/ESM hidden state:
https://huggingface.co/InstaDeepAI/isoformer/blob/afac2a386c8f2241faf8fad3c5971a9307c13775/modeling_isoformer.py
Its tokenizer pads RNA/protein to 1,024 positions without truncation:
https://huggingface.co/InstaDeepAI/isoformer/blob/afac2a386c8f2241faf8fad3c5971a9307c13775/isoformer_tokenizer.py
Enformer's base-index convention and embedding API are defined in its data/forward code:
https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/data.py
https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/modeling_enformer.py
The installed dependency, vocabulary IDs and actual output crop still require a
real-asset check against the selected revisions before training.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import torch
from torch import nn


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an explicit positive integer")


def _sequences(values):
    if isinstance(values, str):
        raise TypeError("Supply a collection of sequences, not one string")
    values = list(values)
    if not values or any(not isinstance(s, str) or not s or any(c.isspace() for c in s)
                         for s in values):
        raise ValueError("Sequences must be nonempty strings without whitespace")
    return values


def _ids(value, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.long
            or value.ndim != 2 or min(value.shape) < 1):
        raise ValueError(f"{name} must be a nonempty long tensor (batch, positions)")
    if bool((value < 0).any()):
        raise ValueError(f"{name} must not contain negative IDs")


def _output(value, expected, device, name):
    if (not isinstance(value, torch.Tensor) or not value.is_floating_point()
            or tuple(value.shape) != tuple(expected)):
        raise ValueError(f"{name} must return floating embeddings of shape {tuple(expected)}")
    if value.device != device:
        raise ValueError(f"{name} output moved devices unexpectedly")
    return value


def dna_base_ids(sequences, *, expected_length):
    """Encode oriented A/C/G/T/N windows for Enformer, without adding a CLS token.

    The release removes its tokenizer's leading CLS before calling Enformer. Here
    the interface starts at that post-CLS boundary: 0/1/2/3 are A/C/G/T and 4 is N,
    which Enformer maps to an all-zero nucleotide vector. No boundary padding,
    uppercasing, U conversion or unsupported-symbol substitution is performed.
    """
    _positive(expected_length, "expected_length")
    sequences = _sequences(sequences)
    if any(len(s) != expected_length for s in sequences):
        raise ValueError("Every DNA window must have the declared length; no padding/crop is applied")
    alphabet = {base: index for index, base in enumerate("ACGTN")}
    if any(set(s).difference(alphabet) for s in sequences):
        raise ValueError("DNA windows must use uppercase A/C/G/T/N; resolve other symbols explicitly")
    return torch.tensor([[alphabet[base] for base in s] for s in sequences], dtype=torch.long)


def tokenize_sequence_batch(tokenizer, sequences, *, modality, padding_length):
    """Tokenize supplied RNA/proteins without extra cropping or batch-dependent padding.

    RNA must already have its nucleotide suffix policy applied and use the supplied
    DNA alphabet. The tokenizer must already use the verified release vocabulary,
    CLS and padding IDs, no RNA EOS, and a protein EOS. Missing proteins are omitted
    by the caller, retaining present rows in their original order.

    Return the keyword inputs and separate boolean mask expected by LiveIsoFormer.
    Unequal per-example lengths are refused: padding such protein rows to the longest
    member would change the released unmasked fusion. A full training collator is not
    implemented here, and no sequence is silently shortened to make a batch fit.
    """
    if modality not in ("rna", "protein"):
        raise ValueError("modality must be 'rna' or 'protein'")
    _positive(padding_length, "padding_length")
    sequences = _sequences(sequences)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0
           for v in (cls_id, pad_id)):
        raise ValueError("The supplied tokenizer must declare its CLS and padding IDs")
    if cls_id == pad_id:
        raise ValueError("CLS and padding must have different IDs")
    if modality == "rna" and eos_id is not None:
        raise ValueError("The released RNA tokenizer has no EOS token; configure it explicitly")
    if modality == "protein" and (isinstance(eos_id, bool)
                                  or not isinstance(eos_id, Integral)
                                  or eos_id < 0 or eos_id in (cls_id, pad_id)):
        raise ValueError("The protein tokenizer must declare a distinct EOS token")

    token_rows, mask_rows = [], []
    for sequence in sequences:
        encoded = tokenizer(sequence, padding="max_length", max_length=padding_length,
                            truncation=False, add_special_tokens=True,
                            return_attention_mask=True)
        if not isinstance(encoded, Mapping):
            raise TypeError("Tokenizer must return an input_ids/attention_mask mapping")
        ids, mask = encoded.get("input_ids"), encoded.get("attention_mask")
        if (not isinstance(ids, (list, tuple)) or not isinstance(mask, (list, tuple))
                or len(ids) != len(mask) or len(ids) < padding_length):
            raise ValueError("Tokenizer must preserve a flat sequence with its declared padding floor")
        if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 0 for v in ids):
            raise ValueError("Tokenizer IDs must be nonnegative integers")
        if any(not isinstance(v, (Integral, bool)) or v not in (0, 1) for v in mask):
            raise ValueError("Tokenizer attention mask must contain only zero/one values")
        real_count = sum(mask)
        minimum_tokens = 3 if modality == "protein" else 2  # CLS, content, optional EOS.
        if (real_count < minimum_tokens
                or list(mask) != [1] * real_count + [0] * (len(mask) - real_count)
                or ids[0] != cls_id or any((i == pad_id) != (not m) for i, m in zip(ids, mask))):
            raise ValueError("Tokenizer must retain CLS/content followed by right padding with matching IDs")
        if modality == "protein" and ids[real_count - 1] != eos_id:
            raise ValueError("Protein EOS must be the last real token")
        token_rows.append(list(ids))
        mask_rows.append(list(mask))
    if len({len(ids) for ids in token_rows}) != 1:
        raise ValueError("Unequal per-example padded lengths: a fidelity-preserving training "
                         "collator is still required; no extra batch padding was applied")
    return {"input_ids": torch.tensor(token_rows, dtype=torch.long)}, torch.tensor(
        mask_rows, dtype=torch.bool)


class EnformerAdapter(nn.Module):
    """Request the backbone's existing cropped embeddings without changing its crop."""

    def __init__(self, backbone, *, input_length, output_length, width=3072):
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an already constructed torch module")
        for value, name in ((input_length, "input_length"), (output_length, "output_length"),
                            (width, "width")):
            _positive(value, name)
        self.backbone = backbone
        self.input_length = input_length
        self.output_length = output_length
        self.width = width

    def forward(self, base_ids):
        _ids(base_ids, "base_ids")
        if base_ids.shape[1] != self.input_length or bool((base_ids > 4).any()):
            raise ValueError("Enformer needs the declared post-CLS length and A/C/G/T/N IDs 0..4")
        # No target_length override: preserve the backbone's configured output crop.
        output = self.backbone(base_ids, return_only_embeddings=True)
        return _output(output, (base_ids.shape[0], self.output_length, self.width),
                       base_ids.device, "Enformer")


class TokenEncoderAdapter(nn.Module):
    """Read the final NT/ESM hidden state without freezing or detaching the backbone.

    Pass a backbone supporting the released masked-LM call contract. Its checkpoint,
    configuration (including dropout) and tokenizer compatibility remain the caller's
    responsibility; sharing feature widths is not sufficient evidence of equivalence.
    """

    def __init__(self, backbone, *, width):
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an already constructed torch module")
        _positive(width, "width")
        self.backbone = backbone
        self.width = width

    def forward(self, input_ids, *, attention_mask):
        _ids(input_ids, "input_ids")
        if (not isinstance(attention_mask, torch.Tensor) or attention_mask.dtype != torch.bool
                or attention_mask.shape != input_ids.shape or attention_mask.device != input_ids.device
                or not bool(attention_mask.any(dim=1).all())):
            raise ValueError("attention_mask must be boolean, aligned, colocated and nonempty per row")
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               encoder_attention_mask=attention_mask,
                               output_hidden_states=True, return_dict=True)
        if not isinstance(output, Mapping):
            raise TypeError("Token backbone must return a mapping containing hidden_states")
        hidden = output.get("hidden_states")
        if not isinstance(hidden, (tuple, list)) or not hidden:
            raise ValueError("Token backbone did not return requested hidden_states")
        return _output(hidden[-1], (*input_ids.shape, self.width), input_ids.device,
                       "NT/ESM")
