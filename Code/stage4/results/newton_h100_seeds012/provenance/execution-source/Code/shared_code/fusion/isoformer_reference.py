"""Resolved operations from the released Enformer-based IsoFormer implementation.

Reference: InstaDeepAI/isoformer revision afac2a386c8f2241faf8fad3c5971a9307c13775,
``modeling_isoformer.py`` and ``modeling_esm.py::MultiHeadAttention``:
https://huggingface.co/InstaDeepAI/isoformer/tree/afac2a386c8f2241faf8fad3c5971a9307c13775

This is a small reimplementation of the released fusion computation, not a loader for
its checkpoint or a recovered historical trainer. DNA queries RNA and then protein;
only the updated DNA stream is pooled and read by the expression head. The released
operator has an output projection and residual, without a fusion LayerNorm, FFN or
active attention dropout. The existing frozen pilot in ``isoformer.py`` is separate.

Pooling coordinates and protein padding treatment are deliberately required choices.
The released 765:770 window is not assumed to be central in an 896-bin output. RNA
mask broadcasting is corrected for batches larger than one. Protein masking can retain
the released omission or apply a declared correction; these are numerically different.
Absent proteins use a separate, explicitly declared bypass: the RNA-updated DNA is
passed unchanged to pooling. This reconstructs the paper's absent-modality rule; it
is not present in the released all-three wrapper.
No model weights, tokenizers, files or remote code are loaded by this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import nn


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ReferenceDimensions:
    """Released feature widths; smaller dimensions support independent tensor tests."""

    dna: int = 3072
    rna: int = 768
    protein: int = 640
    n_heads: int = 8
    n_tissues: int = 30

    def __post_init__(self):
        for name in ("dna", "rna", "protein", "n_heads", "n_tissues"):
            _positive_integer(getattr(self, name), name)
        if self.dna % self.n_heads:
            raise ValueError("DNA width must be divisible by the number of fusion heads")


def _features(value, name, width):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} features must be a torch.Tensor")
    if value.ndim != 3 or min(value.shape) < 1 or value.shape[-1] != width:
        raise ValueError(f"{name} features must have shape (batch, positions, {width}); "
                         f"got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} features must have a floating-point dtype")


def _mask(value, name, shape, device):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError(f"{name} must be a boolean tensor with True for real tokens")
    if tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(value.shape)}")
    if value.device != device:
        raise ValueError(f"{name} must be on the same device as its features")
    if not bool(value.any(dim=1).all()):
        raise ValueError(f"{name} contains an empty sequence; encoded streams must "
                         "contain at least one real token")


def _protein_presence(value, batch, device):
    """Availability is metadata, not an inference from embeddings or padding."""
    if value is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
        raise TypeError("protein_present must be a boolean tensor")
    if tuple(value.shape) != (batch,):
        raise ValueError(f"protein_present must have shape ({batch},)")
    if value.device != device:
        raise ValueError("protein_present must be on the same device as the batch")
    return value


class ReferenceCrossAttention(nn.Module):
    """DNA-width queries read another stream, followed by projection and residual."""

    def __init__(self, dna_width, context_width, n_heads):
        super().__init__()
        for value, name in ((dna_width, "dna_width"), (context_width, "context_width"),
                            (n_heads, "n_heads")):
            _positive_integer(value, name)
        if dna_width % n_heads:
            raise ValueError("DNA width must be divisible by the number of fusion heads")
        self.dna_width = dna_width
        self.context_width = context_width
        self.n_heads = n_heads
        self.head_width = dna_width // n_heads
        self.query = nn.Linear(dna_width, dna_width)
        self.key = nn.Linear(context_width, dna_width)
        self.value = nn.Linear(context_width, dna_width)
        self.dense = nn.Linear(dna_width, dna_width)

    def _heads(self, value):
        batch, length, _ = value.shape
        return value.reshape(batch, length, self.n_heads, self.head_width).transpose(1, 2)

    def forward(self, dna, context, *, context_mask=None):
        _features(dna, "DNA", self.dna_width)
        _features(context, "context", self.context_width)
        if dna.shape[0] != context.shape[0]:
            raise ValueError("DNA and context must have the same batch size")
        if dna.device != context.device or dna.dtype != context.dtype:
            raise ValueError("DNA and context must have the same device and dtype")
        if context_mask is not None:
            _mask(context_mask, "context_mask", context.shape[:2], context.device)

        # The release scales queries before the dot product, with no rotary transform
        # in this fusion block. Preserve that order rather than invoking a generic block.
        queries = self._heads(self.query(dna)) * self.head_width ** -0.5
        keys = self._heads(self.key(context))
        values = self._heads(self.value(context))
        scores = queries @ keys.transpose(-1, -2)
        if context_mask is not None:
            # (B, 1, 1, context positions) broadcasts independently over heads and
            # DNA positions. The release's reshape incorrectly combines B and length.
            masked_score = max(-1e30, torch.finfo(scores.dtype).min)
            scores = scores.masked_fill(~context_mask[:, None, None, :], masked_score)
        attended = torch.softmax(scores, dim=-1) @ values
        attended = attended.transpose(1, 2).contiguous().reshape(dna.shape)
        return self.dense(attended) + dna


class ReferenceFusion(nn.Module):
    """Released fusion/head with an explicit absent-protein reconstruction.

    ``pool_window`` is a half-open interval in the *actual returned DNA embeddings*,
    not a genomic coordinate. An explicit ``expected_dna_length`` can refuse a change
    in the encoder's crop before the same indices silently acquire another meaning.
    Neither setting establishes the biological position of that window.

    ``protein_mask_policy='released_unmasked'`` omits the protein mask at fusion,
    matching the visible release. ``'mask_padding'`` uses it and is a declared numerical
    correction. RNA padding is always masked. Neither choice defines subset models.

    ``protein_present`` marks availability for all B rows. Protein features and their
    mask contain only the P present rows, in their original batch order. Missing rows
    bypass the complete projected protein update, including its bias. For P=0, pass
    ``protein=None`` and ``protein_mask=None``. Omitting availability retains the
    existing all-present interface; no missingness is inferred from zero values.
    """

    def __init__(self, dimensions, *, pool_window, protein_mask_policy,
                 expected_dna_length=None):
        super().__init__()
        if not isinstance(dimensions, ReferenceDimensions):
            raise TypeError("dimensions must be ReferenceDimensions")
        if (not isinstance(pool_window, (tuple, list)) or len(pool_window) != 2
                or any(isinstance(v, bool) or not isinstance(v, Integral)
                       for v in pool_window)
                or not 0 <= pool_window[0] < pool_window[1]):
            raise ValueError("pool_window must explicitly give nonempty [start, end) "
                             "DNA embedding positions; no central window is assumed")
        if protein_mask_policy not in ("released_unmasked", "mask_padding"):
            raise ValueError("protein_mask_policy must explicitly be 'released_unmasked' "
                             "or 'mask_padding'")
        if expected_dna_length is not None:
            _positive_integer(expected_dna_length, "expected_dna_length")
            if pool_window[1] > expected_dna_length:
                raise ValueError("pool_window exceeds expected_dna_length")

        self.dimensions = dimensions
        self.pool_window = tuple(pool_window)
        self.protein_mask_policy = protein_mask_policy
        self.expected_dna_length = expected_dna_length
        self.cross_attention_layer_rna = ReferenceCrossAttention(
            dimensions.dna, dimensions.rna, dimensions.n_heads)
        self.cross_attention_layer_protein = ReferenceCrossAttention(
            dimensions.dna, dimensions.protein, dimensions.n_heads)
        self.head_layer_1 = nn.Linear(dimensions.dna, 2 * dimensions.dna)
        self.head_layer_2 = nn.Linear(2 * dimensions.dna, dimensions.n_tissues)

    def forward(self, dna, rna, protein, *, rna_mask, protein_mask,
                protein_present=None):
        for name, value, width in (("DNA", dna, self.dimensions.dna),
                                   ("RNA", rna, self.dimensions.rna)):
            _features(value, name, width)
        if rna.shape[0] != dna.shape[0]:
            raise ValueError("DNA and RNA must have the same batch size")
        if rna.device != dna.device or rna.dtype != dna.dtype:
            raise ValueError("DNA and RNA must have the same device and dtype")
        _mask(rna_mask, "rna_mask", rna.shape[:2], rna.device)
        present = _protein_presence(protein_present, dna.shape[0], dna.device)
        present_indices = present.nonzero(as_tuple=True)[0]
        count = len(present_indices)
        if count:
            _features(protein, "protein", self.dimensions.protein)
            if protein.shape[0] != count:
                raise ValueError("protein features must contain only the present rows "
                                 "in original batch order")
            if protein.device != dna.device or protein.dtype != dna.dtype:
                raise ValueError("Protein and DNA must have the same device and dtype")
            _mask(protein_mask, "protein_mask", protein.shape[:2], protein.device)
        elif protein is not None or protein_mask is not None:
            raise ValueError("An all-absent batch requires protein=None and protein_mask=None")
        if (self.expected_dna_length is not None
                and dna.shape[1] != self.expected_dna_length):
            raise ValueError(f"Expected {self.expected_dna_length} DNA embedding positions, "
                             f"got {dna.shape[1]}; recheck crop and pooling coordinates")
        start, end = self.pool_window
        if end > dna.shape[1]:
            raise ValueError("pool_window exceeds the actual DNA embedding length")

        rna_to_dna = self.cross_attention_layer_rna(dna, rna, context_mask=rna_mask)
        fusion_protein_mask = (protein_mask if self.protein_mask_policy == "mask_padding"
                               else None)
        if count == dna.shape[0]:
            # Keep the released all-present path unchanged, including padded positions.
            final_dna = self.cross_attention_layer_protein(
                rna_to_dna, protein, context_mask=fusion_protein_mask)
        elif count:
            supplied_update = self.cross_attention_layer_protein(
                rna_to_dna.index_select(0, present_indices), protein,
                context_mask=fusion_protein_mask)
            # Update only supplied rows. Multiplying pre-projection values by zero
            # would leave the output bias; skipping the whole branch removes it too.
            final_dna = rna_to_dna.index_copy(0, present_indices, supplied_update)
        else:
            final_dna = rna_to_dna
        # Slicing the declared window avoids creating a pooling mask on the wrong
        # device. It computes the release's uniform mean over those same positions.
        pooled = final_dna[:, start:end].mean(dim=1)
        predictions = self.head_layer_2(torch.nn.functional.softplus(self.head_layer_1(pooled)))
        return {"gene_expression_predictions": predictions,
                "final_dna_embeddings": final_dna}


class LiveIsoFormer(nn.Module):
    """Connect supplied, differentiable encoder adapters to the all-three fusion.

    Each adapter is an ``nn.Module`` returning a floating tensor (B, positions, width).
    DNA receives its own keyword inputs. RNA/protein additionally receive the explicit
    ``attention_mask``. Their adapters must retain the same token positions as those
    masks. The DNA adapter owns one-hot/CLS handling and the declared Enformer crop;
    RNA/protein adapters own final hidden-state selection. This boundary does not claim
    compatibility with an actual pretrained package before those adapters are checked.

    The caller owns checkpoint loading and its provenance. This class neither loads
    weights nor freezes, detaches, or replaces caller parameters. It is not a trainer.

    Protein keyword tensors and their mask contain only present rows, in original
    batch order. ``protein_present`` still has B entries. An all-absent batch uses an
    empty protein keyword mapping and ``protein_mask=None``; its encoder is not called.
    """

    def __init__(self, encoders, fusion):
        super().__init__()
        names = ("dna", "rna", "protein")
        if not isinstance(encoders, Mapping) or set(encoders) != set(names):
            raise ValueError("encoders must contain exactly dna, rna and protein adapters")
        if not all(isinstance(encoders[name], nn.Module) for name in names):
            raise TypeError("Every encoder adapter must be an nn.Module")
        if not isinstance(fusion, ReferenceFusion):
            raise TypeError("fusion must be a ReferenceFusion")
        self.encoders = nn.ModuleDict({name: encoders[name] for name in names})
        self.fusion = fusion

    def forward(self, inputs, *, rna_mask, protein_mask, protein_present=None):
        if not isinstance(inputs, Mapping) or set(inputs) != set(self.encoders):
            raise ValueError("inputs must contain exactly dna, rna and protein keyword mappings")
        for name, values in inputs.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"{name} inputs must be a keyword mapping")
            if "attention_mask" in values:
                raise ValueError("Pass RNA/protein attention masks through the explicit "
                                 "rna_mask/protein_mask arguments")
        if not isinstance(rna_mask, torch.Tensor) or rna_mask.ndim != 2:
            raise ValueError("rna_mask must be a two-dimensional boolean tensor")
        if min(rna_mask.shape) < 1:
            raise ValueError("rna_mask must describe a nonempty batch and token sequence")
        _mask(rna_mask, "rna_mask", rna_mask.shape, rna_mask.device)
        present = _protein_presence(protein_present, rna_mask.shape[0], rna_mask.device)
        count = int(present.sum().item())
        if count:
            if not isinstance(protein_mask, torch.Tensor) or protein_mask.ndim != 2:
                raise ValueError("protein_mask must be a two-dimensional boolean tensor")
            if protein_mask.shape[0] != count:
                raise ValueError("protein_mask must contain only the present rows")
            _mask(protein_mask, "protein_mask", protein_mask.shape, rna_mask.device)
        elif protein_mask is not None or inputs["protein"]:
            raise ValueError("An all-absent batch requires empty protein inputs and "
                             "protein_mask=None")
        for name, values in inputs.items():
            expected_rows = count if name == "protein" else rna_mask.shape[0]
            for key, value in values.items():
                if isinstance(value, torch.Tensor) and (value.ndim < 1
                                                        or value.shape[0] != expected_rows):
                    raise ValueError(f"{name}.{key} must contain {expected_rows} rows; "
                                     "protein inputs contain only present rows")

        dna = self.encoders["dna"](**inputs["dna"])
        protein = (self.encoders["protein"](**inputs["protein"], attention_mask=protein_mask)
                   if count else None)
        rna = self.encoders["rna"](**inputs["rna"], attention_mask=rna_mask)
        return self.fusion(dna, rna, protein, rna_mask=rna_mask, protein_mask=protein_mask,
                           protein_present=present)
