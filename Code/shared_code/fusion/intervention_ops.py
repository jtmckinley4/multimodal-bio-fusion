"""Small tensor operations for Mina's proposed gating and attention comparisons.

These implement mechanics from Biological_Fusion_Plan A3â€“A6, not complete fusion
architectures. Callers still choose learned projections, gate scorers, attention,
prediction heads and matched-capacity controls. No historical BLF behavior changes.
Equal tensor positions do not establish biological correspondence: the caller must
provide the justified alignment and, for CDS restrictions, its availability mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _floating(tensor, dimensions, name):
    if not isinstance(tensor, torch.Tensor) or tensor.ndim not in dimensions:
        raise ValueError(f"{name} must be a tensor with {dimensions} dimensions.")
    if not tensor.is_floating_point() or any(size == 0 for size in tensor.shape):
        raise ValueError(f"{name} must have floating values and nonempty dimensions.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must contain only finite values, including padding.")


def _mask(mask, shape, device, name):
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(shape) or mask.device != device):
        raise ValueError(f"{name} must be a boolean tensor of shape {tuple(shape)} on {device}.")


def _positions(mask, tokens):
    _mask(mask, tokens.shape[:2], tokens.device, "position mask")
    if not mask.any(dim=1).all():
        raise ValueError("Every example must supply at least one real position.")


def masked_summary(tokens, mask, *, reduction):
    """A3: summarize BÃ—TÃ—D projected tokens over real positions only.

    ``mean_max`` concatenates the mean then maximum, returning BÃ—2D; the other
    reductions return BÃ—D. An entirely empty example is an error, not a zero vector.
    """
    _floating(tokens, (3,), "tokens")
    _positions(mask, tokens)
    if reduction not in ("mean", "max", "mean_max"):
        raise ValueError("Choose reduction='mean', 'max', or 'mean_max'.")
    expanded = mask.unsqueeze(-1)
    if reduction != "max":
        mean = tokens.masked_fill(~expanded, 0).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    if reduction != "mean":
        maximum = tokens.masked_fill(~expanded, -torch.inf).amax(dim=1)
    if reduction == "mean_max":
        return torch.cat((mean, maximum), dim=-1)
    return mean if reduction == "mean" else maximum


@dataclass(frozen=True)
class GateRecord:
    """One detached CPU observation; positions are supplied tensor indices, not CDS coordinates."""

    row_id: str
    weights: torch.Tensor
    positions: torch.Tensor | None


def collect_gate_records(batches):
    """A3â€“A6: retain per-example gates in the caller's batch and row order.

    Each batch is (row IDs, weights, position mask). Sequence weights BÃ—M require
    mask=None. Position weights BÃ—TÃ—M require a boolean BÃ—T mask; only real
    positions are retained, with their original indices. No averaging or sorting
    occurs. Duplicate/empty IDs and inconsistent gate shapes are rejected.
    """
    records, seen = [], set()
    layout = None
    for row_ids, weights, mask in batches:
        _floating(weights, (2, 3), "diagnostic weights")
        if ((weights < 0) | (weights > 1)).any():
            raise ValueError("Diagnostic gate weights must lie between zero and one.")
        if isinstance(row_ids, str):
            raise ValueError("Row IDs must be a sequence of individual strings.")
        row_ids = tuple(row_ids)
        if len(row_ids) != weights.shape[0]:
            raise ValueError("Row IDs must match the diagnostic batch length.")
        current = (weights.ndim, weights.shape[-1])
        if layout is not None and current != layout:
            raise ValueError("Gate rank and modality count must agree across batches.")
        layout = current
        if weights.ndim == 2:
            if mask is not None:
                raise ValueError("Sequence-gate diagnostics require position mask=None.")
        else:
            _positions(mask, weights)
        for index, row_id in enumerate(row_ids):
            if not isinstance(row_id, str) or not row_id.strip() or row_id in seen:
                raise ValueError("Row IDs must be nonempty, unique strings.")
            seen.add(row_id)
            positions = None if mask is None else torch.nonzero(mask[index], as_tuple=True)[0]
            selected = weights[index] if positions is None else weights[index, positions]
            records.append(GateRecord(
                row_id, selected.detach().cpu().clone(),
                None if positions is None else positions.detach().cpu().clone(),
            ))
    if not records:
        raise ValueError("Gate diagnostics require at least one example.")
    return records
