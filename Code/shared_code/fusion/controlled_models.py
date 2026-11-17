"""Learned feature-fusion operations for Mina's controlled A3–A6 comparisons.

These heads do not choose encoders, align biological positions, fit models or make
predictions. A caller supplies aligned streams (A3–A4), or separately masked
streams with an explicit anchor (A5–A6), and attaches its chosen prediction head.
All architecture settings are explicit. Parameter counts describe the constructed
model; a low-rank control is not declared matched merely because it is available.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .heads import GatedMILFusion, MODALITIES
from .intervention_ops import masked_summary


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _widths(widths):
    if not isinstance(widths, dict) or len(widths) < 2:
        raise ValueError("Supply an ordered width mapping for at least two modalities.")
    for name, width in widths.items():
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("Modality names must be nonempty strings without periods.")
        _positive(width, f"{name} width")
    return dict(widths)


def _normalization(width, kind):
    if kind not in ("none", "layer"):
        raise ValueError("Normalization must be 'none' or 'layer'.")
    return nn.Identity() if kind == "none" else nn.LayerNorm(width)


def _projections(widths, width, activation, normalization):
    _positive(width, "projection_width")
    if activation not in ("linear", "tanh"):
        raise ValueError("Projection activation must be 'linear' or 'tanh'.")
    if normalization not in ("none", "layer"):
        raise ValueError("Projection normalization must be 'none' or 'layer'.")
    projects = {}
    for name, input_width in widths.items():
        layers = [nn.Linear(input_width, width)]
        if activation == "tanh":
            layers.append(nn.Tanh())
        if normalization == "layer":
            layers.append(nn.LayerNorm(width))
        projects[name] = nn.Sequential(*layers)
    return nn.ModuleDict(projects)


def _check_streams(streams, widths, masks, *, anchor=None):
    if set(streams) != set(widths) or set(masks) != set(widths):
        raise ValueError("Streams and masks must identify exactly the declared modalities.")
    reference = streams[next(iter(widths))]
    if not isinstance(reference, torch.Tensor) or reference.ndim != 3:
        raise ValueError("Every stream must be a B×T×D tensor.")
    for name, width in widths.items():
        tensor, mask = streams[name], masks[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.ndim != 3
                or tensor.shape[0] != reference.shape[0] or tensor.shape[0] < 1
                or tensor.shape[-1] != width or not tensor.is_floating_point()
                or tensor.device != reference.device or tensor.dtype != reference.dtype):
            raise ValueError("Streams must agree on batch, device and floating dtype, with declared widths.")
        if not torch.isfinite(tensor).all():
            raise ValueError("Stream values, including padding, must be finite.")
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != tensor.shape[:2] or mask.device != tensor.device):
            raise ValueError("Each mask must be a boolean B×T tensor on its stream's device.")
        if (anchor is None or name == anchor) and not mask.any(dim=1).all():
            raise ValueError("Aligned streams and anchor rows need at least one real position.")


def _aligned(streams, widths, mask):
    masks = {name: mask for name in widths}
    _check_streams(streams, widths, masks)
    return masks


def _scorers(names, input_width, scorer_width, shared):
    _positive(scorer_width, "scorer_width")
    if type(shared) is not bool:
        raise ValueError("shared_scorer must be an explicit boolean.")
    if shared:
        v, u = nn.Linear(input_width, scorer_width), nn.Linear(input_width, scorer_width)
        return nn.ModuleDict({m: v for m in names}), nn.ModuleDict({m: u for m in names})
    return (nn.ModuleDict({m: nn.Linear(input_width, scorer_width) for m in names}),
            nn.ModuleDict({m: nn.Linear(input_width, scorer_width) for m in names}))


def _gate_norms(projected, weights, fused, mask):
    """Detached diagnostics only; per-position magnitudes are not causal importance."""
    features = torch.stack([value.detach().norm(dim=-1) for value in projected.values()], -1)
    features = features.masked_fill(~mask.unsqueeze(-1), 0)
    effective = weights.detach()[:, None, :] if weights.ndim == 2 else weights.detach()
    return {"feature_norms": features, "contribution_norms": features * effective,
            "fused_norm": fused.detach().norm(dim=-1)}


class SequenceGateFusion(nn.Module):
    """A3: BLF-style gated scorers with mean or mean–max sequence summaries.

    For mean/tanh/no-normalization and the three BLF modalities, use the released
    project/scorer components directly; their state keys support exact transfer.
    Softmax has the explicitly bounded learned temperature. Sigmoid has no
    temperature parameter, no renormalization and no ``alpha`` entropy output.
    ``alpha`` is exposed only for softmax with the three named BLF modalities;
    the historical trainer's result writer assumes that modality order.
    """

    def __init__(self, widths, *, projection_width, scorer_width, summary,
                 normalization, shared_scorer, temperature_range,
                 projection_activation, projection_norm):
        super().__init__()
        self.widths = _widths(widths)
        self.names, self.out_width = tuple(self.widths), projection_width
        _positive(projection_width, "projection_width")
        _positive(scorer_width, "scorer_width")
        if summary not in ("mean", "mean_max") or normalization not in ("softmax", "sigmoid"):
            raise ValueError("Choose mean/mean_max summaries and softmax/sigmoid gates.")
        if type(shared_scorer) is not bool:
            raise ValueError("shared_scorer must be an explicit boolean.")
        self.summary, self.normalization = summary, normalization
        self.collect_norms = False
        legacy_components = (self.names == MODALITIES and summary == "mean"
                             and projection_activation == "tanh" and projection_norm == "none")
        if legacy_components:
            base = GatedMILFusion(self.widths, out_width=projection_width,
                                  attention_width=scorer_width, shared_projection=shared_scorer)
            self.project, self.gate_v, self.gate_u, self.score = (
                base.project, base.gate_v, base.gate_u, base.score)
        else:
            self.project = _projections(self.widths, projection_width, projection_activation,
                                        projection_norm)
            summary_width = projection_width * (2 if summary == "mean_max" else 1)
            self.gate_v, self.gate_u = _scorers(self.names, summary_width, scorer_width, shared_scorer)
            self.score = nn.Linear(scorer_width, 1, bias=False)
        if normalization == "softmax":
            if (not isinstance(temperature_range, (tuple, list)) or len(temperature_range) != 2
                    or not all(isinstance(v, (float, int)) and math.isfinite(v)
                               for v in temperature_range)
                    or not 0 < temperature_range[0] <= temperature_range[1]):
                raise ValueError("Softmax requires an explicit positive (minimum, maximum) temperature.")
            self.temperature_range = tuple(temperature_range)
            self.log_temperature = nn.Parameter(torch.zeros(()))
        elif temperature_range is not None:
            raise ValueError("Independent sigmoid gates require temperature_range=None.")

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        projected = {m: self.project[m](streams[m]) for m in self.names}
        scores = []
        for m in self.names:
            summary = masked_summary(projected[m], mask, reduction=self.summary)
            hidden = torch.tanh(self.gate_v[m](summary)) * torch.sigmoid(self.gate_u[m](summary))
            scores.append(self.score(hidden))
        logits = torch.cat(scores, dim=-1)
        if self.normalization == "softmax":
            temperature = self.log_temperature.exp().clamp(*self.temperature_range)
            weights = torch.softmax(logits / temperature, dim=-1)
        else:
            weights = torch.sigmoid(logits)
        fused = sum(weights[:, i, None, None] * projected[m] for i, m in enumerate(self.names))
        diagnostics = {"gates": weights, "gate_kind": f"sequence_{self.normalization}",
                       "gate_modalities": self.names}
        if self.normalization == "softmax" and self.names == MODALITIES:
            diagnostics["alpha"] = weights
        fused = fused.masked_fill(~mask.unsqueeze(-1), 0)
        if self.collect_norms:
            diagnostics.update(_gate_norms(projected, weights, fused, mask))
            if self.normalization == "softmax":
                diagnostics["temperature"] = temperature.detach()
        return fused, diagnostics


class PositionGateFusion(nn.Module):
    """A4: each affine sigmoid gate sees all projected modalities at its position.

    ``scorer_width`` remains a validated legacy call argument; the joint linear
    equation has no hidden scorer layer and does not use its value.
    """

    def __init__(self, widths, *, projection_width, scorer_width,
                 projection_activation, projection_norm):
        super().__init__()
        self.widths = _widths(widths)
        self.names, self.out_width = tuple(self.widths), projection_width
        _positive(scorer_width, "scorer_width")
        self.collect_norms = False
        self.project = _projections(self.widths, projection_width, projection_activation, projection_norm)
        self.gate = nn.ModuleDict({m: nn.Linear(len(self.names) * projection_width, 1)
                                   for m in self.names})

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        projected = {m: self.project[m](streams[m]) for m in self.names}
        joint = torch.cat(list(projected.values()), dim=-1)
        scores = [self.gate[m](joint) for m in self.names]
        weights = torch.sigmoid(torch.cat(scores, dim=-1)).masked_fill(~mask.unsqueeze(-1), 0)
        fused = sum(weights[:, :, i, None] * projected[m] for i, m in enumerate(self.names))
        diagnostics = {"gates": weights, "gate_kind": "position_sigmoid", "gate_modalities": self.names}
        if self.collect_norms:
            diagnostics.update(_gate_norms(projected, weights, fused, mask))
        return fused, diagnostics


class JointSequenceGateFusion(PositionGateFusion):
    """A4 sequence comparator: the position gate's scorers see joint masked means.

    Projection and scorer state keys match PositionGateFusion exactly. Only the
    reduction before scoring and broadcast after scoring change; no nonlinear
    hidden scorer, entropy penalty or temperature is introduced.
    """

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        projected = {m: self.project[m](streams[m]) for m in self.names}
        joint = torch.cat(list(projected.values()), dim=-1)
        summary = masked_summary(joint, mask, reduction="mean")
        weights = torch.sigmoid(torch.cat([self.gate[m](summary) for m in self.names], dim=-1))
        fused = sum(weights[:, i, None, None] * projected[m] for i, m in enumerate(self.names))
        fused = fused.masked_fill(~mask.unsqueeze(-1), 0)
        diagnostics = {"gates": weights, "gate_kind": "joint_sequence_sigmoid", "gate_modalities": self.names}
        if self.collect_norms:
            diagnostics.update(_gate_norms(projected, weights, fused, mask))
        return fused, diagnostics


def attention_control_budget(width, modality_count):
    """Nearest active two-layer local mixer budget to joint-gated attention.

    Each auxiliary attention has four d-by-d projections and four d biases.
    Its joint gate has M*d weights and one bias. The local MLP has M*d -> h
    -> d weights and both biases. This is a parameter control, not a claim of
    equal expressivity, optimization difficulty or runtime.
    """
    _positive(width, "projection_width")
    _positive(modality_count, "modality_count")
    if modality_count < 2:
        raise ValueError("The attention control needs at least one auxiliary.")
    target = (modality_count - 1) * (4 * width * width + 4 * width + modality_count * width + 1)
    per_hidden = (modality_count + 1) * width + 1
    hidden = max(1, round((target - width) / per_hidden))
    return {"attention_mixing": target, "hidden_width": hidden,
            "control_mixing": hidden * per_hidden + width}


class ResidualConcatFusion(nn.Module):
    """A5 nongated local nonlinear mixing with an unchanged projected-anchor skip.

    The hidden width is fixed from the joint-gated attention parameter budget,
    before outcomes are inspected. Every registered parameter affects the model;
    no dummy attention or gate is registered to inflate the count.
    """

    def __init__(self, widths, *, anchor, projection_width, hidden_width,
                 projection_activation, projection_norm):
        super().__init__()
        self.widths = _widths(widths)
        self.names, self.out_width = tuple(self.widths), projection_width
        if anchor not in self.widths:
            raise ValueError("Choose an anchor from the declared modalities.")
        expected = attention_control_budget(projection_width, len(self.names))["hidden_width"]
        if type(hidden_width) is not int or hidden_width != expected:
            raise ValueError(f"The declared attention-budget control requires hidden_width={expected}.")
        self.anchor, self.collect_norms = anchor, False
        self.project = _projections(self.widths, projection_width, projection_activation, projection_norm)
        self.mix = nn.Sequential(nn.Linear(len(self.names) * projection_width, hidden_width),
                                 nn.GELU(), nn.Linear(hidden_width, projection_width))

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        projected = {m: self.project[m](streams[m]) for m in self.names}
        anchor = projected[self.anchor]
        update = self.mix(torch.cat(list(projected.values()), dim=-1))
        fused = (anchor + update).masked_fill(~mask.unsqueeze(-1), 0)
        diagnostics = {}
        if self.collect_norms:
            diagnostics = {"anchor_norm": anchor.detach().norm(dim=-1).masked_fill(~mask, 0),
                           "fused_norm": fused.detach().norm(dim=-1)}
        return fused, diagnostics


class PositionalControl(nn.Module):
    """A4: a nongated, low-rank affine mixer at each aligned position.

    Two affine maps factor the concatenated feature-to-output matrix through the
    requested bottleneck. No activation is inserted between them. Compare actual
    parameter counts to the chosen gate head before calling this capacity-matched.
    """

    def __init__(self, widths, *, projection_width, bottleneck_width,
                 projection_activation, projection_norm):
        super().__init__()
        self.widths = _widths(widths)
        self.names, self.out_width = tuple(self.widths), projection_width
        _positive(bottleneck_width, "bottleneck_width")
        self.project = _projections(self.widths, projection_width, projection_activation, projection_norm)
        self.mix = nn.Sequential(nn.Linear(len(self.names) * projection_width, bottleneck_width),
                                 nn.Linear(bottleneck_width, projection_width))

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        joined = torch.cat([self.project[m](streams[m]) for m in self.names], dim=-1)
        return self.mix(joined).masked_fill(~mask.unsqueeze(-1), 0), {}


class ResidualPositionalControl(nn.Module):
    """Full-width fixed sum plus a nongated, low-rank affine learned update.

    The 0.5 scale matches independent sigmoid gates at zero logits; it is not a
    modality mean. The fixed path does not pass through the update bottleneck.
    Ordinary Linear initialization leaves the input factor random; only the
    output factor is zeroed, so the initial update is zero without freezing both
    factors. Parameter proximity to a gate does not imply equal capacity.
    """

    def __init__(self, widths, *, projection_width, bottleneck_width, fixed_scale,
                 update_initialization, projection_activation, projection_norm):
        super().__init__()
        self.widths = _widths(widths)
        self.names, self.out_width = tuple(self.widths), projection_width
        _positive(bottleneck_width, "bottleneck_width")
        if fixed_scale != 0.5 or isinstance(fixed_scale, bool):
            raise ValueError("This residual control requires the explicit fixed_scale=0.5.")
        if update_initialization != "zero_output_random_input":
            raise ValueError("Declare zero_output_random_input initialization for this control.")
        self.fixed_scale = fixed_scale
        self.project = _projections(self.widths, projection_width, projection_activation, projection_norm)
        self.update = nn.Sequential(nn.Linear(len(self.names) * projection_width, bottleneck_width),
                                    nn.Linear(bottleneck_width, projection_width))
        nn.init.zeros_(self.update[1].weight)
        nn.init.zeros_(self.update[1].bias)

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        projected = [self.project[m](streams[m]) for m in self.names]
        fixed = self.fixed_scale * sum(projected)
        update = self.update(torch.cat(projected, dim=-1))
        return (fixed + update).masked_fill(~mask.unsqueeze(-1), 0), {}


class AnchorAttentionFusion(nn.Module):
    """A5–A6: anchor queries each auxiliary independently, then combines updates.

    Modes ordinary/skip use unit update weights; ordinary omits the anchor skip.
    sequence_sigmoid/position_sigmoid retain the skip. Each auxiliary's affine
    sigmoid gate sees the anchor and every auxiliary update (Mina A5's joint u).
    Sequence gates use masked means of those joint features. ``scorer_width`` is
    retained for call compatibility; these linear gates have no hidden width.
    No entropy, temperature or post-skip normalization is applied. Zero effective
    updates recover the projected anchor exactly.

    ``availability`` optionally maps every auxiliary to a boolean B×T_anchor mask.
    It blocks updates after attention and normalization biases, as well as gates.
    No-key auxiliary rows bypass attention entirely, including zero-length inputs.
    The caller, not this class, establishes biological alignment/CDS availability.
    Diagnostics distinguish raw ``gates`` from masked effective ``update_weights``.
    """

    def __init__(self, widths, *, anchor, projection_width, scorer_width, n_heads,
                 dropout, mode, projection_activation, projection_norm, update_norm):
        super().__init__()
        self.widths = _widths(widths)
        if anchor not in self.widths:
            raise ValueError("Anchor must identify one declared modality.")
        _positive(n_heads, "n_heads")
        _positive(scorer_width, "scorer_width")
        _positive(projection_width, "projection_width")
        if projection_width % n_heads:
            raise ValueError("projection_width must be divisible by n_heads.")
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if mode not in ("ordinary", "skip", "sequence_sigmoid", "position_sigmoid"):
            raise ValueError("Choose ordinary, skip, sequence_sigmoid or position_sigmoid mode.")
        self.anchor, self.mode, self.out_width = anchor, mode, projection_width
        self.collect_norms = False
        self.names = tuple(m for m in self.widths if m != anchor)
        self.project = _projections(self.widths, projection_width, projection_activation, projection_norm)
        self.attend = nn.ModuleDict({m: nn.MultiheadAttention(
            projection_width, n_heads, dropout=dropout, batch_first=True) for m in self.names})
        self.update_norm = nn.ModuleDict({m: _normalization(projection_width, update_norm) for m in self.names})
        self.gate = nn.ModuleDict()
        if mode.endswith("sigmoid"):
            self.gate = nn.ModuleDict({m: nn.Linear(len(self.widths) * projection_width, 1)
                                       for m in self.names})

    def forward(self, streams, masks, availability=None):
        _check_streams(streams, self.widths, masks, anchor=self.anchor)
        query_mask = masks[self.anchor]
        if availability is not None and set(availability) != set(self.names):
            raise ValueError("Availability must identify every auxiliary, excluding the anchor.")
        projected = {m: self.project[m](streams[m]).masked_fill(~masks[m].unsqueeze(-1), 0)
                     for m in self.widths}
        query = projected[self.anchor]
        combined = torch.zeros_like(query) if self.mode == "ordinary" else query
        updates, allowed_masks = {}, {}
        for m in self.names:
            allowed = query_mask if availability is None else availability[m]
            if (not isinstance(allowed, torch.Tensor) or allowed.dtype != torch.bool
                    or allowed.shape != query_mask.shape or allowed.device != query.device):
                raise ValueError("Auxiliary availability must be boolean B×T_anchor on the query device.")
            allowed = allowed & query_mask & masks[m].any(dim=1, keepdim=True)
            active = torch.nonzero(allowed.any(dim=1), as_tuple=True)[0]
            update = torch.zeros_like(query)
            if active.numel():
                attended, _ = self.attend[m](query[active], projected[m][active], projected[m][active],
                                             key_padding_mask=~masks[m][active], need_weights=False)
                update = update.index_copy(0, active, self.update_norm[m](attended))
            updates[m] = update.masked_fill(~allowed.unsqueeze(-1), 0)
            allowed_masks[m] = allowed
        # Compute all updates first: a gate can respond to either auxiliary.
        if self.gate:
            features = torch.cat([query, *updates.values()], dim=-1)
            if self.mode == "sequence_sigmoid":
                features = masked_summary(features, query_mask, reduction="mean")
        gates, effective_weights = [], []
        for m, update in updates.items():
            if self.mode.endswith("sigmoid"):
                weight = torch.sigmoid(self.gate[m](features)).squeeze(-1)
                gates.append(weight)
                effective = weight[:, None] if weight.ndim == 1 else weight
            else:
                effective = torch.ones_like(query_mask, dtype=query.dtype)
            effective = effective.masked_fill(~allowed_masks[m], 0)
            effective_weights.append(effective)
            combined = combined + effective.unsqueeze(-1) * update
        diagnostics = {"update_weights": torch.stack(effective_weights, dim=-1),
                       "gate_modalities": self.names}
        if gates:
            diagnostics.update(gates=torch.stack(gates, dim=-1), gate_kind=self.mode)
        combined = combined.masked_fill(~query_mask.unsqueeze(-1), 0)
        if self.collect_norms:
            raw = torch.stack([value.detach().norm(dim=-1) for value in updates.values()], -1)
            diagnostics.update(anchor_norm=query.detach().norm(dim=-1), raw_update_norms=raw,
                               update_norms=raw * diagnostics["update_weights"].detach(),
                               fused_norm=combined.detach().norm(dim=-1))
        return combined, diagnostics


class ProjectedAnchor(nn.Module):
    """A5 control: only the anchor's common projection, before the shared predictor."""

    def __init__(self, widths, *, anchor, projection_width, projection_activation, projection_norm):
        super().__init__()
        declared = _widths(widths)
        if anchor not in declared:
            raise ValueError("Anchor must identify a declared modality.")
        self.anchor, self.out_width = anchor, projection_width
        self.widths = {anchor: declared[anchor]}
        self.project = _projections(self.widths, projection_width, projection_activation, projection_norm)

    def forward(self, streams, mask):
        _aligned(streams, self.widths, mask)
        return self.project[self.anchor](streams[self.anchor]).masked_fill(~mask.unsqueeze(-1), 0), {}


def parameter_count(module):
    """Count distinct trainable parameters; shared scorers are counted only once."""
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
