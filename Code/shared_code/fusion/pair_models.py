"""Pair scorers for two unaligned molecules: baselines, attention and an interleaved Mamba.

The codon-grid heads cannot be reused here. Concatenation, the gated head and Coupled Mamba
all assume that position t means the same thing in every stream; an RNA and the protein that
binds it have no such correspondence, different alphabets and unrelated lengths. What carries
over is the shape of the comparison, not the modules: a fusion claim is only as good as the
single-partner baseline it beats, and every arm sees the same frozen embeddings, the same
pooling and the same training recipe, so a difference is the fusion operation.

Every scorer takes two padded token streams with their masks and returns one logit per pair.

* ``left_only`` and ``right_only`` read one partner and ignore the other. A pair task where
  these score well is a task about the entity, not about the pair, which is the failure mode
  the plan's disjoint splits and partner-reassignment control are built to expose.
* ``concat`` pools each side to one vector and reads the two together: the simple same-input
  fusion that any attention or state-space result has to beat.
* ``bilinear`` scores a low-rank interaction of the two pooled vectors, the standard pair
  scorer in this literature.
* ``cross_attention`` lets each side's tokens attend to the other's, in one direction or both,
  with an optional gate on the update. ``gate="sequence"`` is one weight per sequence, as
  BioLangFusion's head does; ``gate="position"`` is one weight per token of the queried side,
  the non-competing per-position gate the plan's Step 5 asks for; the update is added to the
  side's own tokens, so the partner can only add to what a single encoder already provides.
* ``interleaved_mamba`` is the project's design carried to pairs: place one side's tokens after
  the other's in a single sequence, mark each token with a learned embedding for the molecule
  it came from, and run Mamba over the whole thing. State crosses the boundary, so tokens of
  the second molecule are read with the first already in the state. Order matters in a
  recurrence, so ``order`` names which molecule goes first and ``bidirectional`` runs the
  sequence in both directions and joins the two read-outs, which is how the published pair
  Mamba uses it.

The Mamba block, its initialization and the chunked scan are imported from ``coupled_mamba``
rather than rewritten, so the codon-grid design and the pair design differ only where the task
differs. The interleaved design has no counterpart in Li et al.'s coupled update: there the
modalities share a time axis and their states are pooled at each step, while here one sequence
carries both molecules and the sharing is the state crossing the boundary.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .coupled_mamba import (CHUNK, D_CONV, D_MODEL, D_STATE, EXPAND, MambaBlock, N_LAYERS,
                            previous_states)
from .heads import masked_mean
from .pair_tasks import SIDES

__all__ = ["SIDES", "HIDDEN", "N_HEADS", "DROPOUT", "PAIR_SCORERS", "build",
           "parameter_counts", "SinglePartnerScorer", "ConcatPairScorer",
           "BilinearPairScorer", "CrossAttentionPairScorer", "InterleavedMambaPairScorer"]

HIDDEN = 256           # width of the shared pair representation
N_HEADS = 4
DROPOUT = 0.1


def _mlp(in_width, hidden, dropout):
    return nn.Sequential(nn.Linear(in_width, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, 1))


class PairScorer(nn.Module):
    """One logit per pair, from two padded token streams and their masks.

    ``forward`` returns ``(logit, extras)``; extras carries whatever a diagnostic needs, such
    as the gate values, and is empty for the arms that have none.
    """

    def forward(self, left, right, left_mask, right_mask):  # pragma: no cover - interface
        raise NotImplementedError


class SinglePartnerScorer(PairScorer):
    """Baseline: pool one side and ignore the other entirely."""

    def __init__(self, widths, side, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        if side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}")
        self.side = side
        self.project = nn.Sequential(nn.Linear(widths[side], hidden), nn.Tanh())
        self.score = _mlp(hidden, hidden, dropout)

    def forward(self, left, right, left_mask, right_mask):
        stream, mask = (left, left_mask) if self.side == "left" else (right, right_mask)
        return self.score(masked_mean(self.project(stream), mask)).squeeze(-1), {}


class ConcatPairScorer(PairScorer):
    """Pool both sides and read them together: the same-input fusion reference."""

    def __init__(self, widths, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.project = nn.ModuleDict({
            side: nn.Sequential(nn.Linear(widths[side], hidden), nn.Tanh()) for side in SIDES})
        self.score = _mlp(2 * hidden, hidden, dropout)

    def pooled(self, left, right, left_mask, right_mask):
        return (masked_mean(self.project["left"](left), left_mask),
                masked_mean(self.project["right"](right), right_mask))

    def forward(self, left, right, left_mask, right_mask):
        a, b = self.pooled(left, right, left_mask, right_mask)
        return self.score(torch.cat([a, b], dim=-1)).squeeze(-1), {}


class BilinearPairScorer(ConcatPairScorer):
    """Low-rank bilinear interaction of the two pooled vectors, plus their own terms.

    The rank is explicit because a full bilinear form on two 256-wide vectors would hold more
    parameters than the rest of the model; the own terms keep it a strict extension of the
    pooled baselines rather than a different family.
    """

    def __init__(self, widths, hidden=HIDDEN, dropout=DROPOUT, rank=64):
        super().__init__(widths, hidden=hidden, dropout=dropout)
        self.left_factor = nn.Linear(hidden, rank, bias=False)
        self.right_factor = nn.Linear(hidden, rank, bias=False)
        self.score = _mlp(2 * hidden + rank, hidden, dropout)

    def forward(self, left, right, left_mask, right_mask):
        a, b = self.pooled(left, right, left_mask, right_mask)
        interaction = self.left_factor(a) * self.right_factor(b)
        return self.score(torch.cat([a, b, interaction], dim=-1)).squeeze(-1), {}


class CrossAttentionPairScorer(PairScorer):
    """Each side's tokens attend to the other side's, with an optional gate on the update.

    The attended information is added to the side's own tokens, so the arm is an addition to a
    single-partner representation rather than a replacement for it: with the gate closed it
    returns exactly the pooled single-partner path. ``direction`` chooses which side queries;
    "both" runs the symmetric two-branch design the plan asks for when neither molecule is the
    obvious anchor.
    """

    GATES = (None, "sequence", "position")

    def __init__(self, widths, hidden=HIDDEN, n_heads=N_HEADS, dropout=DROPOUT,
                 direction="both", gate=None):
        super().__init__()
        if direction not in (*SIDES, "both"):
            raise ValueError("direction must be 'left', 'right' or 'both'")
        if gate not in self.GATES:
            raise ValueError(f"gate must be one of {self.GATES}")
        self.direction, self.gate_kind = direction, gate
        self.queries = SIDES if direction == "both" else (direction,)
        self.project = nn.ModuleDict({
            side: nn.Sequential(nn.Linear(widths[side], hidden), nn.Tanh()) for side in SIDES})
        self.attend = nn.ModuleDict({
            side: nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            for side in self.queries})
        self.norm = nn.ModuleDict({side: nn.LayerNorm(hidden) for side in self.queries})
        if gate is not None:
            # One scalar per sequence or per queried token; sigmoid, so the two sides do not
            # compete for a shared budget and the gate cannot collapse by construction.
            self.gate = nn.ModuleDict({side: nn.Linear(2 * hidden if gate == "sequence" else
                                                       2 * hidden, 1) for side in self.queries})
        self.score = _mlp(len(self.queries) * hidden, hidden, dropout)

    def _updated(self, side, query, key, query_mask, key_mask, extras):
        attended, _ = self.attend[side](query, key, key,
                                        key_padding_mask=~key_mask, need_weights=False)
        attended = attended * query_mask.unsqueeze(-1)
        if self.gate_kind is None:
            return self.norm[side](query + attended)
        if self.gate_kind == "sequence":
            summary = torch.cat([masked_mean(query, query_mask),
                                 masked_mean(attended, query_mask)], dim=-1)
            weight = torch.sigmoid(self.gate[side](summary)).unsqueeze(1)
        else:
            weight = torch.sigmoid(self.gate[side](torch.cat([query, attended], dim=-1)))
        extras[f"{side}_gate"] = weight.squeeze(-1)
        return self.norm[side](query + weight * attended)

    def forward(self, left, right, left_mask, right_mask):
        projected = {"left": self.project["left"](left), "right": self.project["right"](right)}
        masks = {"left": left_mask, "right": right_mask}
        extras, summaries = {}, []
        for side in self.queries:
            other = "right" if side == "left" else "left"
            updated = self._updated(side, projected[side], projected[other],
                                    masks[side], masks[other], extras)
            summaries.append(masked_mean(updated, masks[side]))
        return self.score(torch.cat(summaries, dim=-1)).squeeze(-1), extras


class InterleavedMambaPairScorer(PairScorer):
    """One Mamba over both molecules placed in a single sequence, marked by molecule.

    The two streams are projected to a shared width, a learned embedding for "left" or "right"
    is added to every token so the model can tell the molecules apart, and the sequences are
    laid end to end in ``order``. The recurrence then carries the state of the first molecule
    into the second. ``bidirectional`` runs the same layers over the reversed sequence as well
    and joins both read-outs, so neither molecule is only ever read first.

    Padding sits at the end of each side's own stream, so the joined sequence is built from the
    real tokens of each row and padded once at the end; no padded position can enter the state
    before a real one.
    """

    def __init__(self, widths, d_model=D_MODEL, expand=EXPAND, d_state=D_STATE, d_conv=D_CONV,
                 layers=N_LAYERS, dropout=DROPOUT, order="left_right", bidirectional=True,
                 hidden=HIDDEN, chunk=CHUNK):
        super().__init__()
        if order not in ("left_right", "right_left"):
            raise ValueError("order must be 'left_right' or 'right_left'")
        self.order = SIDES if order == "left_right" else SIDES[::-1]
        self.bidirectional, self.chunk = bidirectional, chunk
        d_inner, dt_rank = expand * d_model, math.ceil(d_model / 16)
        self.project = nn.ModuleDict({
            side: nn.Sequential(nn.Linear(widths[side], d_model), nn.Tanh()) for side in SIDES})
        self.molecule = nn.Embedding(len(SIDES), d_model)
        directions = 2 if bidirectional else 1
        self.layers = nn.ModuleList(
            nn.ModuleList(MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
                          for _ in range(directions))
            for _ in range(layers))
        self.norm = nn.LayerNorm(directions * d_model)
        self.score = _mlp(directions * d_model, hidden, dropout)

    def join(self, left, right, left_mask, right_mask):
        """Place both molecules in one sequence per row, real tokens first, padding last."""
        streams = {"left": self.project["left"](left) + self.molecule.weight[0],
                   "right": self.project["right"](right) + self.molecule.weight[1]}
        masks = {"left": left_mask, "right": right_mask}
        first, second = self.order
        lengths = masks[first].sum(1) + masks[second].sum(1)
        total = int(lengths.max())
        joined = streams[first].new_zeros(streams[first].shape[0], total, streams[first].shape[-1])
        mask = masks[first].new_zeros(streams[first].shape[0], total)
        for row in range(joined.shape[0]):
            a, b = masks[first][row], masks[second][row]
            parts = torch.cat([streams[first][row][a], streams[second][row][b]], dim=0)
            joined[row, :parts.shape[0]] = parts
            mask[row, :parts.shape[0]] = True
        return joined, mask

    @staticmethod
    def _reversed_rows(x, mask):
        """Reverse each row's real tokens, leaving its padding at the end.

        Flipping the padded sequence instead would move the padding in front of the real
        tokens, where the causal convolution reads it: a row's result would then depend on how
        long the other rows in its batch are. Applying this twice returns the original order.
        """
        lengths = mask.sum(dim=1, keepdim=True)
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        index = torch.where(positions < lengths, lengths - 1 - positions, positions)
        return torch.gather(x, 1, index.unsqueeze(-1).expand_as(x)), index

    def _scan(self, block, x, mask, reverse):
        """One direction of one layer: Mamba's own recurrence over the joined sequence."""
        index = None
        if reverse:
            x, index = self._reversed_rows(x, mask)
        u, z, delta, b, c = block.prepare(x, mask)
        A, D = block.transition, block.D
        batch, length, inner = u.shape
        state = u.new_zeros(batch, inner, A.shape[-1])
        outputs = []
        recompute = torch.is_grad_enabled() and self.training
        for start in range(0, length, self.chunk):
            part = slice(start, start + self.chunk)
            args = (state, delta[:, part], u[:, part], b[:, part], c[:, part], A, D)
            y, state = (checkpoint(_scan_chunk, *args, use_reentrant=False,
                                   preserve_rng_state=False) if recompute else _scan_chunk(*args))
            outputs.append(y)
        y = torch.cat(outputs, dim=1)
        out = block.finish(x, y, z) * mask.unsqueeze(-1)
        if index is None:
            return out
        return torch.gather(out, 1, index.unsqueeze(-1).expand_as(out))

    def forward(self, left, right, left_mask, right_mask):
        joined, mask = self.join(left, right, left_mask, right_mask)
        streams = [joined] * (2 if self.bidirectional else 1)
        for layer in self.layers:
            streams = [self._scan(block, stream, mask, reverse=bool(index))
                       for index, (block, stream) in enumerate(zip(layer, streams))]
        pooled = torch.cat([masked_mean(stream, mask) for stream in streams], dim=-1)
        return self.score(self.norm(pooled)).squeeze(-1), {}


def _scan_chunk(state, delta, u, b, c, A, D):
    """One chunk of the single-stream recurrence and its read-out (coupled_mamba's, M = 1)."""
    decay = torch.exp(delta.unsqueeze(-1) * A)
    drive = (delta * u).unsqueeze(-1) * b.unsqueeze(2)
    before, state = previous_states(decay, drive, state)
    return torch.einsum("bkin,bkn->bki", decay * before + drive, c) + D * u, state


PAIR_SCORERS = {
    "left_only": lambda widths, **kw: SinglePartnerScorer(widths, "left", **kw),
    "right_only": lambda widths, **kw: SinglePartnerScorer(widths, "right", **kw),
    "concat": ConcatPairScorer,
    "bilinear": BilinearPairScorer,
    "cross_attention": CrossAttentionPairScorer,
    "interleaved_mamba": InterleavedMambaPairScorer,
}


def build(name, widths, **kwargs):
    """One pair scorer by name; ``widths`` gives each side's embedding width."""
    if name not in PAIR_SCORERS:
        raise ValueError(f"unknown pair scorer {name!r}; choose from {sorted(PAIR_SCORERS)}")
    if set(widths) != set(SIDES):
        raise ValueError(f"widths must name exactly {SIDES}")
    return PAIR_SCORERS[name](widths, **kwargs)


def parameter_counts(model):
    """Trainable parameters, split into the fusion path and the final scorer.

    The plan's controls are matched on capacity, so the count a comparison needs is the one
    that excludes the shared scoring head.
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    scorer = sum(p.numel() for p in model.score.parameters() if p.requires_grad)
    return {"total": int(total), "fusion": int(total - scorer), "scorer": int(scorer)}
