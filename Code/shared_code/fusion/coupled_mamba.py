"""Coupled Mamba on the codon grid: the project's first fusion design.

Li et al., "Coupled Mamba: Enhanced Multi-modal Fusion with Coupled State Space Model",
NeurIPS 2024 (arXiv:2405.18014). Each modality runs its own selective state-space (Mamba)
block, and the blocks are coupled through their hidden states: at every step, each
modality's transition is applied to the pooled previous state of all modalities, not to its
own alone. Information therefore accumulates along the sequence while the modalities stay
mixed at every step.

That is the property the three BioLangFusion heads lack. Concatenation mixes each codon
independently of the others. The gated head picks one weighting per sequence. Cross-attention
mixes positions by content but carries no state from one codon to the next. None of them can
let one modality change how a later codon of another modality is read.

The coupled update, the paper's Eq. 6, for modality m at codon t:

    h_t^m = Abar_t^m * H_{t-1} + Bbar_t^m x_t^m,     H_{t-1} = mean over m' of h_{t-1}^{m'}

and each modality reads its own state, y_t^m = C_t^m h_t^m + D^m x_t^m (Algorithm 1).

Four choices here are the project's, not the paper's.

**The pooled state is a mean, not a sum.** Eq. 6 sums the M previous states. Each Abar is an
elementwise decay in (0, 1), and Mamba initializes most of them close to 1, so the summed
transition (Eq. 7's P) approaches M and the state grows like M^t. Over 333 codons with three
modalities that overflows float32. The mean is the same rule scaled by 1/M and keeps every
pooled decay inside (0, 1). The paper's own sequences are far shorter, which may be why it
does not address this.

**The time axis is the codon grid.** Eq. 6 adds the modalities' states at the same step t,
which assumes their steps correspond; the paper does not say how unaligned inputs are matched.
Here they correspond by construction: the codon grid places DNA, RNA and protein on one axis,
one step per codon, before this head sees them.

**Sizes follow the paper where memory allows.** Hidden width 128, expansion 2 (inner width
256) and a causal convolution of 4 are the paper's. Its best state size was 64 and it stacked
3 layers; here the state is Mamba's default 16 and there are 2 layers, to keep a training
batch of 32 full-length rows within a few hundred megabytes on a laptop or an 8 GB card.

**Discretization is Mamba's, not Eq. 2's.** Abar = exp(delta * A) as in Eq. 2, but Bbar is
delta * B, the simplification Mamba's own scan uses, rather than the exact zero-order hold.

The scan is a plain loop over codons in PyTorch, not the fused CUDA kernel of the mamba-ssm
package, which does not build on Apple Silicon. Everything the scan reads (step sizes, inputs,
B and C) is computed for all codons at once. The scan itself runs in chunks of CHUNK codons,
and during training each chunk is recomputed in the backward pass instead of stored, so the
(batch, codons, inner, state) tensors the recurrence needs exist for one chunk at a time. Without
this, one batch of 32 rows at 333 codons held over 5 GB.

Two further variants register beside it, because a coupled head alone could not say what
made a difference:

* ``uncoupled_mamba``: identical blocks and parameter count, but each modality's state
  follows only its own chain. Coupled against uncoupled isolates the coupling.
* ``mamba_bottleneck``: the projections to the Mamba width and the final fusion, with no
  Mamba layers. Uncoupled against bottleneck isolates the recurrence from the narrower width.

All three end the same way: each modality's stream is normalized, the three are concatenated
at each codon and projected to the shared width, so the TextCNN head reads them exactly as it
reads every other head's output.

The scan runs one way, 5' to 3', as in the paper. Padding sits after each row's last real
codon, so it cannot reach a real position, and the TextCNN head reads only real codons. A
codon's state therefore reflects everything upstream of it and nothing downstream; the TextCNN
head's convolutions and pooling see the whole row. A bidirectional variant is a separate
question.

Hyperparameters can be passed to the constructors, but vary them only under a new registered
name or a new result tag: a saved result's identity records the variant name, not these
arguments, so two settings under one name would be indistinguishable on disk.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

# The order and width the BioLangFusion heads use (fusion/heads.py). Repeated rather than
# imported, because heads.py imports this module to register it; a test checks they agree.
MODALITIES = ("dna", "rna", "prot")
SHARED_WIDTH = 600

D_MODEL = 128              # the paper's hidden width
EXPAND = 2                 # the paper's expansion factor
D_CONV = 4                 # the paper's causal convolution
D_STATE = 16               # Mamba's default; the paper's best was 64 (see above)
N_LAYERS = 2               # the paper stacked 3 (see above)
CHUNK = 32                 # codons per scan chunk; changes memory and speed, not the result
DT_RANGE = (1e-3, 1e-1)    # Mamba's range of initial step sizes
DT_FLOOR = 1e-4


def previous_states(decay, drive, state=None):
    """Run h_t = decay_t * h_{t-1} + drive_t across the time axis.

    decay and drive are (batch, time, inner, state); state is h_{-1}, zero if omitted.
    Returns (before, last): before has the shape of drive and holds h_{t-1} at position t, so
    the caller can apply its own transition to it; last is h at the final step.
    """
    if state is None:
        state = drive.new_zeros(drive.shape[0], *drive.shape[2:])
    before = []
    for t in range(drive.shape[1]):
        before.append(state)
        state = decay[:, t] * state + drive[:, t]
    return torch.stack(before, dim=1), state


def scan_chunk(coupled, states, deltas, us, bs, cs, As, Ds):
    """The recurrence and read-out over one chunk of codons, for every modality.

    states: the states entering the chunk; one shared state if coupled, else one per
    modality. Per modality: deltas and us are (batch, chunk, inner), bs and cs are
    (batch, chunk, state), As is the (inner, state) negative transition and Ds the skip.
    Returns (ys, states): each modality's (batch, chunk, inner) output, and the states
    leaving the chunk, in the form they entered.
    """
    decays = [torch.exp(d.unsqueeze(-1) * A) for d, A in zip(deltas, As)]
    drives = [(d * u).unsqueeze(-1) * b.unsqueeze(2) for d, u, b in zip(deltas, us, bs)]
    if coupled:
        # Eq. 6 with a mean in place of the sum: one shared recurrence on the pooled state,
        # whose decay and drive are the modalities' own, averaged.
        before, state = previous_states(sum(decays) / len(decays),
                                        sum(drives) / len(drives), states[0])
        befores, states = [before] * len(decays), [state]
    else:
        scanned = [previous_states(dc, dr, s) for dc, dr, s in zip(decays, drives, states)]
        befores, states = [b for b, _ in scanned], [s for _, s in scanned]
    # Each modality applies its own transition to the state it was given, and reads it out.
    ys = [torch.einsum("bkin,bkn->bki", dc * before + dr, c) + D * u
          for dc, dr, before, c, D, u in zip(decays, drives, befores, cs, Ds, us)]
    return ys, states


class MambaBlock(nn.Module):
    """One modality's Mamba layer before and after the scan (the paper's Algorithm 1).

    The scan itself belongs to the fusion head, because coupling decides which state each
    modality's transition is applied to.
    """

    def __init__(self, d_model, d_inner, d_state, d_conv, dt_rank):
        super().__init__()
        self.d_state, self.dt_rank = d_state, dt_rank
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        # Depthwise and causal: padded on both sides, then cut to the first T outputs, so
        # codon t sees codons t - 3 to t.
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        # S4D-real initialization: A = -(1, 2, ..., state) for every inner channel.
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self._initialize_step()

    def _initialize_step(self):
        """Mamba's initialization of the step size delta: the projection's weight uniform in
        +-rank^-0.5, its bias the inverse softplus of steps drawn log-uniformly from DT_RANGE."""
        bound = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -bound, bound)
        low, high = (math.log(v) for v in DT_RANGE)
        dt = torch.exp(torch.rand(self.dt_proj.out_features) * (high - low) + low)
        dt = dt.clamp(min=DT_FLOOR)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def prepare(self, x, mask):
        """Everything the scan reads, for every codon at once.

        Returns (u, z, delta, b, c): the convolved input and the gate, (batch, codons,
        inner); the step size, (batch, codons, inner); and B and C, (batch, codons, state).
        """
        length = x.shape[1]
        u, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        u = self.conv(u.transpose(1, 2))[..., :length].transpose(1, 2)
        u = F.silu(u) * mask.unsqueeze(-1)
        dt, b, c = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        return u, z, delta, b, c

    @property
    def transition(self):
        """A, negative so that every decay exp(delta * A) lies in (0, 1)."""
        return -torch.exp(self.A_log)

    def finish(self, x, y, z):
        """Everything after the scan: gate the read-out and add the residual."""
        return x + self.out_proj(y * F.silu(z))


class CoupledMambaFusion(nn.Module):
    """Coupled Mamba over the three codon-aligned streams (registered as ``coupled_mamba``)."""

    COUPLED = True
    LAYERS = N_LAYERS

    def __init__(self, widths, out_width=SHARED_WIDTH, d_model=D_MODEL, expand=EXPAND,
                 d_state=D_STATE, d_conv=D_CONV):
        super().__init__()
        self.out_width = out_width
        d_inner = expand * d_model
        dt_rank = math.ceil(d_model / 16)
        # The same input treatment as the BioLangFusion heads: a linear projection and tanh.
        self.project = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(widths[m], d_model), nn.Tanh()) for m in MODALITIES})
        self.layers = nn.ModuleList(
            nn.ModuleDict({m: MambaBlock(d_model, d_inner, d_state, d_conv, dt_rank)
                           for m in MODALITIES})
            for _ in range(self.LAYERS))
        self.norm = nn.ModuleDict({m: nn.LayerNorm(d_model) for m in MODALITIES})
        self.fuse = nn.Linear(len(MODALITIES) * d_model, out_width)

    def modality_streams(self, streams, mask, chunk=CHUNK):
        """Each modality's stream after the Mamba layers, before the final fusion."""
        x = {m: self.project[m](streams[m]) for m in MODALITIES}
        for layer in self.layers:
            prepared = [layer[m].prepare(x[m], mask) for m in MODALITIES]
            ys = self._scan(prepared, [layer[m].transition for m in MODALITIES],
                            [layer[m].D for m in MODALITIES], chunk)
            x = {m: layer[m].finish(x[m], y, p[1])
                 for m, y, p in zip(MODALITIES, ys, prepared)}
        return x

    def _scan(self, prepared, As, Ds, chunk):
        """Each modality's read-out over every codon, chunk by chunk, carrying the state.

        While training, each chunk is recomputed in the backward pass rather than stored.
        """
        batch, length, inner = prepared[0][0].shape
        d_state = prepared[0][3].shape[-1]
        n_states = 1 if self.COUPLED else len(MODALITIES)
        states = [prepared[0][0].new_zeros(batch, inner, d_state) for _ in range(n_states)]
        recompute = torch.is_grad_enabled() and self.training
        outputs = [[] for _ in MODALITIES]
        for start in range(0, length, chunk):
            part = slice(start, start + chunk)
            args = (self.COUPLED, states,
                    [p[2][:, part] for p in prepared], [p[0][:, part] for p in prepared],
                    [p[3][:, part] for p in prepared], [p[4][:, part] for p in prepared],
                    As, Ds)
            # The scan draws no random numbers, so the random state need not be saved for the
            # recomputation (saving it is slower and has been unreliable on Apple's MPS).
            ys, states = (checkpoint(scan_chunk, *args, use_reentrant=False,
                                     preserve_rng_state=False) if recompute
                          else scan_chunk(*args))
            for out, y in zip(outputs, ys):
                out.append(y)
        return [torch.cat(out, dim=1) for out in outputs]

    def forward(self, streams, mask):
        x = self.modality_streams(streams, mask)
        joined = torch.cat([self.norm[m](x[m]) for m in MODALITIES], dim=-1)
        return self.fuse(joined) * mask.unsqueeze(-1), {}


class UncoupledMambaFusion(CoupledMambaFusion):
    """Ablation (``uncoupled_mamba``): the same blocks and parameters, but each modality's
    state follows only its own chain until the final fusion."""

    COUPLED = False


class MambaBottleneckFusion(CoupledMambaFusion):
    """Control (``mamba_bottleneck``): the projections to the Mamba width and the final
    fusion, with no Mamba layers between them."""

    LAYERS = 0


DESIGNS = {
    "coupled_mamba": CoupledMambaFusion,
    "uncoupled_mamba": UncoupledMambaFusion,
    "mamba_bottleneck": MambaBottleneckFusion,
}
