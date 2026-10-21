"""The three BioLangFusion fusion strategies, each taking codon-aligned streams to one.

Every head reads the dictionary CodonGrid returns, three (batch, T', width) streams and
the mask of real codons, and returns a fused (batch, T', out_width) stream together with
whatever the training loop needs from it: the gated head also returns its modality
weights, which carry the entropy term in the loss and the attention figure in the paper.

* ConcatFusion, Section 2.2.1. A learned projection changes the DNA width, the three
  streams are concatenated at each codon, and the result is projected once more before the
  prediction head.
* GatedMILFusion, Section 2.2.2. Each stream is projected to a shared width, each is
  summarized by its mean over codons, and a gated attention over the three summaries gives
  one weight per modality per sequence, used to take a weighted sum.
* CrossModalFusion, Section 2.2.3. The three projected streams are concatenated along the
  codon axis into a joint context; each modality attends to that context, and the three
  updated streams are merged with a residual average and a layer norm.

Two choices the paper leaves open are made explicit.

**The projection width for concatenation.** Section 2.2.1 projects DNA to an unnamed
d'_DNA "since d_DNA is significantly larger than d_RNA and d_Prot". That ordering comes
from Table 3, which gives the Nucleotide Transformer an embedding dimension of 4,107.
4,107 is the checkpoint's vocabulary size, not its hidden size: the model writes 512-wide
states, between RNA-FM's 640 and ESM-2 (8M)'s 320. The stated large-DNA-width rationale
therefore does not describe these checkpoints. The projection is kept as part of the
architecture, using the paper's named common width of 600; here it expands DNA features.

**The mean in gated attention is taken over real codons only.** The paper writes the
summary as an unweighted mean over T' positions. Rows are padded to the longest in the
batch, so an unweighted mean would make a sequence's summary depend on what it was batched
with. The mask is applied instead.
"""

from __future__ import annotations

import torch
from torch import nn

MODALITIES = ("dna", "rna", "prot")
SHARED_WIDTH = 600
ATTENTION_WIDTH = 100
N_HEADS = 4
ATTENTION_DROPOUT = 0.1
TEMPERATURE_RANGE = (0.02, 20.0)


def masked_mean(x, mask):
    """Mean over the real positions of each row of a (batch, time, width) stream."""
    counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / counts


class ConcatFusion(nn.Module):
    """Codon-aware concatenation (Section 2.2.1)."""

    def __init__(self, widths, dna_width=SHARED_WIDTH, out_width=SHARED_WIDTH):
        super().__init__()
        self.dna_project = nn.Sequential(nn.Linear(widths["dna"], dna_width), nn.Tanh())
        self.fuse = nn.Linear(dna_width + widths["rna"] + widths["prot"], out_width)
        self.out_width = out_width

    def forward(self, streams, mask):
        joined = torch.cat([self.dna_project(streams["dna"]), streams["rna"], streams["prot"]],
                           dim=-1)
        return self.fuse(joined) * mask.unsqueeze(-1), {}


class GatedMILFusion(nn.Module):
    """Entropy-regularized gated attention pooling over modalities (Section 2.2.2).

    Returns the modality weights as ``alpha``, a (batch, 3) tensor in the order of
    MODALITIES. The training loop turns them into the entropy penalty.
    """

    def __init__(self, widths, out_width=SHARED_WIDTH, attention_width=ATTENTION_WIDTH,
                 shared_projection=False):
        super().__init__()
        self.out_width = out_width
        self.project = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(widths[m], out_width), nn.Tanh()) for m in MODALITIES})
        if shared_projection:
            # The ablation in Table 4: one attention projection for all three modalities.
            shared_v, shared_u = nn.Linear(out_width, attention_width), nn.Linear(out_width, attention_width)
            self.gate_v = nn.ModuleDict({m: shared_v for m in MODALITIES})
            self.gate_u = nn.ModuleDict({m: shared_u for m in MODALITIES})
        else:
            self.gate_v = nn.ModuleDict(
                {m: nn.Linear(out_width, attention_width) for m in MODALITIES})
            self.gate_u = nn.ModuleDict(
                {m: nn.Linear(out_width, attention_width) for m in MODALITIES})
        self.score = nn.Linear(attention_width, 1, bias=False)
        self.log_temperature = nn.Parameter(torch.zeros(()))

    def temperature(self):
        return self.log_temperature.exp().clamp(*TEMPERATURE_RANGE)

    def forward(self, streams, mask):
        projected = {m: self.project[m](streams[m]) for m in MODALITIES}
        scores = []
        for m in MODALITIES:
            summary = masked_mean(projected[m], mask)
            gated = torch.tanh(self.gate_v[m](summary)) * torch.sigmoid(self.gate_u[m](summary))
            scores.append(self.score(gated))
        alpha = torch.softmax(torch.cat(scores, dim=1) / self.temperature(), dim=1)
        fused = sum(alpha[:, i].view(-1, 1, 1) * projected[m] for i, m in enumerate(MODALITIES))
        return fused * mask.unsqueeze(-1), {"alpha": alpha}


class CrossModalFusion(nn.Module):
    """Cross-modal multi-head attention over the joint context (Section 2.2.3)."""

    def __init__(self, widths, out_width=SHARED_WIDTH, n_heads=N_HEADS,
                 dropout=ATTENTION_DROPOUT):
        super().__init__()
        self.out_width = out_width
        self.project = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(widths[m], out_width), nn.Tanh()) for m in MODALITIES})
        self.attend = nn.ModuleDict({
            m: nn.MultiheadAttention(out_width, n_heads, dropout=dropout, batch_first=True)
            for m in MODALITIES})
        self.norm = nn.ModuleDict({m: nn.LayerNorm(out_width) for m in MODALITIES})
        self.fuse = nn.Linear(out_width * len(MODALITIES), out_width)
        self.fused_norm = nn.LayerNorm(out_width)

    def forward(self, streams, mask):
        projected = {m: self.project[m](streams[m]) for m in MODALITIES}
        context = torch.cat([projected[m] for m in MODALITIES], dim=1)
        # A padded codon is padded in every modality, so the context mask is the row mask
        # repeated once per modality.
        context_pad = ~mask.repeat(1, len(MODALITIES))
        updated = {}
        for m in MODALITIES:
            attended, _ = self.attend[m](projected[m], context, context,
                                         key_padding_mask=context_pad, need_weights=False)
            updated[m] = self.norm[m](projected[m] + attended)
        joined = self.fuse(torch.cat([updated[m] for m in MODALITIES], dim=-1))
        average = sum(updated[m] for m in MODALITIES) / len(MODALITIES)
        return self.fused_norm(average + joined) * mask.unsqueeze(-1), {}


FUSIONS = {"concat": ConcatFusion, "mil": GatedMILFusion, "cross": CrossModalFusion}
