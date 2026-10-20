"""Put the DNA, RNA, and protein token streams on one codon grid.

BioLangFusion aligns the three modalities at the codon, three nucleotides encoding one
amino acid, and takes the protein frame as the reference: an input of T nucleotides
becomes T' = floor(T/3) complete triplets under the chosen origin. This length convention
does not establish a biological reading frame. The protein stream is one token per
codon. The RNA stream, one token per nucleotide, is averaged in non-overlapping groups
of three. The DNA stream, one token per 6-mer, is upsampled by a learned transposed
convolution (BioLangFusion Section 2.1 and Equation 1).

Two details the paper leaves to the implementation are settled here.

**The transposed convolution is specified twice, differently.** Equation 1 gives kernel 2
and stride 2, which doubles the stream exactly: T/6 tokens become T/3. Appendix List 1 and
List 2 give kernel 3, stride 2, padding 2, which returns 2n - 3 positions, three short of
doubling. Both appear in the same paper, so both are offered; the appendix setting is the
default, because the appendix is what the text calls the detailed architecture. Matching
the final array length does not make the settings equivalent in coverage or learned effect.

**Remainder tokens complicate the factor-two mapping.** Nucleotide Transformer v2 tokenizes greedily into
6-mers and falls back to single nucleotides for the remainder, so 999 nucleotides give 169
tokens rather than the 166.5 the paper's T/6 implies. The paper's own answer is the line
"Padding: sequence length aligned to protein" in List 1, so after the convolution the DNA
stream is zero-padded or trimmed to exactly T'. This matches shape without resolving
the nonuniform spans of complete 6-mers and individual remainder nucleotides. For inputs
made entirely of complete 6-mers, the equation setting does match T' before trimming.

That padding has to be done per row, which is why the DNA token count travels with the
batch. A transposed convolution over a batch writes an output position for every input
position, padding included, so a short row sitting in a batch with a long one comes back
with codons its own tokens never reached: real outputs where, on its own, it would have
had the zero padding that aligns it to T'. Those positions are inside the row's mask, so
they would reach the prediction head and make a sequence's representation depend on which
sequences it was batched with. Each row is therefore cut to its own convolution output
length, (n - 1) * stride - 2 * padding + kernel, before the frame mask is applied.

**Other encoders arrive at other resolutions.** Phase B swaps one encoder at a time, and a
replacement need not tokenize like the one it replaces: HyenaDNA reads DNA one nucleotide
per token, and mRNA-FM and CaLM read RNA one codon per token. Each stream is therefore
placed by its resolution rather than by its modality: a 6-mer stream through the transposed
convolution, a single-nucleotide stream through the three-to-one average, and a codon or
residue stream as it is. BioLangFusion's own encoders give the default, and with it the
grid is exactly the one checked in Stage 3.
"""

from __future__ import annotations

import torch
from torch import nn

CODON_LENGTH = 3

# The two transposed-convolution settings the paper gives, as (kernel, stride, padding).
TCONV_SETTINGS = {
    "appendix": (3, 2, 2),   # List 1 and List 2: out = 2n - 3
    "equation": (2, 2, 0),   # Equation 1: out = 2n
}

# How each stream reaches the grid (token_embeddings.grid_resolution names an encoder's), and
# BioLangFusion's own: Nucleotide Transformer 6-mers, RNA-FM nucleotides, ESM-2 residues.
RESOLUTIONS = ("kmer", "nucleotide", "codon")
BLF_RESOLUTIONS = {"dna": "kmer", "rna": "nucleotide", "prot": "codon"}


def fit_length(x, target):
    """Zero-pad or trim a (batch, time, width) stream to exactly target positions."""
    time = x.shape[1]
    if time == target:
        return x
    if time > target:
        return x[:, :target]
    pad = x.new_zeros(x.shape[0], target - time, x.shape[2])
    return torch.cat([x, pad], dim=1)


def mask_of(lengths, target):
    """A (batch, target) boolean mask that is True at each row's real positions."""
    positions = torch.arange(target, device=lengths.device).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


class CodonGrid(nn.Module):
    """Resample a batch of the three streams onto the protein frame.

    The forward pass takes the three padded streams, each (batch, time, width) with its own
    time axis, and the number of codons in every row. It returns the three streams on a
    shared (batch, T', width) grid and the mask marking each row's real codons, where T' is
    the longest row in the batch.
    """

    def __init__(self, dna_width, tconv="appendix", resolutions=None):
        super().__init__()
        if tconv not in TCONV_SETTINGS:
            raise ValueError(f"tconv must be one of {sorted(TCONV_SETTINGS)}, not {tconv!r}")
        self.resolutions = dict(BLF_RESOLUTIONS if resolutions is None else resolutions)
        if set(self.resolutions) != set(BLF_RESOLUTIONS) or any(
                r not in RESOLUTIONS for r in self.resolutions.values()):
            raise ValueError(f"resolutions must give one of {RESOLUTIONS} for each of "
                             f"{sorted(BLF_RESOLUTIONS)}, not {self.resolutions}")
        if self.resolutions["prot"] != "codon" or self.resolutions["rna"] == "kmer":
            raise ValueError("the protein stream is one token per codon, and only the DNA "
                             "stream has a transposed convolution")
        kernel, stride, padding = TCONV_SETTINGS[tconv]
        self.tconv_setting = tconv
        self.kernel, self.stride, self.padding = kernel, stride, padding
        if self.resolutions["dna"] == "kmer":
            self.dna_upsample = nn.ConvTranspose1d(dna_width, dna_width, kernel, stride=stride,
                                                   padding=padding)
        self.rna_downsample = nn.AvgPool1d(CODON_LENGTH, stride=CODON_LENGTH)

    def upsampled_length(self, dna_tokens):
        """Positions the transposed convolution produces from a row's own DNA tokens."""
        return ((dna_tokens - 1) * self.stride - 2 * self.padding + self.kernel).clamp(min=0)

    def place(self, modality, stream, target, tokens):
        """One stream on the (batch, target, width) grid, by its resolution."""
        resolution = self.resolutions[modality]
        if resolution == "kmer":
            upsampled = self.dna_upsample(stream.transpose(1, 2)).transpose(1, 2)
            return fit_length(upsampled, target) * mask_of(
                self.upsampled_length(tokens), target).unsqueeze(-1)
        if resolution == "nucleotide":
            return fit_length(self.rna_downsample(stream.transpose(1, 2)).transpose(1, 2), target)
        return fit_length(stream, target)

    def forward(self, dna, rna, protein, codon_lengths, dna_tokens):
        target = int(codon_lengths.max())
        grid = {
            "dna": self.place("dna", dna, target, dna_tokens),
            "rna": self.place("rna", rna, target, None),
            "prot": self.place("prot", protein, target, None),
        }
        mask = mask_of(codon_lengths, target)
        return {name: stream * mask.unsqueeze(-1) for name, stream in grid.items()}, mask
