"""The fixed TextCNN prediction head every BioLangFusion variant shares.

Parallel one-dimensional convolutions of width 3, 4, and 5 run over the input position axis, each
followed by ReLU and a max over time; the three pooled vectors are concatenated, dropped
out at 0.2, and sent through one fully connected layer to the task output (BioLangFusion
Appendix A.4, List 1 and List 2).

The paper calls this head fixed, "the same fixed TextCNN prediction head to ensure a fair
comparison", but gives it two different widths: List 1 states 1,280 output channels for
the MIL variant and List 2 states 100 for the cross-attention variant. The two differ by
more than a factor of twelve in the head's capacity, so the width is a parameter here. The
default is 100: the narrower setting makes a common-capacity comparison less expensive.
Either width can be shared; this choice does not resolve the paper's conflicting settings.
The build-check notebook offers an optional 1,280-channel MIL sensitivity run, disabled
by default. Fusion models supply codon positions; unimodal baselines supply native tokens,
so the same kernel width need not cover the same nucleotide span.

Max pooling ignores padded positions. A convolution of width k over a row of L positions has
L - k + 1 real outputs, and positions past that are built from padding, so they are set to
negative infinity before the max rather than left to compete with real activations.
"""

from __future__ import annotations

import torch
from torch import nn

KERNEL_SIZES = (3, 4, 5)
DEFAULT_CHANNELS = 100
DEFAULT_DROPOUT = 0.2


class TextCNN(nn.Module):
    """Convolve over the input position axis, pool over time, and predict.

    forward takes a (batch, time, width) stream and the boolean mask marking real
    positions, and returns (batch, n_outputs).
    """

    def __init__(self, width, n_outputs, kernel_sizes=KERNEL_SIZES, channels=DEFAULT_CHANNELS,
                 dropout=DEFAULT_DROPOUT):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.convs = nn.ModuleList(
            [nn.Conv1d(width, channels, kernel) for kernel in self.kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(channels * len(self.convs), n_outputs)

    def forward(self, x, mask):
        lengths = mask.sum(dim=1)
        # A batch shorter than the widest kernel cannot be convolved at all, so it is padded
        # up to it; the padding sits outside every row's mask and is pooled away.
        widest = max(self.kernel_sizes)
        if x.shape[1] < widest:
            x = torch.cat([x, x.new_zeros(x.shape[0], widest - x.shape[1], x.shape[2])], dim=1)
        signal = x.transpose(1, 2)
        pooled = []
        for kernel, conv in zip(self.kernel_sizes, self.convs):
            activated = torch.relu(conv(signal))
            # A row shorter than the kernel has no real output; keep its first position
            # rather than pooling over an empty set, which would give negative infinity.
            real = (lengths - kernel + 1).clamp(min=1)
            positions = torch.arange(activated.shape[2], device=x.device).unsqueeze(0)
            keep = (positions < real.unsqueeze(1)).unsqueeze(1)
            pooled.append(activated.masked_fill(~keep, float("-inf")).max(dim=2).values)
        return self.output(self.dropout(torch.cat(pooled, dim=1)))
