"""BioLangFusion assembled: the cached token streams, the codon grid, a head, a prediction.

The model is the paper's: resample the three streams onto the codon grid, fuse them one of
three ways, and read the fused stream with a fixed TextCNN head. Unimodal baselines are the
same head on one stream at its own resolution, which is what "single-modal baseline" has to
mean when no alignment is needed.

Reading the streams is the part the paper does not describe. The caches written in
Code/stage3/02_token_embeddings.ipynb hold one row per benchmark sequence, in the file's
own order, so a row index selects the same sequence in all three caches and in the
benchmark frame. Rows are padded to the longest in their batch, each stream on its own
time axis, and the number of codons in each row travels with the batch: it is what the
grid resamples to and what the mask is built from.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .. import datasets as datasets_module
from .codon_grid import CODON_LENGTH, CodonGrid, mask_of
from .heads import FUSIONS, SHARED_WIDTH
from .textcnn import DEFAULT_CHANNELS, TextCNN
from .token_embeddings import summarize

# BioLangFusion's own three encoders, in the order the fusion heads expect them.
ENCODER_OF_MODALITY = {"dna": "ntv2_100m", "rna": "rnafm", "prot": "esm2_8m"}
MODALITIES = tuple(ENCODER_OF_MODALITY)


@dataclass
class BenchmarkStreams:
    """One benchmark's rows and its three token caches, opened together."""

    key: str
    rows: object
    caches: dict
    task: str
    label_column: str = "Value"
    split_column: str = "Split"

    @classmethod
    def open(cls, key, cache_root, encoder_of_modality=None):
        encoder_of_modality = encoder_of_modality or ENCODER_OF_MODALITY
        spec = datasets_module.DATASETS[key]
        rows = datasets_module.blf_benchmark_rows(key)
        caches = {m: summarize(f"{cache_root}/{key}/{encoder}")
                  for m, encoder in encoder_of_modality.items()}
        streams = cls(key, rows, caches, spec.task, spec.label_column, spec.split_column)
        streams.check()
        return streams

    def check(self):
        """Every cache must hold one row per sequence, with the token count its length implies."""
        nucleotides = self.rows["length"].to_numpy()
        for modality, cache in self.caches.items():
            if len(cache) != len(self.rows):
                raise ValueError(
                    f"{self.key}/{modality}: {len(cache)} cached rows for {len(self.rows)} "
                    "sequences; the cache was built from a different row set.")
        if not np.array_equal(np.asarray(self.caches["rna"].lengths), nucleotides):
            raise ValueError(f"{self.key}: the RNA cache is not one token per nucleotide.")
        if not np.array_equal(np.asarray(self.caches["prot"].lengths), nucleotides // CODON_LENGTH):
            raise ValueError(f"{self.key}: the protein cache is not one token per codon.")

    @property
    def widths(self):
        return {m: cache.width for m, cache in self.caches.items()}

    @property
    def labels(self):
        return self.rows[self.label_column].to_numpy()

    @property
    def n_outputs(self):
        return int(self.labels.max()) + 1 if self.task == "classification" else 1

    def split_rows(self, split):
        """Positions, into the frame and the caches alike, of one published split's rows."""
        return np.flatnonzero(self.rows[self.split_column].to_numpy() == split)

    def close(self):
        for cache in self.caches.values():
            cache.close()


class GridDataset(Dataset):
    """The three streams and the label of each row in one split."""

    def __init__(self, streams, split):
        self.streams = streams
        self.indices = streams.split_rows(split)
        labels = streams.labels[self.indices]
        self.labels = labels.astype(np.int64 if streams.task == "classification" else np.float32)
        self.codons = streams.rows["length"].to_numpy()[self.indices] // CODON_LENGTH

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = int(self.indices[i])
        # np.array copies, which a memory-mapped shard needs: torch refuses to wrap a
        # read-only array without warning that writing to it is undefined.
        return ({m: torch.from_numpy(np.array(self.streams.caches[m][row]))
                 for m in MODALITIES},
                int(self.codons[i]), self.labels[i])


@dataclass
class GridBatch:
    """One padded batch: the three streams, their real token counts, codons, and labels."""

    streams: dict
    tokens: dict
    codons: object
    labels: object

    def to(self, device):
        return GridBatch({m: s.to(device) for m, s in self.streams.items()},
                         {m: t.to(device) for m, t in self.tokens.items()},
                         self.codons.to(device), self.labels.to(device))


def collate(batch):
    """Pad each stream to the longest row in the batch, on its own time axis."""
    streams, codons, labels = zip(*batch)
    padded, counts = {}, {}
    for modality in MODALITIES:
        rows = [item[modality] for item in streams]
        longest = max(row.shape[0] for row in rows)
        block = torch.zeros(len(rows), longest, rows[0].shape[1], dtype=torch.float32)
        for i, row in enumerate(rows):
            block[i, :row.shape[0]] = row.float()
        padded[modality] = block
        counts[modality] = torch.tensor([row.shape[0] for row in rows], dtype=torch.long)
    return GridBatch(padded, counts, torch.tensor(codons, dtype=torch.long),
                     torch.as_tensor(np.array(labels)))


def loader(streams, split, batch_size=32, shuffle=None, seed=0):
    """A DataLoader over one split; the training split is shuffled, the others are not."""
    shuffle = (split == "train") if shuffle is None else shuffle
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(GridDataset(streams, split), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, generator=generator)


class BioLangFusion(nn.Module):
    """Codon grid, one fusion head, and the TextCNN prediction head."""

    def __init__(self, widths, fusion, n_outputs, channels=DEFAULT_CHANNELS,
                 out_width=SHARED_WIDTH, tconv="appendix", **fusion_kwargs):
        super().__init__()
        if fusion not in FUSIONS:
            raise ValueError(f"fusion must be one of {sorted(FUSIONS)}, not {fusion!r}")
        self.fusion_name = fusion
        self.grid = CodonGrid(widths["dna"], tconv=tconv)
        self.fusion = FUSIONS[fusion](widths, out_width=out_width, **fusion_kwargs)
        self.head = TextCNN(self.fusion.out_width, n_outputs, channels=channels)

    def forward(self, batch):
        grid, mask = self.grid(batch.streams["dna"], batch.streams["rna"],
                               batch.streams["prot"], batch.codons, batch.tokens["dna"])
        fused, aux = self.fusion(grid, mask)
        return self.head(fused, mask), aux


class Unimodal(nn.Module):
    """One stream at its own resolution, read by the same TextCNN head."""

    def __init__(self, widths, modality, n_outputs, channels=DEFAULT_CHANNELS):
        super().__init__()
        self.modality = modality
        self.fusion_name = f"unimodal:{modality}"
        self.head = TextCNN(widths[modality], n_outputs, channels=channels)

    def forward(self, batch):
        stream = batch.streams[self.modality]
        return self.head(stream, mask_of(batch.tokens[self.modality], stream.shape[1])), {}


def build(streams, variant, channels=DEFAULT_CHANNELS, **kwargs):
    """A model by name: a fusion head, or 'unimodal:dna', 'unimodal:rna', 'unimodal:prot'."""
    if variant.startswith("unimodal:"):
        return Unimodal(streams.widths, variant.split(":", 1)[1], streams.n_outputs,
                        channels=channels)
    return BioLangFusion(streams.widths, variant, streams.n_outputs, channels=channels, **kwargs)
