"""IsoFormer's aggregation module: cross-attention between modalities, no codon grid.

IsoFormer (Garau-Luis et al., "Multi-modal Transfer Learning between Biological Foundation
Models") answers the alignment problem by not having one. Where BioLangFusion resamples every
stream onto the codon frame and then combines position by position, IsoFormer leaves each
modality at its own length and its own width and lets cross-attention find the correspondence:

    "The module is applied to each modality, performing cross-attention successively to the
    other modalities to produce multi-modal embeddings that are specific to each modality by
    keeping their dimensionality and which can be also stacked. Note that this aggregation
    method is robust to the absence of a specific modality as the cross-attention term will
    be zeroed out." (Section 4.2)

That robustness is the reason the modality ablation in their Table 2, DNA alone through DNA
plus RNA plus protein, runs through one architecture rather than six. It is reproduced here
by building the module over whichever modalities are present.

Published settings used here: 8 cross-attention heads (Appendix B), Adam at 3e-5, batch 64,
early stopping on 5% of the training set (Section 4.3). Their encoders are Nucleotide
Transformer v2 for both DNA and RNA and ESM-2 150M for protein; ESM-2 150M is 640 wide here
as there, while our Nucleotide Transformer is the 100M checkpoint at 512 rather than their
250M at 768.

Three things the paper leaves to the implementation.

**What "successively" means.** Each modality attends to the others one after another, each
with a residual connection, rather than to all of them at once against a joint context. The
order is fixed so a run is reproducible, and because residual updates do not commute the
order is a choice, recorded here rather than hidden.

**Where the layer norm goes.** The paper says cross-attention with residual connections and
does not say whether anything normalises. A layer norm after each residual is the standard
reading and is what is done here.

**How the pooled modalities are combined.** The paper takes "a global average pooling over
the length of the embedding" and then "a linear layer outputs one value per tissue". Since
each modality keeps its own width and its own length, the natural reading is to pool each
modality separately and concatenate the results before the linear layer, which also keeps
the head's input width fixed as modalities are added or removed.
"""

from __future__ import annotations

import torch
from torch import nn

MODALITIES = ("dna", "rna", "prot")
N_HEADS = 8
DROPOUT = 0.1


def masked_mean(x, mask):
    """Average a (batch, time, width) stream over each row's real positions."""
    counts = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return (x * mask.unsqueeze(-1)).sum(dim=1) / counts


class CrossAttentionAggregator(nn.Module):
    """One aggregation layer: every modality attends to each of the others in turn.

    Each modality keeps its own width, so the attention that reads modality ``other`` into
    modality ``name`` is built with that modality's width as its key and value dimension.
    """

    def __init__(self, widths, n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        self.names = tuple(widths)
        self.attend = nn.ModuleDict()
        self.norm = nn.ModuleDict()
        for name in self.names:
            for other in self.names:
                if other == name:
                    continue
                self.attend[f"{name}<-{other}"] = nn.MultiheadAttention(
                    widths[name], n_heads, kdim=widths[other], vdim=widths[other],
                    dropout=dropout, batch_first=True)
                self.norm[f"{name}<-{other}"] = nn.LayerNorm(widths[name])

    def forward(self, streams, masks):
        updated = dict(streams)
        for name in self.names:
            for other in self.names:
                if other == name:
                    continue
                key = f"{name}<-{other}"
                attended, _ = self.attend[key](updated[name], streams[other], streams[other],
                                               key_padding_mask=~masks[other],
                                               need_weights=False)
                updated[name] = self.norm[key](updated[name] + attended)
            updated[name] = updated[name] * masks[name].unsqueeze(-1)
        return updated


class IsoFormer(nn.Module):
    """Aggregate the present modalities, pool each over length, predict every tissue.

    ``modalities`` selects which streams the model reads, which is how the paper's Table 2
    ablation from one modality to three is run through a single architecture. With one
    modality there is nothing to attend to and the model reduces to that encoder's stream
    pooled and read by the linear head, which is what their single-modality rows are.
    """

    def __init__(self, widths, n_tissues, modalities=MODALITIES, n_layers=1,
                 n_heads=N_HEADS, dropout=DROPOUT):
        super().__init__()
        missing = [m for m in modalities if m not in widths]
        if missing:
            raise ValueError(f"no width given for {missing}")
        # Each modality's own width is its attention embedding dimension, so a swapped-in
        # encoder whose width does not divide the head count fails here with its name
        # rather than inside torch.
        odd = {m: widths[m] for m in modalities if widths[m] % n_heads}
        if odd and len(modalities) > 1:
            raise ValueError(f"{n_heads} attention heads need widths divisible by {n_heads}; "
                             f"got {odd}")
        self.modalities = tuple(modalities)
        self.widths = {m: widths[m] for m in self.modalities}
        self.layers = nn.ModuleList(
            [CrossAttentionAggregator(self.widths, n_heads=n_heads, dropout=dropout)
             for _ in range(n_layers)] if len(self.modalities) > 1 else [])
        self.expression = nn.Linear(sum(self.widths.values()), n_tissues)

    @property
    def variant(self):
        return "+".join(self.modalities)

    def forward(self, batch):
        streams = {m: batch.streams[m] for m in self.modalities}
        masks = {m: batch.masks[m] for m in self.modalities}
        for layer in self.layers:
            streams = layer(streams, masks)
        pooled = torch.cat([masked_mean(streams[m], masks[m]) for m in self.modalities], dim=-1)
        return self.expression(pooled), {}


def build(widths, n_tissues, variant, **kwargs):
    """A model by name: 'dna', 'rna+prot', 'dna+rna+prot', and so on."""
    return IsoFormer(widths, n_tissues, modalities=tuple(variant.split("+")), **kwargs)


# ---------------------------------------------------------------------------
# The GTEx transcript-expression pilot, and the batching the aggregator needs.
#
# The pilot is a 2,000-row sample of InstaDeepAI's own released dataset
# (multi_omics_transcript_expression, GTEx_final.csv), which is the dataset IsoFormer was
# trained on, so the labels already carry their scaling and the thirty tissues are in their
# order. Stage 1 split it by gene with no gene shared between halves, as the paper does.
# What differs is size: 1,000 training rows against their full set. Absolute values are
# therefore not the target; the ordering in their Table 2 is.
#
# Each stream is cached under its own name rather than its encoder's, because IsoFormer
# reads DNA and RNA with the same encoder and the two caches must not collide.
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from .. import datasets as datasets_module  # noqa: E402
from .token_embeddings import summarize  # noqa: E402

GTEX_PILOT = "GTEx_pilot.csv"
# IsoFormer's own encoder roles (Table 1): one DNA model read over both the genomic window
# and the transcript, and ESM-2 150M over the protein. Ours are the checkpoints Stage 1
# already uses, so the protein encoder matches theirs exactly and the DNA one is smaller.
STREAM_COLUMN = {"dna": "DNA", "rna": "RNA", "prot": "Protein"}
STREAM_ENCODER = {"dna": "ntv2_100m", "rna": "ntv2_100m", "prot": "esm2_150m"}


@dataclass
class GTExStreams:
    """The pilot's rows, its thirty tissue labels, and the three token caches."""

    key: str
    rows: object
    caches: dict
    tissues: tuple
    task: str = "expression"
    split_column: str = "split"

    @classmethod
    def open(cls, cache_root, data_dir=None, key="gtex_pilot"):
        import pandas as pd
        directory = data_dir or datasets_module.DEFAULT_DATA_DIR
        rows = pd.read_csv(f"{directory}/{GTEX_PILOT}")
        caches = {m: summarize(f"{cache_root}/{key}/{m}_{STREAM_ENCODER[m]}")
                  for m in MODALITIES}
        streams = cls(key, rows, caches, tuple(datasets_module.GTEX_TISSUES))
        streams.check()
        return streams

    def check(self):
        for modality, cache in self.caches.items():
            if len(cache) != len(self.rows):
                raise ValueError(
                    f"{self.key}/{modality}: {len(cache)} cached rows for {len(self.rows)} "
                    "transcripts; the cache was built from a different row set.")
        if tuple(self.rows.columns[-len(self.tissues):]) != self.tissues:
            raise ValueError("the pilot's tissue columns are not the expected thirty, in order")

    @property
    def widths(self):
        return {m: c.width for m, c in self.caches.items()}

    @property
    def n_tissues(self):
        return len(self.tissues)

    @property
    def labels(self):
        return self.rows[list(self.tissues)].to_numpy(dtype=np.float32)

    def split_rows(self, split):
        return np.flatnonzero(self.rows[self.split_column].to_numpy() == split)

    def close(self):
        for cache in self.caches.values():
            cache.close()


@dataclass
class ExpressionBatch:
    """Three padded streams with their masks, and one expression row per transcript."""

    streams: dict
    masks: dict
    labels: object

    def to(self, device):
        return ExpressionBatch({m: s.to(device) for m, s in self.streams.items()},
                               {m: v.to(device) for m, v in self.masks.items()},
                               self.labels.to(device))


class ExpressionDataset(Dataset):
    def __init__(self, streams, indices):
        self.streams, self.indices = streams, indices
        self.labels = streams.labels[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = int(self.indices[i])
        return ({m: np.array(self.streams.caches[m][row]) for m in MODALITIES},
                self.labels[i])


def collate(batch):
    """Pad each modality to the longest row in the batch, on its own time axis."""
    rows, labels = zip(*batch)
    padded, masks = {}, {}
    for modality in MODALITIES:
        parts = [r[modality] for r in rows]
        longest = max(max(p.shape[0] for p in parts), 1)
        block = torch.zeros(len(parts), longest, parts[0].shape[1], dtype=torch.float32)
        mask = torch.zeros(len(parts), longest, dtype=torch.bool)
        for i, part in enumerate(parts):
            block[i, :part.shape[0]] = torch.from_numpy(part.astype(np.float32))
            mask[i, :part.shape[0]] = True
        padded[modality], masks[modality] = block, mask
    return ExpressionBatch(padded, masks, torch.from_numpy(np.stack(labels)))


# Attention memory, not parameter count, is what limits this model, and it is driven by the
# longest row in a batch rather than the typical one. On the GTEx pilot the DNA window is a
# fixed 1,000 tokens while a transcript ranges from 78 to 2,000 and a protein from 1 to
# 1,022, so the cost of a row spans a factor of eight: about 10 million attention cells for
# a median transcript against 81 million for the longest. A fixed-size batch is padded up to
# its longest member and pays that worst case for every row in it, which is how sixteen rows
# reached 24 GB on a machine where the median batch would have needed under one.
#
# Grouping rows of similar length fixes that. Batches are formed under a budget on attention
# cells, so a batch of short transcripts holds many rows and a batch containing the longest
# holds two or three, and peak memory is roughly the same either way.

DEFAULT_CELL_BUDGET = 300_000_000      # about 1.2 GB of attention maps per batch in float32
DEFAULT_MAX_BATCH = 16


def attention_cells(lengths, n_heads=N_HEADS):
    """Attention cells one row costs: every ordered pair of modalities, times the heads."""
    total = np.zeros(len(next(iter(lengths.values()))), dtype=np.float64)
    for first in MODALITIES:
        for second in MODALITIES:
            if first != second:
                total += lengths[first] * lengths[second]
    return total * n_heads


class BudgetedBatches:
    """Batch positions grouped by cost, under a ceiling on attention cells.

    Rows are ordered by cost so neighbours pad each other cheaply. The order of the batches
    themselves is reshuffled every epoch when shuffling is on, so training still sees them
    in a different sequence each time even though their membership is fixed.
    """

    def __init__(self, lengths, budget, max_size, shuffle, seed, n_heads=N_HEADS):
        # A batch pads every modality to its own longest row, so its cost comes from the
        # per-modality maxima rather than from any single row. Rows are visited in order of
        # their own cost so that neighbours pad each other cheaply.
        def padded_cost(longest):
            return n_heads * sum(longest[a] * longest[b]
                                 for a in MODALITIES for b in MODALITIES if a != b)

        order = np.argsort(attention_cells(lengths, n_heads), kind="stable")
        self.batches, current = [], []
        longest = {m: 0 for m in MODALITIES}
        for position in order:
            row = {m: int(lengths[m][position]) for m in MODALITIES}
            grown = {m: max(longest[m], row[m]) for m in MODALITIES}
            if current and ((len(current) + 1) * padded_cost(grown) > budget
                            or len(current) >= max_size):
                self.batches.append(current)
                current, grown = [], row
            current.append(int(position))
            longest = grown
        if current:
            self.batches.append(current)
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        order = list(range(len(self.batches)))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
            self.epoch += 1
        for index in order:
            yield self.batches[index]


def loaders_for(streams, batch_size=DEFAULT_MAX_BATCH, val_fraction=0.05, seed=0,
                budget=DEFAULT_CELL_BUDGET, n_heads=N_HEADS):
    """Train, validation and test loaders. IsoFormer holds out 5% of train for early
    stopping and reports on the fixed test split, so that is what is built here.
    batch_size is a ceiling: a batch of long transcripts will be smaller."""
    train = streams.split_rows("train")
    shuffled = np.random.default_rng(seed).permutation(train)
    cut = max(1, int(round(len(shuffled) * val_fraction)))
    parts = {"val": shuffled[:cut], "train": shuffled[cut:], "test": streams.split_rows("test")}
    made = {}
    for name, indices in parts.items():
        dataset = ExpressionDataset(streams, indices)
        lengths = {m: np.asarray(streams.caches[m].lengths)[indices] for m in MODALITIES}
        sampler = BudgetedBatches(lengths, budget, batch_size,
                                  shuffle=(name == "train"), seed=seed, n_heads=n_heads)
        made[name] = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    return made


def builder(streams, variant, config, **kwargs):
    """The signature train.run expects, so the resume and checkpoint logic is shared."""
    return build(streams.widths, streams.n_tissues, variant, **kwargs)
