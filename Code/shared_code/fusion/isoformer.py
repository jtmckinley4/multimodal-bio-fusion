"""IsoFormer-style aggregation for the local frozen GTEx pilot, without a codon grid.

Garau-Luis et al., "Multi-modal Transfer Learning between Biological Foundation Models",
describe an aggregation module "performing cross-attention successively to the other
modalities" (Section 4.2). That motivates this pilot's unequal-length stream comparisons;
cross-attention alone does not establish biological correspondence between positions.

Each local layer updates every selected stream against the other streams' original
key/value representations in a fixed order. Each residual update is followed by LayerNorm.
The updated streams are masked, mean-pooled and concatenated for a linear tissue head.
Its input width is the sum of the selected stream widths, so it changes between variants.
Selecting streams at construction is distinct from bypassing absent inputs within a row.

Using eight attention heads follows the paper's Appendix B. The local ordering, LayerNorm,
dropout, layer count and pooling/head choices are explicit implementation choices, not a
verified reconstruction of the historical NT experiment. The released Enformer-based
model instead updates and pools DNA; it is not the implementation provided here.

The pilot freezes NT v2 100M for DNA/RNA and ESM-2 150M for protein. Its notebook records
the input crops and training settings; these differ from the published fine-tuned system.
See docs/sources.md#isoformer-dataset-and-model-cards and the Stage 3 step-4 notebook for
the source distinctions and comparison limits.
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

    ``modalities`` selects this pilot's inputs and the head's input width. With one stream,
    no cross-attention layers are constructed: masked pooling feeds the linear tissue head.
    These local comparisons are motivated by Table 2, not verified copies of its models.
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
# Stage 1's builder samples transcript rows from the release's supplied train/test
# partitions; it does not construct a new gene-grouped split. It copies the thirty tissue
# targets unchanged, without applying the release's separate normalization statistics.
# This loader then reserves transcript rows from training for validation. Cohort size,
# frozen backbones, context and aggregation also differ from the published experiment;
# Table 2's ordering is a question for this pilot, not an outcome it must reproduce.
#
# Each stream is cached under its own name: this pilot reads DNA and RNA with the same
# NT checkpoint, but their different input strings must not share a cache.
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from collections.abc import Mapping  # noqa: E402

import numpy as np  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from .. import datasets as datasets_module  # noqa: E402
from .token_embeddings import summarize  # noqa: E402

GTEX_PILOT = "GTEx_pilot.csv"
# The pilot reuses Stage 1 checkpoints: NT v2 100M for DNA/RNA and ESM-2 150M for protein.
# The protein model family/size matches the NT paper comparison, while NT is smaller.
# This does not establish identical historical weight revisions or producing environments.
STREAM_COLUMN = {"dna": "DNA", "rna": "RNA", "prot": "Protein"}
STREAM_ENCODER = {"dna": "ntv2_100m", "rna": "ntv2_100m", "prot": "esm2_150m"}


def _close_caches(caches):
    """Attempt every close, preserving the first failure if any cache cannot close."""
    first_error = None
    for modality, cache in caches.items():
        try:
            cache.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
            elif hasattr(first_error, "add_note"):
                first_error.add_note(f"Closing the {modality} cache also failed: {error}")
    if first_error is not None:
        raise first_error


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
        caches = {}
        try:
            for modality in MODALITIES:
                caches[modality] = summarize(
                    f"{cache_root}/{key}/{modality}_{STREAM_ENCODER[modality]}")
            streams = cls(key, rows, caches, tuple(datasets_module.GTEX_TISSUES))
            streams.check()
            return streams
        except BaseException as error:
            # Partially opened caches belong to this call until it returns successfully.
            try:
                _close_caches(caches)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"GTEx cache cleanup also failed: {cleanup_error}")
                    for note in getattr(cleanup_error, "__notes__", ()):
                        error.add_note(note)
            raise

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
        _close_caches(self.caches)


@dataclass
class ExpressionBatch:
    """Selected padded streams and masks, with one expression row per transcript."""

    streams: dict
    masks: dict
    labels: object

    def to(self, device):
        return ExpressionBatch({m: s.to(device) for m, s in self.streams.items()},
                               {m: v.to(device) for m, v in self.masks.items()},
                               self.labels.to(device))


def _materialized_modalities(modalities):
    """Validate the nonempty stream selection; None retains the original three streams."""
    if modalities is None:
        return MODALITIES
    if isinstance(modalities, str):
        raise ValueError("modalities must be a sequence of stream names, not one string")
    try:
        selected = tuple(modalities)
    except TypeError as error:
        raise ValueError("modalities must be a sequence of stream names") from error
    if not selected or any(not isinstance(name, str) for name in selected):
        raise ValueError("modalities must contain nonempty, unique DNA/RNA/protein stream names")
    if len(set(selected)) != len(selected) or any(name not in MODALITIES for name in selected):
        raise ValueError(f"modalities must select unique names from {MODALITIES}; got {selected}")
    return selected


class ExpressionDataset(Dataset):
    def __init__(self, streams, indices, modalities=None):
        self.streams, self.indices = streams, indices
        self.modalities = _materialized_modalities(modalities)
        missing = set(self.modalities) - streams.caches.keys()
        if missing:
            raise ValueError(f"selected streams have no caches: {sorted(missing)}")
        self.labels = streams.labels[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        row = int(self.indices[i])
        return ({m: np.array(self.streams.caches[m][row]) for m in self.modalities},
                self.labels[i])


def collate(batch):
    """Pad each modality to the longest row in the batch, on its own time axis."""
    if not batch:
        raise ValueError("cannot collate an empty expression batch")
    rows, labels = zip(*batch)
    if any(not isinstance(row, Mapping) for row in rows):
        raise ValueError("each expression row must map modality names to token arrays")
    modalities = _materialized_modalities(rows[0].keys())
    if any(set(row) != set(modalities) for row in rows):
        raise ValueError("all expression rows in a batch must contain the same modalities")
    padded, masks = {}, {}
    for modality in modalities:
        parts = [r[modality] for r in rows]
        longest = max(max(p.shape[0] for p in parts), 1)
        block = torch.zeros(len(parts), longest, parts[0].shape[1], dtype=torch.float32)
        mask = torch.zeros(len(parts), longest, dtype=torch.bool)
        for i, part in enumerate(parts):
            block[i, :part.shape[0]] = torch.from_numpy(part.astype(np.float32))
            mask[i, :part.shape[0]] = True
        padded[modality], masks[modality] = block, mask
    return ExpressionBatch(padded, masks, torch.from_numpy(np.stack(labels)))


# Padding cost depends on each stream's longest row in a batch. Grouping similar-cost
# rows reduces that overhead; the attention-cell budget is a scheduling proxy rather
# than a measured GPU-memory ceiling. Other activations, parameters and workspaces matter,
# and the attention kernel need not materialize every map. The pilot caps batches at
# sixteen rows and may use fewer for longer inputs.

DEFAULT_CELL_BUDGET = 300_000_000      # 1.2 GB only if every estimated cell occupies 4 bytes
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

    Rows are ordered by cost so neighbors pad each other cheaply. The order of the batches
    themselves is reshuffled every epoch when shuffling is on, so training still sees them
    in a different sequence each time even though their membership is fixed.
    """

    def __init__(self, lengths, budget, max_size, shuffle, seed, n_heads=N_HEADS):
        # Retain the effective batching settings for identity checks before iteration.
        self.budget, self.max_size, self.n_heads = budget, max_size, n_heads
        # A batch pads every modality to its own longest row, so its cost comes from the
        # per-modality maxima rather than from any single row. Rows are visited in order of
        # their own cost so that neighbors pad each other cheaply.
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
                budget=DEFAULT_CELL_BUDGET, n_heads=N_HEADS, *, modalities=None):
    """Pilot loaders with a transcript-row holdout drawn only from supplied training rows.
    The default 5% validation fraction follows the paper's stated fraction; the exact
    historical sampling procedure is not reconstructed here. Supplied test rows stay fixed.
    batch_size is a ceiling: a batch of long transcripts will be smaller.
    modalities controls token materialization only; all three streams still set the
    batch schedule, preserving the original optimizer updates and evaluation order."""
    modalities = _materialized_modalities(modalities)
    train = streams.split_rows("train")
    shuffled = np.random.default_rng(seed).permutation(train)
    cut = max(1, int(round(len(shuffled) * val_fraction)))
    parts = {"val": shuffled[:cut], "train": shuffled[cut:], "test": streams.split_rows("test")}
    made = {}
    for name, indices in parts.items():
        dataset = ExpressionDataset(streams, indices, modalities=modalities)
        # Keep the original cost grouping even when a fit reads fewer token streams.
        lengths = {m: np.asarray(streams.caches[m].lengths)[indices] for m in MODALITIES}
        sampler = BudgetedBatches(lengths, budget, batch_size,
                                  shuffle=(name == "train"), seed=seed, n_heads=n_heads)
        made[name] = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)
    return made


def builder(streams, variant, config, **kwargs):
    """The signature train.run expects, so the resume and checkpoint logic is shared."""
    return build(streams.widths, streams.n_tissues, variant, **kwargs)
