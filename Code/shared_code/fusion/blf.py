"""BioLangFusion assembled: the cached token streams, the codon grid, a head, a prediction.

This implementation follows the paper's overall structure: resample the three streams
onto the codon grid, fuse them one of three ways, and read the fused stream with a shared
TextCNN architecture. Here, unimodal baselines use that head on one stream at its native
token resolution. This is our explicit baseline choice; the paper does not fully specify
the corresponding unimodal input treatment. The prediction head remains trainable.

Reading the streams is the part the paper does not describe. The caches written in
Code/stage3/02_token_embeddings.ipynb hold one row per benchmark sequence, in the file's
own order, so a row index selects the same sequence in all three caches and in the
benchmark frame. Rows are padded to the longest in their batch, each stream on its own
time axis, and the number of codons in each row travels with the batch: it is what the
grid resamples to and what the mask is built from.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .. import datasets as datasets_module
from ..encoders import ENCODERS
from .codon_grid import CODON_LENGTH, CodonGrid, mask_of
from .heads import FUSIONS, SHARED_WIDTH
from .textcnn import DEFAULT_CHANNELS, TextCNN
from .token_embeddings import file_fingerprint, input_fingerprint, inputs_for_rows, summarize

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
    allow_legacy_cache: bool = False
    _file_hashes: dict = field(default_factory=dict, init=False, repr=False)

    @classmethod
    def open(cls, key, cache_root, encoder_of_modality=None, allow_legacy_cache=False):
        encoder_of_modality = encoder_of_modality or ENCODER_OF_MODALITY
        spec = datasets_module.DATASETS[key]
        rows = datasets_module.blf_benchmark_rows(key)
        caches = {}
        try:
            for modality, encoder in encoder_of_modality.items():
                cache_dir = Path(cache_root) / key / encoder
                metadata_path = cache_dir / "meta.json"
                # Check availability before interpreting an absent cache as zero rows.
                if not metadata_path.is_file():
                    raise FileNotFoundError(
                        f"Token-cache metadata is missing: {metadata_path.resolve()}. "
                        "Complete Step 2 (02_token_embeddings.ipynb) for this benchmark, "
                        "or point CACHE_ROOT to an existing complete token cache.")
                try:
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON in token-cache metadata: "
                                     f"{metadata_path.resolve()}.") from error
                if not isinstance(metadata, dict):
                    raise ValueError(f"Token-cache metadata must be a JSON object: "
                                     f"{metadata_path.resolve()}.")
                shard_count = metadata.get("n_shards")
                if type(shard_count) is not int or shard_count <= 0:
                    raise ValueError(f"Token-cache metadata requires a positive integer "
                                     f"n_shards: {metadata_path.resolve()}.")
                caches[modality] = summarize(cache_dir)
            streams = cls(key, rows, caches, spec.task, spec.label_column, spec.split_column,
                          allow_legacy_cache=allow_legacy_cache)
            streams.check()
            status = streams.check_input_identity()
        except BaseException:
            for cache in caches.values():
                cache.close()
            raise
        if status == "unverified_legacy":
            warnings.warn("Legacy token caches accepted explicitly: their original input order "
                          "is unverified. New run identities do not establish that provenance.",
                          RuntimeWarning, stacklevel=2)
        return streams

    def check_input_identity(self):
        """Compare current effective inputs with producer-recorded hashes, when present."""
        status = "recorded_inputs"
        for modality, cache in self.caches.items():
            recorded = cache.meta.get("input_sha256")
            if recorded is None:
                if not self.allow_legacy_cache:
                    raise ValueError(f"{self.key}/{modality}: legacy cache has no ordered-input "
                                     "identity. Use a new extraction directory, or explicitly "
                                     "open with allow_legacy_cache=True for unverified use.")
                status = "unverified_legacy"
                continue
            encoder = ENCODERS[cache.meta["encoder"]]
            expected = input_fingerprint(inputs_for_rows(encoder, self.rows["sequence"]))
            if recorded != expected:
                raise ValueError(f"{self.key}/{modality}: cached inputs differ from current "
                                 "sequences or row order; equal lengths are not sufficient.")
        return status

    def identity(self):
        """Current rows plus exact cache files; legacy provenance stays explicitly unknown.

        Large shards are hashed once per open object. A size/mtime change requires reopening
        the streams, because an existing memory map may otherwise refer to old content.
        Treat caches as read-only while streams are open; this is not a file lock.
        """
        self.check()
        input_status = self.check_input_identity()
        columns = ["sequence", "length", self.label_column, self.split_column]
        row_texts = (json.dumps(row, ensure_ascii=False, separators=(",", ":"),
                               default=lambda value: value.item())
                     for row in self.rows[columns].itertuples(index=False, name=None))
        cache_files = {}
        for modality, cache in self.caches.items():
            paths = [cache.cache_dir / "meta.json"]
            paths += [cache.cache_dir / f"shard_{index:05d}.npz"
                      for index in range(cache.meta["n_shards"])]
            hashes = {}
            for path in paths:
                stat = path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
                previous = self._file_hashes.get(path)
                if previous is not None and previous[0] != signature:
                    raise RuntimeError(f"{path} changed while streams were open; close and reopen.")
                if previous is None:
                    digest = file_fingerprint(path)
                    after = path.stat()
                    if (after.st_size, after.st_mtime_ns) != signature:
                        raise RuntimeError(f"{path} changed while being fingerprinted.")
                    self._file_hashes[path] = (signature, digest)
                hashes[path.name] = self._file_hashes[path][1]
            cache_files[modality] = hashes
        return {"benchmark": self.key, "task": self.task, "row_columns": columns,
                "rows_sha256": input_fingerprint(row_texts), "cache_files": cache_files,
                "input_identity_status": input_status}

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
