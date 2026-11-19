"""Per-entity token embeddings for the partner tasks, cached in resumable shards.

``token_embeddings`` caches one row per *row of a benchmark*: three streams of the same
coding sequence, each long enough to be resampled onto the codon grid, keyed by file order.
The partner tasks need something else. A pair task names two entities per row, the same
entity appears in hundreds of rows, and the two sides are different molecules that share no
grid at all: a 201 bp DNA window against a transcription factor, a transcript against an
RNA-binding protein, a kinase against a SMILES string. Embedding per row would embed the
same protein hundreds of times, and there is no grid to align the sides onto.

So this module caches one row per *entity*, keyed by name:

* **One embedding per entity.** Names are embedded in sorted order and the cache records
  that order, so ``tokens("AAK1")`` is a lookup rather than a scan.
* **No codon grid.** Each side keeps its own token count. The fusion heads already handle
  two streams of different lengths through masking, which is what makes a pair task
  possible in the first place.
* **Truncation is allowed, and recorded.** The grid caches refuse to truncate, because a
  truncated coding sequence loses its 3' end and the streams stop aligning. Here there is
  nothing to align, and a 1,400-residue protein against a 1,024-token encoder has to be cut
  somewhere. Every truncated entity is named in the cache metadata so a result can say how
  many of its entities were not read whole.
* **Resumable.** Shards are written atomically and validated on a later run, so an
  interrupted extraction continues where it stopped.

The cache satisfies ``pair_training.PairEmbeddings``, so a built cache is passed straight to
``pair_training.fit``.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..encoders import ENCODERS, free_memory, load_encoder, protein_input
from ..sequences import dna_to_rna, rna_to_dna
from . import pair_tasks
from .pair_training import PairEmbeddings
from .token_embeddings import open_shard

# Bump when the shard layout or the stored calculation changes.
SCHEMA = "entity-embeddings.v1"
DEFAULT_CHUNK = 128

# The encoder each modality is read with unless a run names another one. RiNALMo and
# HyenaDNA are the encoders the three-seed cohort found the stability gains under, ESM-2
# 150M is the protein encoder of that panel, and ChemBERTa is the only ligand encoder here.
MODALITY_DEFAULTS = {"dna": "hyenadna", "rna": "rinalmo",
                     "protein": "esm2_150m", "ligand": "chemberta"}

# The encoder alphabets each modality may be read with. A protein side may go through either
# protein form; a ligand side only through a chemical one.
MODALITY_ALPHABETS = {"dna": ("dna",), "rna": ("rna",),
                      "protein": ("protein", "protein_spaced"), "ligand": ("smiles",)}


class EntityCacheMismatch(ValueError):
    """An existing cache was built from different inputs or a different encoder."""


def encoder_for(modality, key=None):
    """The Encoder one side is read with, checked against what that side holds.

    A pair task's entities are already in their own alphabet: a protein side holds amino
    acids, not a coding sequence to translate. This refuses an encoder that would read the
    side as something it is not, which is the mistake that produces a cache of embeddings of
    nonsense without ever raising.
    """
    if modality not in pair_tasks.MODALITIES:
        raise ValueError(f"{modality!r} is not one of {pair_tasks.MODALITIES}")
    key = key or MODALITY_DEFAULTS[modality]
    if key not in ENCODERS:
        raise ValueError(f"unknown encoder {key!r}")
    encoder = ENCODERS[key]
    allowed = MODALITY_ALPHABETS[modality]
    if encoder.alphabet not in allowed:
        raise ValueError(f"{key} reads a {encoder.alphabet} input, which cannot be a "
                         f"{modality} entity; that side needs an encoder whose alphabet is "
                         f"one of {allowed}")
    if encoder.codon:
        raise ValueError(f"{key} reads one codon per token, which needs an in-frame coding "
                         "sequence; a pair task's entities are not coding sequences")
    return encoder


def entity_input(encoder, sequence):
    """The string one encoder reads for one entity, which is never translated.

    ``encoders.encoder_input`` exists for retained coding sequences and translates a protein
    stream out of the nucleotides. A pair task's sides are separate molecules already in
    their own alphabet, so each is handed over as it stands, in the form its encoder reads.
    """
    sequence = sequence.strip()
    if not sequence:
        raise ValueError("an entity cannot have an empty sequence")
    if encoder.alphabet == "dna":
        return rna_to_dna(sequence).upper()
    if encoder.alphabet == "rna":
        return dna_to_rna(sequence).upper()
    if encoder.alphabet == "smiles":
        return sequence
    return protein_input(encoder, sequence.upper())


def inputs_for(encoder, sequences):
    """Every entity's encoder input, in sorted name order, with that order returned too."""
    names = sorted(sequences)
    return names, [entity_input(encoder, sequences[name]) for name in names]


def identity_fingerprint(names, texts):
    """Hash the exact (name, input) pairs in order, with unambiguous boundaries.

    The name is hashed alongside the string: two entities with identical sequences are still
    two entities, and a cache keyed by name is only reusable if the names match too.
    """
    digest = hashlib.sha256()
    for name, text in zip(names, texts):
        for part in (name.encode("utf-8"), text.encode("utf-8")):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _meta_path(cache_dir):
    return Path(cache_dir) / "entities.json"


def _shard_path(cache_dir, index):
    return Path(cache_dir) / f"shard-{index:05d}.npz"


def _identity(encoder, modality, names, texts, chunk_size, dtype, max_tokens):
    if not names:
        raise ValueError("an entity cache needs at least one entity")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    return {"schema": SCHEMA, "modality": modality, "encoder": encoder.key,
            "checkpoint": encoder.checkpoint, "revision": encoder.revision or "",
            "max_len": encoder.max_len, "max_tokens": int(max_tokens),
            "dtype": np.dtype(dtype).name, "n_entities": len(names),
            "chunk_size": chunk_size,
            "n_shards": (len(names) + chunk_size - 1) // chunk_size,
            "input_sha256": identity_fingerprint(names, texts)}


def token_limit(encoder, max_tokens=None):
    """How many content tokens one entity may keep.

    An encoder's ``max_len`` counts the framing tokens it adds, so the content limit is two
    below it. HyenaDNA's million-token window is far past anything a pair task holds, and a
    limit that large would make the padded batches unusable, so an explicit cap is honoured
    whenever it is the smaller of the two.
    """
    inherent = max(int(encoder.max_len) - 2, 1)
    return inherent if max_tokens is None else min(inherent, int(max_tokens))


def _states(text, tokenizer, model, limit, dtype):
    """One entity's content-token states, truncated to ``limit`` tokens if it is longer.

    Framing tokens are found in the ids the model actually consumed rather than in the
    tokenizer's ``special_tokens_mask``, for the reason ``token_embeddings.token_states``
    gives: at least one encoder here returns a mask that does not line up with its own ids.
    """
    import torch

    from .token_embeddings import structural_token_ids

    with torch.no_grad():
        encoded = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=limit + 2)
        ids = encoded["input_ids"]
        framing = structural_token_ids(tokenizer)
        keep = np.array([int(i) not in framing for i in ids[0]], dtype=bool)
        device = next(model.parameters()).device
        out = model(**{k: v.to(device) for k, v in encoded.items()})
        states = out[0][0].float().cpu().numpy()
    if states.shape[0] != keep.shape[0]:
        raise ValueError(
            f"the model returned {states.shape[0]} states for {keep.shape[0]} token ids; "
            "this encoder changes sequence length inside the model and needs its own "
            "handling")
    return states[keep][:limit].astype(dtype, copy=False)


def _write_shard(path, rows):
    """Write one shard atomically: every token in one array, plus each row's length."""
    tmp = Path(path).with_name(Path(path).name + ".partial")
    with open(tmp, "wb") as handle:
        np.savez(handle, tokens=np.concatenate(rows, axis=0),
                 lengths=np.array([len(r) for r in rows], dtype=np.int32))
    os.replace(tmp, path)


def _shard_is_valid(path, expected_rows, width=None):
    """Whether a shard on disk holds the rows this plan expects of it."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            if {"tokens.npy", "lengths.npy"} - set(archive.namelist()):
                return False
        mapped = open_shard(path)
        lengths = np.asarray(mapped["lengths"])
        tokens = mapped["tokens"]
        if len(lengths) != expected_rows:
            return False
        if int(lengths.sum()) != int(tokens.shape[0]):
            return False
        return width is None or int(tokens.shape[1]) == int(width)
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def _write_metadata(cache_dir, metadata):
    """Replace the metadata file atomically, so an interrupted write leaves the old one."""
    cache_dir = Path(cache_dir)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir,
                                         prefix="entities-", suffix=".partial", delete=False)
    try:
        with handle:
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        os.replace(handle.name, _meta_path(cache_dir))
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def inspect(sequences, cache_dir, *, modality, encoder=None, chunk_size=DEFAULT_CHUNK,
            dtype=np.float16, max_tokens=None):
    """What an extraction would have to do, without loading an encoder or writing anything.

    Returns the planned identity, the shards still missing, and the stored metadata if a
    cache is already there. A stored cache whose identity differs from the plan is reported
    rather than overwritten, because its embeddings may be the ones a finished result was
    computed from.
    """
    cache_dir = Path(cache_dir)
    chosen = encoder_for(modality, encoder)
    names, texts = inputs_for(chosen, sequences)
    limit = token_limit(chosen, max_tokens)
    planned = _identity(chosen, modality, names, texts, chunk_size, dtype, limit)
    stored = None
    if _meta_path(cache_dir).exists():
        stored = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
        differing = sorted(k for k, v in planned.items() if stored.get(k) != v)
        if differing:
            return {"status": "mismatch", "cache_dir": cache_dir, "identity": planned,
                    "stored": stored, "differing": differing, "names": names,
                    "texts": texts, "encoder": chosen, "limit": limit, "missing": []}
    counts = [len(names[i:i + chunk_size]) for i in range(0, len(names), chunk_size)]
    if stored is None or "width" not in stored:
        # Shard files without the metadata that says what is in them are not a cache: the
        # names, lengths and width live only in that file, so there is nothing to read by.
        missing = list(range(len(counts)))
    else:
        missing = [i for i, rows in enumerate(counts)
                   if not _shard_is_valid(_shard_path(cache_dir, i), rows, stored["width"])]
    status = "complete" if not missing else ("partial" if stored else "absent")
    return {"status": status, "cache_dir": cache_dir, "identity": planned, "stored": stored,
            "differing": [], "names": names, "texts": texts, "encoder": chosen,
            "limit": limit, "missing": missing, "shard_rows": counts}


def extract(sequences, cache_dir, device, *, modality, encoder=None,
            chunk_size=DEFAULT_CHUNK, dtype=np.float16, max_tokens=None, progress=print):
    """Embed one side's entities into a resumable shard cache, and return its summary.

    ``sequences`` maps an entity name to its string, as a ``PairFrame`` side holds it. The
    encoder is loaded only when a shard is actually missing, so calling this again on a
    complete cache costs nothing but the validation.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    plan = inspect(sequences, cache_dir, modality=modality, encoder=encoder,
                   chunk_size=chunk_size, dtype=dtype, max_tokens=max_tokens)
    if plan["status"] == "mismatch":
        raise EntityCacheMismatch(
            f"{cache_dir} was built from different inputs: "
            f"{', '.join(plan['differing'])} differ. Point this run at another directory, or "
            "delete that cache if nothing was computed from it.")

    names, texts, chosen, limit = plan["names"], plan["texts"], plan["encoder"], plan["limit"]
    metadata = dict(plan["stored"] or {})
    metadata.update(plan["identity"])
    if plan["status"] == "complete":
        if progress:
            progress(f"  {cache_dir.name}: {len(names)} entities already cached")
        return summarize(cache_dir)

    truncated = dict(metadata.get("truncated", {}))
    lengths = list(metadata.get("lengths", [])) or [0] * len(names)
    if len(lengths) != len(names):
        lengths = [0] * len(names)
    tokenizer = model = None
    try:
        for index in plan["missing"]:
            if tokenizer is None:
                if progress:
                    progress(f"  loading {chosen.key} on {device}")
                tokenizer, model = load_encoder(chosen, device)
            start = index * chunk_size
            block = list(range(start, min(start + chunk_size, len(names))))
            rows = []
            for position in block:
                row = _states(texts[position], tokenizer, model, limit, dtype)
                if len(row) == 0:
                    raise ValueError(f"{names[position]} produced no content tokens; its "
                                     "string is not something this encoder reads")
                rows.append(row)
                lengths[position] = int(len(row))
                if len(row) == limit:
                    # Either exactly at the limit or cut at it; both are recorded, because a
                    # result has to be able to say how much of each side was read.
                    truncated[names[position]] = int(len(row))
            _write_shard(_shard_path(cache_dir, index), rows)
            metadata["width"] = int(rows[0].shape[1])
            metadata["lengths"] = lengths
            metadata["truncated"] = truncated
            metadata["names"] = names
            _write_metadata(cache_dir, metadata)
            if progress:
                progress(f"  {cache_dir.name}: shard {index + 1}/{metadata['n_shards']} "
                         f"({block[-1] + 1}/{len(names)} entities)")
    finally:
        if model is not None:
            del model, tokenizer
            free_memory()
    metadata["names"] = names
    metadata["lengths"] = lengths
    metadata["truncated"] = truncated
    _write_metadata(cache_dir, metadata)
    return summarize(cache_dir)


def summarize(cache_dir):
    """A built cache's metadata plus the two numbers a write-up quotes."""
    metadata = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
    lengths = np.asarray(metadata.get("lengths", []), dtype=np.int64)
    truncated = metadata.get("truncated", {})
    return {"cache_dir": str(cache_dir), "modality": metadata["modality"],
            "encoder": metadata["encoder"], "revision": metadata["revision"],
            "entities": int(metadata["n_entities"]), "width": int(metadata.get("width", 0)),
            "token_limit": int(metadata["max_tokens"]),
            "tokens": {"total": int(lengths.sum()), "min": int(lengths.min()) if len(lengths) else 0,
                       "median": float(np.median(lengths)) if len(lengths) else 0.0,
                       "max": int(lengths.max()) if len(lengths) else 0},
            "at_token_limit": len(truncated),
            "at_token_limit_share": round(len(truncated) / max(int(metadata["n_entities"]), 1), 4),
            "input_sha256": metadata["input_sha256"]}


@dataclass
class EntityCache(PairEmbeddings):
    """One side's token embeddings, read by entity name.

    ``tokens(name)`` returns a ``(length, width)`` tensor. Shards are memory-mapped on first
    use and kept, so reading a whole task touches each shard once; entities are looked up in
    pair order, which is random, so a side whose cache does not fit in memory will reread.
    """

    cache_dir: Path
    meta: dict
    index: dict
    lengths: np.ndarray
    shard_of: np.ndarray
    row_in_shard: np.ndarray
    row_start: np.ndarray
    width: int = field(init=False)
    _mapped: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.width = int(self.meta["width"])

    def __len__(self):
        return len(self.lengths)

    @property
    def names(self):
        return list(self.meta["names"])

    @property
    def modality(self):
        return self.meta["modality"]

    @property
    def encoder(self):
        return self.meta["encoder"]

    @property
    def truncated(self):
        """The entities that were cut at the token limit, with the length each kept."""
        return dict(self.meta.get("truncated", {}))

    def missing(self, names):
        """Which of these entity names this cache does not hold."""
        return sorted(set(names) - set(self.index))

    def tokens_of_shard(self, shard):
        mapped = self._mapped.get(shard)
        if mapped is None:
            mapped = open_shard(_shard_path(self.cache_dir, shard))["tokens"]
            self._mapped[shard] = mapped
        return mapped

    def close(self):
        """Release the mapped shards. One file stays open per shard until this is called."""
        self._mapped.clear()

    def row(self, name):
        """One entity's embedding as a ``(length, width)`` array.

        The returned array is a copy rather than a view of the mapped shard. A shard is mapped
        read-only, and handing a caller a read-only view of it means every tensor built from it
        is read-only too, which PyTorch accepts with a warning and then cannot be written to.
        An entity is read once per batch and is small, so the copy is cheaper than the class of
        bug it removes.
        """
        try:
            position = self.index[name]
        except KeyError:
            raise KeyError(f"{self.cache_dir.name} holds no entity named {name!r}") from None
        start = int(self.row_start[position])
        shard = self.tokens_of_shard(int(self.shard_of[position]))
        return np.array(shard[start:start + int(self.lengths[position])])

    def tokens(self, name):
        import torch

        return torch.as_tensor(self.row(name), dtype=torch.float32)


def load(cache_dir):
    """Open a built entity cache for reading."""
    import torch  # noqa: F401  (fail here rather than on the first lookup)

    cache_dir = Path(cache_dir)
    metadata = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"{cache_dir}: unsupported entity cache schema "
                         f"{metadata.get('schema')!r}")
    names = list(metadata["names"])
    lengths = np.asarray(metadata["lengths"], dtype=np.int64)
    if len(names) != len(lengths) or len(names) != int(metadata["n_entities"]):
        raise ValueError(f"{cache_dir}: metadata names and lengths disagree; the cache is "
                         "incomplete and has to be extracted again")
    chunk = int(metadata["chunk_size"])
    positions = np.arange(len(names))
    shard_of = positions // chunk
    row_in_shard = positions % chunk
    row_start = np.zeros(len(names), dtype=np.int64)
    for shard in np.unique(shard_of):
        rows = np.flatnonzero(shard_of == shard)
        row_start[rows] = np.concatenate([[0], np.cumsum(lengths[rows])[:-1]])
    for shard in np.unique(shard_of):
        rows = np.flatnonzero(shard_of == shard)
        if not _shard_is_valid(_shard_path(cache_dir, int(shard)), len(rows),
                               metadata.get("width")):
            raise ValueError(f"{cache_dir}: shard {int(shard)} is missing or does not hold "
                             "the rows its metadata claims; extract the cache again")
    return EntityCache(cache_dir, metadata, {name: i for i, name in enumerate(names)},
                       lengths, shard_of, row_in_shard, row_start)


def cache_dir_for(root, frame, side, encoder_key):
    """Where one side of one table is cached, named so two encoders never collide."""
    return Path(root) / frame.name / f"{side}__{frame.modality(side)}__{encoder_key}"


def build(frame, root, device, *, encoders=None, max_tokens=None, chunk_size=DEFAULT_CHUNK,
          dtype=np.float16, progress=print):
    """Extract or open both sides of one pair table, ready to hand to ``pair_training.fit``.

    ``encoders`` names the encoder per side, defaulting to the modality's own. Returns the
    two caches keyed by side, and the two summaries, so a results file can record which
    encoder read which side and how many entities hit the token limit.
    """
    encoders = dict(encoders or {})
    # One cap for both sides, or one per side.
    caps = dict(max_tokens) if isinstance(max_tokens, dict) else {
        side: max_tokens for side in pair_tasks.SIDES}
    caches, summaries = {}, {}
    for side in pair_tasks.SIDES:
        modality = frame.modality(side)
        chosen = encoder_for(modality, encoders.get(side))
        directory = cache_dir_for(root, frame, side, chosen.key)
        if progress:
            progress(f"{frame.name} {side} ({modality}) through {chosen.key}")
        summaries[side] = extract(frame.sequences(side), directory, device,
                                  modality=modality, encoder=chosen.key,
                                  chunk_size=chunk_size, dtype=dtype,
                                  max_tokens=caps.get(side), progress=progress)
        caches[side] = load(directory)
        absent = caches[side].missing(frame.pairs[side])
        if absent:
            raise ValueError(f"{frame.name}: the {side} cache is missing "
                             f"{len(absent)} entities the pair table names, starting with "
                             f"{absent[:3]}")
    return caches, summaries
