"""Per-token embeddings from frozen encoders, cached in resumable shards.

Stage 1 stored one mean-pooled vector per sequence (shared_code.embeddings). The
BioLangFusion heads read the full token sequence instead, so this module keeps the
final hidden state of every content token.

Three differences from the pooled cache matter:

* **Size.** A pooled cache is one vector per sequence; a token cache is one vector per
  token. The four BioLangFusion benchmarks come to roughly 34 GB in float16 across the
  three encoders, so the cache is written in shards and is never loaded whole.
* **Ragged rows.** Sequences differ in length, so a shard stores every token of its
  sequences in one array with a lengths vector, rather than a rectangular matrix.
* **Resumability.** Extraction is an overnight job on a laptop. Each shard is written
  atomically and skipped on a later run, so an interrupted run continues where it
  stopped instead of starting again.

Special tokens are removed, so a stored row holds exactly the tokens that correspond to
the sequence: one per nucleotide for a single-nucleotide encoder, one per residue for a
protein encoder, one per k-mer for a k-mer encoder. The codon-grid resampling that
BioLangFusion applies to those streams belongs to the fusion model, not to this cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..encoders import ENCODERS, encoder_input, load_encoder, protein_input
from ..sequences import rna_to_dna

# Bump when the shard layout or the stored calculation changes.
SHARD_VERSION = 1
DEFAULT_CHUNK = 256


class CacheRecoveryRequired(ValueError):
    """A cache must be preserved until its recorded identity can be verified."""


@dataclass
class CacheVerification:
    """Reuse SHA checks within a notebook session, invalidating on file-stat changes."""

    _sha256: dict = field(default_factory=dict, repr=False)

    def sha256(self, path):
        path = Path(path).resolve()
        before = path.stat()
        identity = (str(path), before.st_size, before.st_mtime_ns, before.st_ctime_ns,
                    before.st_ino)
        if identity not in self._sha256:
            digest = file_fingerprint(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
                raise ValueError(f"{path} changed during its SHA check; inspect it again.")
            self._sha256[identity] = digest
        return self._sha256[identity]


_VERIFICATION = CacheVerification()


@dataclass(frozen=True)
class CacheReadiness:
    """Read-only preflight result; a partial cache can compute only its missing shards."""

    cache_dir: Path
    status: str
    metadata: dict
    reusable_shards: tuple
    missing_shards: tuple
    issues: tuple = ()
    restore_metadata: bool = False
    provenance: str = "recorded_input"
    warnings: tuple = ()

    @property
    def ready(self):
        return not self.issues

    @property
    def summary(self):
        detail = "; ".join(self.issues) if self.issues else (
            f"{len(self.reusable_shards)} reusable, {len(self.missing_shards)} missing shards"
            + ("; metadata can be restored from historical evidence" if self.restore_metadata else ""))
        if self.warnings:
            detail += "; " + "; ".join(self.warnings)
        return f"{self.cache_dir}: {self.status}: {detail}"

    def require_ready(self):
        if not self.ready:
            raise CacheRecoveryRequired(self.summary + ". Preserve this cache. Supply its recorded "
                                        "historical metadata, ordered input SHA and shard SHA values "
                                        "as recovery_evidence, or choose a fresh cache directory for "
                                        "an explicitly authorized recomputation.")
        return self


def structural_token_ids(tokenizer):
    """The ids of tokens that frame a sequence rather than stand for part of it.

    Only the framing tokens are listed: start, separator, end, and padding. The unknown
    and mask ids are deliberately left out, because an unknown token sits where a real
    character was and dropping it would silently shorten the row.
    """
    named = ("cls_token_id", "bos_token_id", "sep_token_id", "eos_token_id", "pad_token_id")
    return {i for i in (getattr(tokenizer, name, None) for name in named) if i is not None}


@torch.no_grad()
def token_states(text, tokenizer, model, max_len, dtype=np.float16):
    """Return one sequence's content-token final hidden states as a (tokens, width) array.

    Framing tokens are found in the token ids the model actually consumed, not in the
    tokenizer's ``special_tokens_mask``. That mask cannot be trusted: Nucleotide
    Transformer v2 returns a mask one element longer than its own ``input_ids``, built as
    though a trailing end token were present when it is not, which silently misaligns
    every row if it is used as a filter.

    Raises when the input would be truncated: a truncated coding sequence loses its 3' end,
    and the streams would stop aligning on the codon grid.
    """
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"]
    if ids.shape[1] > max_len:
        raise ValueError(
            f"{ids.shape[1]} tokens exceeds the {max_len}-token limit for an input of "
            f"{len(text)} characters; the benchmark rows are filtered to fit, so this "
            "signals the wrong rows or the wrong encoder."
        )
    framing = structural_token_ids(tokenizer)
    keep = np.array([int(i) not in framing for i in ids[0]], dtype=bool)
    device = next(model.parameters()).device
    out = model(**{k: v.to(device) for k, v in encoded.items()})
    states = out[0][0].float().cpu().numpy()
    if states.shape[0] != keep.shape[0]:
        raise ValueError(
            f"the model returned {states.shape[0]} states for {keep.shape[0]} token ids; "
            "this encoder changes sequence length inside the model and needs its own handling."
        )
    return states[keep].astype(dtype, copy=False)


def input_fingerprint(texts):
    """Hash exact encoder input strings in order, including unambiguous row boundaries."""
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def file_fingerprint(path):
    """Hash a file in bounded memory, including large token-cache shards."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _meta_path(cache_dir):
    return Path(cache_dir) / "meta.json"


def _shard_path(cache_dir, index):
    return Path(cache_dir) / f"shard_{index:05d}.npz"


def _replace_with_retry(source, destination, *, preserve_destination=False):
    """Retry Windows sharing/lock errors when publishing an already closed file."""
    # Ten attempts, with at most 11.1 seconds of total waiting. Never repeat inference
    # or serialization, and let other I/O errors and cancellation propagate immediately.
    for attempt in range(10):
        if preserve_destination and Path(destination).exists():
            raise CacheRecoveryRequired(
                f"{destination} appeared before publication; preserve both files and inspect "
                "the cache before resuming.")
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (32, 33):
                raise
            if attempt == 9:
                if hasattr(error, "add_note"):
                    error.add_note(
                        f"Could not publish {source} -> {destination} after 10 attempts. "
                        "The file remained locked. Existing completed files were retained; "
                        "release the lock and rerun extraction to resume verified shards.")
                raise
            time.sleep(min(0.1 * 2 ** attempt, 2.0))


def _write_metadata(cache_dir, metadata):
    """Publish metadata atomically, retaining every existing shard on failure."""
    temporary = None
    primary_error = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir,
                                         prefix=".meta.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(metadata, handle, indent=2)
            handle.write("\n")
        _replace_with_retry(temporary, _meta_path(cache_dir))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(
                        f"Metadata temporary file retained at {temporary}: {cleanup_error}")


def tokens_for_input(encoder, text):
    """Expected content-token count for an already prepared encoder input, if fixed."""
    if encoder.modality == "protein":
        return len(text.replace(" ", ""))
    return tokens_for_length(encoder, len(text))


def _extraction_identity(encoder, texts, chunk_size, dtype):
    if not texts:
        raise ValueError("Token extraction requires at least one input sequence.")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    return {
        "shard_version": SHARD_VERSION,
        "encoder": encoder.key,
        "checkpoint": encoder.checkpoint,
        "revision": encoder.revision or "",
        "max_len": encoder.max_len,
        "dtype": np.dtype(dtype).name,
        "n_sequences": len(texts),
        "chunk_size": chunk_size,
        "n_shards": (len(texts) + chunk_size - 1) // chunk_size,
        "input_sha256": input_fingerprint(texts),
    }


def _shard_arrays(path):
    """Read bounded NPY headers and map stored arrays after checking payload sizes."""
    mapped = {}
    with zipfile.ZipFile(path) as archive, Path(path).open("rb") as handle:
        infos = archive.infolist()
        if sorted(info.filename for info in infos) != ["lengths.npy", "tokens.npy"]:
            raise ValueError("shard must contain exactly tokens.npy and lengths.npy")
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError("compressed shard requires separate recovery; expected np.savez layout")
            handle.seek(info.header_offset + 26)
            name_len, extra_len = np.frombuffer(handle.read(4), dtype="<u2")
            start = info.header_offset + 30 + int(name_len) + int(extra_len)
            handle.seek(start)
            major, _ = np.lib.format.read_magic(handle)
            if major not in (1, 2):
                raise ValueError(f"unsupported NPY format {major}")
            reader = (np.lib.format.read_array_header_1_0 if major == 1
                      else np.lib.format.read_array_header_2_0)
            shape, fortran_order, dtype = reader(handle)
            if fortran_order or dtype.hasobject:
                raise ValueError("shard arrays must be numeric and in C order")
            payload = int(np.prod(shape, dtype=object)) * dtype.itemsize
            if handle.tell() - start + payload != info.file_size:
                raise ValueError(f"{info.filename} payload size disagrees with its header")
            # A zero-token row array cannot be memory-mapped; its shape is still checked.
            mapped[info.filename[:-4]] = (np.memmap(path, dtype=dtype, mode="r",
                                                   offset=handle.tell(), shape=shape)
                                          if payload else np.empty(shape, dtype=dtype))
    return mapped


def _check_shard(path, encoder, texts, dtype, width=None):
    shard = _shard_arrays(path)
    tokens, lengths = shard["tokens"], np.asarray(shard["lengths"])
    if (tokens.ndim != 2 or tokens.shape[1] < 1 or tokens.dtype != np.dtype(dtype)
            or lengths.ndim != 1 or len(lengths) != len(texts) or lengths.dtype.kind not in "iu"):
        raise ValueError("token shape/dtype or number of length entries disagrees with extraction")
    if np.any(lengths < 0) or int(lengths.sum(dtype=np.int64)) != tokens.shape[0]:
        raise ValueError("lengths do not sum to the stored token count")
    if width is not None and tokens.shape[1] != width:
        raise ValueError("token width differs between shards")
    expected = [tokens_for_input(encoder, text) for text in texts]
    if any(count is not None and int(lengths[row]) != count
           for row, count in enumerate(expected)):
        raise ValueError("cached token lengths disagree with the current ordered inputs")
    return tokens.shape[1]


def _recorded_shas(record):
    shards = record.get("shards", [])
    if isinstance(shards, dict):
        return {name: value.get("sha256") if isinstance(value, dict) else value
                for name, value in shards.items()}
    return {item["name"]: item["sha256"] for item in shards}


def inspect_cache(encoder, texts, cache_dir, chunk_size=DEFAULT_CHUNK, dtype=np.float16,
                  recovery_evidence=None, verification=None, *, allow_legacy=False):
    """Inspect input identity and shard headers without loading an encoder or writing files.

    A legacy/orphan cache needs a historical cache record containing ``metadata``,
    ``input_sha256`` and ``shards`` (name/SHA pairs), as recorded in run identities.
    These records attest to old bytes; current inputs or current code cannot supply
    missing historical identity. SHA verification is reused only while file stats match.
    ``allow_legacy=True`` explicitly permits complete legacy caches after config/header/
    length checks, while flagging their input identity as unverified and leaving their
    metadata untouched. It never permits continuation of a partial legacy extraction.
    """
    cache_dir, texts = Path(cache_dir), list(texts)
    expected = _extraction_identity(encoder, texts, chunk_size, dtype)
    verification = verification if verification is not None else _VERIFICATION
    present = tuple(i for i in range(expected["n_shards"]) if _shard_path(cache_dir, i).exists())
    missing = tuple(i for i in range(expected["n_shards"]) if i not in present)
    metadata, issues, warnings = {}, [], []
    provenance = "new"
    exists = _meta_path(cache_dir).exists()
    try:
        if exists:
            metadata = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("meta.json must contain an object")
            different = [key for key, value in expected.items()
                         if key != "input_sha256" and metadata.get(key) != value]
            if different:
                raise ValueError("different extraction (differs in: " + ", ".join(different) + ")")
        extra = sorted(p.name for p in cache_dir.glob("shard_*.npz")
                       if p.name not in {_shard_path(cache_dir, i).name
                                         for i in range(expected["n_shards"])})
        if extra:
            raise ValueError("unexpected shards: " + ", ".join(extra))
        identified = "input_sha256" in metadata
        if identified:
            provenance = "recorded_input"
        if identified and metadata["input_sha256"] != expected["input_sha256"]:
            raise ValueError("different ordered inputs (input_sha256 differs)")
        shas = _recorded_shas(metadata)
        legacy_complete = (present and not missing and exists and not identified
                           and recovery_evidence is None and allow_legacy)
        if legacy_complete:
            provenance = "legacy_unverified"
            warnings.append("legacy input identity is UNVERIFIED: config/header/length compatibility "
                            "does not establish original input order or producer")
        elif present and (not identified or (recovery_evidence is not None and not shas)):
            if recovery_evidence is None:
                raise ValueError("existing shards lack verified ordered-input metadata")
            historical = recovery_evidence["metadata"]
            different = [key for key, value in expected.items() if key != "input_sha256"
                         and historical.get(key) != value]
            historical_input = recovery_evidence.get("input_sha256", historical.get("input_sha256"))
            if different or historical_input != expected["input_sha256"]:
                raise ValueError("historical evidence differs from current extraction/input order"
                                 + (" (" + ", ".join(different) + ")" if different else ""))
            if historical.get("input_sha256", historical_input) != historical_input:
                raise ValueError("historical input SHA values disagree")
            if metadata and any(historical.get(key) != value for key, value in metadata.items()):
                raise ValueError("existing metadata differs from the recorded historical metadata")
            shas = _recorded_shas(recovery_evidence)
            if any(not shas.get(_shard_path(cache_dir, i).name) for i in present):
                raise ValueError("historical evidence lacks SHA values for existing shards")
            metadata = {**historical, "input_sha256": historical_input,
                        "shards": [{"name": name, "sha256": sha} for name, sha in sorted(shas.items())],
                        "recovery_source": recovery_evidence.get("source", "historical cache record")}
            provenance = "historical_recovery"
        elif not exists and not present:
            metadata = {**expected, "extraction_code_sha256": verification.sha256(__file__), "shards": []}
        elif not identified:
            # No stored tokens exist, so no old bytes are being relabelled.
            metadata = {**metadata, **expected,
                        "extraction_code_sha256": verification.sha256(__file__), "shards": []}
        width = None
        for index in present:
            path = _shard_path(cache_dir, index)
            if path.name in shas and verification.sha256(path) != shas[path.name]:
                raise ValueError(f"{path.name}: SHA differs from recorded historical/cache bytes")
            chunk = texts[index * chunk_size:(index + 1) * chunk_size]
            width = _check_shard(path, encoder, chunk, dtype, width)
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile) as error:
        issues.append(str(error))
    status = ("recovery_required" if issues else "complete" if not missing
              else "partial" if present else "new")
    return CacheReadiness(cache_dir, status, metadata, present if not issues else (), missing,
                          tuple(issues), bool(present and not exists and not issues),
                          provenance, tuple(warnings))


def _write_shard(path, rows):
    """Write one shard atomically: all tokens in one array, plus each row's length."""
    tmp = path.with_name(path.name + ".partial")
    # Write through a file object: np.savez appends ".npz" to a name that lacks it.
    with open(tmp, "wb") as handle:
        np.savez(handle, tokens=np.concatenate(rows, axis=0),
                 lengths=np.array([len(r) for r in rows], dtype=np.int32))
    _replace_with_retry(tmp, path, preserve_destination=True)


def open_shard(path):
    """Memory-map a shard's arrays instead of reading them, and return them by name.

    ``np.load`` ignores ``mmap_mode`` for a ``.npz``: asking for one of its arrays reads the
    whole member, which is 180 MB for an RNA-FM shard and a fifth of a second. Training reads
    single rows tens of thousands of times, so the members are mapped in place instead. A
    shard written by ``np.savez`` is an uncompressed zip of ``.npy`` members, so each member's
    data begins at a known offset in the file and maps directly; a compressed archive has no
    such offset and is read the ordinary way.
    """
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
            with np.load(path) as loaded:
                return {name: loaded[name] for name in loaded.files}
        mapped = {}
        with open(path, "rb") as handle:
            for info in infos:
                handle.seek(info.header_offset + 26)
                name_len, extra_len = np.frombuffer(handle.read(4), dtype="<u2")
                handle.seek(info.header_offset + 30 + int(name_len) + int(extra_len))
                major, _ = np.lib.format.read_magic(handle)
                reader = (np.lib.format.read_array_header_1_0 if major == 1
                          else np.lib.format.read_array_header_2_0)
                shape, fortran_order, dtype = reader(handle)
                if fortran_order:
                    raise ValueError(f"{path} stores {info.filename} in Fortran order.")
                mapped[info.filename.removesuffix(".npy")] = np.memmap(
                    path, dtype=dtype, mode="r", offset=handle.tell(), shape=shape)
    return mapped


def extract(encoder, texts, cache_dir, device, chunk_size=DEFAULT_CHUNK, dtype=np.float16,
            progress=print, *, recovery_evidence=None, verification=None, allow_legacy=False):
    """Embed texts token by token into a resumable shard cache, and return its summary.

    texts are the strings this encoder reads, as Dataset.inputs_for or encoder_input
    supplies them. Compatible validated shards are kept, so calling this again after an
    interruption only does the work that is left. The encoder is loaded only when at
    least one shard is missing. Historical evidence can restore missing identity;
    allow_legacy explicitly accepts only complete legacy caches with unverified input
    identity, emits a warning, and preserves their metadata byte for byte.
    """
    texts = list(texts)
    verification = verification if verification is not None else _VERIFICATION
    plan = inspect_cache(encoder, texts, cache_dir, chunk_size, dtype,
                         recovery_evidence, verification, allow_legacy=allow_legacy).require_ready()
    if progress:
        for warning in plan.warnings:
            progress(f"    {encoder.key}: WARNING: {warning}")
    cache_dir = plan.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    metadata = dict(plan.metadata)
    # Bind identity before the first shard, so an interruption leaves a recoverable run.
    saved_metadata = (json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
                      if _meta_path(cache_dir).exists() else None)
    if (saved_metadata is None or plan.restore_metadata
            or (not plan.reusable_shards and metadata != saved_metadata)):
        _write_metadata(cache_dir, metadata)
    missing = plan.missing_shards
    if missing:
        tokenizer, model = load_encoder(encoder, device)
        primary_error = None
        try:
            model.eval()
            producers = dict(metadata.get("shard_extraction_code_sha256", {}))
            code_sha = verification.sha256(__file__)
            shas = _recorded_shas(metadata)
            width = None
            if plan.reusable_shards:
                first = plan.reusable_shards[0]
                width = _check_shard(_shard_path(cache_dir, first), encoder, chunks[first], dtype)
            for done, index in enumerate(missing, start=1):
                path = _shard_path(cache_dir, index)
                if path.exists():
                    raise CacheRecoveryRequired(f"{path} appeared after preflight; preserve it and rerun inspection.")
                rows = [token_states(t, tokenizer, model, encoder.max_len, dtype)
                        for t in chunks[index]]
                for text, row in zip(chunks[index], rows):
                    if (row.ndim != 2 or row.shape[1] < 1 or row.dtype != np.dtype(dtype)
                            or (width is not None and row.shape[1] != width)):
                        raise ValueError(f"{encoder.key}: computed token shape/dtype differs; "
                                         f"shard {index} was not written.")
                    if not np.isfinite(row).all():
                        raise ValueError(f"{encoder.key}: computed non-finite token values; "
                                         f"shard {index} was not written.")
                    width = row.shape[1]
                    expected_count = tokens_for_input(encoder, text)
                    if expected_count is not None and len(row) != expected_count:
                        raise ValueError(f"{encoder.key}: computed {len(row)} tokens, expected "
                                         f"{expected_count}; shard {index} was not written.")
                _write_shard(path, rows)
                _check_shard(path, encoder, chunks[index], dtype)
                shas[path.name] = verification.sha256(path)
                producers[path.name] = code_sha
                metadata["shards"] = [{"name": name, "sha256": sha}
                                      for name, sha in sorted(shas.items())]
                metadata["shard_extraction_code_sha256"] = producers
                _write_metadata(cache_dir, metadata)
                if progress:
                    progress(f"    {encoder.key}: shard {index + 1}/{len(chunks)} "
                             f"({done}/{len(missing)} this run)")
        except BaseException as error:
            primary_error = error
            raise
        finally:
            del model, tokenizer
            try:
                device_type = str(device).split(":", 1)[0]
                if device_type == "mps":
                    torch.mps.empty_cache()
                elif device_type == "cuda":
                    torch.cuda.empty_cache()
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(f"Token-cache GPU cleanup also failed: {cleanup_error}")
    if metadata != json.loads(_meta_path(cache_dir).read_text(encoding="utf-8")):
        _write_metadata(cache_dir, metadata)
    return summarize(cache_dir), bool(missing)


@dataclass
class TokenCache:
    """One encoder's token embeddings for one benchmark, read shard by shard.

    Indexing returns a (tokens, width) array for one sequence. Shards are memory-mapped
    and the most recent one is held open, so reading rows in order touches each shard
    once; random access works but rereads.
    """

    cache_dir: Path
    meta: dict
    lengths: np.ndarray
    shard_of: np.ndarray
    row_in_shard: np.ndarray
    row_start: np.ndarray
    _mapped: dict = field(default_factory=dict, repr=False)

    def __len__(self):
        return len(self.lengths)

    @property
    def width(self):
        return int(self.meta["width"])

    def tokens_of_shard(self, index):
        """The token array of one shard, mapped on first use and kept for later rows."""
        mapped = self._mapped.get(index)
        if mapped is None:
            mapped = open_shard(_shard_path(self.cache_dir, index))["tokens"]
            self._mapped[index] = mapped
        return mapped

    def close(self):
        """Release the mapped shards. One file stays open per shard until this is called."""
        self._mapped.clear()

    def __getitem__(self, i):
        start = int(self.row_start[i])
        return np.asarray(
            self.tokens_of_shard(int(self.shard_of[i]))[start:start + int(self.lengths[i])])


def pooled_matrix(cache, progress=None, progress_every=5000):
    """Mean-pool every row of a token cache into one vector, giving a (rows, width) matrix.

    Stage 1's pooled cache and these token shards hold the same encoder reading different
    row sets: Stage 1 embedded its own sample, these hold a benchmark's rows in file order.
    Pooling the shards is how a pooled vector for the benchmark's rows is obtained without
    loading an encoder again, which is what a probe comparison against LucaOne needs: the
    same rows, the same frozen weights, the same pooling, and no second extraction.

    Rows are read in order, so each shard is touched once.
    """
    matrix = np.empty((len(cache), cache.width), dtype=np.float32)
    for i in range(len(cache)):
        matrix[i] = np.asarray(cache[i], dtype=np.float32).mean(axis=0)
        if progress and progress_every and (i + 1) % progress_every == 0:
            progress(f"    pooled {i + 1}/{len(cache)}")
    cache.close()
    return matrix


def summarize(cache_dir):
    """Read the shard lengths and return a TokenCache describing the whole extraction."""
    cache_dir = Path(cache_dir)
    meta = json.loads(_meta_path(cache_dir).read_text()) if _meta_path(cache_dir).exists() else {}
    lengths, shard_of, row_in_shard, row_start, width = [], [], [], [], None
    for index in range(meta.get("n_shards", 0)):
        path = _shard_path(cache_dir, index)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; rerun extract to finish this cache.")
        shard = open_shard(path)
        counts = np.asarray(shard["lengths"])
        width = shard["tokens"].shape[1]
        lengths.append(counts)
        shard_of.append(np.full(len(counts), index, dtype=np.int32))
        row_in_shard.append(np.arange(len(counts), dtype=np.int32))
        row_start.append(np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64))
    lengths = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int32)
    meta = {**meta, "width": int(width or 0),
            "total_tokens": int(lengths.sum()),
            "bytes": int(lengths.sum()) * int(width or 0) * np.dtype(meta.get("dtype", "float16")).itemsize}
    return TokenCache(cache_dir, meta, lengths,
                      np.concatenate(shard_of) if shard_of else np.zeros(0, dtype=np.int32),
                      np.concatenate(row_in_shard) if row_in_shard else np.zeros(0, dtype=np.int32),
                      np.concatenate(row_start) if row_start else np.zeros(0, dtype=np.int64))


# Nucleotides per token for the encoders that consume a fixed number of them. A byte-pair
# vocabulary such as DNABERT-2's has no fixed rate and is deliberately absent.
NUCLEOTIDES_PER_TOKEN = {
    "rnafm": 1, "rinalmo": 1, "hyenadna": 1,     # one token per nucleotide
    "mrnafm": 3, "calm": 3,                      # one token per codon
}

# k-mer encoders, which tokenize greedily and fall back to single nucleotides for whatever
# does not fill a final k-mer: 999 nucleotides is 166 six-mers plus 3 single tokens, 169 in
# all, not the 167 a plain division would suggest.
KMER_SIZE = {"ntv2_100m": 6, "nt500m": 6}


def expected_tokens(encoder, seq):
    """Tokens one coding sequence contributes to an encoder's stream.

    Taken from the coding sequence rather than from the string the encoder reads, so that a
    spaced protein alphabet (ProtBERT) and a k-mer vocabulary are both counted correctly. A
    protein encoder sees the grid-aligned translation, one token per codon. Returns None for
    an encoder whose tokenization has no fixed rate, such as DNABERT-2's byte-pair vocabulary.
    """
    return tokens_for_length(encoder, len(seq))


def tokens_for_length(encoder, nucleotides):
    """expected_tokens for a coding sequence of the given length."""
    if encoder.modality == "protein":
        return nucleotides // 3
    k = KMER_SIZE.get(encoder.key)
    if k is not None:
        return nucleotides // k + nucleotides % k
    per_token = NUCLEOTIDES_PER_TOKEN.get(encoder.key)
    return None if per_token is None else -(-nucleotides // per_token)


def grid_resolution(encoder_key):
    """How one encoder's stream reaches the codon grid: "kmer", "nucleotide", or "codon".

    kmer: one token per six nucleotides, upsampled by the learned transposed convolution
    (Nucleotide Transformer v2 and 500M). nucleotide: one token per nucleotide, averaged in
    threes (RNA-FM, RiNALMo, HyenaDNA). codon: one token per codon or residue, used as it is
    (mRNA-FM, CaLM, and every protein encoder). An encoder with no fixed rate, DNABERT-2's
    byte-pair vocabulary, raises: placing it on the grid needs an adapter of its own.
    """
    encoder = ENCODERS[encoder_key]
    if encoder.modality == "protein":
        return "codon"
    if KMER_SIZE.get(encoder_key) == 6:
        return "kmer"
    per_token = NUCLEOTIDES_PER_TOKEN.get(encoder_key)
    if per_token in (1, 3):
        return "nucleotide" if per_token == 1 else "codon"
    raise ValueError(f"{encoder_key} has no fixed number of nucleotides per token, so the "
                     "codon grid cannot place its stream without a dedicated adapter.")


def grid_protein(seq):
    """Translate every complete codon, so the stream has exactly len(seq)//3 tokens.

    Stage 1's translation (sequences.translate_cds) stops at the first stop codon, which
    is right for a pooled vector of a protein but wrong for the codon grid: the grid needs
    one protein token per codon, and a clean coding sequence would otherwise be one token
    short while a fragment with an internal stop would be far shorter. Stop codons become
    "X", which every protein encoder's vocabulary accepts.

    This matters most on CoV-Vac, whose 81-nucleotide fragments are cut from a construct
    rather than being coding sequences: under the stop-at-first-stop rule only 0.7% of them
    translate to a full reading frame and 6.2% translate to nothing at all. The choice is
    recorded as a deviation in Code/stage3/01_blf_datasets.ipynb; BioLangFusion does not
    state how it fed such fragments to its protein encoder.
    """
    from Bio.Seq import Seq

    trimmed = seq[: len(seq) - (len(seq) % 3)]
    return str(Seq(rna_to_dna(trimmed)).translate()).replace("*", "X")


def inputs_for_rows(encoder, sequences):
    """The strings one encoder reads for a list of coding sequences, in row order.

    Nucleotide encoders read the sequence in their own alphabet, as in Stage 1. Protein
    encoders read the grid-aligned translation of grid_protein rather than Stage 1's
    stop-at-first-stop translation, so that every stream has a token count the codon grid
    can resample.
    """
    if encoder.modality != "protein":
        return [encoder_input(encoder, s) for s in sequences]
    return [protein_input(encoder, grid_protein(s)) for s in sequences]
