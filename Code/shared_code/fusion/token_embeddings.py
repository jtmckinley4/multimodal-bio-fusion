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


def _write_shard(path, rows):
    """Write one shard atomically: all tokens in one array, plus each row's length."""
    tmp = path.with_name(path.name + ".partial")
    # Write through a file object: np.savez appends ".npz" to a name that lacks it.
    with open(tmp, "wb") as handle:
        np.savez(handle, tokens=np.concatenate(rows, axis=0),
                 lengths=np.array([len(r) for r in rows], dtype=np.int32))
    os.replace(tmp, path)


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
            progress=print):
    """Embed texts token by token into a resumable shard cache, and return its summary.

    texts are the strings this encoder reads, as Dataset.inputs_for or encoder_input
    supplies them. Shards already on disk are kept, so calling this again after an
    interruption only does the work that is left. The encoder is loaded only when at
    least one shard is missing.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = list(texts)
    if not texts:
        raise ValueError("Token extraction requires at least one input sequence.")

    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    expected = {
        "shard_version": SHARD_VERSION,
        "encoder": encoder.key,
        "checkpoint": encoder.checkpoint,
        "revision": encoder.revision or "",
        "max_len": encoder.max_len,
        "dtype": np.dtype(dtype).name,
        "n_sequences": len(texts),
        "chunk_size": chunk_size,
        "n_shards": len(chunks),
    }
    saved = json.loads(_meta_path(cache_dir).read_text()) if _meta_path(cache_dir).exists() else None
    if saved is not None and {k: saved.get(k) for k in expected} != expected:
        differing = sorted(k for k in expected if saved.get(k) != expected[k])
        raise ValueError(
            f"{cache_dir} holds a different extraction (differs in: {', '.join(differing)}). "
            "Use a new directory rather than mixing two runs in one cache."
        )

    missing = [i for i in range(len(chunks)) if not _shard_path(cache_dir, i).exists()]
    if missing:
        tokenizer, model = load_encoder(encoder, device)
        try:
            model.eval()
            for done, index in enumerate(missing, start=1):
                _write_shard(_shard_path(cache_dir, index),
                             [token_states(t, tokenizer, model, encoder.max_len, dtype)
                              for t in chunks[index]])
                if progress:
                    progress(f"    {encoder.key}: shard {index + 1}/{len(chunks)} "
                             f"({done}/{len(missing)} this run)")
        finally:
            del model, tokenizer
            if device == "mps":
                torch.mps.empty_cache()
            elif str(device).startswith("cuda"):
                torch.cuda.empty_cache()

    _meta_path(cache_dir).write_text(json.dumps(expected, indent=2) + "\n")
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
