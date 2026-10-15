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

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..encoders import encoder_input, load_encoder, protein_input
from ..sequences import rna_to_dna

# Bump when the shard layout or the stored calculation changes.
SHARD_VERSION = 1
DEFAULT_CHUNK = 256


@torch.no_grad()
def token_states(text, tokenizer, model, max_len, dtype=np.float16):
    """Return one sequence's content-token final hidden states as a (tokens, width) array.

    Special tokens are dropped using the tokenizer's own mask, so the row holds only
    tokens standing for the sequence. Raises when the input would be truncated: a
    truncated coding sequence loses its 3' end, and the streams would stop aligning.
    """
    encoded = tokenizer(text, return_tensors="pt", return_special_tokens_mask=True)
    n_tokens = encoded["input_ids"].shape[1]
    if n_tokens > max_len:
        raise ValueError(
            f"{n_tokens} tokens exceeds the {max_len}-token limit for an input of "
            f"{len(text)} characters; the benchmark rows are filtered to fit, so this "
            "signals the wrong rows or the wrong encoder."
        )
    special = encoded.pop("special_tokens_mask")[0].numpy().astype(bool)
    device = next(model.parameters()).device
    out = model(**{k: v.to(device) for k, v in encoded.items()})
    states = out[0][0].float().cpu().numpy()
    return states[~special].astype(dtype, copy=False)


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

    def __len__(self):
        return len(self.lengths)

    @property
    def width(self):
        return int(self.meta["width"])

    def __getitem__(self, i):
        shard = np.load(_shard_path(self.cache_dir, int(self.shard_of[i])), mmap_mode="r")
        starts = np.concatenate([[0], np.cumsum(shard["lengths"])])
        row = int(self.row_in_shard[i])
        return np.asarray(shard["tokens"][starts[row]:starts[row + 1]])


def summarize(cache_dir):
    """Read the shard lengths and return a TokenCache describing the whole extraction."""
    cache_dir = Path(cache_dir)
    meta = json.loads(_meta_path(cache_dir).read_text()) if _meta_path(cache_dir).exists() else {}
    lengths, shard_of, row_in_shard, width = [], [], [], None
    for index in range(meta.get("n_shards", 0)):
        path = _shard_path(cache_dir, index)
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; rerun extract to finish this cache.")
        with np.load(path, mmap_mode="r") as shard:
            counts = np.asarray(shard["lengths"])
            width = shard["tokens"].shape[1]
        lengths.append(counts)
        shard_of.append(np.full(len(counts), index, dtype=np.int32))
        row_in_shard.append(np.arange(len(counts), dtype=np.int32))
    lengths = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int32)
    meta = {**meta, "width": int(width or 0),
            "total_tokens": int(lengths.sum()),
            "bytes": int(lengths.sum()) * int(width or 0) * np.dtype(meta.get("dtype", "float16")).itemsize}
    return TokenCache(cache_dir, meta, lengths,
                      np.concatenate(shard_of) if shard_of else np.zeros(0, dtype=np.int32),
                      np.concatenate(row_in_shard) if row_in_shard else np.zeros(0, dtype=np.int32))


# Nucleotides per token for the encoders with a fixed tokenization rate. A byte-pair
# vocabulary such as DNABERT-2's has no fixed rate and is deliberately absent.
NUCLEOTIDES_PER_TOKEN = {
    "rnafm": 1, "rinalmo": 1,      # one token per nucleotide
    "mrnafm": 3, "calm": 3,        # one token per codon
    "ntv2_100m": 6, "nt500m": 6,   # one token per 6-mer
}


def expected_tokens(encoder, seq):
    """Tokens one coding sequence contributes to an encoder's stream.

    Taken from the coding sequence rather than from the string the encoder reads, so that a
    spaced protein alphabet (ProtBERT) and a k-mer vocabulary are both counted correctly. A
    protein encoder sees the grid-aligned translation, one token per codon. Returns None for
    an encoder whose tokenization has no fixed rate, such as DNABERT-2's byte-pair vocabulary.
    """
    if encoder.modality == "protein":
        return len(seq) // 3
    per_token = NUCLEOTIDES_PER_TOKEN.get(encoder.key)
    return None if per_token is None else -(-len(seq) // per_token)


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
