"""Embeddings from frozen encoders: pooled vectors, per-layer vectors, and attention.

embed_with_cache saves one matrix per encoder and reuses it only when the cache
version, checkpoint, requested revision, token limit, and effective inputs match.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch

from .analysis import aggregate_attention, expand_token_attention_to_nucleotides
from .encoders import encoder_input, free_memory, load_encoder


# Bump when the NPZ schema or embedding calculation (for example pooling) changes.
# Old versions require deliberate recovery; they are not silently recomputed.
CACHE_VERSION = 1


@torch.no_grad()
def embed(seq, tokenizer, model, max_len, device):
    """Mean-pool one sequence's final hidden state into a NumPy vector."""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    out = model(**inputs)
    token_embeddings = out[0]  # (1, T, d): last_hidden_state for every encoder used here
    return token_embeddings.mean(dim=1).squeeze().float().cpu().numpy()


def count_tokens(seq, tokenizer, max_len):
    """Return (tokens after truncation, tokens without truncation) for one input."""
    full = len(tokenizer(seq)["input_ids"])
    return min(full, max_len), full


def sequences_fingerprint(seqs):
    """Identify ordered effective input strings without ambiguous separators."""
    payload = json.dumps(list(seqs), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_embeddings(matrix, n_sequences):
    """Check the stored/computed row contract without guessing a model's width."""
    if (matrix.ndim != 2 or matrix.shape[0] != n_sequences
            or matrix.shape[1] == 0 or matrix.dtype.kind not in "fiu"):
        raise ValueError("embeddings must be a real numeric matrix with one row per input and positive width")


def _read_cache(path, expected):
    """Return a matching matrix, None for a valid mismatch, or a recovery error."""
    try:
        with np.load(path, allow_pickle=False) as saved:
            required = set(expected) | {"embeddings", "device"}
            if not required.issubset(saved.files):
                raise ValueError("legacy cache or missing required metadata")
            metadata = {}
            for key in required - {"embeddings"}:
                value = saved[key]
                if value.ndim != 0:
                    raise ValueError(f"metadata {key} must be scalar")
                metadata[key] = value.item()
            for key in ("cache_version", "max_len", "n_sequences"):
                if type(metadata[key]) is not int or metadata[key] < 1:
                    raise ValueError(f"metadata {key} must be a positive integer")
            for key in ("checkpoint", "revision", "fingerprint", "device"):
                if not isinstance(metadata[key], str):
                    raise ValueError(f"metadata {key} must be text")
            if metadata["cache_version"] != CACHE_VERSION:
                raise ValueError(f"unsupported cache version {metadata['cache_version']}")
            if (not metadata["checkpoint"] or len(metadata["fingerprint"]) != 64
                    or any(c not in "0123456789abcdef" for c in metadata["fingerprint"])):
                raise ValueError("invalid checkpoint or input fingerprint")
            matrix = saved["embeddings"]
            _validate_embeddings(matrix, metadata["n_sequences"])
            if all(metadata[key] == value for key, value in expected.items()):
                return matrix
            return None
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(
            f"Cannot reuse embedding cache {path}: {error}. Preserve this file and "
            "choose a fresh cache directory for an explicitly authorized recomputation."
        ) from error


def _write_cache(path, matrix, device, expected):
    """Publish a checked sibling archive without exposing an incomplete replacement."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez(handle, embeddings=matrix, device=str(device), **expected)
        if _read_cache(temporary, expected) is None:
            raise ValueError("New embedding cache does not match the requested computation.")
        os.replace(temporary, path)
    except BaseException as error:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Temporary cache remains at {temporary}: {cleanup_error}")
        raise


def _free_memory_preserving_error(primary_error):
    """Report cleanup failure without replacing this call's primary exception."""
    try:
        free_memory()
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        if hasattr(primary_error, "add_note"):
            primary_error.add_note(f"Memory cleanup also failed: {cleanup_error}")


def embed_with_cache(encoder, seqs, cache_dir, device, progress_every=200, prepared=False):
    """Embed seqs with one encoder, reusing a saved matrix when it matches.

    With prepared=True, seqs already contain the strings the encoder reads, such as
    Dataset.inputs_for(encoder); otherwise encoder_input converts each coding sequence.
    These effective strings identify the cache. Valid current-format identity mismatches
    recompute on this call; legacy or damaged files raise before loading a model.
    Checkpoint revisions are requested revisions, not necessarily resolved commits.
    """
    texts = list(seqs) if prepared else [encoder_input(encoder, seq) for seq in seqs]
    if not texts:
        raise ValueError("Embedding requires at least one input sequence.")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{encoder.key}.npz")
    expected = {
        "cache_version": CACHE_VERSION,
        "checkpoint": encoder.checkpoint,
        "revision": encoder.revision or "",
        "max_len": encoder.max_len,
        "fingerprint": sequences_fingerprint(texts),
        "n_sequences": len(texts),
    }
    if os.path.exists(path):
        saved = _read_cache(path, expected)
        if saved is not None:
            return saved, True

    tokenizer, model = load_encoder(encoder, device)
    vectors = []
    primary_error = None
    try:
        for i, text in enumerate(texts, start=1):
            vectors.append(embed(text, tokenizer, model, encoder.max_len, device))
            if progress_every and i % progress_every == 0:
                print(f"    {encoder.key}: {i}/{len(texts)}")
    except BaseException as error:
        primary_error = error
        raise
    finally:
        del model, tokenizer
        _free_memory_preserving_error(primary_error)
    embeddings = np.vstack(vectors)
    _validate_embeddings(embeddings, len(texts))
    _write_cache(path, embeddings, device, expected)
    return embeddings, False


@torch.no_grad()
def embed_all_layers(seq, encoder, tokenizer, model, device, prepared=False):
    """Mean-pool every returned hidden state for one sequence.

    Every encoder contributes its embedding-layer output followed by each transformer
    layer. DNABERT-2's own all-layer option fails, so its states are captured with
    forward hooks on the embedding module and each encoder layer.
    """
    text = seq if prepared else encoder_input(encoder, seq)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=encoder.max_len).to(device)
    if encoder.loader == "remote_dnabert2":
        captured = []
        modules = [model.embeddings] + list(model.encoder.layer)
        hooks = [m.register_forward_hook(lambda _m, _i, out: captured.append(out)) for m in modules]
        try:
            model(**inputs)
        finally:
            for hook in hooks:
                hook.remove()
        layers = captured
    else:
        layers = model(**inputs, output_hidden_states=True).hidden_states
    return [h.reshape(-1, h.shape[-1]).mean(dim=0).float().cpu().numpy() for h in layers]


@torch.no_grad()
def attention_maps(text, tokenizer, model, device, max_len, layers=None):
    """Return self-attention arrays and token strings for one input string.

    Each array has shape (heads, query_tokens, key_tokens). layers selects which layers
    to return, all by default; converting only the needed layers saves memory for long
    inputs. The model must be loaded with eager attention so that it returns weights.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    attentions = model(**inputs, output_attentions=True).attentions
    chosen = range(len(attentions)) if layers is None else layers
    arrays = [attentions[i][0].float().cpu().numpy() for i in chosen]
    return arrays, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])


def nucleotide_attention(seq, encoder, tokenizer, model, device, layer=-1):
    """Return one layer's incoming attention, min-max scaled and expanded to nucleotides."""
    arrays, tokens = attention_maps(
        encoder_input(encoder, seq), tokenizer, model, device, encoder.max_len, layers=[layer]
    )
    return expand_token_attention_to_nucleotides(aggregate_attention(arrays, layer=0), tokens)


def embed_layers(encoder, seqs, device, prepared=False):
    """Return one (n, d) matrix per hidden state for the given retained sequences.

    The encoder is loaded, applied to each sequence with embed_all_layers, and released.
    """
    tokenizer, model = load_encoder(encoder, device)
    primary_error = None
    try:
        per_sequence = [embed_all_layers(s, encoder, tokenizer, model, device, prepared) for s in seqs]
    except BaseException as error:
        primary_error = error
        raise
    finally:
        del model, tokenizer
        _free_memory_preserving_error(primary_error)
    return [np.vstack([states[k] for states in per_sequence]) for k in range(len(per_sequence[0]))]


def attention_profiles(encoder, seqs, device, layer=-1):
    """Return each sequence's incoming attention, expanded to nucleotides, for one encoder.

    The encoder is loaded with eager attention and released afterwards. An array is
    shorter than its sequence when the encoder truncates its input.
    """
    tokenizer, model = load_encoder(encoder, device, eager_attention=True)
    primary_error = None
    try:
        return [nucleotide_attention(s, encoder, tokenizer, model, device, layer=layer) for s in seqs]
    except BaseException as error:
        primary_error = error
        raise
    finally:
        del model, tokenizer
        _free_memory_preserving_error(primary_error)
