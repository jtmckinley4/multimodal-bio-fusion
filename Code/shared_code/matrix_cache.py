"""One matrix per file, reused only when the settings that produced it match exactly.

Stage 1's ``embeddings.embed_with_cache`` caches pooled vectors for the encoders in
``shared_code.encoders``, keyed on that module's fields. Stage 4 adds two embedders those
fields do not describe: LucaOne, whose stream type is part of the calculation, and the
chunked pooling the half-life transcripts need, where the window size is. Rather than widen
Stage 1's metadata and invalidate caches that took days to fill, both keep their own
metadata here and share the mechanics: write atomically, refuse a file written under
different settings instead of silently reusing it, and never load a model when the cache
already answers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def read(path, expected):
    """Return the stored matrix when every metadata field matches, else raise.

    A missing file returns None, which the caller reads as "compute it". A file whose
    metadata differs is an error rather than a recomputation: two runs that differ in what
    they embedded must not share a filename, and finding out silently is how a table ends
    up mixing them.
    """
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        missing = [k for k in expected if k not in saved.files]
        if missing:
            raise ValueError(f"{path} is missing the metadata {', '.join(sorted(missing))}; "
                             "it was written by an older rule and is not reused.")
        stored = {k: saved[k].item() for k in expected}
        if stored != expected:
            differing = sorted(k for k in expected if stored[k] != expected[k])
            raise ValueError(
                f"{path} holds a different run (differs in: {', '.join(differing)}). "
                "Use a new file rather than mixing two runs in one cache."
            )
        matrix = saved["embeddings"]
    if matrix.ndim != 2 or matrix.shape[0] != expected.get("n_sequences", matrix.shape[0]):
        raise ValueError(f"{path} does not hold one row per input sequence.")
    return matrix


def write(path, matrix, expected):
    """Publish a matrix and its metadata atomically, so an interrupted write leaves nothing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".npz", delete=False)
    try:
        with handle:
            np.savez(handle, embeddings=matrix, **expected)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return matrix


def load_or_compute(path, expected, compute):
    """Return (matrix, reused). compute() runs only when the cache cannot answer."""
    found = read(path, expected)
    if found is not None:
        return found, True
    return write(path, compute(), expected), False
