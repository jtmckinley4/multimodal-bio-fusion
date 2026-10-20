"""LucaOne as a frozen baseline: one jointly pretrained model where this project fuses three.

Research plan Priority 2 (Gantt 41). LucaOne is trained from scratch on a single 39-token
vocabulary spanning nucleotides and amino acids, so whatever relates the two modalities is
learned in pretraining rather than fused afterwards. The question it answers here is narrow
and worth asking exactly: does one jointly pretrained representation beat three separately
pretrained encoders and their fusion, on the same rows, under the same probe?

Why this is its own module rather than another entry in ``shared_code.encoders``:

* **Its tokenizer takes an argument no other tokenizer takes.** Every call needs
  ``seq_type="gene"`` or ``seq_type="prot"``; the same letters mean different things in the
  two modes and the model is told which through ``token_type_ids``. ``encoders.load_encoder``
  and ``embeddings.embed`` have no place to put that.
* **It needs its own package.** The checkpoint ships remote code that imports ``lucaone``
  (``pip install lucaone==1.1.1``), so a registry that can be read without Transformers
  installed should not list it.
* **It is 1.8B parameters.** Loading it next to the Stage 1 encoders would invite a sweep
  that silently costs hours, so it is loaded explicitly, by one function, with a dtype the
  caller chooses.

Input limit. LucaOne's published maximum input is 1,280 tokens and its tokenization is
character level, one token per nucleotide or per residue with no codon grouping, so a
coding sequence costs its full length. Stage 3 step 4 measured what that admits: all of the
BioLangFusion benchmarks one stream at a time, and none of the GTEx pilot's 6,000-nucleotide
DNA windows. The checkpoint's own ``config.json`` reports ``max_position_embeddings`` 4096
with ``no_position_embeddings`` true, so longer inputs may well run; ``CONTEXT_TOKENS``
stays at the published 1,280 because that is the number the paper supports, and anything
longer is chunked rather than silently extrapolated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import matrix_cache
from .embeddings import sequences_fingerprint

# The three checkpoints the group publishes on the hub, with the training step each one is
# named for. "default" is the joint gene-and-protein model, the one Priority 2 asks about;
# the other two are its single-modality ablations and are here for a controlled comparison.
CHECKPOINTS = {
    "default": "LucaGroup/LucaOne-default-step36M",
    "gene": "LucaGroup/LucaOne-gene-step36.8M",
    "prot": "LucaGroup/LucaOne-prot-step38.2M",
}

# Pinned commits, as for every other encoder in shared_code.encoders. These repositories
# ship their own model and tokenizer code, which Transformers downloads and executes, so the
# revision pins the code as well as the weights.
REVISIONS = {
    "LucaGroup/LucaOne-default-step36M": "f0d2807eb097a1507b911022d979b502cd62b74c",
    "LucaGroup/LucaOne-gene-step36.8M": "77194618c8063a7fc3a17287cfd44b168db1cca2",
    "LucaGroup/LucaOne-prot-step38.2M": "139edef751b605d9695fded1924fe404b64b52da",
}

# Published maximum input, in tokens, including [CLS] and [SEP].
CONTEXT_TOKENS = 1280
SPECIAL_TOKENS = 2

# Bump when the stored calculation changes; a cache written by an older rule is refused
# rather than silently reused.
CACHE_VERSION = 1


def tokens_for(text):
    """Tokens one sequence costs, specials included: one per character, as the vocabulary is."""
    return len(text) + SPECIAL_TOKENS


def fits(text, limit=CONTEXT_TOKENS):
    """Whether one sequence fits the context without chunking."""
    return tokens_for(text) <= limit


def joint_tokens_for(nucleotides):
    """Tokens a coding sequence and its translation cost in one joint input.

    The gene stream costs its length, the protein stream a third of it, and the pair needs
    [CLS], a separator, and [SEP]. This is the mode that makes LucaOne distinctive, and it
    is also the mode the context limits hardest.
    """
    return nucleotides + nucleotides // 3 + SPECIAL_TOKENS + 1


def longest_joint_cds(limit=CONTEXT_TOKENS):
    """The longest coding sequence whose joint gene-and-protein input fits the context."""
    return (limit - SPECIAL_TOKENS - 1) * 3 // 4


def load(checkpoint="default", device="cpu", dtype=None):
    """Return (tokenizer, model) for one LucaOne checkpoint, frozen and in evaluation mode.

    dtype is passed through to Transformers: float16 halves the 1.8B parameters to about
    3.6 GB, which is what lets the model sit on an 8 GB consumer GPU beside its activations.
    Keep it None (float32) on CPU, where float16 arithmetic is slower rather than faster.

    Checked against Transformers 5.15.1, the version this project pins: the repository's own
    model and tokenizer code loads and runs there, and needs no ``lucaone`` package despite
    what the model card's installation section says.
    """
    from transformers import AutoModel, AutoTokenizer

    repo = CHECKPOINTS.get(checkpoint, checkpoint)
    revision = REVISIONS.get(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, revision=revision)
    model = AutoModel.from_pretrained(
        repo, task_level="token_level", task_type="embedding", revision=revision,
        trust_remote_code=True, **({"dtype": dtype} if dtype is not None else {}),
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model


def _content_states(text, seq_type, tokenizer, model, device, limit):
    """Final hidden states of one window's content tokens, without [CLS] and [SEP]."""
    import torch

    encoded = tokenizer(text, seq_type=seq_type, return_tensors="pt",
                        add_special_tokens=True, truncation=True, max_length=limit)
    moved = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**moved)
    states = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
    return states[0, 1:-1, :].float().cpu().numpy()


def pooled(text, seq_type, tokenizer, model, device="cpu", limit=CONTEXT_TOKENS):
    """Mean-pool one sequence's content tokens into a vector, chunking past the context.

    A sequence longer than the context is cut into consecutive windows of the largest size
    that fits, every window is embedded, and the token states are concatenated before the
    mean is taken. That is mRNABench's rule for models whose context is shorter than a
    transcript, and it is recorded as a deviation wherever it is used: a window has no view
    of what lies outside it, so a chunked embedding is not the embedding of the whole
    sequence. Nothing in the BioLangFusion benchmarks needs it; the half-life and
    translation-efficiency transcripts do.
    """
    content = limit - SPECIAL_TOKENS
    if len(text) <= content:
        return _content_states(text, seq_type, tokenizer, model, device, limit).mean(axis=0)
    windows = [text[i:i + content] for i in range(0, len(text), content)]
    states = [_content_states(w, seq_type, tokenizer, model, device, limit) for w in windows]
    return np.concatenate(states, axis=0).mean(axis=0)


def _cache_metadata(checkpoint, seq_type, texts, limit):
    return {
        "cache_version": CACHE_VERSION,
        "checkpoint": CHECKPOINTS.get(checkpoint, checkpoint),
        "seq_type": seq_type,
        "limit": int(limit),
        "n_sequences": len(texts),
        "fingerprint": sequences_fingerprint(texts),
    }


def embed_with_cache(texts, seq_type, cache_path, device="cpu", checkpoint="default",
                     dtype=None, limit=CONTEXT_TOKENS, progress_every=200, progress=print):
    """Pool every text with LucaOne, reusing a saved matrix when it matches exactly.

    The cache records the checkpoint, the stream type, the context limit and a fingerprint
    of the input strings in order, so a matrix computed for a different checkpoint, a
    different stream, or a different set of rows is never reused by accident. The model is
    loaded only when there is work to do, which matters when the model is 1.8B parameters.
    """
    texts = list(texts)
    if not texts:
        raise ValueError("Embedding requires at least one input sequence.")
    expected = _cache_metadata(checkpoint, seq_type, texts, limit)

    def compute():
        import torch

        tokenizer, model = load(checkpoint, device, dtype)
        try:
            vectors = []
            for i, text in enumerate(texts, start=1):
                vectors.append(pooled(text, seq_type, tokenizer, model, device, limit))
                if progress and progress_every and i % progress_every == 0:
                    progress(f"    lucaone/{seq_type}: {i}/{len(texts)}")
            return np.vstack(vectors)
        finally:
            del model, tokenizer
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()
            elif str(device) == "mps":
                torch.mps.empty_cache()

    return matrix_cache.load_or_compute(cache_path, expected, compute)


def coverage(lengths, limit=CONTEXT_TOKENS):
    """What share of sequences of these nucleotide lengths each mode admits.

    The three modes are the ones Priority 2 can be asked in: the nucleotide stream alone,
    the translated protein alone, and the joint input that carries both. Returned as a plain
    dictionary so a notebook can table it beside a benchmark's row counts.
    """
    lengths = np.asarray(lengths)
    return {
        "rows": int(len(lengths)),
        "longest nt": int(lengths.max()) if len(lengths) else 0,
        "gene stream": float(np.mean(lengths + SPECIAL_TOKENS <= limit)),
        "protein stream": float(np.mean(lengths // 3 + SPECIAL_TOKENS <= limit)),
        "joint input": float(np.mean(lengths + lengths // 3 + SPECIAL_TOKENS + 1 <= limit)),
    }


def write_manifest(path, **fields):
    """Record what a LucaOne run embedded, beside its cache, for the run record."""
    Path(path).write_text(json.dumps(fields, indent=2, sort_keys=True) + "\n")
