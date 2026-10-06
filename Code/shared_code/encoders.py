"""Registry of frozen pretrained encoders, and the code that loads them.

Adding an encoder means adding one Encoder entry to ENCODERS. PyTorch, Transformers, and
multimolecule are imported only when a model is loaded, so the registry can be read
without them.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass

from .sequences import dna_to_rna, rna_to_dna, translate_cds


@dataclass(frozen=True)
class Encoder:
    """A frozen pretrained encoder and the input it reads.

    alphabet: "dna" (U replaced by T), "rna" (T replaced by U), "protein" (translated
    coding sequence), or "protein_spaced" (the translation with residues separated by
    spaces and the rare amino acids U, Z, O, and B written as X, as ProtBERT expects).
    loader: "auto" for checkpoints that load with the Transformers auto classes,
    "remote_esm" for Nucleotide Transformer v2, "remote_dnabert2" for DNABERT-2, and
    "bert" for ProtBERT, whose checkpoint does not name its model type.
    attention: False when the model does not return attention weights (DNABERT-2's
    model code, and HyenaDNA, which has no attention layers). codon: True when the
    tokenizer reads one codon per token, so the encoder needs an in-frame coding sequence
    when a dataset supplies a transcript and a coding sequence separately.
    """

    key: str
    label: str
    modality: str
    checkpoint: str
    alphabet: str
    max_len: int
    loader: str = "auto"
    revision: str | None = None
    attention: bool = True
    codon: bool = False


ENCODERS = {
    e.key: e
    for e in [
        Encoder("nt500m", "NT 500M human-ref", "DNA",
                "InstaDeepAI/nucleotide-transformer-500m-human-ref", "dna", 1000),
        Encoder("ntv2_100m", "NT v2 100M multi-species", "DNA",
                "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species", "dna", 2048,
                loader="remote_esm", revision="f34324c6fde36a4f635f0f1f06cac5d25acd6798"),
        Encoder("dnabert2", "DNABERT-2", "DNA",
                "zhihan1996/DNABERT-2-117M", "dna", 1024,
                loader="remote_dnabert2", revision="7bce263b15377fc15361f52cfab88f8b586abda0",
                attention=False),
        Encoder("hyenadna", "HyenaDNA large", "DNA", "multimolecule/hyenadna-large", "dna", 1_000_000,
                revision="928e4a8cf67eab56be6e9c517b1a155faca0b1e1", attention=False),
        Encoder("rnafm", "RNA-FM", "RNA", "multimolecule/rnafm", "rna", 1024),
        Encoder("rinalmo", "RiNALMo 150M", "RNA", "multimolecule/rinalmo-mega", "rna", 1024,
                revision="5e3a682eb856148fe81f12a25301962c9ffaf0ed"),
        Encoder("mrnafm", "mRNA-FM", "RNA", "multimolecule/mrnafm", "rna", 1024, codon=True),
        Encoder("calm", "CaLM", "RNA", "multimolecule/calm", "rna", 1026,
                revision="63fb4125a6139353defa40cbdb0c2f5e76a8cc75", codon=True),
        Encoder("esm2_8m", "ESM-2 8M", "protein", "facebook/esm2_t6_8M_UR50D", "protein", 1024),
        Encoder("esm2_35m", "ESM-2 35M", "protein", "facebook/esm2_t12_35M_UR50D", "protein", 1024),
        Encoder("esm2_150m", "ESM-2 150M", "protein", "facebook/esm2_t30_150M_UR50D", "protein", 1024),
        Encoder("protbert", "ProtBERT", "protein", "Rostlab/prot_bert", "protein_spaced", 1024,
                loader="bert", revision="7a894481acdc12202f0a415dd567f6cfdb698908"),
    ]
}


def protein_input(encoder, protein):
    """Return a protein sequence in the form a protein encoder reads."""
    if encoder.alphabet == "protein_spaced":
        return " ".join(re.sub(r"[UZOB]", "X", protein))
    return protein


def encoder_input(encoder, seq):
    """Return the string an encoder reads for one retained coding sequence."""
    if encoder.alphabet == "dna":
        return rna_to_dna(seq)
    if encoder.alphabet == "rna":
        return dna_to_rna(seq)
    return protein_input(encoder, translate_cds(seq))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

# Configuration fields that Transformers 5 no longer sets by default but that the
# remote model code reads.
_REMOTE_CONFIG_DEFAULTS = {
    "remote_esm": {"is_decoder": False, "add_cross_attention": False},
    "remote_dnabert2": {"is_decoder": False},
}


# Remote base-model class, checkpoint key prefix, and weights file for each remote loader.
_REMOTE_CLASSES = {
    "remote_esm": ("modeling_esm.EsmModel", "esm.", "model.safetensors"),
    "remote_dnabert2": ("bert_layers.BertModel", "bert.", "pytorch_model.bin"),
}


def select_device():
    """Return "cuda", "mps", or "cpu", in that order of preference."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_transformers5_compat():
    """Let the Nucleotide Transformer v2 and DNABERT-2 remote code run under Transformers 5.

    1. DNABERT-2 optionally imports `triton`, which is unavailable on macOS. Its code falls
       back to PyTorch attention when that import fails, but Transformers rejects the file
       before the fallback can run. `triton` is removed from that pre-import check only.
    2. Nucleotide Transformer v2 imports `find_pruneable_heads_and_indices` and calls
       `get_head_mask`, which Transformers 5 removed. Equivalent helpers are restored;
       head masking is never used here.
    """
    import torch
    import transformers.dynamic_module_utils as dmu
    import transformers.pytorch_utils as pu
    from transformers.modeling_utils import PreTrainedModel

    if not getattr(dmu, "_stage1_triton_patch", False):
        original_get_imports = dmu.get_imports
        dmu.get_imports = lambda filename: [
            m for m in original_get_imports(filename) if m != "triton"
        ]
        dmu._stage1_triton_patch = True

    if not hasattr(pu, "find_pruneable_heads_and_indices"):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    if not hasattr(PreTrainedModel, "get_head_mask"):
        def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is not None:
                raise NotImplementedError("Head masking is not supported here.")
            return [None] * num_hidden_layers

        PreTrainedModel.get_head_mask = get_head_mask


def _load_checkpoint_state(repo, filename, revision):
    """Return a checkpoint's state dict, using the local Hugging Face cache when available."""
    import torch
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(repo, filename, revision=revision, local_files_only=True)
    except Exception:
        path = hf_hub_download(repo, filename, revision=revision)
    if filename.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_remote_model(encoder):
    """Build a remote-code base model on the CPU and load its checkpoint weights.

    Transformers 5 constructs models on a placeholder device before loading weights.
    DNABERT-2 computes its ALiBi tensor and Nucleotide Transformer v2 its rotary
    frequencies during construction, so both are built directly instead. Only the
    unused pooler may be absent from the checkpoint.
    """
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    class_ref, prefix, weights_file = _REMOTE_CLASSES[encoder.loader]
    config = AutoConfig.from_pretrained(
        encoder.checkpoint, trust_remote_code=True, revision=encoder.revision
    )
    for name, value in _REMOTE_CONFIG_DEFAULTS[encoder.loader].items():
        if getattr(config, name, None) is None:
            setattr(config, name, value)
    if encoder.loader == "remote_dnabert2" and getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = 3  # [PAD] in the DNABERT-2 tokenizer

    model_class = get_class_from_dynamic_module(
        class_ref, encoder.checkpoint, revision=encoder.revision
    )
    model = model_class(config)
    state = {
        k[len(prefix):]: v
        for k, v in _load_checkpoint_state(encoder.checkpoint, weights_file, encoder.revision).items()
        if k.startswith(prefix)
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("pooler.")]
    if missing or unexpected:
        raise RuntimeError(
            f"{encoder.key}: weights did not match (missing={missing}, unexpected={unexpected})"
        )
    return model


def load_tokenizer(encoder):
    """Return an encoder's tokenizer without loading its model, for counting input tokens."""
    import multimolecule  # noqa: F401  (registers the multimolecule tokenizers with Transformers)
    from transformers import AutoTokenizer, BertTokenizer

    if encoder.loader == "bert":
        return BertTokenizer.from_pretrained(encoder.checkpoint, do_lower_case=False, revision=encoder.revision)
    return AutoTokenizer.from_pretrained(encoder.checkpoint, trust_remote_code=True, revision=encoder.revision)


def load_encoder(encoder, device, eager_attention=False):
    """Return (tokenizer, model) for an Encoder, on device and in evaluation mode."""
    import multimolecule  # noqa: F401  (registers the RNA-FM, mRNA-FM, RiNALMo, CaLM, and HyenaDNA classes with Transformers)
    from transformers import AutoModel, AutoTokenizer

    if encoder.loader == "bert":
        from transformers import BertModel, BertTokenizer

        tokenizer = BertTokenizer.from_pretrained(
            encoder.checkpoint, do_lower_case=False, revision=encoder.revision
        )
        kwargs = {"attn_implementation": "eager"} if eager_attention else {}
        model = BertModel.from_pretrained(encoder.checkpoint, revision=encoder.revision, **kwargs)
        return tokenizer, model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        encoder.checkpoint, trust_remote_code=True, revision=encoder.revision
    )
    if encoder.loader == "auto":
        kwargs = {"attn_implementation": "eager"} if eager_attention else {}
        model = AutoModel.from_pretrained(
            encoder.checkpoint, trust_remote_code=True, revision=encoder.revision, **kwargs
        )
    else:
        apply_transformers5_compat()
        model = _load_remote_model(encoder)
    return tokenizer, model.to(device).eval()


def free_memory():
    """Release cached memory after a model is deleted, before loading the next one."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
