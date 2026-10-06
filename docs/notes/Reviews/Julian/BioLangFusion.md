# 1. Paper Information

**Title:** BioLangFusion: Multimodal Fusion of DNA, mRNA, and Protein Language Models
**Authors:** Mollaysa et al.
**Year / Venue:** 2025, ICML 2025 Workshop on Multi-modal Foundation Models and LLMs for Life Sciences
**Paper Type:** Method
**Biological Domain:** Central-dogma molecular biology (DNA, mRNA, protein)
**Primary Sequence Representation(s):** Mixed (nucleotide, codon-resolved mRNA, amino acid)

**One-sentence summary:** The paper proposes three lightweight, training-free fusion schemes that align frozen DNA, RNA, and protein language-model embeddings to a shared codon resolution and combine them, improving molecular property prediction over any single modality.

# 2. Problem and Motivation

Pretrained DNA, RNA, and protein language models are trained separately and tokenize at different granularities, so they cannot be combined directly even though the sequences they model are causally linked by the central dogma. The paper addresses how to fuse these frozen embeddings cheaply, without retraining or architectural changes to the base models. It explicitly deals with three representations simultaneously: DNA (6-mer tokens), mRNA (single-nucleotide tokens), and protein (amino-acid/codon tokens), unified at codon resolution (Sec. 2.1).

# 3. Input Representation and Tokenization

Each modality keeps its own pretrained tokenizer: Nucleotide Transformer v2 for DNA (6-mers), RNA-FM for mRNA (single nucleotides), ESM-2 for protein (amino acids, i.e. 3-mers of nucleotide). For an mRNA of length $T$:

$$E_{DNA}\in\mathbb{R}^{T/6\times d_{DNA}},\quad E_{RNA}\in\mathbb{R}^{T\times d_{RNA}},\quad E_{Prot}\in\mathbb{R}^{T/3\times d_{Prot}}$$

with $d_{DNA}=4107$, $d_{RNA}=640$, $d_{Prot}=320$ (Table 3). No special tokens, reading-frame or strand handling are specified beyond using start/stop codons to map an mRNA to its DNA/protein counterparts; max input length is 1000 tokens (RNA truncated beyond this, Table 2).

**Mixed-representation handling.** The protein frame ($T'=T/3$) is the reference resolution. DNA is upsampled and RNA downsampled onto it:

$$\tilde{E}_{DNA}=\mathrm{TConv}_{k=2,s=2}(E_{DNA})\in\mathbb{R}^{T'\times d_{DNA}},\quad \tilde{E}_{RNA}=\mathrm{AvgPool}_{k=3,s=3}(E_{RNA})\in\mathbb{R}^{T'\times d_{RNA}}$$

so position $t$ denotes the same codon in all three modalities. Vocabularies are separate (one per pretrained model); the model does not use an explicit modality/type token — the modality identity is implicit in which embedding stream a vector arrives on.

# 4. Model Architecture

**Backbone:** none learned end-to-end for representation — three frozen pretrained encoders (Transformer-based) feed a downstream fusion head plus a TextCNN prediction head.
**Representation-specific components:** separate frozen encoders per modality; a shared codon-resolution grid (not a shared latent space); modality-specific learnable MLP projections; one of three fusion mechanisms.

Three fusion heads, all after alignment (Sec. 2.2):
(1) Concatenation: $Z_{concat}(t)=\mathrm{MLP}(\tilde{E}_{DNA}[t])\,\|\,\tilde{E}_{RNA}[t]\,\|\,E_{Prot}[t]$.
(2) MIL with entropy regularization: gated (tanh/sigmoid) attention produces one weight per modality per sequence from mean-pooled summaries; fused vector is the attention-weighted sum, trained with an added negative-entropy loss term to discourage near-uniform weights.
(3) Cross-modal multi-head attention: the three projected streams are concatenated along the sequence axis into one context of length $3T'$; each modality queries this joint context; outputs are averaged with a residual connection and layer-normalized.

Flow: $x_{DNA},x_{RNA},x_{Prot}\to E_{DNA},E_{RNA},E_{Prot}\to$ codon-resolution alignment $\to$ {concat | MIL-attention | cross-attention} $\to$ TextCNN head $\to y$.

# 5. Training Strategy

Base encoders stay frozen; only the fusion head and prediction head are trained. Objective is task-specific supervised loss (regression/classification) on five downstream datasets — not an explicit cross-modal alignment loss beyond the entropy regularizer in method (2). No joint pretraining; training is single-stage on the fusion+prediction modules only.

# 6. Evaluation and Main Findings

Five tasks: CoV-Vac, Fungal, E. coli expression, mRNA stability, Ab1 (Table with per-task metrics). All three fusion methods beat single-modality baselines. MIL with entropy regularization is best/tied-best on 4/5 datasets; cross-attention wins on E. coli protein abundance. Ablation: removing the codon-alignment step (projecting and concatenating without resampling) measurably underperforms the aligned version, and removing the entropy term hurts MIL performance — direct evidence the alignment step and the entropy regularizer both matter.

# 7. Architectural Strengths and Limitations

**Strengths**
- No retraining of pretrained FMs needed — cheap, modular (Sec. 1).
- Ablation isolates the value of codon-resolution alignment itself.
- Attention-pooling adapts modality weighting per sequence rather than fixing it.

**Limitations**
- Alignment is purely positional/resolution matching, not learned semantic alignment — our inference.
- Codon-grid alignment assumes a clean reading frame throughout; no mechanism for non-coding regions.
- Separate frozen encoders cannot interact during pretraining, limiting deep cross-modal learning.

# 8. Relevance to Our Project

**Direct relevance:** High. BioLangFusion is the closest existing precedent for this project: it fuses frozen Nucleotide Transformer and ESM-2 embeddings (plus RNA-FM) for molecular property prediction, and mRNA stability is one of its five benchmark tasks. Our mRNA_Stability sequences are coding-only, start at a start codon, and stay in frame, which is exactly the setting where its codon-resolution alignment holds at every position.

**What can be transferred?**
Our inference: three pieces map directly onto our setup, which now follows BioLangFusion in giving the same coding sequence to DNA, RNA, and protein encoders. Our Stage 1 DNA encoder, Nucleotide Transformer 500M human-ref, was pretrained on the human genome, so it plays the role of BioLangFusion's DNA stream, and our Stage 1 RNA encoder is RNA-FM, the same RNA encoder BioLangFusion uses.

1. The concatenation baseline becomes our Stage 3 concatenation-plus-MLP baseline, the model every fusion candidate must beat. Stage 1's linear version already shows no gain: RNA plus protein reaches mean $R^2$ 0.103 and all three modalities 0.098, versus 0.103 for protein alone. This is the bar a learned fusion module has to clear.
2. Codon-resolution resampling puts all three streams on one grid. Nucleotide Transformer's 6-mer tokens cover two codons each, so the DNA stream is upsampled onto ESM-2's one-token-per-residue grid, and a single-nucleotide RNA encoder such as RNA-FM is average-pooled over each codon:

$$\tilde{E}_{DNA}=\mathrm{TConv}_{k=2,s=2}(E_{DNA}),\qquad \tilde{E}_{RNA}=\mathrm{AvgPool}_{k=3,s=3}(E_{RNA}),\qquad E_{Prot}\in\mathbb{R}^{T/3\times d_{Prot}}$$

Position $t$ then refers to the same codon in every stream, which is what per-position fusion (and Coupled Mamba's step-matched recurrence) needs.
3. The entropy-regularized MIL head produces one weight per modality per sequence, a direct, interpretable readout of how much the model leans on each encoder. That is a useful check given Stage 1 found protein carries most of the linear stability signal (mean $R^2$ 0.103, versus 0.015 for the DNA encoder and 0.028 for the RNA encoder).

Its reported mRNA-stability scores are also an external comparison point for Stage 4. Matching them needs its exact encoders (Nucleotide Transformer v2 100M multi-species, RNA-FM, ESM-2 8M), the CodonBERT splits, and Spearman correlation. It reports 41,123 raw and 23,929 used stability sequences, while our file has 65,356 rows, so the data version must be checked before comparing numbers.

**What would need to change?** Stage 1 used one mean-pooled vector per sequence; per-codon fusion needs the encoders' token-level outputs, so the embedding step must keep full hidden-state sequences. Resampling is also only a positional alignment, not a representational one. Our Stage 2 decision tree reserves resampling-only alignment for the high-alignment, redundancy-dominated case. Stage 1 finds low alignment between the protein and nucleotide encoders (CKA 0.128 and 0.199), most of which disappears once sequence composition is removed, so resampling alone is not expected to explain any fusion gain.

**Key architectural takeaway:** The most useful ideas from this paper are the codon-grid resampling (exact for our coding-only data) and the concatenation baseline, with the MIL modality weights as a cheap interpretability check on which encoder the fused model relies on.

# 9. Final Verdict for Literature Review

**Category:** Multimodal fusion of frozen encoders; Representation alignment
**Priority for our project:** High
**Reason:** Same encoder families, same task family, and it supplies our Stage 3 baseline, a codon-grid alignment that fits our coding-only data exactly, and an external Stage 4 comparison.

## Compact Comparison Record

| Input | Tokenization | Backbone | Interaction | Objective | Our relevance |
|---|---|---|---|---|---|
| DNA (6-mer) + mRNA (nt) + protein (aa/codon) | Separate pretrained tokenizers per modality | 3 frozen pretrained Transformers + fusion head + TextCNN | Codon-grid resampling then concat / MIL-attention / cross-attention | Downstream supervised loss + entropy regularizer (MIL) | High: same encoders and task family; concatenation baseline, codon-grid resampling, external stability benchmark |
