# 1. Paper Information

**Title:** Central Dogma Transformer
**Authors:** Ota (independent researcher)
**Year / Venue:** Preprint (arXiv 2601.01089)
**Paper Type:** Model
**Biological Domain:** Gene regulation (enhancer–gene effect prediction, CRISPRi)
**Primary Sequence Representation(s):** Mixed (DNA, RNA gene-level, protein), all as frozen pre-computed embeddings, not raw tokens

**One-sentence summary:** CDT enforces one-directional DNA→RNA→protein cross-attention over frozen foundation-model embeddings, mirroring transcription/translation so attention weights are directly interpretable as regulatory hypotheses, validated by a CTCF/Hi-C case study.

# 2. Problem and Motivation

The paper addresses how to combine DNA, RNA, and protein information for gene-regulation prediction while keeping the model's internal computation interpretable as biological causality, not just accurate. It explicitly uses three representations: DNA (Enformer embeddings over a 114 kb window), RNA (scGPT gene-token embeddings), and protein (ESM-C/ProteomeLM embeddings) (Table, Sec. "Frozen foundation models").

# 3. Input Representation and Tokenization

CDT does not tokenize raw sequence itself; it consumes pre-computed frozen embeddings: Enformer gives DNA embeddings over 896 positional bins spanning a 114 kb window (128 bp/bin), 3072-dim; scGPT gives 512-dim gene-level embeddings via vocabulary lookup; ESM-C/ProteomeLM give 768-dim protein embeddings. Reading-frame/strand handling: Not specified — CDT centers DNA windows on the enhancer rather than the TSS. No special tokens or boundary markers are described; modality identity is structural (dedicated encoder per stream).

**Mixed-representation handling.** The three modalities are processed by separate frozen encoders, never converted into a shared vocabulary. They are connected only through learned cross-attention layers after self-attention, not through positional resampling: DNA keeps 896 positions, RNA/protein keep per-gene/per-protein token counts, and cross-attention directly relates differently-shaped representations via query/key/value projections rather than forcing a shared length.

# 4. Model Architecture

**Backbone:** Transformer self-attention + directional cross-attention over frozen embeddings (~60M trainable parameters on top of ~712M frozen).
**Representation-specific components:** separate encoders per modality; one-directional cross-attention (not bidirectional/symmetric); a "Virtual Cell Embedder" (VCE) fusion/pooling layer; no shared latent space or type embeddings — direction of information flow substitutes for them.

Each modality first passes self-attention (DNA: 2 layers/896 bins; RNA/protein: 1 layer each). Cross-attention is strictly directional:

$$\text{DNA}\to\text{RNA}:\ Q=\text{genes},\ K,V=\text{DNA positions}\ \Rightarrow\ \text{attn}\in\mathbb{R}^{n_{genes}\times 896}$$
$$\text{RNA}\to\text{Protein}:\ Q=\text{proteins},\ K,V=\text{RNA}_{fused}\ \Rightarrow\ \text{attn}\in\mathbb{R}^{n_{proteins}\times n_{genes}}$$

so protein-layer attention indirectly carries DNA context. A learned query per modality attention-pools each cross-informed representation into a 768-dim summary; the three summaries (2304-dim) are concatenated and passed through a small feed-forward network (GELU) to produce the Virtual Cell Embedding, fed to an MLP head predicting a scalar effect size (Huber loss).

Flow: $x_{DNA}\to E_{DNA}\to$ self-attn $\to$ DNA$\to$RNA cross-attn $\to$ RNA$_{fused}\to$ RNA$\to$Protein cross-attn $\to$ Protein$_{fused}\to$ VCE pooling $\to$ concat $\to$ FFN $\to$ effect-size head.

# 5. Training Strategy

Objective: Huber-loss regression on CRISPRi enhancer effect sizes; no masked-language-modeling or generative pretraining — CDT trains only its ~60M parameters (projections, self/cross-attention, VCE, task head) on top of frozen foundation-model embeddings, computed once and cached. Training is single-task, single-stage (not multi-task or alternating). Data: Gasperini et al. K562 CRISPRi screen, 4,605 training / 996 validation enhancer–gene pairs, split at the enhancer level (Sec. "Task and data").

# 6. Evaluation and Main Findings

Main result: Pearson $r=0.503$ predicting CRISPRi effect sizes on held-out pairs; an independent K562 replicate (STING-seq) correlates with Gasperini's at only $r=0.797$, a ceiling CDT reaches ~63% of. Key qualitative finding: a case study on the FNDC5 enhancer–gene pair (726 kb apart) where gradient analysis (not attention) identified a CTCF site as most prediction-critical, independently confirmed by Hi-C contact data as a physical loop anchor — attention and gradient top-20 positions overlapped only ~10% across 100 samples, showing the two methods answer different questions.

# 7. Architectural Strengths and Limitations

**Strengths**
- Directional cross-attention gives every attention weight one unambiguous interpretation by construction.
- Cheap to train (~60M trainable params) by freezing large foundation models and caching embeddings.
- Case study shows sequence-only training recovers 3D chromatin-contact structure never explicitly supervised.

**Limitations**
- Static, non-cell-specific RNA/protein tokens (v1) make cross-modal attention nearly uniform across samples (correlation >0.99), limiting cell-state specificity.
- 114 kb DNA window can miss one anchor of a chromatin loop pair, as in the FNDC5 case.
- Single cell line/dataset with a visible train/validation gap; generalization unproven.

# 8. Relevance to Our Project

**Direct relevance:** Medium. CDT uses different encoders (Enformer, scGPT, ESM-C) and a different task (CRISPRi enhancer effects), and it fuses pre-computed gene-level embeddings rather than per-sequence token embeddings. Its value to us is a design idea, directional cross-attention, plus two findings that bear on our Stage 1 results.

**What can be transferred?**
Our inference: on the codon grid, the one-directional pattern becomes DNA to RNA to protein cross-attention. Its translation step has each protein residue query the nucleotide positions:

$$\text{RNA}\to\text{Protein}:\ Q=\text{protein residues},\ K,V=\text{nucleotide codon positions}\ \Rightarrow\ \text{attn}\in\mathbb{R}^{T/3\times T/3}$$

Because our data is coding-only, both axes sit on the same codon grid, so each attention weight reads directly as "how much this codon informs this residue." That makes it an interpretable variant of the Isoformer-style cross-attention candidate. Two findings also carry over. First, the CDT-III sequel reports that RNA and protein changes often move in opposite directions (66.7 percent of genes with observable mRNA changes), which supports our Stage 1 reading of high uniqueness between these modalities. Second, CDT's attention and gradient attributions overlapped only about 10 percent, which suggests gradient-based attribution as a follow-up to our Track C attention-motif test, since attention alone may not be the right tool for locating what the encoder uses.

**What would need to change?** CDT's direction is built for DNA to RNA to protein over large genomic windows; on the coding-only stability data, the DNA and RNA encoders read the same sequence, so a DNA-to-RNA step carries no transcriptional information and the translation direction is the meaningful one. The full direction fits data with genomic context, such as IsoFormer's GTEx set. Enforcing one direction also discards whatever protein-to-RNA information a bidirectional model keeps, so it belongs as a variant tested against bidirectional cross-attention, not as a replacement.

**Key architectural takeaway:** The most useful idea from this paper is one-way nucleotide-to-protein cross-attention as an interpretable Stage 3 variant, with gradient attribution as a Track C follow-up.

# 9. Final Verdict for Literature Review

**Category:** Directional cross-attention; Interpretability
**Priority for our project:** Medium
**Reason:** It offers an interpretable fusion variant and evidence for RNA-protein uniqueness, but its encoders, task, and gene-level embeddings limit direct reuse.

## Compact Comparison Record

| Input | Tokenization | Backbone | Interaction | Objective | Our relevance |
|---|---|---|---|---|---|
| Frozen DNA (Enformer bins) + RNA (scGPT gene tokens) + protein (ESM-C) embeddings | Not raw-tokenized; pre-computed per-modality embeddings | Self-attention per modality + directional cross-attention + VCE pooling | One-directional DNA→RNA→protein cross-attention | Huber-loss regression on CRISPRi effect size | Medium: interpretable one-way nucleotide-to-protein cross-attention variant; evidence for RNA-protein uniqueness |
