# 1. Paper Information

**Title:** Multi-modal Transfer Learning between Biological Foundation Models (IsoFormer)
**Authors:** InstaDeep / BioNTech (de Almeida et al.)
**Year / Venue:** 2024, NeurIPS
**Paper Type:** Model
**Biological Domain:** Gene splicing / RNA isoform expression across tissues
**Primary Sequence Representation(s):** Mixed (DNA nucleotide, RNA nucleotide, amino acid)

**One-sentence summary:** IsoFormer fuses frozen pretrained DNA, RNA, and protein encoders via successive cross-attention to predict tissue-specific RNA-isoform expression, a task no single modality can solve alone.

# 2. Problem and Motivation

A single gene's DNA can be spliced into many RNA transcript isoforms, each expressed at different levels per tissue; DNA sequence alone cannot determine this because splicing/expression outcomes depend on RNA and protein-level information too. The paper explicitly targets three representations at once: DNA (Enformer or Nucleotide Transformer, 1- or 6-nucleotide tokens), RNA (Nucleotide Transformer reused, with U substituted for T), and protein (ESM-2, amino-acid tokens) (Table 1, Sec. 4.3).

# 3. Input Representation and Tokenization

Each modality uses its own pretrained tokenizer, none modified for this work: DNA — Enformer (1-nucleotide tokens, up to 190,000 bp) or Nucleotide Transformer v2 (6-nucleotide tokens, up to 12,282 bp); RNA — Nucleotide Transformer v2 reused with U→T substitution (same 6-mer scheme); protein — ESM-2 (1-amino-acid tokens, up to 2,047 aa). No new vocabulary, special tokens, reading-frame, or strand handling is introduced; DNA sequences are centered on each transcript's transcription start site, and RNA sequences longer than 12 kb are left-cropped to keep the 3' UTR (Sec. 4.2, Appendix).

**Mixed-representation handling.** The three modalities are processed by separate encoders, not concatenated or converted into one another. Each encoder's own vocabulary stays intact; there is no shared vocabulary or explicit modality/type token — modality identity is structural (which encoder produced the embedding). They are aligned only implicitly, through the aggregation module described below, rather than through explicit positional resampling.

# 4. Model Architecture

**Backbone:** three independent pretrained Transformer encoders (one per modality), each unchanged internally.
**Representation-specific components:** separate encoders; a cross-attention aggregation/fusion module; no shared latent space, no explicit type embeddings — fusion happens after encoding, per modality.

For hidden states $h_{dna},h_{rna},h_{prot}$ from the three encoders, a shared aggregation function $f_\phi^{agg}$ produces modality-centered multi-modal embeddings by successive cross-attention with residual connections (Eq. 1–2):

$$h'_{dna}=f_\phi^{agg}(h_{dna},h_{rna},h_{prot}),\quad h'_{rna}=f_\phi^{agg}(h_{rna},h_{dna},h_{prot}),\quad h'_{prot}=f_\phi^{agg}(h_{prot},h_{dna},h_{rna})$$

concatenated into the final embedding $h_{multi}=[h'_{dna},h'_{rna},h'_{prot}]$ (Eq. 3), fed to a linear head predicting per-tissue expression. If a modality is missing (e.g. non-coding transcripts with no protein), its cross-attention term is simply zeroed, so the architecture degrades gracefully.

Flow: $x_{dna}\to E_{dna}$, $x_{rna}\to E_{rna}$, $x_{prot}\to E_{prot}$ → successive cross-attention (each modality attends to the concatenation of the other two, residual, repeated across layers) → $h_{multi}$ → linear head → per-tissue expression.

# 5. Training Strategy

Objective: MSE regression on log-transformed expression values, end-to-end fine-tuning of the aggregation module and (per ablation) the pretrained encoders together — not a separate cross-modal alignment loss. Training is joint (all three encoders and the aggregator trained together in one pass), starting from pretrained weights. Data: GTEx transcript-level TPM across 30 tissues, ~170,000 transcripts (90,000 protein-coding), gene-level train/test split (Sec. 4.2).

# 6. Evaluation and Main Findings

Modality-ablated variants using Nucleotide Transformer for DNA and RNA (Table 2): DNA only $R^2=0.13$; RNA only $R^2=0.36$; protein only $R^2=0.20$; DNA+RNA $R^2=0.39$; all three $R^2=0.43$, best. Swapping Enformer in as DNA encoder raises the full model to $R^2=0.53$ (Table 3). An aggregation-module ablation (Table 4) shows the paper's cross-attention matches the best alternative (C-Abstractor) and beats Perceiver-Resampler and linear-projection variants. A transfer ablation (Table 5): reinitializing only the DNA encoder drops $R^2$ to 0.41, only RNA to 0.48 (vs. 0.53 pretrained) — evidence of genuine inter-modality transfer.

# 7. Architectural Strengths and Limitations

**Strengths**
- Cross-attention aggregation degrades gracefully when a modality is missing, without redesign.
- Demonstrated genuine cross-modal transfer via encoder-reinitialization ablations.
- Modular: a better single-modality encoder (Enformer) improves the whole system without touching fusion.

**Limitations**
- No explicit alignment step across token resolutions — cross-attention must learn correspondence implicitly, unverified directly — our inference.
- Modality-centered embeddings require running all three encoders even for single-modality-adjacent tasks.
- No evaluation on sequences mixing nucleotide and codon resolution within one input stream.

# 8. Relevance to Our Project

**Direct relevance:** High. IsoFormer is the source of one of the two fusion candidates Mina asked for: cross-attention over frozen pretrained encoders, using Nucleotide Transformer and ESM-2, the same encoder families as our pipeline. It is also the source of the encoder-reinitialization ablation we plan to use to prove the fused model uses both modalities.

**What can be transferred?**
Our inference: the aggregation function applies directly to our three modalities, with DNA, RNA, and protein encoders each producing its own hidden states:

$$h'_{dna}=f_\phi^{agg}(h_{dna},h_{rna},h_{prot}),\quad h'_{rna}=f_\phi^{agg}(h_{rna},h_{dna},h_{prot}),\quad h'_{prot}=f_\phi^{agg}(h_{prot},h_{dna},h_{rna})$$

followed by a linear head regressing mRNA stability. On the stability data all three inputs derive from one coding sequence. IsoFormer's own GTEx dataset, public on Hugging Face, supplies genomic DNA around the transcription start site, so running our Stage 1 analyses there tests the same encoder families where DNA carries information the protein lacks. The reinitialization ablation (their Table 5) transfers as is: reinitialize one encoder, retrain, and confirm the score drops. That check matters here because Stage 1 shows protein alone carries most of the signal (mean $R^2$ 0.103, versus 0.015 for the DNA encoder and 0.028 for the RNA encoder), so a fused model could score well while ignoring the nucleotide streams entirely. Graceful degradation when a modality is missing also lets one model handle examples that lack an input.

**What would need to change?** IsoFormer fine-tunes the encoders together with the aggregator; our plan keeps the encoders frozen and trains only the fusion module and head, which is cheaper but may give up some of the gain they attribute to end-to-end training. Their training set (about 170,000 transcripts) is far larger than our 981-sequence pilot, so cross-attention needs the scaled-up Stage 3 training set. Their RNA inputs include UTRs and splicing context; ours are coding-only, so the RNA-specific signal available to cross-attention is narrower. Notably, RNA outperformed protein alone in their task ($R^2$ 0.36 versus 0.20), the reverse of our Stage 1 stability result.

**Key architectural takeaway:** The most useful ideas from this paper are bidirectional cross-attention between token-level DNA, RNA, and protein encoder outputs, our primary Stage 3 candidate, and the reinitialization ablation that tests whether the fused model actually uses both inputs.

# 9. Final Verdict for Literature Review

**Category:** Multimodal fusion of frozen encoders; Cross-attention architecture
**Priority for our project:** High
**Reason:** It defines one of Mina's two primary fusion candidates, uses the same encoder families as our pipeline, and provides the ablation we need to verify genuine cross-modal use.

## Compact Comparison Record

| Input | Tokenization | Backbone | Interaction | Objective | Our relevance |
|---|---|---|---|---|---|
| DNA + RNA + protein sequences | Enformer (1-nt) / NT v2 (6-nt) DNA & RNA, ESM-2 (1-aa) protein | 3 frozen pretrained Transformer encoders + cross-attention aggregator | Successive cross-attention with residual connections, per modality | End-to-end MSE regression on isoform expression | High: one of Mina's two primary fusion candidates, plus the encoder-reinitialization ablation |
