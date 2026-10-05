# 1. Paper Information

**Title:** Understanding the Emergence of Multimodal Representation Alignment
**Authors:** Megan Tjandrasuwita, Chanakya Ekbote, Liu Ziyin, Paul Pu Liang (MIT, NTT Research)
**Year / Venue:** 2025, ICML (Proceedings of the 42nd International Conference on Machine Learning, PMLR 267)
**Paper Type:** Empirical analysis / theoretical framework (not a method paper; proposes no new architecture)
**Biological Domain:** None. General multimodal representation learning, validated on vision-language models and non-biological multimodal benchmarks. Relevant here because Mina cited it as the theoretical grounding for what "alignment" should mean before designing a biological fusion method.
**Primary Sequence Representation(s):** Not applicable. The paper studies frozen embeddings from independently trained unimodal encoders (vision, language, and synthetic feature vectors), not sequence data.

**One-sentence summary:** The paper shows that representation alignment between independently trained unimodal models is not a simple function of scale, as the "Platonic Representation Hypothesis" suggested, but is instead capped by how much modality-specific, task-relevant information (uniqueness) and modality dissimilarity (heterogeneity) exist in the data, and that alignment only reliably predicts downstream performance when redundancy dominates uniqueness.

# 2. Problem and Motivation

The Platonic Representation Hypothesis (Huh et al., 2024) claimed that independently trained vision and language models become more aligned as they scale, and that higher alignment predicts better multimodal performance. This paper treats that claim as too simple and asks two concrete questions: (1) when and why does alignment emerge implicitly between independently trained unimodal models, and (2) is alignment a reliable indicator of downstream task performance. This matters directly for any project, like ours, considering whether to build an explicit alignment objective into a fusion pipeline: if alignment does not reliably track performance, optimizing for it could waste effort or actively hurt.

# 3. Input Representation and Tokenization

No tokenization scheme is proposed or used; this is not a sequence modeling paper. The paper studies alignment along two controlled dimensions: interactions (redundancy R versus uniqueness U, meaning how much task-relevant information is shared between modalities versus unique to one) and heterogeneity (how dissimilar the two modalities are as data types, independent of the task).

Three experimental settings are used. First, fully synthetic data: task-relevant information is decomposed into a shared component and per-modality unique components, sampled as binary vectors, with heterogeneity introduced by passing one modality through an MLP of varying depth (Dφ). Redundancy and uniqueness are controlled directly by which components feed into the label. Second, real vision-language foundation models: alignment between DINOv2 vision models and language models (BLOOM, OpenLLaMA, LLaMA) is measured on a Wikipedia image-caption dataset, with uniqueness varied by using GPT-4 to generate captions that progressively diverge in meaning from the paired image. Third, real multimodal benchmarks from MultiBench (MOSEI, MOSI, URFUNNY, MUStARD, AVMNIST), spanning vision, audio, and language, plus a finetuning experiment on MM-IMDb.

**Mixed-representation handling.** Modalities are never merged into one stream or converted into each other. Each is encoded by its own independently trained model, and the two representations are compared only after the fact through alignment metrics on the same set of examples. There is no shared vocabulary, type embedding, or positional resampling.

# 4. Model Architecture

**Backbone:** none proposed; the paper analyzes existing encoders (synthetic MLPs, DINOv2 with BLOOM/OpenLLaMA/LLaMA, MultiBench models, and CLIP for the finetuning experiment).
**Representation-specific components:** separate, independently trained encoders per modality; no fusion module, shared latent space, or type embedding; alignment is measured post hoc rather than built in.

No trainable fusion architecture is proposed. Alignment itself is the object of study, computed with two metrics: Centered Kernel Alignment (CKA) with a linear kernel on synthetic data, and mutual k-nearest-neighbor alignment (mutual-KNN), a more scalable variant used in the original Platonic Representation Hypothesis paper, for the large real-world models. Both metrics take two mean-centered feature matrices from the same set of examples and return a single similarity score. For real vision-language models, alignment is computed across all layer pairs of the two encoders and the best pairwise score is reported, following Schrimpf et al. (2018).

The one place an explicit alignment objective is trained is the MM-IMDb experiment: CLIP vision and language encoders are finetuned with a combined loss, a supervised classification term plus a weighted CLIP contrastive alignment term, to test whether the alignment-performance relationship measured beforehand on frozen encoders actually predicts the effect of adding that explicit alignment pressure.

Flow: $x_A\to E_A$ and $x_B\to E_B$ (trained independently) $\to$ frozen features on shared examples $\to$ CKA / mutual-KNN alignment score $\to$ compared against each setting's task performance.

# 5. Training Strategy

For the synthetic experiments, single or multi-layer MLP encoders are trained independently per modality with AdamW, varying encoder depth and the heterogeneity transform's depth as separate knobs, then alignment is measured between the resulting frozen encoders after training, never during a joint objective. For the vision-language and MultiBench experiments, the base encoders are pretrained, off-the-shelf models used as-is; no additional training happens except in the MM-IMDb finetuning experiment described above, which trains only a linear classification head plus optionally weights an added CLIP alignment loss during finetuning.

# 6. Evaluation and Main Findings

Three findings, corresponding to the paper's three research questions. First, on when alignment emerges: the maximum achievable alignment falls as uniqueness and heterogeneity rise, consistently across model depths and sizes, with strong negative Spearman correlations between uniqueness and peak alignment (as strong as ρ = -0.833 to -1.000 across synthetic heterogeneity levels, and ρ = -0.950 on the real DINOv2-language experiment). Increasing model capacity compensates for heterogeneity only when uniqueness is low; that compensating effect disappears as uniqueness increases, meaning scale alone cannot force alignment when real modality-specific information exists.

Second, on whether alignment predicts performance: in highly redundant settings the alignment-performance correlation is strong and consistent. As uniqueness increases, that correlation decays toward zero and, for a meaningful share of cases, turns negative. Model depth, by contrast, correlates positively with performance regardless of alignment level, a more universal effect that does not depend on the alignment story at all. On real MultiBench datasets, this same pattern holds: alignment-performance correlation is weak or negative for tasks needing modality-specific information (sentiment and emotion recognition from video, audio, and text), but strongly positive on AVMNIST, a highly redundant task where both modalities encode the same digit identity.

Third, the practical payoff: the alignment-performance slope, measured using only frozen, never-explicitly-aligned unimodal encoders, predicts whether adding an explicit alignment loss during finetuning will help or hurt on a given task. On MM-IMDb, classes with a low or negative alignment-performance slope got worse as more weight was put on the CLIP alignment loss during finetuning, while classes with a high slope improved. This is demonstrated per-class on a real, held-out finetuning task, not just on synthetic data.

# 7. Architectural Strengths and Limitations

**Strengths**
- Isolates redundancy, uniqueness, and heterogeneity as independent variables using controlled synthetic data, then validates the same pattern on three different families of real data (vision-language foundation models, MultiBench, MM-IMDb), which is a rare degree of both controlled and real-world validation in one paper.
- Produces one directly actionable diagnostic, the alignment-performance slope, that can be computed cheaply from frozen encoders before committing to any explicit alignment objective or fine-tuning run.
- Generalizes rather than contradicts the Platonic Representation Hypothesis: it holds under redundancy and breaks down under uniqueness, rather than being simply right or wrong.

**Limitations**
- The paper itself states the "ideal" amount of alignment is dataset- and task-specific, meaning the diagnostic has to be recomputed for every new task rather than reused as a fixed threshold, our inference from their own conclusion in Section 6.2.
- The synthetic experiments define redundancy and uniqueness through simple binary feature masks, which may not capture the more continuous, biologically structured kind of redundancy and uniqueness present in a real central-dogma relationship, our inference.
- Most of their alignment measurements come from encoders that were trained end-to-end on that experiment's own task (synthetic MLPs, the MM-IMDb finetuning run); the vision-language and MultiBench experiments are closer to our frozen off-the-shelf setting, but the paper does not explicitly separate how much of their capacity finding depends on task-specific training versus being fully task-agnostic pretraining, our inference.

# 8. Relevance to Our Project

**Direct relevance:** High. DNA, RNA, and protein form exactly the kind of modality pair this paper is warning about: the central dogma is lossy in both directions, RNA carries UTR, intron, and splicing information that protein does not, and protein carries folding and post-translational information that RNA does not, meaning real, biologically motivated uniqueness almost certainly exists. This paper's central claim, that alignment stops reliably predicting fusion performance once uniqueness becomes non-trivial, is a direct caution against assuming a high CKA score means fusion will help, or that a low one means it will not.

**What can be transferred?**
The CKA and mutual-KNN alignment computation on frozen encoders, already implemented in our Stage 1 pilot. The redundancy, uniqueness, and heterogeneity vocabulary, useful for describing our own results to Mina in the terms her own framing already uses, rather than reporting a bare CKA number with no interpretive frame. Most valuably, the alignment-performance slope diagnostic. Stage 1 ran an exploratory sample-level adaptation of it on our frozen DNA, RNA, and protein embeddings and the mRNA stability label (the alignment-error association). The Spearman correlation between per-sample alignment and prediction error is 0.010 (p = 0.86) for DNA-RNA and -0.013 (p = 0.83) for DNA-protein, with no detectable relationship, and -0.131 (p = 0.025) for RNA-protein, a weak association that does not survive correction for three pairs. This is consistent with this paper's prediction that alignment stops tracking performance once uniqueness is high. The paper's own across-model version, comparing alignment and performance across several encoder configurations, has not been replicated.

**What would need to change?** Their synthetic experiments control the exact amount of redundancy and uniqueness by construction, which we cannot do with real biological sequence data. Our approximation has to come from the probe, CKA, CCA, and layer-wise CKA battery already built, read together as an indirect estimate of where each encoder pair sits (so far the DNA-RNA, DNA-protein, and RNA-protein pairs) on their redundancy-uniqueness-heterogeneity map, rather than a clean, dialed-in ground truth.

**Key architectural takeaway:** The most useful idea from this paper is the alignment-performance slope diagnostic. Our own Stage 1 results already look like the pattern this paper predicts for a uniqueness-containing pair: CKA and mutual-KNN show modest global alignment while CCA and retrieval reveal a strong, narrow shared subspace, and our layer-wise CKA breakdown suggested that shared subspace is driven mostly by composition. The composition control confirms it: removing letter, codon, and amino-acid frequencies cuts CKA by 72% to 81% and CCA Recall@1 to between 0.02 and 0.11. Our adapted alignment-error association already tests this warning on our own data and finds alignment does not predict error, which supports skipping an explicit alignment loss in Stage 2; the full across-model slope remains a possible follow-up.

# 9. Final Verdict for Literature Review

**Category:** Alignment theory; representation analysis methodology
**Priority for our project:** High
**Reason:** This is the paper Mina specifically pointed to as the theoretical grounding for defining what alignment means before designing a fusion method. It gives our Stage 1 pilot both a vocabulary (redundancy, uniqueness, heterogeneity) and a concrete, cheap diagnostic (the alignment-performance slope), an adaptation of which Stage 1 has already run on our own frozen nucleotide and protein embeddings.

## Compact Comparison Record

| Input | Tokenization | Backbone | Interaction | Objective | Our relevance |
|---|---|---|---|---|---|
| Synthetic redundant/unique feature vectors; real image-caption pairs (DINOv2 + BLOOM/OpenLLaMA/LLaMA); MultiBench video/audio/language | Not applicable, representation-level analysis rather than a tokenization scheme | Independently trained per-modality encoders (synthetic MLPs; pretrained vision and language foundation models; MultiBench transformers) | None learned end-to-end; alignment is measured post-hoc between frozen, independently trained encoders, except one finetuning experiment that adds an explicit CLIP alignment loss to test the slope diagnostic | Per-modality task loss; CKA and mutual-KNN used as an analysis metric, not a training objective | High: defines the vocabulary and the alignment-performance slope diagnostic our Stage 1 findings should be read through |
