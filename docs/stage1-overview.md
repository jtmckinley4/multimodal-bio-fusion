# Stage 1 overview: DNA, RNA, and protein representations

This overview brings together the settings and retained results for Stage 1's twelve frozen DNA, RNA, and protein encoders. The [Stage 1 README](../Code/stage1/README.md) connects the investigation and evaluation limits; the two independent notebooks below own the calculations and saved outputs.

| Notebook | Inputs and target | Contents |
| --- | --- | --- |
| [mRNA stability](../Code/stage1/Stage1_stability.ipynb) | Coding sequences and stability labels from `mRNA_Stability.csv` | Input checks, embeddings, Tracks A–C, synonymous-recoding control, and the BioLangFusion published-split comparison |
| [GTEx transcript expression](../Code/stage1/Stage1_gtex.ipynb) | Genomic DNA, transcripts, proteins, and expression in 30 tissues from `GTEx_pilot.csv` | Independent setup, input checks, embeddings, Tracks A–C, and the IsoFormer published-split comparison |

Read stability first for the shared methods, then GTEx for the changed inputs and target. Each notebook has its own setup; reading saved results requires no execution. The [Summary](#summary) compares their retained evidence.

## Models and input settings

In the stability experiment, each retained coding sequence supplies one input per encoder: the sequence in DNA letters for a DNA encoder, the same sequence in RNA letters for an RNA encoder, and its translated amino acid sequence for a protein encoder. All encoders remain frozen during the analysis. Their parameters are not updated using the pilot dataset.

| Modality | Encoder | Pretraining data | Reason for inclusion |
|---|---|---|---|
| DNA | Nucleotide Transformer 500M human-ref | Human reference genome | Reference DNA encoder in `REFERENCE_KEYS` |
| DNA | Nucleotide Transformer v2 100M multi-species | Genomes of many species | DNA encoder in [BioLangFusion](../Papers/core/BioLangFusion.pdf), Table 3 |
| DNA | DNABERT-2 | Genomes of many species | Multi-species DNA encoder with byte-pair-encoding tokens |
| DNA | HyenaDNA large | Human reference genome | Long-context DNA encoder built from Hyena convolutions rather than attention; reads up to one million nucleotides, one token per nucleotide |
| RNA | RNA-FM | Non-coding RNA sequences from RNAcentral | Reference RNA encoder; RNA encoder in BioLangFusion |
| RNA | RiNALMo 150M | Non-coding RNA sequences from RNAcentral, Rfam, Ensembl, and Nucleotide | Second RNA encoder that reads one nucleotide per token, from a different group than RNA-FM |
| RNA | mRNA-FM | Messenger RNA coding sequences | RNA-FM variant pretrained on messenger RNA, with one token per codon |
| RNA | CaLM | Coding sequences from the European Nucleotide Archive | Codon-level encoder from a different group than mRNA-FM; grouped with the RNA encoders because it also reads the coding sequence one codon at a time |
| Protein | ESM-2 8M | UniRef50 protein sequences | Protein encoder in BioLangFusion, Table 3 |
| Protein | ESM-2 35M | UniRef50 protein sequences | Reference protein encoder |
| Protein | ESM-2 150M | UniRef50 protein sequences | Protein encoder in [IsoFormer](../Papers/core/Multi-Modal-Transfer-Learning.pdf) |
| Protein | ProtBERT | UniRef100 protein sequences | Protein encoder outside the ESM-2 family |

The stability inputs share one coding-sequence origin. DNA/RNA letter conversion preserves the nucleotide sequence, while translation loses synonymous codon distinctions, so the protein input does not retain all nucleotide information. The comparison therefore examines how pretrained encoders, trained on different corpora with different tokenizers and at different sizes, represent related biological information. This follows the modality definition used by [BioLangFusion](../Papers/core/BioLangFusion.pdf), Section 2.1, which gives the same coding sequence to DNA, RNA, and protein language models. The stability dataset carries no gene or transcript identifiers, so genomic DNA around each gene, such as promoters or introns, is not available here.

The `shared_code.datasets` registry records each dataset's setting, and `shared_code.encoders` holds the encoder definitions. The stability notebook uses this derived setting. [GTEx transcript expression](../Code/stage1/Stage1_gtex.ipynb#Distinct-inputs:-GTEx-transcript-expression) repeats the main analyses in the distinct setting, on IsoFormer's GTEx data, where DNA encoders read genomic DNA around the transcription start site, the single-nucleotide RNA encoders read the transcript, the codon-level RNA encoders read its coding sequence, and protein encoders read the protein.

## Research questions

- **Track A: Decodability.** Which properties can simple predictive models recover from each encoder's embedding?
- **Track B: Cross-modal geometry.** How similarly do the embedding spaces organize the same examples, including pairs within a modality, and how much agreement remains after the specified linear composition control?
- **Track C: Attention and diagnostics.** How does the selected attention summary of the nucleotide encoders relate to matches for RNA stability-associated sequence patterns, and do sample-level alignment and representational similarity relate to one another and to prediction?
- **Distinct inputs.** What changes on GTEx, where the DNA input includes promoter context and the transcript includes untranslated regions absent from the protein? This also changes the population and target, so it does not isolate the effect of additional sequence context.

## Role in the project

These analyses nominate encoders, controls, and hypotheses for later fusion comparisons. Agreement within a modality provides a reference for cross-modal agreement, but neither agreement nor a linear-probe gain determines which fusion operation to use. That requires controlled comparisons on matched inputs and evaluation.

The notebooks link to [sequence preparation](methods/sequences-to-predictions.md#from-sequences-to-embeddings), [linear probing](methods/sequences-to-predictions.md#linear-probing), and maintained functions in `Code/shared_code/`. Their [stability](../Code/stage1/Stage1_stability.ipynb#Implementation-displays) and [GTEx](../Code/stage1/Stage1_gtex.ipynb#Implementation-displays) display settings control whether full helper source is shown. For an intended execution, follow [Stage 1's running guidance](../Code/stage1/README.md#notebooks-and-supporting-files) and root [Setup](../README.md#setup).

---

## Summary

The saved outputs and execution counts come from a top-to-bottom run of each experiment notebook, identified in the [run records](run-records.md#recorded-runs). The results summarized here cover twelve frozen encoders, four per modality, on 981 retained mRNA stability sequences, where every input derives from one coding sequence, and on the GTEx pilot, where DNA, transcript, and protein are distinct inputs (999 train and 996 test transcripts).

**What the frozen embeddings make accessible.**

- On stability, the protein encoders and the two codon-level RNA encoders carry the most linearly accessible signal ($R^2$ about 0.11 to 0.14). The encoders that read single nucleotides, six-nucleotide tokens, or byte-pair tokens reach 0.03 to 0.08, and sequence length alone predicts almost nothing ($R^2$ 0.003).
- Nucleotide embeddings support stronger GC3 prediction than protein embeddings. Translation loses the original synonymous codon choice, but protein embeddings still predict GC3 partly through correlated sequence properties ($R^2$ 0.38 to 0.44); this is not recovery of the original codons. See the [GC3 probe](../Code/stage1/Stage1_stability.ipynb#GC3-prediction) and retained values below.
- On GTEx, RNA and protein encoders predict expression (0.09 to 0.18) far better than the DNA encoders reading the region around the start site (0.03 to 0.05), but four transcript-length features (0.24) beat every single encoder.
- Identical stability sequences in the full file have differing targets, with a median within-sequence standard deviation of about 0.49. Repeated rows can represent different cells or assays, so this variation is not established as measurement noise. The [dataset audit's ceiling estimate](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) is conditional, not a universal predictive limit.

**How the representations relate.**

- Strong agreement clusters among RNA-FM and RiNALMo, the ESM-2 models, and encoders reading single letters or short tokens of the same sequence. This pattern crosses modality labels but does not isolate the effects of tokenization, pretraining, or size. mRNA-FM has low whole-embedding CKA despite some stronger fitted-subspace correspondence.
- In the saved stability notebook, the linear control using letter, codon, and amino-acid frequencies reduces CKA by 52% to 90% for every pair involving DNA. This shows sensitivity to those features, not a complete separation of composition from other biological information. The [composition control](../Code/stage1/representation-analysis.md#composition-control) uses all analysis rows before the CCA split; nonlinear composition effects can remain. Some saved sources disagree on controlled values, as noted beside the [result tables](#result-tables).
- Three groups keep strong agreement after the control on both datasets: the protein encoders among themselves, RNA-FM with RiNALMo, and CaLM with the protein encoders. CaLM still retrieves its protein partner 58% to 69% of the time on stability, about 170 to 200 times chance.
- On GTEx, DNA-to-other-stream retrieval is low under this setup (Recall@1 at most 0.041). The changed sequence regions, population, and target do not isolate why the comparison differs from stability. The GTEx composition control describes genomic windows only through GC fraction, so it cannot remove their full composition.
- The ten dominant principal components carry much of the composition signal and measured agreement. They also support stability prediction, but whole embeddings outperform them for most encoders. ESM-2 models and CaLM with protein encoders retain agreement in lower-variance directions; mRNA-FM also has correspondence there. The [PCA comparison](../Code/stage1/Stage1_stability.ipynb#High--and-low-variance-directions) qualifies comparisons between blocks of different widths.

**Combining representations.**

- On stability, every pairing of a codon-level encoder with a protein encoder improves on the better encoder under all ten further fold assignments, by about 0.01 to 0.02. Larger concatenations give little or inconsistent additional benefit under the linear probe; the [four-protein combination](../Code/stage1/Stage1_stability.ipynb#Stability-prediction-from-concatenated-embeddings) has a small pinned improvement.
- On GTEx, 47 of 66 pairs gain under every assignment, every DNA encoder gains with every protein and codon-level encoder, and all twelve encoders together reach 0.239, the level of the length features.
- On the published splits, the frozen probes keep the cross-validated order. BioLangFusion's three encoders concatenated reach Spearman 0.364 against its published 0.539 to 0.563, and the GTEx three-modality sets reach $R^2$ 0.216 to 0.249 against IsoFormer's 0.43, in the same order of modalities.

**Diagnostics.**

- No alignment/error association survives the stated multiple-comparison correction across 66 pairs on either dataset. These diagnostics use the CCA test subsets, not an experiment training with an alignment loss; [stability](../Code/stage1/Stage1_stability.ipynb#Run-and-interpret-the-alignment-diagnostics) and [GTEx](../Code/stage1/Stage1_gtex.ipynb#Alignment,-prediction-error,-and-representational-similarity-on-GTEx) retain their evaluation limits.
- The tested position null moves labels within 100-nucleotide bins and leaves the observed attention gaps largely intact. Several gaps shrink under the codon-and-frame null. CaLM's remaining ARE-pentamer signal is exploratory: 500 shuffles cannot reach the multiplicity-corrected threshold for 36 tests. The [attention controls](../Code/stage1/Stage1_stability.ipynb#Controlling-the-attention-test-for-position-and-codons) do not establish regulation. On GTEx's 3' untranslated regions, RNA-FM shows no detected association and RiNALMo's ARE gap is not retained under the local three-nucleotide context null; those UTR blocks are not biological codons.

**Implications for alignment and fusion.** Report agreement before and after the stated composition control; raw similarity alone does not establish useful complementarity. The alignment/error diagnostic leaves the value of an alignment loss open. mRNA-FM and CaLM both read codons and gain when combined with protein embeddings, yet their geometries differ. These observations motivate testing codon-level RNA/protein pairs on stability and DNA with protein or codon-level encoders on GTEx. Include simple concatenation and transcript-length baselines: the GTEx concatenation score is close to the length baseline. These are hypotheses from the retained evidence, not a validated rule for choosing fusion.

**Qualification of the earlier summary.** Earlier wording treated repeated-target variation as a label-noise bound, attributed the published-score gap to training scale and token-level heads, and described absent diagnostic associations or position effects categorically. The [dataset audit](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons), linked diagnostic cautions above, and [published-comparison limits](#published-comparisons-and-distinct-inputs) support the narrower interpretations here. This prose correction changes no retained score or analysis and records no new execution; the composition-controlled source discrepancy remains unresolved.

### Result tables

These tables summarize the saved notebook outputs. Stability uses 981 retained sequences; GTEx cross-validation uses the pilot's 999 train transcripts. Probe entries give mean $R^2$ over five pinned folds with fold standard deviation, plus the mean over ten further assignments where shown.

Probes use ridge regression over five outer folds. Stability groups identical sequences; GTEx groups transcripts by gene. A hash of each group identifier and seed pins the assignment across machines. The ten further assignments reuse the same examples, so their spread is not an independent-repeat confidence interval. Inner ridge selection follows scaling and does not enforce the outer groups; see the [current evaluation limitation](methods/sequences-to-predictions.md#current-evaluation-limitation). The concatenated-embedding probe uses a linear model, distinct from the concatenation + MLP reference baseline.

**Saved-source distinction.** The controlled values below retain the saved notebook summary. The [swap notebook](../Code/stage4/tier2_encoder_swaps.ipynb#Relate-the-comparisons-to-Stage-1-alignment) instead reads the [figure capture](figures/stability/results.json), where some controlled values differ: RNA-FM/RiNALMo CKA is 0.486 there versus 0.378 here. Their producing states have not been reconciled; treat exact controlled-value comparisons as provisional. No score has been substituted between sources.

| Encoder | Modality | Stability $R^2$, pinned (fold SD) | Stability $R^2$, ten further assignments | GC3 $R^2$ | Share of embedding variance explained by composition | GTEx expression $R^2$, pinned (fold SD) |
| --- | --- | --- | --- | --- | --- | --- |
| Nucleotide Transformer 500M human-ref | DNA | 0.032 (0.043) | 0.046 | 0.968 | 0.618 | 0.034 (0.025) |
| Nucleotide Transformer v2 100M multi-species | DNA | 0.080 (0.032) | 0.074 | 0.986 | 0.804 | 0.036 (0.030) |
| DNABERT-2 | DNA | 0.054 (0.031) | 0.057 | 0.948 | 0.610 | 0.033 (0.028) |
| HyenaDNA large | DNA | 0.026 (0.017) | 0.019 | 0.896 | 0.924 | 0.031 (0.028) |
| RNA-FM | RNA | 0.052 (0.032) | 0.046 | 0.917 | 0.677 | 0.149 (0.037) |
| RiNALMo 150M | RNA | 0.078 (0.021) | 0.079 | 0.949 | 0.648 | 0.149 (0.047) |
| mRNA-FM | RNA | 0.127 (0.032) | 0.110 | 0.946 | 0.296 | 0.172 (0.044) |
| CaLM | RNA | 0.127 (0.058) | 0.132 | 0.982 | 0.472 | 0.120 (0.032) |
| ESM-2 8M | Protein | 0.127 (0.044) | 0.131 | 0.379 | 0.479 | 0.091 (0.022) |
| ESM-2 35M | Protein | 0.114 (0.049) | 0.123 | 0.387 | 0.446 | 0.104 (0.031) |
| ESM-2 150M | Protein | 0.144 (0.035) | 0.142 | 0.436 | 0.437 | 0.130 (0.043) |
| ProtBERT | Protein | 0.143 (0.041) | 0.135 | 0.440 | 0.415 | 0.163 (0.040) |

On GTEx, four transcript-length features alone reach an expression $R^2$ of 0.242 (0.036).

| Encoder pairs, stability data | Linear CKA | CKA after composition control | CCA Recall@1 (chance 0.0034) | Recall@1 after composition control |
| --- | --- | --- | --- | --- |
| Within ESM-2 | 0.553 to 0.744 | 0.376 to 0.643 | 0.942 to 0.955 | 0.701 to 0.869 |
| ProtBERT with ESM-2 | 0.199 to 0.312 | 0.093 to 0.162 | 0.818 to 0.873 | 0.481 to 0.622 |
| Within DNA | 0.270 to 0.736 | 0.038 to 0.113 | 0.591 to 0.773 | 0.103 to 0.162 |
| RNA-FM with RiNALMo | 0.809 | 0.378 | 0.931 | 0.704 |
| mRNA-FM with CaLM | 0.163 | 0.084 | 0.619 | 0.124 |
| DNA with RNA-FM or RiNALMo | 0.171 to 0.613 | 0.044 to 0.164 | 0.491 to 0.677 | 0.052 to 0.244 |
| DNA with mRNA-FM or CaLM | 0.030 to 0.517 | 0.012 to 0.075 | 0.271 to 0.632 | 0.027 to 0.086 |
| DNA with protein | 0.058 to 0.163 | 0.010 to 0.072 | 0.168 to 0.540 | 0.021 to 0.082 |
| RNA-FM or RiNALMo with protein | 0.116 to 0.339 | 0.035 to 0.161 | 0.196 to 0.509 | 0.021 to 0.223 |
| mRNA-FM with protein | 0.037 to 0.295 | 0.021 to 0.261 | 0.405 to 0.464 | 0.069 to 0.113 |
| CaLM with protein | 0.309 to 0.448 | 0.246 to 0.401 | 0.794 to 0.866 | 0.581 to 0.687 |

### Published comparisons and distinct inputs

The frozen encoders are compared with published fusion studies in two settings: [Setting A in the stability notebook](../Code/stage1/Stage1_stability.ipynb#Published-split-comparison), fitting probes on rows sampled from the published train split and scoring its test split, and [Setting B in the GTEx notebook](../Code/stage1/Stage1_gtex.ipynb#GTEx-published-split-comparison), whose rows carry separate DNA, transcript, and protein sequences. The Overleaf sections on the published split and on GTEx give the full comparison.

| Setting | Dataset | How DNA enters | Published comparison | Frozen encoders here |
| --- | --- | --- | --- | --- |
| A: derived modalities | CodonBERT mRNA stability (the current CSV) | A DNA encoder reads the coding sequence in DNA letters, an RNA encoder reads it in RNA letters, and a protein encoder reads its translation. | [BioLangFusion](../Papers/core/BioLangFusion.pdf), Table 1: best fusion Spearman 0.563 versus 0.553 for the best single encoder. | BioLangFusion's three encoders concatenated reach Spearman 0.364 on 981 published test rows; ProtBERT alone reaches 0.396. |
| B: distinct modalities | [IsoFormer GTEx transcript expression](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression) | Genomic DNA centered on the transcription start site, alongside the full transcript and the protein. | [IsoFormer](../Papers/core/Multi-Modal-Transfer-Learning.pdf), Table 2: three modalities reach $R^2$ 0.43 versus 0.36 for RNA alone. | The three-modality sets reach $R^2$ 0.216 to 0.249 on 996 published test transcripts, with the modalities in IsoFormer's order. |

Setting A supplies different representations of one coding-sequence origin; Setting B adds genomic and untranslated-region context absent from the protein. The frozen probes and published systems also differ in cohort, encoders, input handling, pooling, heads, and training or fine-tuning, so their score gap cannot identify the effect of any one difference. Using the published partitions does not reproduce the published systems. The [stability comparison](../Code/stage1/README.md#how-the-stability-data-are-evaluated) retains train/test sequence overlap, while the [GTEx comparison](../Code/stage1/README.md#how-gtex-is-evaluated) uses a separate test half from the within-training CCA holdout. These limits remain attached to the values above.
