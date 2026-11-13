# Stage 1: representation analysis

Stage 1 asks what the chosen pretrained encoders offer before deciding how to combine them: what a common predictor can recover, how representations relate, and whether simply joining them helps prediction. The retained analyses provide baselines and hypotheses for later alignment and fusion work; they do not establish a rule for choosing a fusion architecture.

## Why stability and GTEx

The two main notebooks examine different relationships between biological inputs and targets.

| Notebook | Prediction target | What the inputs contribute |
| --- | --- | --- |
| [mRNA stability](Stage1_stability.ipynb) | A profile-standardized, decay-derived score; higher means relatively more stable, not hours of half-life. | DNA and RNA versions of the same coding sequence, plus its protein translation. Translation loses synonymous codon distinctions. |
| [GTEx transcript expression](Stage1_gtex.ipynb) | Processed RNA-abundance targets across 30 human tissues. | Genomic DNA around the transcription start site; full transcripts for single-nucleotide RNA encoders or CDS for codon encoders; corresponding proteins. |

Stability compares representations of a common sequence origin. GTEx adds genomic and untranslated-region context absent from the protein input. It also changes the population and target, so differences between the notebooks cannot isolate the effect of extra sequence context. The [target guide](../../docs/methods/data-and-targets.md) traces the measurements and transformations; the [input comparison](../../docs/stage1-overview.md#published-comparisons-and-distinct-inputs) connects them to the papers.

Read stability first for the shared methods, then GTEx for the changed inputs and evaluation. Neither notebook requires the other to run. The separate [mRFP control](mrfp_representation_control.ipynb) extends the synonymous-recoding analysis, and [LucaOne](lucaone_representation_comparison.ipynb) adds a matched comparison of another representation family. Neither locally reproduces a learned fusion architecture.

## Comparing the encoders

The retained panel contains twelve frozen encoders and all 66 distinct pairs, including pairs within a modality. The [encoder table](../../docs/stage1-overview.md#models-and-input-settings) owns checkpoint and input settings. Selected compatible nucleotide encoders also supply the separate attention diagnostic.

Each encoder's final-layer token representations are averaged into one sequence vector. Its weights stay fixed while standardized ridge predictors are fitted to the vectors. This tests what the chosen predictor makes accessible, not the best performance possible with another head or fine-tuning. See [sequence preparation and pooling](../../docs/methods/sequences-to-predictions.md#from-sequences-to-embeddings) and [linear probing](../../docs/methods/sequences-to-predictions.md#linear-probing) for the computational details.

## Why the analyses have three tracks

| Track | Question | How it contributes |
| --- | --- | --- |
| **A: Decodability** | What can a predictor recover from the embeddings? | Biological-target probes test accessible predictive signal. Stability's GC/GC3 and calculated folding-proxy targets test other recoverable sequence properties. |
| **B: Representation geometry** | How do encoders organize corresponding examples? | Linear centered kernel alignment (CKA), neighborhood overlap and canonical correlation analysis (CCA) retrieval measure different forms of agreement. CKA and CCA retrieval are also compared before and after composition control. |
| **C: Attention and diagnostics** | Which exploratory relationships might help explain behavior? | Selected attention summaries, candidate RNA-pattern matches and agreement/error associations generate hypotheses. |

Calculated length and composition baselines use simpler sequence features as **inputs** to predict the biological target; this differs from predicting GC/GC3 as **targets** in Track A. The folding target is a computational pairing proxy, not a measured RNA structure.

Concatenation supplies a separate test of **predictive complementarity**: append two embedding vectors and compare their ridge prediction with the stronger individual under the same folds. Stability places it in Track B and GTEx beside its expression probes. Similar representation geometry does not guarantee a concatenation gain.

The [representation guide](representation-analysis.md) explains the metrics and controls. Removing the fitted linear composition component does not remove every composition effect. That fit uses all analysis rows before the PCA/CCA split, so post-control retrieval is not fully held out from preprocessing. Candidate motif matches and attention associations likewise do not establish regulatory mechanisms; GTEx's three-base UTR control blocks are not biological codons.

The **mRFP control** in [the stability notebook](Stage1_stability.ipynb#Pipeline-control:-synonymous-recoding) uses a separate expression dataset with different nucleotide sequences encoding the same protein. The dedicated [mRFP notebook](mrfp_representation_control.ipynb) asks whether nucleotide representations predict expression beyond calculated controls.

## How the stability data are evaluated

The main pilot samples rows without using the supplied `Split` assignments, then keeps identical retained sequences together in its five outer folds. The later [published-split comparison](Stage1_stability.ipynb#Published-split-comparison) samples the supplied training and test partitions instead. Those partitions overlap in sequence identity; the notebook also scores test sequences absent from the full supplied training partition.

The supplied validation rows are unused in that comparison, but `RidgeCV` performs internal penalty selection within training. Scaling precedes that inner selection, and it does not enforce the outer sequence groups. These [evaluation limitations](../../docs/methods/sequences-to-predictions.md#current-evaluation-limitation) remain attached to the scores. The grouped pilot and supplied-split probes are different evaluations, and neither reproduces BioLangFusion's complete training procedure.

## How GTEx is evaluated

The main GTEx probes use gene-grouped outer folds within the pilot's training half. They predict all 30 tissue targets and report mean tissue-specific $R^2$ on the supplied transformed-abundance scale. Length and composition baselines use the same rows and folds. The [target trace](../../docs/methods/data-and-targets.md#gtex-tissue-expression-labels) retains the log-base uncertainty and the pilot's omission of upstream normalization statistics.

The [published-split comparison](Stage1_gtex.ipynb#GTEx-published-split-comparison) separately evaluates the supplied test half, whose genes are excluded from training. CCA retrieval has a different holdout within the training half. Inner ridge selection does not enforce the outer gene groups and follows scaling fitted on the outer training subset. These small frozen probes are not a reproduction of IsoFormer's end-to-end training or published score.

## Representation controls and alternatives

### mRFP: changing the nucleotide sequence while keeping the protein fixed

The [mRFP notebook](mrfp_representation_control.ipynb) uses synonymous coding variants expressed in *E. coli*. Different codons can specify the same amino acid—`GCU` and `GCC` both specify alanine—so nucleotide sequences differ while the protein input stays identical. Protein encoders therefore provide a negative control; nucleotide encoders can distinguish the variants.

The target is the natural log of red fluorescence corrected using the green GFPuv reference. It reflects reporter production under the assay conditions, not a direct translation rate. The [target guide](../../docs/methods/data-and-targets.md#gfpuv-normalization-and-the-natural-logarithm) explains the reference correction, logarithm and worked examples. Scores concern prediction of that transformed target, not increases in protein production.

The experiment reuses Stage 1's sample and pooled embeddings. Ridge predictors compare embeddings with GC/GC3 content, codon frequencies and calculated folding summaries; a further comparison tests whether embeddings predict errors left by the controls. Identical sequences stay together in the grouped outer folds. The [notebook](mrfp_representation_control.ipynb#Build-sequence-controls) owns the precise controls, inherited prefix lengths and evaluation settings; the [ridge guide](../../docs/methods/sequences-to-predictions.md#linear-probing) explains the procedure.

The [recorded analysis](../../docs/run-records.md#2026-10-19-stage4-mrfp-control-the-mrfp-synonymous-control-research-plan-item-5) found positive average nucleotide-embedding gains above the tested controls, with mRNA-FM strongest in that sample. Fuller controls reduced the gains, and improvement was not uniform across all assignments. This concerns one reporter and selected linear controls; it does not establish RNA-structure accuracy, exclude nonlinear composition explanations or guarantee transfer to another task.

### LucaOne: comparing jointly pretrained representations

#### Separate inputs and the matched ridge comparison

[The LucaOne notebook](lucaone_representation_comparison.ipynb) compare representations on stability. The local interface encodes nucleotide and translated protein inputs separately, pools their token outputs, then fits ridge predictors to individual vectors or their concatenation. The same predictor family is applied to pooled BLF representations. This compares representation sets, rather than trained token-level fusion operators. The notebook explains the context-limit reason for separate encoding.

#### Stability findings and the GTEx limit

The [recorded comparison](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) reports **0.483 Spearman** for the two LucaOne vectors together versus **0.390** for concatenated BLF encoder vectors. It does not isolate joint pretraining as the cause. The historical best-single reference is selected by test performance, and supplied training/test sequence overlap limits unseen-sequence claims.

LucaOne's GTEx comparison is excluded under the selected 1,280-token whole-input scope. That limits this experiment, not every possible LucaOne approach. The [constituent follow-up](lucaone_representation_comparison.ipynb#Focused-follow-up:-do-the-two-LucaOne-streams-complement-each-other?) separately asks whether joining LucaOne's vectors improves on its own individual streams.

## What carries into the next stages

The retained [result tables](../../docs/stage1-overview.md#result-tables) support three useful observations:

- **Stability:** codon-level RNA and protein encoders are useful candidates under the linear probes.
- **GTEx:** transcript-length features provide a strong reference for subsequent models.
- **Representation agreement:** the specified composition features account for a substantial part of the measured agreement.

These findings nominate encoders and controls. They do not establish that enforcing alignment improves prediction, that learned fusion beats concatenation, or that attention reveals a biological mechanism. Some composition-controlled values also disagree across saved sources used by the [swap notebook](../stage4/encoder_swaps.ipynb); exact alignment-to-performance comparisons must identify their source and remain provisional where unresolved.

[Stage 2](../../README.md#stage-2-defining-alignment) distinguishes biological correspondence, representation agreement and predictive complementarity. [Stage 3](../stage3/README.md) establishes reference methods and baselines, and [Stage 4](../stage4/README.md) investigates encoder substitutions, fusion methods and biological context. Reusing a dataset does not make their scores interchangeable: filtering, pooling, heads and evaluation can change.

A prospective rule for selecting pairs or a new alignment objective remains a hypothesis requiring controlled evaluation on evidence not already used to develop it. The [Stage 4 proposals](../stage4/README.md#proposed-tests-of-fusion-explanations) pursue such explanations alongside the existing comparative experiments; they remain separate from completed Stage 1 analyses. The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx) retain tasks 8/9 for input mapping and published-split probes, and 7/15/16/19 for definitions, protocols, baselines and interpretation.

## Notebooks and supporting files

The notebooks retain calculations and saved outputs; [the Stage 1 overview](../../docs/stage1-overview.md) brings together encoder settings and result tables. [Representation analysis](representation-analysis.md) owns the detailed metrics and diagnostics, while [the shared method guides](../../docs/README.md#understand-the-methods) explain targets and prediction procedures.

[build_gtex_pilot.py](build_gtex_pilot.py) rebuilds the pilot dataset; reading existing results does not require rebuilding it. Embeddings and older caches remain local under `embeddings/` and `embeddings_legacy/`, following the [cache policy](../../README.md#results-caches-and-intermediate-files).

Saved outputs can be read without executing cells. The stability and GTEx notebooks use `Code/stage1/` as their working directory; follow each notebook’s setup and [the root environment guide](../../README.md#setup) before intended execution. The mRFP and LucaOne notebooks retain their historical result locations under `Code/stage4/results/`; their reading switches and explicitly enabled calculations are documented in the [runbook](../stage4/RUNBOOK.md#read-saved-results-without-starting-work). Moving these readers does not move caches or combine run cohorts.
