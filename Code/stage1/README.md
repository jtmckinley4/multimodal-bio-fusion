# Stage 1: representation analysis

Stage 1 asks what the chosen pretrained encoders offer before deciding how to combine them: what a common predictor can recover, how representations relate, and whether simply joining them helps prediction. The retained analyses provide baselines and hypotheses for later alignment and fusion work; they do not establish a rule for choosing a fusion architecture.

## Why stability and GTEx

The two main notebooks examine different relationships between biological inputs and targets.

| Notebook | Prediction target | What the inputs contribute |
| --- | --- | --- |
| [mRNA stability](Stage1_stability.ipynb) | A profile-standardized, decay-derived score; higher means relatively more stable, not hours of half-life. | DNA and RNA versions of the same coding sequence, plus its protein translation. Translation loses synonymous codon distinctions. |
| [GTEx transcript expression](Stage1_gtex.ipynb) | Processed RNA-abundance targets across 30 human tissues. | Genomic DNA around the transcription start site; full transcripts for single-nucleotide RNA encoders or CDS for codon encoders; corresponding proteins. |

Stability compares representations of a common sequence origin. GTEx adds genomic and untranslated-region context absent from the protein input. It also changes the population and target, so differences between the notebooks cannot isolate the effect of extra sequence context. The [target guide](../../docs/methods/data-and-targets.md) traces the measurements and transformations; the [input comparison](../../docs/stage1-overview.md#published-comparisons-and-distinct-inputs) connects them to the papers.

Read stability first for the shared methods, then GTEx for the changed inputs and evaluation. Neither notebook requires the other to run. A separate mRFP dataset supplies the synonymous-recoding control below.

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

The **mRFP control** in [the stability notebook](Stage1_stability.ipynb#Pipeline-control:-synonymous-recoding) uses a separate expression dataset with different nucleotide sequences encoding the same protein. Stage 1 checks input/embedding behavior under that constraint. The [Stage 4 mRFP analysis](../stage4/README.md#mrfp-changing-the-nucleotide-sequence-while-keeping-the-protein-fixed) separately asks whether nucleotide representations predict expression beyond calculated controls.

## How the stability data are evaluated

The main pilot samples rows without using the supplied `Split` assignments, then keeps identical retained sequences together in its five outer folds. The later [published-split comparison](Stage1_stability.ipynb#Published-split-comparison) samples the supplied training and test partitions instead. Those partitions overlap in sequence identity; the notebook also scores test sequences absent from the full supplied training partition.

The supplied validation rows are unused in that comparison, but `RidgeCV` performs internal penalty selection within training. Scaling precedes that inner selection, and it does not enforce the outer sequence groups. These [evaluation limitations](../../docs/methods/sequences-to-predictions.md#current-evaluation-limitation) remain attached to the scores. The grouped pilot and supplied-split probes are different evaluations, and neither reproduces BioLangFusion's complete training procedure.

## How GTEx is evaluated

The main GTEx probes use gene-grouped outer folds within the pilot's training half. They predict all 30 tissue targets and report mean tissue-specific $R^2$ on the supplied transformed-abundance scale. Length and composition baselines use the same rows and folds. The [target trace](../../docs/methods/data-and-targets.md#gtex-tissue-expression-labels) retains the log-base uncertainty and the pilot's omission of upstream normalization statistics.

The [published-split comparison](Stage1_gtex.ipynb#GTEx-published-split-comparison) separately evaluates the supplied test half, whose genes are excluded from training. CCA retrieval has a different holdout within the training half. Inner ridge selection does not enforce the outer gene groups and follows scaling fitted on the outer training subset. These small frozen probes are not a reproduction of IsoFormer's end-to-end training or published score.

## What carries into the next stages

The retained [result tables](../../docs/stage1-overview.md#result-tables) support three useful observations:

- **Stability:** codon-level RNA and protein encoders are useful candidates under the linear probes.
- **GTEx:** transcript-length features provide a strong reference for subsequent models.
- **Representation agreement:** the specified composition features account for a substantial part of the measured agreement.

These findings nominate encoders and controls. They do not establish that enforcing alignment improves prediction, that learned fusion beats concatenation, or that attention reveals a biological mechanism. Some composition-controlled values also disagree across saved sources used by the [swap notebook](../stage4/tier2_encoder_swaps.ipynb); exact alignment-to-performance comparisons must identify their source and remain provisional where unresolved.

[Stage 2](../../README.md#stage-2-defining-alignment) distinguishes biological correspondence, representation agreement and predictive complementarity. [Stage 3](../stage3/README.md) builds the comparison machinery, and [Stage 4](../stage4/README.md) tests encoder substitutions and fusion methods. Reusing a dataset does not make their scores interchangeable: filtering, pooling, heads and evaluation can change.

A prospective rule for selecting pairs or a new alignment objective remains a hypothesis requiring controlled evaluation on evidence not already used to develop it. The [Stage 5 proposals](../stage5/README.md) pursue such explanations separately from completed Stage 1 analyses. The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx) retain tasks 8/9 for input mapping and published-split probes, and 7/15/16/19 for definitions, protocols, baselines and interpretation.

## Notebooks and supporting files

The notebooks retain calculations and saved outputs; [the Stage 1 overview](../../docs/stage1-overview.md) brings together encoder settings and result tables. [Representation analysis](representation-analysis.md) owns the detailed metrics and diagnostics, while [the shared method guides](../../docs/README.md#understand-the-methods) explain targets and prediction procedures.

[build_gtex_pilot.py](build_gtex_pilot.py) rebuilds the pilot dataset; reading existing results does not require rebuilding it. Embeddings and older caches remain local under `embeddings/` and `embeddings_legacy/`, following the [cache policy](../../README.md#results-caches-and-intermediate-files).

Saved outputs can be read without executing cells. For an intended execution, use `Code/stage1/` as the notebook working directory and follow [Setup](../../README.md#setup); these notebooks include computation and figure-export paths.
