# Stage 4: evaluating fusion and encoder choices

Stage 4 asks whether combining DNA, RNA and protein representations improves biological prediction, and what accounts for any improvement. Stage 1 examined what pretrained representations make accessible; Stage 2 developed the alignment questions; Stage 3 prepared the inputs and reference implementations. Here we compare encoder choices, fusion methods and sequence regions.

The retained prediction experiments use **frozen encoders**: pretrained weights stay fixed while a predictor, and sometimes fusion layers, learns from their outputs. A baseline is the complete prediction setup used as a reference, including its trained head and evaluation procedure. Stage 4 remains active: additional seeds, missing matched comparisons and the separate stability evaluation without sequence overlap strengthen these existing investigations.

## How the experiments connect

Three questions organize the investigation:

- **Representation usefulness:** what can a predictor recover from one encoder?
- **Predictive complementarity:** does combining representations improve on the relevant strong single-encoder predictor?
- **Fusion-method value:** does an interaction such as attention improve on a simpler combination of the same inputs?

A stronger replacement can improve a fused model without establishing complementarity. Likewise, successful concatenation does not establish a benefit from attention or gating. A controlled comparison that finds no gain is still useful evidence; each baseline tests a particular claim, rather than setting a score every valid experiment must beat.

| Analysis | Question and role |
| --- | --- |
| [mRFP control](tier1_mrfp_control.ipynb) | Do nucleotide embeddings add predictive information beyond calculated controls when every protein input is identical? |
| [BioLangFusion Phase A](tier2_blf_phaseA.ipynb) | How do the original encoders perform alone and with different fusion heads? |
| [Encoder swaps](tier2_encoder_swaps.ipynb) | Does replacing one encoder improve the representation, the combination, or the value of a fusion method? |
| [LucaOne, Parts 1–3](tier2_lucaone_isoformer.ipynb) | Does another source of representations help under the same pooled ridge predictor family? |
| [GTEx pilot, Part 4](tier2_lucaone_isoformer.ipynb) | How do combinations of genomic, transcript and protein inputs perform in a local cross-attention adaptation? |
| [GTEx data audit](tier2_isoformer_gtex_audit.ipynb) | What do the released targets, annotations, partitions and sequence lengths establish, including limits on cohort choices? |
| [Half-life and translation efficiency](tier3_halflife_te.ipynb) | Does extra RNA context help, and does protein add value once that context is available? |
| [Phase A appendix](tier2_blf_phaseA_appendix.ipynb) | What does a separate historical convolution configuration show? It is supporting evidence, not another required stage. |

Encoders reading a common coding sequence can expose different predictive patterns without adding an independently measured molecule. Genomic flanks and untranslated regions can instead supply sequence context absent from a translated protein. Neither guarantees a fusion gain. Cross-notebook differences also change datasets, targets, predictors or evaluation, so they cannot isolate one biological cause.

The `tier` prefixes group investigations within Stage 4; they are not execution dependencies. The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx) retain tasks 17/24 for Phase A/swaps, 18 for mRFP, 41 for LucaOne, 42 for half-life/TE and 44 for IsoFormer preparation.

## Read completed results from the top

Saved notebook outputs can be read without running cells. For a reading-only Run All of the retained experiment notebooks, explicitly set `READ_SAVED_RESULTS = True` and leave the separate follow-up switches off. A notebook saved after computation may retain different settings. This route reads available results and refreshes displays; missing or incompatible evidence is reported rather than computed. Follow the [reading instructions](RUNBOOK.md#read-saved-results-without-starting-work) for the applicable switches and the separate audit/preparation modes.

Reading historical records does not recover unknown producer settings or establish that current code reproduces them. For computation, use the [runbook](RUNBOOK.md): encoder swaps have a missing-work continuation route, but setting another notebook's reading flag to false does not guarantee that only missing work will run. Cache recovery, checkpoint identity and repair history remain in the [repair record](RUNBOOK.md#concrete-repair-list).

## The current experiments

### mRFP: changing the nucleotide sequence while keeping the protein fixed

The [mRFP notebook](tier1_mrfp_control.ipynb) uses synonymous coding variants expressed in *E. coli*. Different codons can specify the same amino acid—`GCU` and `GCC` both specify alanine—so nucleotide sequences differ while the protein input stays identical. Protein encoders therefore provide a negative control; nucleotide encoders can distinguish the variants.

The target is the natural log of red fluorescence corrected using the green GFPuv reference. It reflects reporter production under the assay conditions, not a direct translation rate. The [target guide](../../docs/methods/data-and-targets.md#gfpuv-normalization-and-the-natural-logarithm) explains the reference correction, logarithm and worked examples. Scores concern prediction of that transformed target, not increases in protein production.

The experiment reuses Stage 1's sample and pooled embeddings. Ridge predictors compare embeddings with GC/GC3 content, codon frequencies and calculated folding summaries; a further comparison tests whether embeddings predict errors left by the controls. Identical sequences stay together in the grouped outer folds. The [notebook](tier1_mrfp_control.ipynb#Build-sequence-controls) owns the precise controls, inherited prefix lengths and evaluation settings; the [ridge guide](../../docs/methods/sequences-to-predictions.md#linear-probing) explains the procedure.

The [recorded analysis](../../docs/run-records.md#2026-10-19-stage4-mrfp-control-the-mrfp-synonymous-control-research-plan-item-5) found positive average nucleotide-embedding gains above the tested controls, with mRNA-FM strongest in that sample. Fuller controls reduced the gains, and improvement was not uniform across all assignments. This concerns one reporter and selected linear controls; it does not establish RNA-structure accuracy, exclude nonlinear composition explanations or guarantee transfer to another task.

### BioLangFusion Phase A

[Phase A](tier2_blf_phaseA.ipynb) extends Stage 3's original-encoder comparison to E. coli, CoV-Vac and stability. Fungal expression supplies the earlier [Stage 3 build comparison](../stage3/03_blf_build_check.ipynb); Ab1 remains unavailable. These benchmarks use different biological targets despite sharing the column name `Value`.

| Benchmark | What the supplied target measures |
| --- | --- |
| [mRNA stability](../../docs/methods/data-and-targets.md#mrna-stability-standardized-decay-profiles) | Decay-derived scores standardized within experimental profiles; higher means relatively more stable, not hours of half-life. |
| [E. coli protein expression](../../docs/methods/data-and-targets.md#protein-expression-in-e-coli) | Gel-based low, medium and high recombinant protein-production categories. |
| [CoV-Vac degradation](../../docs/methods/data-and-targets.md#cov-vac-chemical-degradation-of-designed-rna) | Standardized chemical degradation under magnesium/heat exposure; higher means more degradation. |
| [Fungal expression](../../docs/methods/data-and-targets.md#fungal-expression) | Natural log of the source RNA-expression quantity; original normalization and units remain unresolved. |

DNA and RNA streams use alphabet-specific versions of the supplied sequence; protein uses its computational translation. For CoV-Vac fragments, that translation does not establish a biological coding frame. The [dataset audit](../stage3/01_blf_datasets.ipynb) and [target guide](../../docs/methods/data-and-targets.md#fungal-and-e-coli-benchmark-targets) retain the preprocessing choices and unresolved match to the paper.

**Models and evaluation choices.** The original encoders are NT v2 100M, RNA-FM and ESM-2 8M. Single-encoder and concatenation, gated-fusion and cross-attention comparisons use the local BLF implementation and TextCNN predictor family. The supplied train/validation/test partitions are retained; validation selects the epoch and single-encoder reference before test scoring. Regression uses Spearman correlation and E. coli uses classification accuracy.

**Equation 1 and RNA-control integration.** The main Equation 1 comparison and its RNA-FM codon-grid control are distinct from the [appendix-convolution cohort](tier2_blf_phaseA_appendix.ipynb). Their configurations and uncertainty procedures must not be pooled. The [Phase A record](../../docs/run-records.md#2026-10-20-stage4-blf-phasea-biolangfusion-phase-a-on-cov-vac-e-coli-and-mrna-stability-research-plan-item-1) and [appendix record](../../docs/run-records.md#2026-10-19t220959z-phasea-ecoli-three-seed-classification-comparison) preserve the respective settings and results.

#### Findings and interpretation

The recorded Phase A comparison found positive paired intervals for E. coli concatenation and gated fusion against the selected single; cross-attention's interval included zero. CoV-Vac did not show the same advantage. Stability retains extensive training/test sequence overlap. Evaluating an unseen-sequence subset is informative, but does not convert the original fit into a clean-split experiment. Matching some published scores also does not establish faithful reproduction; the [notebook](tier2_blf_phaseA.ipynb#What-this-comparison-establishes) retains the implementation qualifications.

### Phase B: changing one encoder at a time

The [swap notebook](tier2_encoder_swaps.ipynb) replaces one encoder while retaining the other two and the surrounding BLF comparison recipe. It distinguishes replacement fusion versus original fusion, fusion versus the validation-selected single, and attention/gating versus same-input concatenation. The [comparison section](tier2_encoder_swaps.ipynb#Report-representation,-complementarity-and-method-differences) presents these separately.

Swaps inherit the gated-fusion penalty selected for the original encoders. They therefore test substitutions under that recipe, not the best performance each replacement could achieve after individual tuning. Coverage tables identify available evidence by seed; missing repeats must not become an incomplete three-seed average. [Replacement settings and continuation](RUNBOOK.md#complete-the-encoder-swaps-one-seed-at-a-time).

#### RNA-FM and mRNA-FM: connecting pretraining to the fusion question

The [worked encoder comparison](tier2_encoder_swaps.ipynb#RNA-FM-and-mRNA-FM-as-a-worked-encoder-comparison) examines pretraining, supported capabilities, our results and competing explanations. RNA-FM's noncoding-RNA/nucleotide-token training differs from mRNA-FM's coding-sequence/codon-token training. Corpus, tokenization, capacity and predictor input dimensions change together, so the swap cannot isolate a pretraining mechanism.

On fungal expression, the recorded concatenation score rises from about **0.793 to 0.853 Spearman**, while **mRNA-FM alone scores 0.852**. This supports a better RNA encoder; the [paired comparison](tier2_encoder_swaps.ipynb#3.-What-the-saved-RNA-FM-and-mRNA-FM-results-show) leaves additional fusion benefit unresolved. The recorded single-encoder ordering favors mRNA-FM on fungal expression, E. coli and stability, but reverses on CoV-Vac. RNA abundance and chemical degradation are different targets, so this is not a universal encoder ranking.

Together with mRFP, these observations motivate asking what another representation adds beyond a strong task-specific predictor. They do not establish which biological features caused the scores, that other streams contain no useful information, or that one training seed establishes reliability.

### LucaOne: comparing jointly pretrained representations

#### Separate inputs and the matched ridge comparison

Parts 1–3 of [the LucaOne/GTEx notebook](tier2_lucaone_isoformer.ipynb) compare representations on stability. The local interface encodes nucleotide and translated protein inputs separately, pools their token outputs, then fits ridge predictors to individual vectors or their concatenation. The same predictor family is applied to pooled BLF representations. This compares representation sets, rather than trained token-level fusion operators. The notebook explains the context-limit reason for separate encoding.

#### Stability findings and the GTEx limit

The [recorded comparison](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) reports **0.483 Spearman** for the two LucaOne vectors together versus **0.390** for concatenated BLF encoder vectors. It does not isolate joint pretraining as the cause. The historical best-single reference is selected by test performance, and supplied training/test sequence overlap limits unseen-sequence claims.

LucaOne's GTEx comparison is excluded under the selected 1,280-token whole-input scope. That limits this experiment, not every possible LucaOne approach. The [constituent follow-up](tier2_lucaone_isoformer.ipynb#Focused-follow-up:-do-the-two-LucaOne-streams-complement-each-other?) separately asks whether joining LucaOne's vectors improves on its own individual streams.

### IsoFormer: combining GTEx sequence inputs

Part 4 of [the same notebook](tier2_lucaone_isoformer.ipynb#Part-4:-IsoFormer-style-GTEx-comparisons-across-seeds) reads a **separate GTEx pilot**. Each transcript supplies genomic DNA context, RNA, protein and 30 tissue-expression targets. The targets are transformed tissue-averaged RNA abundance, not tissue inputs or protein production; the [GTEx target trace](../../docs/methods/data-and-targets.md#gtex-tissue-expression-labels) records the transformation and normalization uncertainties.

The frozen pilot combines multiple updated streams through a local cross-attention model. It differs from the released IsoFormer computation described [below](#isoformer-reference-preparation). Its additional training seeds vary initialization and the validation subset while retaining the test partition; the notebook's coverage table distinguishes available fits from missing repeats. Original evidence remains in the [Stage 3 modality records](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders).

A matched comparison with pooled concatenation would test whether the local block adds value on the same inputs. It would assess the complete block, including extra parameters and normalization, rather than isolate attention weights. That proposed comparison is not a result established by the pilot's presence.

### Full-transcript half-life and translation efficiency

The [half-life/TE notebook](tier3_halflife_te.ipynb) introduces transcript regions absent from the earlier coding-sequence inputs. The 5′ untranslated region can affect translation initiation, while the 3′ untranslated region can contain sites relevant to RNA persistence. Both are absent from the translated protein. This motivates two questions: does extra RNA context improve prediction, and does adding protein help once that context is available?

These are different datasets and target constructions from the earlier stability benchmark. Human half-life is a **consensus stability score**, not hours. Human translation efficiency is an **RNA-abundance-adjusted ribosome-occupancy score**, not direct protein output or a raw ribosome/RNA ratio. The [target guide](../../docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explains their processing and examples.

The retained human encoder comparisons use pooled frozen embeddings and ridge, not reconstructed Saluki, RiboNN or BLF heads. Long transcripts are encoded in separate windows; pooling their outputs does not model cross-window interactions. The notebook retains the exact region conditions, suitability checks and partition procedure. Inspection of additional tables does not establish completed encoder comparisons on them.

#### Human findings and the limits of their scope

The [historical human results](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) report mean Pearson improvements of about **0.053 for half-life** and **0.030 for TE** when joining full-transcript and protein representations, relative to a single selected by mean test performance. These are descriptive correlation gains, not percentage increases in lifetime or protein output. The reported standard errors are not paired confidence intervals for the gains.

The repeated seeds vary held-out partitions; they are not neural-initialization repeats. Gene-grouped splits do not establish independence between related gene families, and human evidence does not establish mouse replication. The [matched follow-up](tier3_halflife_te.ipynb#Part-4:-test-complementarity-with-matched-evaluation) uses separately identified comparisons rather than silently replacing these historical scores.

### Focused comparisons using completed inputs

Three follow-ups make the comparison questions explicit: [swaps](tier2_encoder_swaps.ipynb) compare saved predictions against original, single-encoder and simple-fusion references; [LucaOne](tier2_lucaone_isoformer.ipynb) compares the combined vector with each constituent; [half-life/TE](tier3_halflife_te.ipynb) compares RNA, protein, concatenation and composition on common actual partitions.

The latter two use validation-selected ridge penalties and retained test predictions, separately from the historical internal-`RidgeCV` evaluations. Their paired group-resampled intervals condition on the fitted models; they do not repair split overlap or establish family-level generalization. The notebooks report available evidence, while the [focused run instructions](RUNBOOK.md#run-focused-reports-and-cpu-probes) own switches, reuse rules and destinations.

## IsoFormer reference preparation

The [GTEx audit](tier2_isoformer_gtex_audit.ipynb) connects targets, transcript annotations, partition identities and sequence lengths to possible cohort policies. Its saved JSONs describe the released table; they do not recover undocumented historical training membership or demonstrate predictive performance.

**Deferred from the main investigation.** The [shared reference implementation](../shared_code/fusion/isoformer_reference.py) follows the released Enformer/NT/ESM pathway: DNA queries RNA, then protein, before pooling updated DNA for prediction. Input/model foundations, missing-protein handling and adapters have synthetic checks. Pretrained integration, remaining validation/target/training choices and reproduction training remain unfinished. The [source record](../../docs/sources.md#isoformer-dataset-and-model-cards) retains the public evidence and source differences. The [runbook status](RUNBOOK.md#isoformer-reproduction-specification) distinguishes these shared foundations from privately retained reproduction preparation.

The GTEx audit and frozen pilot retain their evidence here. Future Stage 5 comparisons may reuse a verified reference operation with fixed encoders, but that is a controlled adaptation; the fuller published-training reconstruction remains deferred.

## How these comparisons inform the proposed designs

The strongest connection across these experiments is the distinction between choosing a better representation, adding useful context and improving the operation that combines inputs. Similar representations need not help when combined; a higher fusion score does not identify a biological mechanism.

The [Stage 5 guide](../stage5/README.md) develops proposed tests of those explanations: auxiliary error correction, sequence- versus position-wise gates, anchor-preserving attention and CDS/UTR correspondence. Each needs a stated expectation, an appropriate single and simpler-fusion reference, and an outcome that would challenge the explanation. Existing repeats and the grouped stability evaluation remain Stage 4 work, with a separate identity for a changed split; Stage 5 comparisons own their new matched baselines and repeats.

The existing [`tier4_designs.ipynb`](tier4_designs.ipynb) is the **Coupled Mamba comparison**: coupled recurrence, uncoupled recurrence and a projection-bottleneck control, compared against the BLF references. Its broad filename does not name a separate general design stage. The notebook supplies an implementation, not evidence by itself of completed fits or paper fidelity. A selected controlled Mamba comparison could belong to Stage 5, conditional on its resource question; full published-experiment reproduction is a separate scope. Species- or condition-held-out evaluations would ask further transfer questions where suitable data exist.

## Results and supporting files

Small result JSON/CSV files under [results/](results/) retain evidence; the [run records](../../docs/run-records.md#recorded-runs) identify producing settings and limitations. The notebooks provide the route from those files to interpreted tables. A stored score, an available prediction array and a verified repeat are different kinds of evidence.

Large data, embeddings and weights follow the [repository cache policy](../../README.md#results-caches-and-intermediate-files). Compatible frozen caches can be reused, but existence alone does not establish compatibility. Use the [runbook](RUNBOOK.md) for execution and recovery; its dated assignments are not a live report of what is running.

Shared [fusion utilities](../shared_code/fusion/), [LucaOne helpers](../shared_code/lucaone.py), [full-transcript helpers](../shared_code/mrnabench.py) and [pooled-cache handling](../shared_code/matrix_cache.py) support the notebooks. The [test guide](../../tests/README.md) documents their checks. Experimental choices and interpretation remain with the notebooks; reusable background remains in [the method guides](../../docs/README.md#understand-the-methods).
