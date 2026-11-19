# Stage 4: comparative and explanatory experiments

Stage 4 asks when combining representations improves biological prediction and what accounts for that improvement. [Stage 1](../stage1/README.md) characterizes representations and controls; [Stage 3](../stage3/README.md) establishes reference methods and baselines, including their validation comparisons and repeats. Here we use those references to investigate encoder choices, sequence context and fusion behavior.

The retained prediction experiments use **frozen encoders**: pretrained weights stay fixed while a predictor, and sometimes fusion layers, learns from their outputs. A baseline is the complete prediction setup used as a reference, including its trained head and evaluation. Comparative results and proposed explanatory interventions belong to the same investigation, but an available implementation, an executed result and an unrun proposal remain different evidence.

## How the experiments connect

Three questions organize the investigation:

- **Representation usefulness:** what can a predictor recover from one encoder?
- **Predictive complementarity:** does combining representations improve on the relevant strong single-encoder predictor?
- **Fusion-method value:** does an interaction such as attention improve on a simpler combination of the same inputs?

A stronger replacement can improve a fused model without establishing complementarity. Likewise, successful concatenation does not establish a benefit from attention or gating. A controlled comparison that finds no gain is still useful evidence; each baseline tests a particular claim.

| Investigation | Question and current role |
| --- | --- |
| [Encoder swaps](encoder_swaps.ipynb) | Does replacing one encoder improve the representation, the combination, or the value of a fusion method? Completed Newton three-seed comparisons, with historical results kept separate. |
| [Grouped stability](grouped_stability.ipynb) | Do the original and mRNA-FM comparisons survive testing on unseen exact nucleotide sequences? The 36-fit protocol runs separately. Its results are outside this Newton return. |
| [Half-life and translation efficiency](halflife_translation_efficiency.ipynb) | Does extra RNA context help, and does protein add value once that context is available? Existing pooled comparisons and separately enabled matched follow-ups. |
| [Auxiliary correction, gates and attention](#proposed-tests-of-fusion-explanations) | Which controlled change helps a predictor use another representation? Three seeds of gate and attention comparisons are available for CoV-Vac and E. coli. Auxiliary correction has its own workflow. |
| [Transcript-region correspondence](region_aware_fusion.ipynb) | Does mapping protein features only to annotated coding regions help beyond simpler controls? Implemented A.6 pipeline for a matched cohort and new token caches; preparation, extraction and training have not run. |
| [Molecular-partner tasks](#transfer-to-other-molecular-relationships) | Do selected explanations transfer to RNA–protein, DNA–protein or protein–ligand pairs? Five tables built; arms not yet established. |

Encoders reading a common coding sequence can expose different predictive patterns without adding an independently measured molecule. Genomic flanks and untranslated regions can instead supply sequence context absent from a translated protein. Neither guarantees a fusion gain. Cross-notebook differences can also change datasets, targets, predictors or evaluation, so they cannot isolate one biological cause.

The BLF benchmark and appendix comparisons, frozen GTEx reference, GTEx audit and adapted Coupled Mamba now belong to [Stage 3](../stage3/README.md). The mRFP control and LucaOne ridge comparison belong to [Stage 1](../stage1/README.md). Their original result directories and dated run identities are retained.

## Read completed results from the top

Saved notebook outputs can be read without running cells. For a reading-only Run All of the retained experiment notebooks, set `READ_SAVED_RESULTS = True` and leave separate follow-up switches off. This reads available results and refreshes displays; missing or incompatible evidence is reported rather than computed. The [reading instructions](RUNBOOK.md#read-saved-results-without-starting-work) also cover the relocated reference and representation notebooks.

Reading historical records does not recover unknown producer settings or establish that current code reproduces them. For computation, use the [runbook](RUNBOOK.md): encoder swaps have a missing-work continuation route, but setting another notebook's reading flag to false does not guarantee that only missing work will run. Cache recovery, checkpoint identity and repair history remain in the [repair record](RUNBOOK.md#concrete-repair-list).

## The current experiments

### Phase B: changing one encoder at a time

The [swap notebook](encoder_swaps.ipynb) replaces one encoder while retaining the other two and the surrounding BLF comparison recipe. It distinguishes replacement fusion versus original fusion, fusion versus the validation-selected single, and attention/gating versus same-input concatenation. The [comparison section](encoder_swaps.ipynb#Report-representation,-complementarity-and-method-differences) presents these separately.

The Newton cohort contains 504 completed fits across four benchmarks and training seeds 0, 1 and 2. Single-encoder baselines and the inherited MIL penalty (`lambda=0.01`) retain the historical seed-0 validation choices; all scores in the current three-seed reports come from Newton. Swaps test substitutions under that fixed recipe, not the best performance each replacement could achieve after individual tuning. [Result-set separation](RUNBOOK.md#select-a-returned-result-set) and [replacement settings](RUNBOOK.md#complete-the-encoder-swaps-one-seed-at-a-time).

**Focused E. coli comparison.** Newton ran both BLF heads with RNAErnie and ERNIE-RNA across seeds 0, 1 and 2, retaining the original DNA and protein encoders. RiNALMo had the highest mean test accuracy under both heads. RNAErnie’s two heads finished within 0.05 percentage points of each other. For ERNIE-RNA, concatenation scored higher on all three seeds. [Three-seed table in encoder swaps](encoder_swaps.ipynb#Replacing-RiNALMo-with-RNAErnie-and-ERNIE-RNA) · [Verified return](../../docs/run-records.md#2026-11-19-verified-newton-return-for-e-coli-gates-and-attention) · [Execution recipe](../../execution/arcc/encoder_swaps/README.md#focused-e-coli-rna-extension) (Gantt/Details task 24).

#### Newton three-seed findings

The [saved three-seed reports and interpretation](encoder_swaps.ipynb#Newton-three-seed-findings) distinguish improvement over a selected single from improvement over concatenation with the same inputs. Scores are means of three fitted models' scores, not prediction ensembles. The counts below use saved 95% paired test-sequence-group intervals; these condition on the three fitted seeds and are unadjusted for multiple comparisons.

| Benchmark | Fusion versus its selected single | Cross-attention versus concatenation |
| --- | --- | --- |
| CoV-Vac degradation | Only RiNALMo cross-attention has an interval wholly above zero against its selected single. | Higher in all eight replacements, with all eight intervals above zero. |
| E. coli production | Concatenation improves on its selected single in five of eight replacements. | Lower mean in all eight; seven intervals are entirely below zero. |
| Fungal expression | Strong RNA singles leave little demonstrated additional value for fusion. | Positive intervals for NT 500M and ESM-2 150M: two of eight replacements. |
| mRNA stability | mRNA-FM concatenation remains close to mRNA-FM alone. | Lower mean in all eight; four intervals are entirely below zero. |

No MIL-versus-concatenation interval is wholly positive across the 32 replacements. This concerns the implemented gate and inherited penalty, not every gating design. A gain over the validation-selected single within one encoder combination also does not establish a gain over the strongest single elsewhere in the experiment. The notebook makes these baseline qualifications explicit; the retained historical intervals and seed-0 examples are separate evidence.

#### RNA-FM and mRNA-FM: connecting pretraining to the fusion question

The [worked encoder comparison](encoder_swaps.ipynb#RNA-FM-and-mRNA-FM-as-a-worked-encoder-comparison) examines pretraining, supported capabilities, our results and competing explanations. RNA-FM's noncoding-RNA/nucleotide-token training differs from mRNA-FM's coding-sequence/codon-token training. Corpus, tokenization, capacity and predictor input dimensions change together, so the swap cannot isolate a pretraining mechanism.

On fungal expression, the Newton mean concatenation score rises from **0.7956 to 0.8514 Spearman** after replacing RNA-FM with mRNA-FM, while **mRNA-FM alone scores 0.8510**. The additional fusion difference is only **+0.0004 [−0.0156, 0.0162]**. The earlier values of about 0.793, 0.853 and 0.852 remain [historical seed-0 evidence](encoder_swaps.ipynb#3.-What-the-historical-RNA-FM-and-mRNA-FM-results-show), not part of the Newton mean. Both cohorts support investigating the stronger RNA representation while leaving its additional fusion benefit unresolved. The Newton single-encoder ordering favors mRNA-FM over RNA-FM on fungal expression, E. coli and stability, but reverses on CoV-Vac. RNA abundance and chemical degradation are different targets, so this is not a universal encoder ranking.

Together with mRFP, these observations motivate asking what another representation adds beyond a strong task-specific predictor. They do not establish which biological features caused the scores or that other streams contain no useful information.

### Full-transcript half-life and translation efficiency

The [half-life/TE notebook](halflife_translation_efficiency.ipynb) introduces transcript regions absent from the earlier coding-sequence inputs. The 5′ untranslated region can affect translation initiation, while the 3′ untranslated region can contain sites relevant to RNA persistence. Both are absent from the translated protein. This motivates two questions: does extra RNA context improve prediction, and does adding protein help once that context is available?

These are different datasets and target constructions from the earlier stability benchmark. Human half-life is a **consensus stability score**, not hours. Human translation efficiency is an **RNA-abundance-adjusted ribosome-occupancy score**, not direct protein output or a raw ribosome/RNA ratio. The [target guide](../../docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explains their processing and examples.

The retained human encoder comparisons use pooled frozen embeddings and ridge, not reconstructed Saluki, RiboNN or BLF heads. Long transcripts are encoded in separate windows; pooling their outputs does not model cross-window interactions. The notebook retains the exact region conditions, suitability checks and partition procedure. Inspection of additional tables does not establish completed encoder comparisons on them.

#### Human findings and the limits of their scope

The [historical human results](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) report mean Pearson improvements of about **0.053 for half-life** and **0.030 for TE** when joining full-transcript and protein representations, relative to a single selected by mean test performance. These are descriptive correlation gains, not percentage increases in lifetime or protein output. The reported standard errors are not paired confidence intervals for the gains.

The repeated seeds vary held-out partitions; they are not neural-initialization repeats. Gene-grouped splits do not establish independence between related gene families, and human evidence does not establish mouse replication. The [matched follow-up](halflife_translation_efficiency.ipynb#Part-4:-test-complementarity-with-matched-evaluation) uses separately identified comparisons rather than silently replacing these historical scores.

### Focused comparisons using completed inputs

Three follow-ups make the comparison questions explicit: [swaps](encoder_swaps.ipynb) compare saved predictions against original, single-encoder and simple-fusion references; [LucaOne](../stage1/lucaone_representation_comparison.ipynb) compares the combined vector with each constituent; [half-life/TE](halflife_translation_efficiency.ipynb) compares RNA, protein, concatenation and composition on common actual partitions.

The latter two use validation-selected ridge penalties and retained test predictions, separately from the historical internal-`RidgeCV` evaluations. Their paired group-resampled intervals condition on the fitted models; they do not repair split overlap or establish family-level generalization. The notebooks report available evidence, while the [focused run instructions](RUNBOOK.md#run-focused-reports-and-cpu-probes) own switches, reuse rules and destinations.

## Reliability and the next comparison

The Newton swap batch supplies the three independent training seeds and saved paired reports; its seed variation remains distinct from uncertainty over test-sequence groups. Those completed runs retain the supplied partitions: an unseen test subset or grouped bootstrap does not remove training overlap.

The [grouped-stability experiment](grouped_stability.ipynb) implements the agreed follow-up to [section A.1 of the advisor protocol](../../docs/Biological_Fusion_Plan.docx). It keeps identical nucleotide sequences together in a fixed 70/15/15 partition and compares the original BLF encoder combination with its mRNA-FM replacement. Six unique single-input controls and three fusion methods for each combination give 12 fits per training seed, 36 across seeds 0/1/2. Targets, existing observations and frozen caches are retained; prediction heads start fresh. Split seed 42 and this bounded panel are project choices, not values prescribed by the advisor. The [runbook](RUNBOOK.md#run-the-grouped-stability-comparison) separates reading, fitting and reporting.

The [Newton run record](../../docs/run-records.md#2026-11-19-newton-e-coli-encoder-extension-and-controlled-studies) documents completion of the seed-0 panel. Exact-sequence grouping tests generalization to new nucleotide sequences. Holding out related families, species or conditions would require different grouping. The new partition changes training and test membership, so an old-to-new score difference measures sensitivity to that split. The panel was informed by existing test findings and remains exploratory.

Unfinished reference repeats remain with their [Stage 3 methods](../stage3/README.md); completed Newton swap comparisons and separately enabled matched half-life/TE comparisons remain with their experiments here. A new question need not wait for every notebook to finish, but it needs the evidence and evaluation relevant to that question. Each new experiment includes matched baselines, training seeds and uncertainty analysis. The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx) retain tasks 17/24/28 for reference comparisons, swaps and reliability, and 42 for transcript-region work; the Newton reports provide newer swap-repeat evidence than the historical inventory in task 24.

## Proposed tests of fusion explanations

The [advisor's revised protocol](../../docs/Biological_Fusion_Plan.docx) starts from frozen encoders with trainable prediction and fusion components. The [gate diagnostics](gate_diagnostics.ipynb) and [anchor cross-attention](anchor_cross_attention.ipynb) notebooks read three-seed Newton comparisons for CoV-Vac and E. coli. Each compares the original encoder combination with its mRNA-FM replacement. Use the [saved-results settings](RUNBOOK.md#read-saved-results-without-starting-work) to refresh and save their tables and figures locally. Correction and transcript-region studies use separate inputs and result sets. Partner-task extensions remain conditional.

### Can auxiliary features correct the anchor's errors?

The **anchor** is the strongest eligible single-encoder predictor selected using validation, including its trained head. The comparison asks whether auxiliary representations improve on that complete predictor under matched evaluation, making useful information accessible after the strong individual predictor has been fitted.

Fit auxiliary representations to the anchor's regression errors and add the predicted correction to its output. The prepared panel covers CoV-Vac, fungal expression and grouped stability; classification logit correction remains outside this panel. Obtain training errors from **group-aware out-of-fold predictions**, so each example's group was excluded from the head that predicted it. Validation selects the correction; final test data stay outside selection.

Compare auxiliaries separately and jointly, with shuffled-input and parameter-matched random-feature controls. The [mRFP analysis](../stage1/mrfp_representation_control.ipynb) offers a related precedent starting from calculated controls, not an encoder anchor. Success supports accessible complementarity; failure need not mean that no useful information exists.

[Liang et al.](../../docs/sources.md#liang-et-al-quantifying-and-modeling-multimodal-interactions) provide prior work on task-related redundancy, unique information and synergy, including model selection. Our proposed correction test measures predictive improvement; it does not implement their information decomposition. Connecting representation relationships to fusion choices therefore needs a specific biological contribution beyond that existing idea.

### Do gates preserve useful contributions?

If an auxiliary representation helps under a controlled comparison, a gate should not be assumed to retain that contribution. The existing BLF gate assigns one weight per modality per sequence. Compare its softmax competition with and without entropy regularization, independent sigmoid gates, and mean-plus-max summaries. Record per-example weights alongside prediction: a small weight alone does not establish harmful suppression or biological importance. Control feature-scale changes when comparing gate normalizations.

The contribution may also vary along a sequence. Compare joint-input sequence and position gates with the same affine scorers, alongside a nongated positional model. The nongated control keeps the full-width half-sum of projected features and adds a rank-2 affine update. Its 5,402 update parameters nearly match the joint gates' 5,403; equal counts do not imply equal expressive capacity. The older BLF sequence gates remain separate comparisons because their scorers also differ. Keep encoders, alignment and the downstream predictor fixed. The [token-cache route](../stage3/blf_token_embeddings.ipynb) supplies a starting point; equal widths do not establish correspondence, and codon pooling changes information. CoV-Vac's translated fragment has no established biological reading frame. Gate patterns can motivate perturbations, not establish local biological mechanisms.

### Should attention retrieve auxiliary information while retaining the anchor?

Let anchor positions query auxiliary streams, potentially of different lengths, then add gated updates through a skip connection. Compare anchor-only, concatenation, ordinary attention, anchor-skip attention and sequence- versus position-gated updates under the common training budget. An additional nonlinear concatenation control retains the anchor skip and nearly matches the gated-attention parameter count, testing whether attention helps beyond similarly sized local processing. Parameter matching does not equalize expressive capacity or runtime.

BLF's joint-context attention is a different computation. A skip retains a representation, not a guaranteed prediction score. After the correction, gate and attention comparisons, an optional experiment could compare raw auxiliary embeddings with their reconstruction residuals and both together. Those feature residuals differ from target errors; they are not necessarily uniquely biological information.

### Does respecting transcript regions improve prediction?

Start with human half-life, then translation efficiency, building on the [region checks and pooled comparisons](halflife_translation_efficiency.ipynb). Keep their [different target meanings](../../docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explicit. Protein residues correspond to coding-sequence (CDS) nucleotides, with no corresponding translated residue in the untranslated regions (UTRs).

Retain nucleotide-resolution RNA and map protein features to annotated CDS positions:

| Comparison | Question |
| --- | --- |
| RNA alone → add protein, zero-filled outside CDS | Does the additional representation help? |
| Add a CDS/UTR availability mask | Does region information help beyond zero filling? |
| Add position-wise gates | Does adaptive weighting help beyond the mask? |
| Add CDS-restricted gated attention with an RNA skip | Does this complete design improve on local gating? Its different skip/readout prevents attributing the contrast to attention alone. |

Compare native-resolution with codon-pooled RNA and true with appropriately permuted masks. Supplying a mask as a feature does not enforce exclusion: multiply the protein-update gate by the CDS mask to block direct updates outside CDS. This tests correspondence, not whether proteins can regulate UTRs. These are new token-level experiments; existing pooled probes cannot recover positions lost through averaging.

Within local gating, compare sequence and position gates with a full-width nongated control. Within attention, hold CDS update availability fixed while comparing unit, sequence and position weights. Select the strongest single on seed-0 validation across the RNA-only configurations and native protein-only, then hold that choice across training seeds. RNA remains the specified attention anchor even if protein wins that baseline comparison. The [control plan](../../docs/research-execution-plan.md#control-repairs-before-newton) records these adaptations, fit counts and optional comparisons.

### Transfer to other molecular relationships

Three pairings, five built tables, one notebook: [`partner_tasks.ipynb`](partner_tasks.ipynb). These are protocol steps 7 and 8 plus a ninth this project proposes, and they are Julian's; Part A, steps 1 to 6, is Chase's. The advisor's protocol names two of the three; step 9, DNA–protein binding, is this project's own addition and is not yet approved. Tables are under `data/pairs/`, and the code is `pair_tasks`, `pair_models`, `pair_training`, `pair_sources`, `pair_dna`, `entity_embeddings` and `pair_study` in `shared_code/fusion/`. The data is built and the code is tested; arms are being fitted, and no comparison is established.

#### Step 7: RNA–protein binding

Binding pairs an RNA with a separate protein partner rather than its translation. Two tables: `rpi2241` (4,479 pairs; 2,238 measured positives with the 2,241 random negatives the source supplies) and `npinter-matched` (22,516 pairs; 11,258 positives with degree-matched sampled negatives). The two are not interchangeable. Measurement before fitting found that degree matching inverted the popularity signal in `npinter-matched` rather than removing it, so its negatives pose a different question from the supplied negatives of `rpi2241`, and the two should not be pooled.

#### Step 9: DNA–protein binding (this project's addition, pending approval)

A DNA region paired with the transcription factor that binds it. This is **binding, not affinity**: the label is whether a factor bound a region, not a measured strength. It is proposed because it gives the DNA encoder a task where DNA carries information the protein does not, which no existing benchmark here does. Built from DAP-seq as `dapseq`: 15,905 pairs, 7,905 positive 201 bp windows on peak summits for 40 factors in 15 families, each with one same-chromosome negative matched on GC content within 0.05 and clear of every peak of that factor. Regions group into 10 kb genomic bins (15,384 regions in 7,816 bins) and factors group by family; a split may not break either. Two limits belong with any result: DAP-seq is an *in vitro* assay, so it measures what a factor can bind on naked DNA rather than what it occupies in a cell, and 15 families leaves four in the test partition under a dual-disjoint split.

#### Step 8: protein–ligand affinity

Protein–small-molecule pairs provide another transfer test, and the only one of the three with a continuous label. Two tables taken directly from Harvard Dataverse: `davis` (25,772 measured pKd over 379 proteins and 68 ligands) and `kiba` (117,657 KIBA scores over 229 proteins and 2,068 ligands). Affinity and inhibition measurements are not interchangeable, and neither table pools endpoints. One property of `davis` qualifies every number from it: 18,343 of its values (71.2%) sit at exactly the pKd floor of 5.0, which is the assay's 10,000 nM censoring limit and not a measurement, so a rank correlation there largely separates the censored floor from everything else. `kiba` has no such floor. The ligand side reads through ChemBERTa 77M at a pinned revision, added to the encoder registry for this task.

#### Evaluation, common to all three

Splits hold out left entities, right entities or both, over groups rather than individual entities. The dual-disjoint split is the one a method should be judged on, and it is also the thinnest: it discards 38% to 49% of each table and leaves 132 to 2,088 test pairs. Two controls run with every evaluation. The degree baseline predicts from each entity's training labels alone and reads no sequence; measured before any fit, it reaches 0.867 AUROC on `rpi2241` with proteins held out, so that split is close to a popularity contest and an arm there must clear 0.867 rather than 0.5. On the dual-disjoint split it scores exactly 0.500 by construction, since neither entity of a test pair was seen in training. Partner reassignment keeps each positive pair's anchor and gives it a partner nobody measured. Group membership is exact-identity only, so every number these tables can produce is optimistic until entities are clustered at a stated threshold.

### Organize the preparation notebooks around questions

The four notebooks keep the advisor's steps connected while giving each question its own settings, controls and results. Supporting inspections examine fold feasibility, control strength, capacity, contribution scales and region/window assumptions. Position and attention-update gates use the protocol's joint-input affine sigmoid formulation; the notebooks distinguish those source requirements from the team's implemented control choices. They default to reading saved evidence, with separate switches for preparation, extraction, fitting, diagnostic inference and statistical reporting. The [Newton recipe](../../execution/arcc/fusion_studies/README.md) prepares a source-only snapshot and points to existing inputs rather than copying caches.

| Project home or preparation notebook | the advisor's protocol steps | Gantt task |
| --- | --- | --- |
| Existing investigations and separate stability evaluation | 1: reliability and grouped evaluation | 24, 28, 41, 42 |
| [Anchor correction](anchor_correction.ipynb) | 2: auxiliary target-error correction | 45 |
| [Gate diagnostics](gate_diagnostics.ipynb) | 3–4: sequence and position gates | 46 |
| [Anchor cross-attention](anchor_cross_attention.ipynb) | 5: skip, attention and gated updates | 25 |
| [Region-aware fusion](region_aware_fusion.ipynb) | 6: CDS/UTR correspondence | 42 |
| [Partner tasks](partner_tasks.ipynb) (Julian) | 7–8: RNA–protein binding and protein–ligand affinity, plus 9: DNA–protein binding as this project's addition | 47, 48 |

[Gantt and Details](../../docs/Research_Project_Plan_Gantt.xlsx) owns dependencies and schedule qualifications. Task 42 spans two scopes: the existing pooled comparisons remain in Stage 4, and the proposed positional intervention is a new experiment within the same investigation.

### Keep published references and proposed methods distinct

The [IsoFormer audit](../stage3/isoformer_gtex_audit.ipynb) has saved evidence; the [shared model/input foundations](../shared_code/fusion/isoformer_reference.py) have synthetic checks. Pretrained integration and research training remain unfinished. [Full experimental reproduction is deferred](RUNBOOK.md#isoformer-reproduction-specification) from the main investigation route; the private preparation notebook and specification are not supplied with this checkout. Faithful reference operations can support a controlled comparison without reconstructing the paper's whole experiment; generic anchor-query attention should not be labeled an IsoFormer reproduction.

The [Coupled Mamba reference](../stage3/coupled_mamba_reference.ipynb) supplies an adapted implementation and controls. Its corresponding DNA/RNA/protein codon streams do not by themselves establish a pipeline for separately paired molecules. Further Mamba work is conditional on a measured long-sequence runtime or memory bottleneck (tasks 20/21), not a prerequisite for gate/attention tests. The reference methods remain in Stage 3; using one on a new partner task requires a separate input and evaluation design.

### Decide what the evidence would justify

If true masks help but local gates do not, region information has stronger support than adaptive weighting. If concatenation performs similarly to attention, extra complexity has not established its value. These are possible outcomes, not findings.

Select candidate gains, inputs, grouping rules, controls and tuning budgets before fitting. Existing test results informed these hypotheses; stronger prospective transfer claims need an independent final evaluation. Reusing caches saves extraction, but out-of-fold fits and long token inputs still cost computation. Measure a representative workload before expanding the experiment matrix.

### Source and decision provenance

The [advisor protocol](../../docs/Biological_Fusion_Plan.docx) retains the original equations, controls and dataset candidates; its [source entry](../../docs/sources.md#minas-revised-biological-fusion-protocol) records attribution and reading scope. The [execution plan](../../docs/research-execution-plan.md#control-repairs-before-newton) identifies the selected panels and project-specific controls. Earlier result inventories remain in the [historical runbook](RUNBOOK.md#historical-execution-snapshots); current completed Newton findings are summarized above. Additional datasets and architecture comparisons remain conditional on their own question, inputs and evaluation design.


## Results and supporting files

Small result JSON/CSV files under [results/](results/) retain evidence; the [run records](../../docs/run-records.md#recorded-runs) identify producing settings and limitations. The notebooks provide the route from those files to interpreted tables. A stored score, an available prediction array and a verified repeat are different kinds of evidence.

Large data, embeddings and weights follow the [repository cache policy](../../README.md#results-caches-and-intermediate-files). Compatible frozen caches can be reused, but existence alone does not establish compatibility. Use the [runbook](RUNBOOK.md) for execution and recovery; its dated assignments are not a live report of what is running.

Shared [fusion utilities](../shared_code/fusion/), [LucaOne helpers](../shared_code/lucaone.py), [full-transcript helpers](../shared_code/mrnabench.py) and [pooled-cache handling](../shared_code/matrix_cache.py) support the notebooks. The [test guide](../../tests/README.md) documents their checks. Experimental choices and interpretation remain with the notebooks; reusable background remains in [the method guides](../../docs/README.md#understand-the-methods).
