# Stage 4: evaluating fusion and encoder choices

Stage 4 asks whether combining DNA, RNA, and protein representations improves predictions of biological outcomes such as expression and RNA stability, and what accounts for any improvement. Stage 1 examined which properties could be predicted from the pretrained representations and how those representations relate. Stage 2 developed a working definition of alignment, and Stage 3 built and checked the implementations used here. We now compare encoder choices, ways of combining their outputs, and the sequence regions supplied to them.

The encoders remain **frozen**: their pretrained weights are not updated during these experiments. Separate predictors, such as ridge regression or a TextCNN neural network, are fitted to predict the biological target from the representations. Some experiments also train layers that combine encoder outputs. A **baseline** is the complete prediction setup used to judge what a change contributes, evaluated on the same held-out examples and target. For example, a predictor using one encoder provides a reference for a predictor combining several encoders. Baselines can be trained and tuned.

The experiments below move from the mRFP sequence control to BioLangFusion's original encoders and their replacements, then to LucaOne, IsoFormer and full-transcript comparisons. Each section explains the biological question, the chosen comparison and what the retained evidence supports. The `tier` prefixes group investigations by purpose; the dependencies within each experiment determine which earlier inputs it needs.

## The current experiments

### mRFP: changing the nucleotide sequence while keeping the protein fixed

**Notebook:** [mRFP expression control](tier1_mrfp_control.ipynb) · **Plan:** `Details` task 18.

#### What changes when the protein stays the same?

This experiment reuses the [mRFP expression dataset](../../README.md#mrfp-expression-control) introduced as a control in Stage 1. Different codons can specify the same amino acid, so these nucleotide sequences can differ while encoding the same red fluorescent protein, expressed in E. coli. Stage 1 checked that the protein input remains unchanged while nucleotide representations can vary. We now use each row's expression score to ask whether the nucleotide embeddings help predict expression beyond the selected composition and folding controls.

#### Comparing sequence controls with encoder embeddings

The controls turn each coding sequence into a small set of numerical inputs for predicting its expression score. **GC content** is the fraction of nucleotides whose base is guanine (G) or cytosine (C). **GC3 content** calculates that fraction only at the third nucleotide in each codon—positions 3, 6, 9, and so on in the supplied coding frame. These are inputs calculated directly from the sequence; the encoder embeddings may also capture these properties.

| Calculated input | Part of the supplied coding sequence used | Meaning |
| --- | --- | --- |
| GC content | Whole sequence | Fraction of nucleotides that are G or C. |
| GC3 content | Positions 3, 6, 9, and so on | G-or-C fraction at third codon positions, using the supplied coding frame. |
| 64 codon frequencies | Nonoverlapping triplets throughout the sequence | Fraction of complete codons belonging to each possible triplet. |
| Nussinov pairing fraction | First 200 nucleotides | A calculated base-pairing summary. |
| ViennaRNA minimum folding energy | First 60 nucleotides, and separately the whole sequence | Two calculated folding-energy summaries. |

The folding prefixes are fixed regions, not sliding windows. Ridge regression receives the calculated numbers and learns their relationship to expression across training rows; it does not move a window along a sequence. The folding values are computational proxies, not measured RNA structures. Length and amino-acid composition are omitted from these controls because they are constant across this sample. Protein embeddings supply a further control: an unchanged protein input should not distinguish the expression differences.

The experiment reuses Stage 1's 1,000-row sample, pooled embeddings, and standardized ridge procedure. Ridge predicts a weighted sum of its inputs while penalizing large coefficients. Applying the same predictor family to the controls and embeddings tests what their different inputs make accessible to this linear procedure. For the incremental comparison, one ridge model predicts expression from the controls; another uses embeddings to predict errors left by the controls, estimated within the training data. Their predictions are added and evaluated on held-out rows.

The sample is drawn without using the CSV's supplied `Split` assignments. Instead, five-fold evaluation keeps identical sequence strings together, so copies do not appear on both sides of an outer training/test comparison; ten further assignments assess sensitivity to that partition. This continues Stage 1's pilot procedure and differs from the published-split experiments below. The 200-nucleotide prefix is also inherited from Stage 1. Reuse explains continuity, but the original justification for the exact sample size, folding-prefix lengths, and ridge penalty range has not been established. The notebook gives the settings; the [ridge guide](../../docs/methods/linear-probing.md) explains the fitting procedure and its internal-validation limits.

#### What the results suggest and their limits

The [October 19 record](../../docs/run-records.md#2026-10-19-stage4-mrfp-control-the-mrfp-synonymous-control-research-plan-item-5) reports positive average gains from nucleotide embeddings above the tested controls, with mRNA-FM strongest in that sample. The fullest controls reduced those gains, and CaLM's gain was positive in nine of ten assignments, so the evidence does not support an improvement in every assignment for every encoder. These observations concern one reporter protein and the specified controls. A gain over linear composition controls does not rule out a nonlinear relationship with the same composition features. The results motivate testing candidate encoders on other tasks, without establishing that an advantage will transfer.

### BioLangFusion Phase A

**Notebook:** [BioLangFusion Phase A](tier2_blf_phaseA.ipynb) · **Plan:** `Details` task 17.

#### Question and benchmark inputs

Phase A carries the BioLangFusion implementation built in Stage 3 onto the remaining available benchmarks, establishing the reference for later encoder substitutions. It reuses the token embeddings prepared in Stage 3.

The notebook uses three benchmark tables examined in the [Stage 3 audit](../stage3/01_blf_datasets.ipynb), applying the same preparation rules: E. coli protein-expression classification, CoV-Vac degradation, and mRNA stability. Fungal expression supplies the earlier [Stage 3 build comparison](../stage3/03_blf_build_check.ipynb); together, these cover the four available BioLangFusion benchmarks, with Ab1 still unavailable. Stability is the same source table examined in Stage 1, now under different filtering and a trained token-level prediction procedure. The local BioLangFusion preparation limits sequences to 999 nucleotides but does not apply Stage 1's start-codon filter, which would discard most CoV-Vac fragments. The exact match between that local length filter and the paper's preprocessing remains unresolved in the [target guide](../../docs/methods/biological-targets-and-controls.md#fungal-and-e-coli-benchmark-targets).

DNA and RNA encoders receive alphabet-specific versions of the supplied nucleotide sequence; the protein encoder receives its computational translation. For CoV-Vac fragments, that translation remains an input-construction assumption, not confirmation of a biological coding sequence. The [dataset descriptions](../../README.md#other-biolangfusion-benchmarks) and [target guide](../../docs/methods/biological-targets-and-controls.md#fungal-and-e-coli-benchmark-targets) explain the targets without treating expression classes, stability scores, and degradation scores as interchangeable measurements.

#### Models and evaluation choices

BioLangFusion's original encoders are Nucleotide Transformer v2 100M for the DNA input, RNA-FM for RNA, and ESM-2 8M for protein. Concatenation joins their representations, gated fusion learns how to weight their contributions, and cross-attention learns interactions between them. A convolutional prediction head, TextCNN, converts the resulting representations into predictions. Single-encoder baselines use the same prediction-head family on one encoder's outputs. The pretrained encoders remain frozen while the prediction and fusion components are fitted.

Here the supplied train, validation, and test partitions are retained to compare against the published benchmark setup. Validation results select settings such as the training epoch and the single-encoder reference, before inspecting test performance. This differs from Stage 1's initial sampled, grouped-fold analysis and from its separate published-split ridge comparison. Matching the source dataset alone therefore does not make their scores interchangeable.

#### Reading the separate Phase A records

##### The appendix-convolution cohort on E. coli (Chase, October 19 to 20)

The [appendix comparison notebook](tier2_blf_phaseA_appendix.ipynb) reads a separate historical cohort produced by [run_blf_phaseA.py](run_blf_phaseA.py). Its [execution record](../../docs/run-records.md#2026-10-19t220959z-phasea-ecoli-three-seed-classification-comparison) identifies 27 E. coli fits across three seeds, using the appendix convolution with kernel 3, stride 2, and padding 2. These results live in `results/blf_phaseA/ecoli_proteins/`; the Equation 1 cohort lives in `results/phaseA/ecoli_proteins/`. Keep their configurations and uncertainty procedures distinct when comparing them.

The retained runner currently calls `train.run_identity`, which is absent from the current shared trainer. That is an interface mismatch to resolve before rerunning it. The saved-result reader has a different role; this static finding does not establish that it cannot read the historical results. Current runtime compatibility remains unverified.

##### Equation 1 and RNA-control integration

The Equation 1 convolution and the RNA-FM codon-grid control belong to the corresponding Phase A comparison. They should not be inferred from results produced by the appendix runner. A software check of the grid establishes a property of that implementation, not the predictive effect of the control. Read the [Phase A record](../../docs/run-records.md#2026-10-20-stage4-blf-phasea-biolangfusion-phase-a-on-cov-vac-e-coli-and-mrna-stability-research-plan-item-1) for the reported control runs and the [integration record](../../docs/run-records.md#2026-10-19-stage3-integration-preservation-of-completed-results-no-research-execution) for preservation and version boundaries. Historical per-seed intervals and the Equation 1 comparison's seed-averaged intervals describe different analyses and should retain those labels.

#### Findings and interpretation

The [October 20 comparison](../../docs/run-records.md#2026-10-20-stage4-blf-phasea-biolangfusion-phase-a-on-cov-vac-e-coli-and-mrna-stability-research-plan-item-1) found that fusion's benefit depended on the benchmark and head. E. coli concatenation and gated fusion had positive paired intervals relative to the selected single encoder; cross-attention's interval included zero. CoV-Vac did not show the same advantage. On stability, extensive sequence overlap between the supplied training and test partitions limits interpretation. Scoring the subset of test sequences absent from training adds a useful comparison, but does not turn the original fit into a new clean-split experiment.

Matching some published scores is evidence to inspect alongside the implementation differences; it does not by itself establish faithful reproduction. The notebook and run record retain the convolution, input preparation, model selection, and repetition details.

### Phase B: changing one encoder at a time

**Notebook:** [Encoder swaps](tier2_encoder_swaps.ipynb) · **Plan:** `Details` task 24.

#### Why replace an encoder?

Once the original-encoder comparison is available, we can ask whether its outcome depends on which encoder fills each input role. Phase B reuses the same benchmark source tables, replacing one encoder while retaining the surrounding BioLangFusion method and evaluation conditions. Stage 1's stability and GTEx observations help identify candidates; testing them on fungal expression, E. coli, or CoV-Vac also asks whether those observations transfer to other prediction tasks.

There are two different improvements to examine. A replacement may improve the same fusion head relative to its original version. We also need to compare that fused model with the strongest single encoder in the replacement setting. If the replacement alone accounts for the improvement, the experiment supports choosing a better encoder without establishing an additional benefit from fusion.

#### Replacements and inherited evaluation choices

The configured replacements make that comparison concrete. Each replaces one input role while the other two encoders remain in place:

| Input role | Original encoder | Configured replacements |
| --- | --- | --- |
| DNA | Nucleotide Transformer v2 100M | Nucleotide Transformer 500M human-reference; HyenaDNA |
| RNA | RNA-FM | RiNALMo; mRNA-FM; CaLM |
| Protein | ESM-2 8M | ESM-2 35M; ESM-2 150M; ProtBERT |

These are configured comparisons, not a list of completed runs. The stability sweep is limited to mRNA-FM, CaLM, and ESM-2 150M because of its larger data and cache requirements. The notebook connects mRNA-FM's selection to the mRFP result, CaLM to another codon-level RNA representation, and ESM-2 150M to a larger protein encoder. Those are reasons to test the candidates, not evidence that their advantages transfer to stability.

Phase B also keeps the gated-fusion penalty selected for the original encoders on Phase A's validation data, rather than tuning it again for each replacement. This tests replacements under the inherited training recipe; it does not measure the best performance each replacement could achieve after separate tuning. The paper's architecture and the chosen local settings therefore need to remain visible alongside the encoder names.

#### Retained results and unresolved comparisons

The [October 22 restoration record](../../docs/run-records.md#2026-10-22-phaseb-result-restoration-reuse-four-completed-swaps-on-main) preserves four benchmark/encoder configurations: mRNA-FM and CaLM, each on CoV-Vac and E. coli, at seed 0. These are four configurations involving two replacement encoders, not four distinct encoders or the complete swap study. The [October 25 reset record](../../docs/run-records.md#2026-10-25-encoder-swap-unverified-cache-reset) identifies which evidence was retained after the cache review. Those records describe dated states; follow later records for additional completed work.

Those recorded seed-0 comparisons do not measure variation across training seeds. The notebook also distinguishes its imported Stage 1 alignment measurements from alignment measured on each Phase B benchmark. Some composition-controlled values disagree between the saved Stage 1 sources and remain provisional, as explained in the notebook; their relationship to swap performance must be read with that uncertainty.

### LucaOne: comparing jointly pretrained representations

**Notebook:** [LucaOne and IsoFormer comparisons](tier2_lucaone_isoformer.ipynb), Parts 1–3 · **Plan:** `Details` task 41.

#### Question and stability data

**LucaOne** provides gene and protein representations from a jointly pretrained model family. On the same stability source table used in Stages 1 and 3, each coding sequence supplies a nucleotide input and a translated protein input. This comparison asks how those representations perform alongside BioLangFusion's original encoders when the same predictor family is used.

#### Separate inputs and the matched ridge comparison

The local LucaOne interface encodes the nucleotide and protein inputs separately and averages token outputs into one vector per sequence for each input. We fit ridge regression to either vector or their concatenation, and apply the same procedure to pooled outputs from BioLangFusion's original encoders. This holds the predictor family constant while comparing representations; it differs from Phase A's trained token-level fusion heads. The BioLangFusion vectors reuse the token caches prepared in Stage 3; this comparison does not require Phase B's encoder swaps to finish.

The separate encoding is a deliberate input choice: the notebook records that a joint LucaOne input would exceed the selected context limit for part of this benchmark. Consequently, “both streams” means concatenating separately obtained vectors, not supplying the nucleotide and protein together to the encoder. The probe uses the supplied training and test partitions. The supplied validation partition remains unused because `RidgeCV` selects its penalty internally on the training data. That describes the implemented procedure; the original reason for preferring internal selection over the supplied validation partition has not been established.

#### Stability findings and the GTEx limit

The [October 22 record](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) reports stability Spearman correlations of 0.483 for the two LucaOne representations together and 0.390 for the three BioLangFusion representations concatenated, under the matched ridge procedure. The supplied training/test sequence overlap remains a limitation. LucaOne's GTEx comparison is excluded under the selected published 1,280-token context rule; this is a restriction of the chosen comparison, not a claim that every possible LucaOne configuration is incapable of processing GTEx inputs.

### IsoFormer: combining GTEx sequence inputs

**Notebook:** [LucaOne and IsoFormer comparisons](tier2_lucaone_isoformer.ipynb), Part 4 · **Plan:** `Details` task 44.

#### Question and GTEx inputs

**IsoFormer-style cross-attention** reuses the GTEx pilot from Stages 1 and 3. Each transcript record supplies a genomic DNA window, a transcript sequence, a protein sequence, and expression values across 30 tissues. The sequence representations are inputs; the 30 tissue-expression values are prediction targets. The comparisons vary which sequence inputs the predictor receives, not which tissues become input features. Cross-attention combines representations without requiring all three sequences to have the same number of positions.

#### What the additional training seeds test

This part of the notebook is intended to add training seeds to the [Stage 3 modality comparison](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders). The supplied test partition stays fixed; each seed changes model initialization and the 5% validation subset taken from the training partition. Thus the repeats assess sensitivity to training and validation assignment. The notebook records three seeds rather than the paper's five as a pilot computation compromise. Our encoders also remain frozen, whereas the paper's regime includes fine-tuning, so this evaluates the local frozen-encoder adaptation. The fits reuse the GTEx token embeddings prepared in Stage 3 and do not require the Phase B swaps.

#### Existing evidence and remaining repeat comparisons

The [October 22 record](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) adds no repeat-seed evidence and flags a summary-reader issue that combines distinct IsoFormer configurations. Use the individual run records when interpreting those comparisons. The [Stage 3 modality comparison](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders) supplies the existing seed-0 evidence; the [runbook](RUNBOOK.md#the-order) distinguishes it from the additional repeat runs.

### Full-transcript half-life and translation efficiency

**Notebook:** [Full-transcript half-life and translation efficiency](tier3_halflife_te.ipynb) · **Plan:** `Details` task 42.

#### Why introduce full-transcript datasets?

The earlier stability experiment starts from coding sequences. This notebook broadens the input question: what changes when we include the untranslated regions surrounding the coding sequence, and does adding the corresponding protein representation help? Untranslated regions are parts of the transcript outside the protein-coding region; they remain in the RNA but do not specify the translated protein sequence. This experiment introduces datasets distributed through mRNABench, rather than adding regions to the earlier `mRNA_Stability.csv` rows. Region comparisons within the new tables can test the contribution of the supplied sequence regions. A score difference from the earlier stability benchmark would also involve different data and target construction.

Half-life concerns persistence before degradation. Translation efficiency concerns translation activity relative to RNA abundance; the translation-efficiency targets use estimates derived from ribosome profiling. Half-life and translation efficiency are separate prediction targets. The human half-life table's numerical target is a summary across studies, specifically a first principal component, rather than a measurement in hours.

The notebook inspects six tables: human and mouse half-life, human and mouse translation efficiency, human mean ribosome load, and a synthetic-mRNA table with separate half-life and ribosome-load targets. It checks sequence regions, coding frames, targets, and grouping information, then evaluates simple composition baselines on eligible tables. The main encoder comparisons focus on two human tables, `rnahl-human` and `translation-efficiency-human`. Inspection of the other tables does not establish completed encoder experiments on them. The [dataset catalog](../shared_code/mrnabench.py) records the source and target for each table.

#### Region comparisons and evaluation choices

The composition baselines calculate features for the coding sequence, either untranslated region, coding sequence plus the 3′ untranslated region, and the full transcript. The encoder comparisons instead include the 5′ untranslated region, coding sequence, full transcript, and translated protein. They do not currently include an encoder condition for coding sequence plus the 3′ untranslated region. A DNA encoder also reads the transcript in its accepted alphabet; this adds another representation of that transcript, not the genomic flanking sequence supplied by GTEx.

These human experiments use pooled frozen embeddings and ridge probes, including concatenated representations. They do not reconstruct Saluki, RiboNN, or the BioLangFusion trained heads. Unlike LucaOne's use of the supplied partitions, this procedure constructs partitions targeting 70% training, 15% validation, and 15% test rows. Records from the same gene stay together; that does not keep all related genes together. The split proportions and nine-seed comparison follow mRNABench; gene-only grouping is the local fallback when its homology map is unavailable. Ridge chooses its penalty internally on training data; validation scores are calculated but do not choose that penalty. The nine split seeds vary the held-out partitions, unlike the fixed-test IsoFormer repeats.

Long transcripts are divided into consecutive, nonoverlapping windows to fit the encoders' input limits, following the benchmark's short-context procedure. Their outputs are averaged into a sequence representation. This includes sequence across the transcript, but each window is encoded separately, so the encoder cannot model interactions across window boundaries in that pass. This windowing serves a different purpose from the fixed folding prefixes in the mRFP control.

#### Human findings and the limits of their scope

The [October 22 evidence](../../docs/run-records.md#2026-10-22-stage4-human-probes-completed-half-life-translation-efficiency-and-lucaone-results) reports mean Pearson improvements of about 0.053 for half-life and 0.030 for translation efficiency when combining full-transcript and protein representations, relative to the best single encoder. The reported standard errors are not paired confidence intervals for those improvements. The gene-grouped splits do not establish generalization to unrelated gene families, and the record does not include mouse encoder replication.

## How these comparisons inform the proposed designs

The current experiments help us judge the strength of the baselines and whether an apparent gain comes from the encoder, the fusion operation, or the available sequence regions. That evidence feeds back into Stage 3, where the project's proposed designs are built and checked, before Stage 4 evaluates them against the established comparisons. It can also require revisiting Stage 1's interpretation or Stage 2's working definition of alignment. Agreement between representations and an improvement in biological prediction remain separate questions: similar representations need not help when combined, and useful fusion does not establish agreement under every alignment metric.

The planned `tier4_designs.ipynb` and `tier5_ablations.ipynb` are not present in this folder. The [Gantt workbook](../../docs/Research_Project_Plan_Gantt.xlsx), `Details` tasks 20 and 21, places Coupled Mamba implementation and evaluation after the baseline and dataset work; task 25 places the project's cross-attention design after those comparisons. Task 23 is a conditional alignment-objective ablation: compare the same setup with and without that objective. The existing plan also identifies codon-grid and modality-dropping ablations. These are planned investigations, not conclusions that the current results establish.

## Supporting code

The notebooks keep the experimental choices and interpretations near their results. Shared modules implement operations reused by these comparisons; their presence does not mean every Stage 4 notebook uses every module.

| Implementation | Role in these experiments | Software checks |
| --- | --- | --- |
| [`shared_code/fusion/`](../shared_code/fusion/) | Token handling, fusion models, training, and comparison utilities used by the trained-head experiments. | [Test guide](../../tests/README.md) |
| [`lucaone.py`](../shared_code/lucaone.py) | LucaOne loading, stream-specific tokenization, context handling, and pooled gene/protein embeddings. Its separate interface also keeps the large checkpoint out of ordinary encoder sweeps. | [LucaOne tests](../../tests/test_lucaone.py) |
| [`mrnabench.py`](../shared_code/mrnabench.py) | Dataset catalog, transcript-region preparation, feasibility checks, grouped splits, ridge probes, and pooling across transcript windows. | [mRNABench tests](../../tests/test_mrnabench.py) |
| [`matrix_cache.py`](../shared_code/matrix_cache.py) | Saves and checks pooled matrices for LucaOne and mRNABench reuse. | Covered by the two suites above. |
| [`fusion/token_embeddings.py`](../shared_code/fusion/token_embeddings.py) | `pooled_matrix` reuses cached BioLangFusion token embeddings for the matched LucaOne ridge comparison. | [Fusion tests](../../tests/test_fusion_model.py) |

## Results, caches, and execution

Small result JSON and CSV files are shared under `results/`. The main destinations are `tier1_mrfp/`, `phaseA/<benchmark>/`, `phaseB/<benchmark>/<encoder>/`, `lucaone/`, `gtex_isoformer/`, and `halflife_te/`. JSON records can include per-row test predictions; aggregate probe CSVs do not provide the same evidence for reconstructing uncertainty. The [run records](../../docs/run-records.md#recorded-runs) identify the producing code, settings, inputs, and interpretation limits.

Large embeddings, downloaded data, checkpoints, and model weights remain local and ignored by Git. Swapped-encoder token caches sit under `../stage3/embeddings/`; mRFP reuses the Stage 1 caches. Cache existence alone does not establish that it matches the current inputs or settings. Follow the [results and cache policy](../../README.md#results-caches-and-intermediate-files) and the relevant recovery record before reuse.

The [research plan](<../../docs/Multimodal_Research Plan .docx>) and [Gantt workbook](../../docs/Research_Project_Plan_Gantt.xlsx) record the broader priorities: `Details` task 24 follows Phase A (17), LucaOne (41) follows the swaps, and task 42 allows early full-transcript data checks before the later comparisons. That planned order is distinct from the file dependencies described in each experiment. LucaOne and IsoFormer both use Stage 3 inputs, so neither technically requires completed encoder swaps.

Saved outputs can be read without executing cells. Some notebooks also perform extraction or fitting, and their saved execution switches may be enabled. The [runbook](RUNBOOK.md) explains the recorded run assignments, where each notebook saves its outputs, and how interrupted extraction or training can resume. Its assignments and the findings above are dated records, not a live report of what is running.

