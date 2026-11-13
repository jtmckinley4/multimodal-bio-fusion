# Stage 3: reference methods and baselines

Stage 3 establishes the reference computations and evidence used by later fusion experiments. [Stage 1](../stage1/README.md) characterizes representations; Stage 2 clarifies alignment; [Stage 4](../stage4/README.md) investigates encoder substitutions, biological context and proposed explanations of fusion behavior. Establishing a reference includes input checks, implementation validation, predictive comparisons and repeated fits.

Read the BioLangFusion route from datasets through token caches and build checks to the reference benchmarks. The frozen GTEx reference and adapted Coupled Mamba are separate method families, not subsequent steps required to complete BLF. Software checks establish what an implementation computes; held-out comparisons establish its predictive behavior. Neither alone explains a biological mechanism or reproduces a paper's entire experiment.

| Reference route | Notebook and role |
| --- | --- |
| BioLangFusion inputs | [Dataset audit](blf_datasets.ipynb) and [token embeddings](blf_token_embeddings.ipynb) establish the retained rows and frozen token inputs. |
| BioLangFusion validation | [Build check](blf_build_check.ipynb) examines fungal expression and convolution choices; [reference benchmarks](blf_reference_benchmarks.ipynb) extends the original-encoder comparison. |
| Historical BLF alternative | [Appendix comparison](blf_appendix_comparison.ipynb) reads a distinct convolution configuration and its own selection/uncertainty protocol. |
| GTEx inputs and frozen fusion | [Released-data audit](isoformer_gtex_audit.ipynb) examines input policies; [frozen fusion reference](gtex_frozen_fusion_reference.ipynb) combines the pilot and its repeat route. |
| Adapted Coupled Mamba | [Reference and controls](coupled_mamba_reference.ipynb) compares bottleneck, uncoupled and coupled recurrence under the BLF input recipe. |

## BioLangFusion: inputs, implementation and reference comparisons

### Establish which benchmark data the models receive

[BioLangFusion dataset audit](blf_datasets.ipynb) reads the supplied CSVs and checks targets, splits, lengths and repeated sequences. Four of BioLangFusion's five benchmarks are available; Ab1 was absent from the public release inspected here. GTEx and mRFP belong to separate investigations.

| Benchmark | Meaning of the supplied target |
| --- | --- |
| mRNA stability | Profile-standardized, decay-derived score; higher means more stable. |
| Fungal expression | Natural log of the supplied RNA-expression quantity; earlier normalization and units remain unresolved. |
| E. coli expression | Low, medium or high recombinant protein-production class derived from visual whole-cell gel scores. |
| CoV-Vac degradation | Standardized average chemical-degradation measurement; higher means more degradation under the assay conditions. |

The [target guide](../../docs/methods/data-and-targets.md) traces these measurements and transformations. Stability reuses Stage 1's source table under different preparation and evaluation rules.

The ≤999-nucleotide filter matches the four reported used-row counts, without establishing identical row membership or preprocessing. About 68% of retained stability test rows reuse a training sequence. Preserving that split supports a published-protocol comparison, but limits claims about unseen sequences.

### Keep the token information needed for fusion

[Token embedding extraction](blf_token_embeddings.ipynb) caches final hidden states from Nucleotide Transformer v2 100M, RNA-FM and ESM-2 8M. These encoders remain frozen while downstream fusion and prediction heads are trained. Unlike Stage 1's pooled vectors, token caches retain positions for those heads.

The streams use different token resolutions. A codon grid maps them onto corresponding triplet positions; matching array lengths alone does not establish biological correspondence. The `grid_protein` convention translates complete triplets from the supplied origin and represents stops as `X`. For CoV-Vac fragments, that convention preserves positions without establishing a biological coding frame.

The notebook checks token counts against the selected rows; cache reuse also depends on input and extraction identities. The [extraction record](../../docs/run-records.md#2026-10-16-stage3-token-embeddings-token-level-caches-for-the-four-biolangfusion-benchmarks) retains the historical storage and execution evidence.

### Assemble BioLangFusion and investigate the first comparison

[BioLangFusion build and comparison](blf_build_check.ipynb) combines the codon-grid streams through concatenation, gated fusion or cross-attention, then applies a TextCNN prediction head. Single-encoder baselines use a prediction head without fusion. These fitted neural heads differ from Stage 1's ridge probes.

Fungal expression provides the first build comparison: it has regression targets, published reference scores and no exact sequence crossing the retained splits in the audit. That does not rule out related sequences or resolve its target provenance.

Dimension, masking and position-mapping checks exposed an ambiguity between the paper's appendix and Equation 1 transposed-convolution settings. The [completed comparison](../../docs/run-records.md#2026-10-17-stage3-tconv-rerun-the-papers-two-transposed-convolution-settings-compared) tested both, motivating the Equation 1 reference used in the broader benchmark comparisons and subsequent encoder swaps. Both configurations remain recorded.

The [fungal saved-result verification](../../docs/run-records.md#fungal-equation-1-saved-result-verification) recovered three-seed summaries from retained predictions. Cross-attention's mean Spearman score exceeded the fixed DNA baseline, but all three fusion-versus-DNA paired intervals included zero. The gain remains unresolved under that comparison; this does not prove equivalence or that fusion cannot help. Historical selection and producer-provenance limitations remain attached to the evidence.

### Extend the BLF reference across benchmarks

The [reference benchmark notebook](blf_reference_benchmarks.ipynb), historically Phase A, reuses the original NT v2 100M, RNA-FM and ESM-2 8M caches on E. coli, CoV-Vac and stability. Validation selects the epoch, single-encoder reference and positive entropy weight. Regression uses Spearman correlation; E. coli uses classification accuracy. It also adds a codon-grid RNA baseline and revisits fungal RNA resolution and prediction-head width. These controls qualify the local reference without identifying unreported details of the paper's implementation.

The main Equation 1 comparison remains distinct from the [appendix-convolution reader](blf_appendix_comparison.ipynb). The main protocol fixes validation choices at seed 0 and averages scores across matched fitted seeds for its uncertainty analysis. The appendix selects a single encoder and entropy weight separately for each training seed and reports per-seed intervals. Configuration, eligible baselines and uncertainty procedures differ; their results must not be pooled. The [main record](../../docs/run-records.md#2026-10-20-stage4-blf-phasea-biolangfusion-phase-a-on-cov-vac-e-coli-and-mrna-stability-research-plan-item-1) and [appendix record](../../docs/run-records.md#2026-10-19t220959z-phasea-ecoli-three-seed-classification-comparison) preserve the original evidence.

The recorded main comparison found positive paired intervals for E. coli concatenation and gated fusion against the selected single; cross-attention's interval included zero. CoV-Vac did not show the same advantage. Stability retains extensive training/test sequence overlap: an unseen-sequence subset is informative but does not convert the original fit into a clean-split experiment. Matching some published scores likewise does not establish faithful reproduction. The [reference notebook](blf_reference_benchmarks.ipynb#What-this-comparison-establishes) retains those limits.

## GTEx: input suitability and a frozen fusion reference

The [GTEx audit](isoformer_gtex_audit.ipynb) connects released targets, transcript annotations, partition identities and sequence lengths to possible input policies. Its saved JSONs describe the released table; they do not recover undocumented historical training membership or demonstrate predictive performance. The audit does not adopt a training protocol.

The [frozen fusion reference](gtex_frozen_fusion_reference.ipynb) brings together the original pilot and its additional-seed route. It predicts Stage 1's 30 tissue RNA-abundance targets with frozen NT v2 100M for genomic DNA and transcript RNA and ESM-2 150M for protein. Six configurations compare individual inputs, DNA+protein, DNA+RNA and all three. These are transformed tissue-averaged RNA-abundance targets, not tissue inputs or protein production; the [target trace](../../docs/methods/data-and-targets.md#gtex-tissue-expression-labels) records the transformation and normalization uncertainties.

The local pilot updates each available stream through cross-attention, normalizes, pools and concatenates them for prediction. The released Enformer-based design instead updates and pools DNA. The [shared source record](../../docs/sources.md#isoformer-dataset-and-model-cards) documents that difference. Avoiding a codon grid changes the computation without establishing learned biological correspondence; comparing fungal and GTEx results also changes the task.

The [recorded one-seed pilot](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders) found higher average tissue Spearman correlation for DNA+RNA than RNA alone, without resolving further benefit from protein. Row-bootstrap intervals do not establish robustness across training seeds. Additional seeds vary initialization and the validation subset while retaining the test partition; coverage distinguishes available fits from missing repeats. Different encoders, population and implementation choices prevent attributing the gap to published scores solely to freezing.

A matched comparison with pooled concatenation would test whether the local block adds value on the same inputs. It would assess the complete block, including extra parameters and normalization, rather than isolate attention weights. That proposed comparison is not established by the pilot's presence. LucaOne's separate input-feasibility and ridge-probe investigation is now in [Stage 1](../stage1/lucaone_representation_comparison.ipynb); these are not a combined LucaOne–IsoFormer model or a direct contest.

### Released-reference foundations and deferred reproduction

The [shared reference implementation](../shared_code/fusion/isoformer_reference.py) follows the released Enformer/NT/ESM pathway: DNA queries RNA, then protein, before pooling updated DNA for prediction. Input/model foundations, missing-protein handling and adapters have synthetic checks. Pretrained integration, remaining validation/target/training choices and reproduction training remain unfinished.

[Fuller experimental reproduction is deferred](../stage4/RUNBOOK.md#isoformer-reproduction-specification), separately from the shared frozen pilot. Its private preparation notebook and specification are not supplied with this checkout. Faithful reference operations can support a controlled comparison without reconstructing the paper's entire experiment; generic anchor-query attention should not be labeled an IsoFormer reproduction.

## Coupled Mamba: establish the adapted reference and controls

The [Coupled Mamba notebook](coupled_mamba_reference.ipynb) assembles three registered heads: a projection bottleneck without recurrence, independent recurrent streams, and coupled recurrence. Their contrasts distinguish the bottleneck, recurrence and state coupling, then compare the adapted method with BLF concatenation, cross-attention and the validation-selected single encoder. These controls qualify the reference for later matched experiments.

The [shared implementation](../shared_code/fusion/coupled_mamba.py) records the project's deviations: mean rather than summed pooled state, a codon-grid time axis, reduced state size/layer count and Mamba-style discretization. It is an adapted reference, not evidence of a full published reproduction. Its corresponding DNA/RNA/protein streams do not by themselves establish inputs or controls for independently paired RNA–protein or protein–ligand examples.

The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx), tasks 20/21, make further Mamba work conditional on a measured long-sequence runtime or memory bottleneck and a selected matched protocol. The available notebook is not evidence of completed design fits or a requirement to cross every design with every benchmark. Reference work is not a prerequisite for the proposed gate and attention tests in [Stage 4](../stage4/README.md#proposed-tests-of-fusion-explanations).

## Reading, execution and retained evidence

Saved outputs can be read without executing cells. Reading flags differ between notebooks: the BLF build notebook and original GTEx pilot contain direct fit cells, and Mamba's default skips training but can still bootstrap eligible saved predictions. The GTEx repeat section has its own reading switch; it does not guard the earlier pilot. Consult the [runbook](../stage4/RUNBOOK.md#read-saved-results-without-starting-work) and each notebook's setup before execution; a file move does not change what Run All computes.

The notebooks explain and assemble the references; [shared fusion code](../shared_code/fusion/) supplies reused extraction, grid, head and training operations. [Software tests](../../tests/README.md) check controlled cases, while notebooks and [run records](../../docs/run-records.md) retain research evidence. Gantt/Details tasks 16/17/41/44 retain the baseline and reference work; task 25 covers proposed attention comparisons.

Results and caches retain their existing paths. The main BLF benchmarks use `Code/stage4/results/phaseA/`; the separate appendix uses `Code/stage4/results/blf_phaseA/` for E. coli and `Code/stage3/results/fungal_expression/reproducible/` for fungal records. GTEx audit/repeat evidence and Mamba destinations likewise retain their historical Stage 4 directories. Token caches stay in `Code/stage3/embeddings/`. The [cache policy](../../README.md#results-caches-and-intermediate-files) and run records govern reuse, not the current notebook folder.

[Fungal results](results/fungal_expression/) include a separate [Windows seed-0 cohort](results/fungal_expression/reproducible/) without per-row predictions; it must not be mixed with the verified three-seed prediction cohort. [GTEx records](results/gtex_pilot/) preserve separately tagged configurations. The historical appendix exporter under the E. coli records remains a producer snapshot with its old notebook path, not a current execution command; the maintained analysis route is [the appendix notebook](blf_appendix_comparison.ipynb).
