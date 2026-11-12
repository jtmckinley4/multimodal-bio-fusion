# Stage 3: building and checking the models

Stage 1 examined what frozen representations make accessible and how they relate; Stage 2 clarified what alignment should mean. Stage 3 turns those questions into models with inspectable inputs, fusion operations and prediction heads. [Stage 4](../stage4/README.md) extends their comparisons across benchmarks and encoder substitutions, returning here when results raise implementation questions.

Read notebooks 01–04 as a progression from dataset checks to token caches, BioLangFusion and the GTEx pilot. Software checks establish what an implementation computes; held-out comparisons establish its predictive behavior. Neither alone explains a biological mechanism or reproduces a paper's entire experiment.

## Building the baseline comparisons

### 1. Establish which benchmark data the models receive

[01 — BioLangFusion dataset audit](01_blf_datasets.ipynb) reads the supplied CSVs and checks targets, splits, lengths and repeated sequences. Four of BioLangFusion's five benchmarks are available; Ab1 was absent from the public release inspected here. GTEx and mRFP belong to separate investigations.

| Benchmark | Meaning of the supplied target |
| --- | --- |
| mRNA stability | Profile-standardized, decay-derived score; higher means more stable. |
| Fungal expression | Natural log of the supplied RNA-expression quantity; earlier normalization and units remain unresolved. |
| E. coli expression | Low, medium or high recombinant protein-production class derived from visual whole-cell gel scores. |
| CoV-Vac degradation | Standardized average chemical-degradation measurement; higher means more degradation under the assay conditions. |

The [target guide](../../docs/methods/data-and-targets.md) traces these measurements and transformations. Stability reuses Stage 1's source table under different preparation and evaluation rules.

The ≤999-nucleotide filter matches the four reported used-row counts, without establishing identical row membership or preprocessing. About 68% of retained stability test rows reuse a training sequence. Preserving that split supports a published-protocol comparison, but limits claims about unseen sequences.

### 2. Keep the token information needed for fusion

[02 — Token embedding extraction](02_token_embeddings.ipynb) caches final hidden states from Nucleotide Transformer v2 100M, RNA-FM and ESM-2 8M. These encoders remain frozen while downstream fusion and prediction heads are trained. Unlike Stage 1's pooled vectors, token caches retain positions for those heads.

The streams use different token resolutions. A codon grid maps them onto corresponding triplet positions; matching array lengths alone does not establish biological correspondence. The `grid_protein` convention translates complete triplets from the supplied origin and represents stops as `X`. For CoV-Vac fragments, that convention preserves positions without establishing a biological coding frame.

The notebook checks token counts against the selected rows; cache reuse also depends on input and extraction identities. The [extraction record](../../docs/run-records.md#2026-10-16-stage3-token-embeddings-token-level-caches-for-the-four-biolangfusion-benchmarks) retains the historical storage and execution evidence.

### 3. Assemble BioLangFusion and investigate the first comparison

[03 — BioLangFusion build and comparison](03_blf_build_check.ipynb) combines the codon-grid streams through concatenation, gated fusion or cross-attention, then applies a TextCNN prediction head. Single-encoder baselines use a prediction head without fusion. These fitted neural heads differ from Stage 1's ridge probes.

Fungal expression provides the first build comparison: it has regression targets, published reference scores and no exact sequence crossing the retained splits in the audit. That does not rule out related sequences or resolve its target provenance.

Dimension, masking and position-mapping checks exposed an ambiguity between the paper's appendix and Equation 1 transposed-convolution settings. The [completed comparison](../../docs/run-records.md#2026-10-17-stage3-tconv-rerun-the-papers-two-transposed-convolution-settings-compared) tested both, motivating the Equation 1 reference used in the main Stage 4 comparisons. Both configurations remain recorded.

The [fungal saved-result verification](../../docs/run-records.md#fungal-equation-1-saved-result-verification) recovered three-seed summaries from retained predictions. Cross-attention's mean Spearman score exceeded the fixed DNA baseline, but all three fusion-versus-DNA paired intervals included zero. The gain remains unresolved under that comparison; this does not prove equivalence or that fusion cannot help. Historical selection and producer-provenance limitations remain attached to the evidence.

### 4. Check LucaOne's scope and build the GTEx comparison

[04 — LucaOne feasibility and IsoFormer-style GTEx comparison](04_lucaone_isoformer_check.ipynb) contains two separate investigations, not a combined LucaOne–IsoFormer model. Its first part checks sequence lengths and LucaOne release metadata without extracting LucaOne embeddings or fitting predictors. Its second part builds and trains the frozen-encoder GTEx pilot described below. Notebooks 01–03 are the BLF preparation/build route; notebook 04 introduces this separate reference family.

The selected LucaOne 1,280-token context rule excludes the intended full 6,000-nucleotide GTEx windows. This limits that setup, not every possible LucaOne approach. The later [stability comparison](../stage4/README.md#lucaone-comparing-jointly-pretrained-representations) instead encodes nucleotide and protein inputs separately and compares pooled representations with ridge predictors.

The GTEx pilot reuses Stage 1's 30 tissue RNA-abundance targets, with frozen NT v2 100M for genomic DNA and transcript RNA and ESM-2 150M for protein. Six configurations compare individual inputs, DNA+protein, DNA+RNA and all three. The pilot updates each available stream through cross-attention, normalizes, pools and concatenates them for prediction. The released Enformer-based design instead updates and pools DNA; the [shared source record](../../docs/sources.md#isoformer-dataset-and-model-cards) documents that difference.

The [recorded one-seed pilot](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders) found higher average tissue Spearman correlation for DNA+RNA than RNA alone, without resolving further benefit from protein. Row-bootstrap intervals do not establish robustness across training seeds. Different encoders, cohort and implementation choices prevent attributing the gap to published scores solely to freezing. Comparing this result with fungal expression also changes the task, not just the architecture.

## How the baseline work leads into the proposed designs

Stage 3 establishes reference computations; Stage 4 asks where combinations and replacements help. [Stage 5](../stage5/README.md) proposes controlled tests of why, including error correction, gating and biological correspondence. Those proposals are distinct from the completed comparisons above.

The earlier `05_coupled_mamba_check.ipynb` and `06_cross_attention_check.ipynb` names remain uncreated proposals. Julian's [design notebook](../stage4/tier4_designs.ipynb) supplies a separate implementation route. The [Gantt and Details sheets](../../docs/Research_Project_Plan_Gantt.xlsx), tasks 20/21, make further Mamba work conditional on a measured long-sequence resource bottleneck; task 25 covers proposed attention comparisons. Tasks 16/17/41/44 retain the baseline and reference work.

[Fuller IsoFormer reproduction is deferred](../stage4/RUNBOOK.md#isoformer-reproduction-specification), separately from the shared frozen pilot. Reference training is unfinished and is not a prerequisite for the proposed Stage 5 diagnostics. The later full-transcript half-life/TE analysis is another experiment, in [Stage 4](../stage4/tier3_halflife_te.ipynb); it is not the GTEx pilot in notebook 04. The [Stage 4 runbook](../stage4/RUNBOOK.md) owns execution routes.

## Shared implementations and retained evidence

The notebooks explain and assemble experiments; [shared fusion code](../shared_code/fusion/) supplies reused extraction, grid, head and training operations. [Software tests](../../tests/README.md) check controlled cases, while notebooks and [run records](../../docs/run-records.md) retain research evidence.

Token caches and large training artifacts remain local under the [cache policy](../../README.md#results-caches-and-intermediate-files). Small result records are retained. [Fungal results](results/fungal_expression/) include a separate [Windows seed-0 cohort](results/fungal_expression/reproducible/) without per-row predictions; it must not be mixed with the verified three-seed prediction cohort. [GTEx records](results/gtex_pilot/) likewise preserve separately tagged configurations. Use their run records to identify compatible evidence before combining scores.
