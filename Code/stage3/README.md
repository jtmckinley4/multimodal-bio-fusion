# Stage 3: building and checking the models

Stage 1 examined what frozen encoder representations make accessible to simple predictors and how those representations relate. Stage 2 used those observations to define alignment and identify comparisons worth making. Stage 3 turns those comparisons into models we can train and inspect: first the existing methods that provide our reference points, then the project's proposed fusion designs.

The first four notebooks connect data preparation, token representations, fusion, and prediction. Their checks matter because a model can produce a plausible score even when its inputs do not represent the intended comparison. Early fits also provide research results, so Stage 3 and Stage 4 overlap: we build and inspect an implementation here, compare it more broadly in Stage 4, and return here when those comparisons expose a question about the implementation.

## Building the baseline comparisons

### 1. Establish which benchmark data the models receive

The BioLangFusion comparison begins with its original encoder combination and available benchmarks. This gives later encoder substitutions a reference: we can change an encoder while retaining the surrounding method. It serves a different purpose from Stage 1's initial stability and GTEx probes, which examined representations before fitting these fusion architectures.

Alongside the stability dataset already used in Stage 1, this comparison introduces fungal expression, E. coli protein expression, and CoV-Vac degradation. Together, these cover four of BioLangFusion's five benchmarks. Ab1 is the missing fifth; it was not found in the public release used here. Stage 1's GTEx and mRFP datasets serve other comparisons and are not part of that suite.

The new datasets broaden what the same model pipeline is asked to predict. Fungal sequences are paired with released expression scores whose cited source uses RNA sequencing. The E. coli task predicts three stored protein-expression categories for genes expressed in that host. CoV-Vac pairs short RNA vaccine-construct fragments with degradation scores. The [dataset descriptions](../../README.md#other-biolangfusion-benchmarks) distinguish these targets and their unresolved assay-to-table details.

The dataset notebook reads the existing CSVs; it does not download or create the original datasets. It checks the sequence and target columns, supplied training/validation/test assignments, lengths, and repeated sequences. Stability therefore returns as the same source table under the BioLangFusion preparation rules, rather than as a new dataset.

Two findings affect every later comparison. Keeping sequences of at most 999 nucleotides matches the four reported used-row counts, but count agreement does not establish that our retained rows or preprocessing are identical to the authors'. On stability, about 68% of retained test rows reuse a sequence present in training. Preserving that split permits a published-protocol comparison; it limits what the score says about prediction for unseen sequences. The audit records these choices and limitations before any encoder is run.

**Notebook:** [01 — BioLangFusion dataset audit](01_blf_datasets.ipynb). The [benchmark target guide](../../docs/methods/biological-targets-and-controls.md#fungal-and-e-coli-benchmark-targets) explains the biological targets and remaining provenance gaps.

### 2. Keep the token information needed for fusion

The main Stage 1 probes summarize each sequence as one embedding vector. BioLangFusion's fusion heads instead operate on a sequence of vectors, so this notebook extracts and caches the final hidden state of each content token from Nucleotide Transformer v2 100M, RNA-FM, and ESM-2 8M. The encoders remain frozen; later training changes the components that combine and predict from these stored representations.

The token streams have different resolutions. RNA-FM supplies a vector per nucleotide, ESM-2 a vector per amino acid, and Nucleotide Transformer uses six-base tokens with separate handling of remaining bases. On a coding sequence, three nucleotides specify one amino acid. The subsequent codon-grid operation uses that relationship to place the streams on a common position axis; equal array lengths alone would not establish the correspondence.

The input construction must also cover the benchmark's nonstandard cases. `grid_protein` translates complete triplets from the supplied sequence origin and represents stop codons as `X`, retaining one protein-input position per triplet. For CoV-Vac fragments, this is a computational convention, not evidence that the fragment is a complete biological coding sequence. It differs from stopping translation at the first stop codon.

The [October 16 extraction record](../../docs/run-records.md#2026-10-16-stage3-token-embeddings-token-level-caches-for-the-four-biolangfusion-benchmarks) reports approximately 34 GB across 405 cache shards for the four benchmarks. The notebook checks token counts against the selected rows; current cache reuse also depends on the recorded input and extraction identities.

**Notebook:** [02 — Token embedding extraction](02_token_embeddings.ipynb).

### 3. Assemble BioLangFusion and investigate the first comparison

With the inputs established, notebook 03 combines the token streams through concatenation, gated fusion, or cross-attention, followed by a TextCNN prediction head. The single-encoder baselines use a prediction head without combining modalities. These trainable predictors differ from Stage 1's ridge probes, so their scores answer a different comparison.

Fungal expression was chosen for the initial build comparison because it provides regression targets, published scores for the relevant variants, and no exact sequence crossing the retained splits in the audit. Stability has substantial exact-sequence overlap, CoV-Vac adds the fragment-translation assumption, and E. coli introduces a classification loss. This makes fungal expression a useful starting point, not a proof that its splits are independent in every biological sense or that its preprocessing is fully resolved.

The notebook first checks dimensions, padding and masking, and the mapping between token positions. A controlled input with one nonzero token makes the mapping visible: we can see where a DNA-token signal lands on the codon grid. These checks answer whether the intended operation is implemented. The subsequent fits ask how the resulting models compare with the published values and with one another. Neither check substitutes for the other.

This distinction uncovered a consequential ambiguity: the paper's appendix and Equation 1 specify different transposed convolutions for the DNA stream. The [October 17 comparison](../../docs/run-records.md#2026-10-17-stage3-tconv-rerun-the-papers-two-transposed-convolution-settings-compared) ran both. Their different position mappings changed the scores, motivating the Equation 1 configuration used in the main Stage 4 notebook comparisons. Both settings remain recorded; the rerun is no longer an open task.

The later [fungal saved-result verification](../../docs/run-records.md#fungal-equation-1-saved-result-verification) reproduced the three-seed summaries from retained predictions. Cross-attention had a higher mean Spearman score than the fixed DNA baseline, but the paired interval for that difference included zero; the other fusion-versus-DNA intervals did too. This leaves the improvement unresolved under that comparison. It neither establishes a fusion advantage nor proves that fusion cannot help. The record also identifies historical selection and producer-provenance limitations.

**Notebook:** [03 — BioLangFusion build and comparison](03_blf_build_check.ipynb).

### 4. Check the scope of LucaOne and build the GTEx comparison

LucaOne and IsoFormer supply different reference points. LucaOne asks what jointly pretrained nucleotide/protein representations offer relative to separately pretrained encoders. IsoFormer-style cross-attention asks how to combine GTEx's genomic window, transcript, and protein when the full inputs do not share a single codon grid.

For LucaOne, notebook 04 first compares sequence lengths with the selected 1,280-token context rule. The GTEx DNA windows are 6,000 nucleotides, so that rule excludes the intended full-window comparison. This is a constraint on the chosen setup, not a statement that every possible LucaOne configuration is incapable of using GTEx data. The later [Stage 4 stability comparison](../stage4/README.md#lucaone-comparing-jointly-pretrained-representations) encodes nucleotide and protein inputs separately and compares their pooled representations with a common ridge predictor. It does not feed both sequences jointly to the model.

For GTEx, notebook 04 reuses the pilot introduced in Stage 1, keeping its 30 tissue-expression targets while replacing the simple probe with a trainable aggregator and expression predictor. It implements successive cross-attention between streams at their own lengths, using frozen Nucleotide Transformer v2 100M for the genomic and transcript inputs and ESM-2 150M for protein. Its six input combinations compare DNA, RNA, and protein individually with DNA+protein, DNA+RNA, and all three on the same pilot split. Removing an input tests what it contributes within this setup; the six combinations are six model configurations, not six datasets.

The [October 19 pilot record](../../docs/run-records.md#2026-10-19-stage3-isoformer-gtex-isoformers-aggregator-on-the-gtex-pilot-frozen-encoders) reports higher average tissue Spearman correlation for DNA+RNA than RNA alone, while adding protein to DNA+RNA did not resolve a further improvement. This is one-seed evidence on a small pilot, with row-bootstrap intervals. Our frozen encoders, smaller training sample, checkpoint choice, and implementation adaptations differ from the paper, so the gap to its published scores cannot isolate the effect of freezing. Likewise, comparing this GTEx result with the fungal result does not isolate architecture: the data and prediction task change too.

**Notebook:** [04 — LucaOne feasibility and IsoFormer-style GTEx comparison](04_lucaone_isoformer_check.ipynb).

## How the baseline work leads into the proposed designs

Stage 4 extends these implementations to further benchmarks, encoder substitutions, and repeated comparisons. Stage 1 helps nominate encoders, Stage 2 specifies what we mean by agreement, and the trained comparisons test whether those choices improve prediction. A later result can therefore send us back to an input rule, a baseline, or an alignment interpretation; the stages are connected responsibilities rather than a one-way sequence.

The current plan places the project's proposed designs after the baseline and dataset work. These notebooks are planned and do not yet exist:

| Planned notebook | Purpose |
| --- | --- |
| `05_coupled_mamba_check.ipynb` | Build and inspect Coupled Mamba using explicit correspondence between the input streams, then carry it into the matched Stage 4 evaluation. |
| `06_cross_attention_check.ipynb` | Develop cross-attention with the encoder choices informed by Stage 1. This is distinct from the IsoFormer-style pilot baseline in notebook 04. |

The [Gantt workbook](../../docs/Research_Project_Plan_Gantt.xlsx), `Details` tasks 20, 21, and 25, records that sequence and its dependencies. The earlier baseline work corresponds to tasks 16, 17, 41, and 44. A planned date or an existing implementation does not establish that its evaluation is complete.

**Continue with:** [Stage 4 comparisons](../stage4/README.md). The guide explains which question each experiment adds; the [runbook](../stage4/RUNBOOK.md) handles execution coordination.

## Shared implementations and retained evidence

The notebooks explain the choices and assemble the experiments. [`shared_code/fusion/`](../shared_code/fusion/) contains the implementations reused by Stage 3 and Stage 4: token extraction, codon-grid operations, fusion heads, prediction heads, and training. Sharing these operations keeps a correction to the implementation available to both stages. [Software tests](../../tests/README.md) check controlled cases; the notebooks and [run records](../../docs/run-records.md) retain the research comparisons and their interpretation limits.

Token caches under `embeddings/` and large training artifacts remain local and excluded from Git; small result JSONs and verification records are retained as evidence. [`results/fungal_expression/`](results/fungal_expression/) holds the results read by notebook 03. Its [`reproducible/` subfolder](results/fungal_expression/reproducible/) preserves a separate Windows seed-0 cohort with run identities but no per-row predictions; it is not interchangeable with the three-seed prediction cohort used in the saved-result verification. The [GTEx result folder](results/gtex_pilot/) similarly retains separately tagged configurations. Use the corresponding run record to establish what a result represents before combining it with another.
