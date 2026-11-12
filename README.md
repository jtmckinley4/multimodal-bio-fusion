# multimodal-bio-fusion

A research project in the [Complex Adaptive Systems Laboratory](https://complexity.cecs.ucf.edu/directors-welcome/) at the University of Central Florida, investigating **when and how combining pretrained DNA, RNA and protein encoders improves biological prediction**. The team's manuscript is maintained in [Overleaf](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f).

Read the [research progression](#research-progression) for the investigation, [Data](#data) for target meanings, or [Setup](#setup) to prepare an environment. Saved notebook outputs can be read without executing cells.

## Project background

An encoder turns a biological sequence into numerical features. Encoders differ in pretraining data, sequence coverage and model design, so even the same input can yield differently useful representations. Combining those representations is fusion. Another encoder may add predictive information, duplicate what a strong model already provides, or introduce information the chosen fusion method fails to use.

We separate three questions:

- **Representation usefulness:** what can a fitted predictor recover from one encoder?
- **Predictive complementarity:** does another representation improve on an appropriate single-encoder reference?
- **Fusion-method value:** does a learned interaction improve on a simpler combination of the same inputs?

A stronger replacement can improve a fused model without making fusion beneficial. Likewise, a concatenation gain does not establish that attention is needed. The aim is to develop evidence for choosing whether and how to fuse, eventually including multiple encoders per modality. The current benchmarks supply testable biological outcomes; they have not yet established a general selection rule or a particular wet-lab application. The [meeting direction](docs/notes/Slides/Mina_Meeting_Slides_2026-09-23.pptx) and [planning route](docs/README.md#follow-the-research-plan) retain that scope.

## Research progression

Stage 1 characterizes representations; Stage 2 defines what alignment means; Stage 3 checks implementations; Stage 4 compares their predictions; Stage 5 proposes tests of the explanations. This is a reading route, not a requirement to execute every stage in order. Findings can send us back to an input rule, baseline or implementation.

### Stage 1 analysis

The [stability and GTEx notebooks](Code/stage1/README.md) compare twelve frozen encoders using pooled vectors and fitted ridge predictors. Stability supplies DNA/RNA views and a translation of the same coding sequence. GTEx supplies genomic, transcript and protein contexts with 30 tissue-expression targets. Because both inputs and targets change, differences between these tasks cannot isolate the value of extra context.

The analyses examine individual prediction, representation agreement and pairwise concatenation; selected attention diagnostics generate hypotheses. The mRFP synonymous control checks behavior when nucleotide sequences vary but the protein stays fixed. Detailed encoder settings and pair coverage belong in the [Stage 1 overview](docs/stage1-overview.md).

**Findings.** Retained probes make codon-level RNA and protein encoders useful stability candidates. Composition explains substantial measured agreement, and transcript length is a strong GTEx reference. These are findings under the chosen probes, not proof that enforcing alignment or adding complex fusion will improve prediction. See the [result tables and qualifications](docs/stage1-overview.md#result-tables).

**Published comparisons.** The [published-split probes](docs/stage1-overview.md#published-comparisons-and-distinct-inputs) compare with BioLangFusion and IsoFormer inputs and partitions. Small frozen probes are distinct from reproducing their complete trained systems; split overlap and preprocessing limits remain attached to the scores.

### Stage 2: defining alignment

**Biological correspondence** relates inputs, such as a CDS and its translation. **Representation agreement** concerns their numerical features. **Predictive complementarity** asks whether combining them improves prediction. CKA, neighborhood overlap, canonical correlation and retrieval measure different aspects of agreement; none substitutes for the prediction comparison.

Our working definition concerns agreement between corresponding representations on held-out examples after specified composition control, reporting composition-explained agreement separately. Removing a fitted linear composition component does not isolate purely biological information. The current Stage 1 composition fit uses all analysis rows before the later PCA/CCA split, so those controlled calculations do not fully meet the intended held-out preprocessing boundary. The [representation guide](Code/stage1/representation-analysis.md#comparing-representations) and [Stage 1 limitations](Code/stage1/README.md#why-the-analyses-have-three-tracks) retain the procedures and unresolved source discrepancies.

Initial designs begin without an explicit alignment loss and use positional correspondence where the inputs support it. An exploratory agreement/error analysis did not test whether adding that loss helps; the alignment-loss comparison remains conditional. Codon-level RNA/protein pairings and GTEx genomic context are candidates to evaluate, not established fusion rules. The [alignment source review](docs/sources.md#tjandrasuwita-et-al-understanding-multimodal-representation-alignment) distinguishes the paper's findings from our proposed selection heuristics.

Stage 2 is a definition and interpretation step, without a separate executable folder. The [October alignment discussion](docs/notes/Slides/Mina_Meeting_Slides_2026-10-07.pptx), manuscript and [Gantt Details tasks 7/23](docs/Research_Project_Plan_Gantt.xlsx) retain its rationale and conditional comparison.

### Stage 3: building and checking models

The [Stage 3 guide](Code/stage3/README.md) connects four notebooks: benchmark audit → token extraction → BioLangFusion build comparison → LucaOne feasibility and the IsoFormer-style GTEx pilot. Token-level fusion retains positions that Stage 1's pooled vectors summarize away. Grid and input checks establish what the implementation computes; matching array dimensions alone does not establish biological correspondence.

The available BLF comparison uses four benchmarks; Ab1 was absent from the inspected public release. Initial fungal fits and the GTEx pilot provide research evidence as well as implementation checks, so Stage 3 and Stage 4 overlap in purpose.

#### Architecture priorities

The [September 23 meeting](docs/notes/Slides/Mina_Meeting_Slides_2026-09-23.pptx), slide 6, prioritized Coupled Mamba, with IsoFormer-style attention and fallback designs. Mina's [revised protocol](docs/Biological_Fusion_Plan.docx) now places controlled anchor/gate comparisons first. Further Mamba work is conditional on a measured long-sequence memory or runtime bottleneck; the earlier order is historical.

Fuller IsoFormer reproduction is [deferred](Code/stage4/RUNBOOK.md#isoformer-reproduction-specification), distinct from the retained frozen GTEx pilot. Its preparation is preserved for possible reuse; controlled fusion comparisons do not require completion of the paper's entire training pipeline. The [runbook](Code/stage4/RUNBOOK.md#current-investigation-scope) retains the current scope and earlier decisions.

### Stage 4: controlled comparisons

The [Stage 4 investigation](Code/stage4/README.md) separates original BLF comparisons from one-encoder substitutions, mRFP controls, LucaOne representations, the GTEx pilot, and full-transcript half-life/translation-efficiency probes. These ask different questions; LucaOne stability and IsoFormer-style GTEx scores are not a direct contest.

Stage 4 remains active. Completed evidence shows task-dependent outcomes and the importance of a strong single-encoder reference. Seed-0 swaps cover the available benchmark/replacement combinations, while repeat coverage varies. Existing-comparison repeats, missing matched comparisons and the separate stability evaluation without sequence overlap belong here. Read each notebook's saved-result inventory and [run records](docs/run-records.md) before forming a multi-seed claim; repetitions do not remove overlap.

The findings characterize these settings and motivate hypotheses. A prospective rule needs a stated expectation, simpler controls and an evaluation not already used to develop it. Fungal species/condition transfer remains a candidate setting, not a selected dataset or implemented experiment.

### Stage 5: controlled tests of fusion hypotheses

The [Stage 5 plan](Code/stage5/README.md) tests explanations suggested by Stage 4: auxiliary correction of a strong predictor, gate competition and position-specific contributions, anchor-preserving attention, and CDS/UTR correspondence. Its proposed notebooks have not been implemented or run. Molecular-partner tasks are conditional transfer studies.

Existing results and their follow-ups remain in Stage 4; each new Stage 5 comparison includes its own matched baselines and repeats. A new split requires a distinct experiment identity, not automatically a new stage. The [Gantt and Details sheets](docs/Research_Project_Plan_Gantt.xlsx) map reliability to tasks 24/28, correction to 45, gates to 46, attention to 25, regions to 42 and conditional partner tasks to 47/48. Recorded statuses and historical dates do not establish execution.

## Repo structure

| Location | What to read or find there |
| --- | --- |
| [Code/](Code/) | Stage guides connect experiments; notebooks retain their inputs, choices, code, outputs and interpretation. |
| [Code/shared_code/](Code/shared_code/) | Reused methods, including [fusion implementations](Code/shared_code/fusion/). Comments and docstrings explain local mechanics. |
| [datasets/](datasets/) | Included input CSVs; additional downloads follow the cache policy below. |
| [docs/](docs/README.md) | Shared methods, planning, selected sources, meeting records and figure guidance. |
| [Papers/](Papers/README.md) | Retained paper versions and source locations. |
| [tests/](tests/README.md) | Controlled software checks, separate from scientific validation. |

The [target guide](docs/methods/data-and-targets.md), [sequence-to-prediction guide](docs/methods/sequences-to-predictions.md) and [representation guide](Code/stage1/representation-analysis.md) own reusable explanations. [Individual reviews](docs/notes/Reviews/) retain contributor attribution; Obsidian supports personal study without becoming a dependency for understanding the shared project.

[AGENTS.md](AGENTS.md) routes repository-aware agents to the maintained workflows. [Agent context](docs/agent-context.md) points to evidence cautions; [run records](docs/run-records.md) connect outputs with their producing inputs, code and environment.

## Data

Six supplied CSVs can be read without rebuilding them. Counts below describe source tables before notebook filtering or sampling. The same dataset name does not establish the same rows, inputs or evaluation; the same `Value` column name does not establish the same biological quantity.

### mRNA stability

[mRNA_Stability.csv](datasets/mRNA_Stability.csv) contains 65,356 observations of 29,949 distinct coding-sequence strings. `Value` is a decay-derived score standardized within experimental profiles, with higher meaning relatively more stable, not hours of half-life. Repeated sequences can have different context-dependent values, and supplied training/test partitions overlap.

The ≤999-nucleotide filter matches BLF's reported used-row count without proving identical membership. Its reported raw count still differs from this checked CodonBERT release. Published-split comparisons and new sequence-disjoint evaluations must remain separate; grouping exact sequences does not address every generalization concern. See the [dataset audit](Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) and [measurement trace](docs/methods/data-and-targets.md#mrna-stability-standardized-decay-profiles). The benchmark files came from [CodonBERT](https://github.com/Sanofi-Public/CodonBERT/tree/master/benchmarks/CodonBERT/data/fine-tune).

### mRFP expression control

[mRFP_Expression.csv](datasets/mRFP_Expression.csv) contains 1,459 observations of synonymous variants encoding one red fluorescent protein. Stage 1 checks representation behavior; Stage 4 tests expression prediction beyond calculated controls.

`Value` is the natural log of mean red fluorescence normalized against GFPuv. Normalization addresses variation shared proportionally by the reporters; the log expresses equal fold changes as equal differences. It is not a molecule count, and the loader does not log it again. The [target guide](docs/methods/data-and-targets.md#mrfp-fluorescence-to-the-value-column) retains the formulas, example and limitations; the [control guide](docs/methods/sequences-to-predictions.md#pipeline-control-synonymous-recoding) explains its role.

### GTEx transcript expression

[GTEx_pilot.csv](datasets/GTEx_pilot.csv) contains 2,000 protein-coding human transcripts: 1,000 from each published train/test partition. Rows supply transcript/CDS/UTR information, protein, a 6,000-nucleotide genomic window around the transcription start site, and 30 tissue targets from [IsoFormer's released dataset](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression).

Targets are released `log(x + 1)` tissue-averaged TPM values. The inspected release does not identify the log base; the pilot copies these values without the upstream loader's separate normalization statistics. They describe processed RNA abundance, not transcription rates or protein counts. See the [target trace](docs/methods/data-and-targets.md#gtex-tissue-expression-labels).

**Optional rebuilding:** from the repository root, `python Code/stage1/build_gtex_pilot.py` downloads/caches the full table, samples transcripts and retrieves genomic windows from Ensembl. This requires network access and **replaces the pilot CSV**; reading supplied inputs does not require it. The [builder](Code/stage1/build_gtex_pilot.py) checks columns and tissue order, publishes only completed validated writes, and stops on invalid caches. Preserve questionable files before choosing a fresh cache. Its [bounded network recovery](docs/sources.md#ensembl-sequence-requests-and-rate-limits) does not validate the biological design.

### Other BioLangFusion benchmarks

The [Stage 3 audit](Code/stage3/01_blf_datasets.ipynb) introduces these alongside stability; GTEx and mRFP are separate from BLF's five-benchmark suite.

| Supplied table | Target and interpretation |
| --- | --- |
| [Fungal_expression.csv](datasets/Fungal_expression.csv), 7,089 rows | Natural log of the supplied RNA-expression quantity; earlier normalization, units and organism metadata remain unresolved. [Trace](docs/methods/data-and-targets.md#fungal-expression). |
| [E.Coli_proteins.csv](datasets/E.Coli_proteins.csv), 6,348 rows | Classes 0/1/2 group gel-based low/medium/high recombinant protein production. E. coli is the expression host, not necessarily the gene's source organism. [Trace](docs/methods/data-and-targets.md#protein-expression-in-e-coli). |
| [CoV_Vaccine_Degradation.csv](datasets/CoV_Vaccine_Degradation.csv), 2,400 rows | Designed 81-nucleotide RNA fragments with standardized chemical-degradation targets; higher means more degradation. These are not verified coding sequences or cellular half-life measurements. [Trace](docs/methods/data-and-targets.md#cov-vac-chemical-degradation-of-designed-rna). |

### Full-transcript half-life and translation efficiency

The [Stage 4 notebook](Code/stage4/tier3_halflife_te.ipynb) downloads separate mRNABench tables with transcript-region information; they are not expanded versions of the stability CSV. The [stage guide](Code/stage4/README.md#full-transcript-half-life-and-translation-efficiency) distinguishes screened tables from the primary human encoder comparisons.

The primary half-life target is a consensus stability component, not time in hours. Human TE is an RNA-abundance-adjusted ribosome-occupancy score, not a raw ribosome/RNA ratio or completed proteins per second. [Target provenance](docs/methods/data-and-targets.md#full-transcript-targets-half-life-and-translation) explains their transformations. Better prediction of these scores does not mean proportional increases in lifetime or protein production.

## Setup

### Notebook dependencies

Use the project's Python environment. Install PyTorch through its [official selector](https://pytorch.org/get-started/locally/) for the operating system/backend, then install these notebook dependencies:

```sh
python -m pip install numpy pandas scipy scikit-learn umap-learn matplotlib biopython "transformers==5.15.1" multimolecule einops ipykernel ViennaRNA
```

`ViennaRNA` is imported as `RNA` for folding calculations in the [mRFP notebook](Code/stage4/tier1_mrfp_control.ipynb). Its calculation mode also uses pooled embeddings from [Stage 1's control analysis](Code/stage1/Stage1_stability.ipynb); reading saved results requires neither ViennaRNA nor those embeddings.

In VS Code, use the Python/Jupyter extensions and [select that environment as the notebook kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management). Stage 1 expects `Code/stage1/` as the kernel working directory for imports and relative embedding paths; dataset loaders resolve the repository's `datasets/` independently. Other notebooks retain their own settings and prerequisites.

**Optional viewers:** [Office Viewer](https://marketplace.visualstudio.com/items?itemName=imfing.office-viewer-vscode) for Office files, [PDF Viewer](https://marketplace.visualstudio.com/items?itemName=mathematic.vscode-pdf), and [Spreadsheet Viewer](https://marketplace.visualstudio.com/items?itemName=GrapeCity.gc-excelviewer) for CSVs. For CSVs, choose **Open Preview**; `csv-preview.formatValues = never` displays values without the default two-significant-digit formatting.

Stage 1 selects CUDA, then MPS, then CPU through `encoders.select_device`. MPS fallback must be configured before importing PyTorch; the notebook Imports cells set `PYTORCH_ENABLE_MPS_FALLBACK=1` only if absent. Record device, versions and reused-embedding origins with shared results.

`multimolecule` supplies RNA-FM, RiNALMo, mRNA-FM, CaLM and HyenaDNA. [encoders.py](Code/shared_code/encoders.py) contains checkpoint identifiers and compatibility loading for NT v2/DNABERT-2; DNABERT-2 also needs `einops`. The [earlier compatibility record](docs/run-records.md#legacy-observations-without-complete-run-identities) explains the Transformers pin. Inspect actual versions and traceback before changing dependencies.

Loading downloads missing checkpoints into the usual [Hugging Face user cache](https://huggingface.co/docs/transformers/installation#cache-directory), not a repository-configured `.model-cache/`. Embedding matrices are separate outputs. **A fresh-environment run has not been verified:** packages other than Transformers remain unpinned. NT v2, DNABERT-2, HyenaDNA, RiNALMo, CaLM and ProtBERT have pinned model revisions; the other six registry encoders use their checkpoint's current revision.

### Reading implementation code

In the [stability](Code/stage1/Stage1_stability.ipynb#Implementation-displays) and [GTEx](Code/stage1/Stage1_gtex.ipynb#Implementation-displays) notebooks, `SHOW_IMPLEMENTATION = False` makes `show_source` display function links. Set it to `True`, run the setting and rerun a display cell for expanded source; these cells do not run the analyses. The display describes current code, not proof of which version produced a saved result.

After editing the helper, reload `shared_code.notebook` and reimport `show_source`, or restart and rerun Setup. The [helper](Code/shared_code/notebook.py) retains full listings by default when `full` is omitted. Links assume a stage-folder notebook; use the displayed function and line number if a viewer does not jump to the line.

### Interpreting loading messages

Unavailable Windows symlinks can increase cache disk use without preventing downloads. See the [Hub cache limitations](https://huggingface.co/docs/huggingface_hub/v2.1.1/guides/manage-cache#limitations); that documentation version does not identify the installed library.

For the inspected encoders, unused `lm_head.*`, RNA-FM `ss_head.*`, or ProtBERT `cls.*` weights can be reported as unexpected; absent pooler parameters may be newly initialized. These are different cases. [embeddings.py](Code/shared_code/embeddings.py) averages `last_hidden_state` rather than using those heads or `pooler_output`. These particular messages do not identify missing weights in that embedding path; **other missing or unexpected parameters require inspection**. See the [loading definitions](https://huggingface.co/docs/transformers/main_classes/model).

NT v2/DNABERT-2 compatibility loaders instead compare weights directly and reject mismatches other than the unused pooler. A completed load establishes that check, not successful extraction or downstream analysis.

### Checking notebook and shared-code changes

Follow [tests/README.md](tests/README.md) from the repository root in the notebook environment. Its controlled checks do not run research analyses or download models; passing them does not establish scientific reproducibility. [Research software readings](docs/sources.md#organizing-and-checking-research-software) explain the relevant practices.

## Results, caches, and intermediate files

Keep routine model caches and regenerable intermediates local under specific ignore paths; retain supplied datasets and notebooks in Git. Share selected results deliberately, with [run records](docs/run-records.md) connecting inputs, code, checkpoints, settings, environment and outputs. Avoid blanket scientific-format ignores. Preserve evidence before any deliberate recomputation that could overwrite it.

When an output changes, update this inventory and its handling, adjusting [.gitignore](.gitignore) only for the relevant paths. Cache reuse must match the recorded input/extraction identity; an existing file alone is insufficient. The [cache guide](docs/methods/sequences-to-predictions.md#embedding-matrices-and-cache-reuse) explains verification and recovery.

| Location | Purpose and handling |
| --- | --- |
| `Code/stage1/embeddings/` | Local, ignored per-encoder matrices for sampled stability, published train/test subsets and `gtex_pilot_train/` / `gtex_pilot_test/`. Matching metadata govern reuse; preserve old evidence and choose a fresh directory for deliberate recomputation. |
| `Code/stage1/embeddings_legacy/` | Local historical caches; current Stage 1 notebooks do not read them. |
| `.data-cache/GTEx_final.csv` | Local, ignored full source table for pilot preparation. The GTEx audit can read it; explicit `run_audit` temporarily downloads it if absent. Reading saved audit JSON needs no CSV. |
| [Code/stage4/results/isoformer_cohort_audit/](Code/stage4/results/isoformer_cohort_audit/) | Shared audit/provenance JSONs. `gtex_audit_v2.json` remains separate from predecessor/historical records. Identical reruns retain existing bytes; conflicting identities/measurements cannot overwrite them. Temporary downloads are removed; inline plots create no separate figure files. |
| [docs/figures/stability/](docs/figures/stability/) and [docs/figures/gtex/](docs/figures/gtex/) | Deliberately shared figures, numeric data and manifests. The [figure guide](docs/figures/README.md) describes export from saved snapshots without encoder or analysis reruns. Export does not commit or push. |
| `.data-cache/mrnabench/` | Local, ignored downloaded tables for the half-life/TE notebook. |
| `Code/stage3/results/` and `Code/stage4/results/` | Small JSON/CSV and verification records are shared. Best-validation `*.weights.pt` files, recovery checkpoints, interrupted publications and `Code/*/embeddings/` remain local and ignored. Existing results without weights are reused without retraining merely to fill that gap. |
| [Windows E. coli records](Code/stage4/results/blf_phaseA/ecoli_proteins/) and [fungal seed-0 records](Code/stage3/results/fungal_expression/reproducible/) | Separate historical configurations, not extra repetitions of another cohort. Consult their [run records](docs/run-records.md) before combining evidence; historical runner/verifier scripts depended on a trainer refactor absent from this branch. |

## Contributors

- Julian McKinley ([@jtmckinley4](https://github.com/jtmckinley4))
- Chase Di Maria Breisinger ([@Lqvy](https://github.com/Lqvy))
