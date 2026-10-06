# multimodal-bio-fusion

A research project in the [Complex Adaptive Systems Laboratory](https://complexity.cecs.ucf.edu/directors-welcome/) at the University of Central Florida, studying how to choose and combine pretrained DNA, RNA, and protein foundation models. The team's manuscript is the Overleaf project [multimodal-bio-fusion](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f), which records the Stage 1 methods, metric definitions, and results.

## Start here

1. Read the [Stage 1 overview](docs/stage1-overview.md) for the research questions, the twelve encoders, and how the two experiments relate. Its [summary](docs/stage1-overview.md#summary) brings their findings together.
2. Follow the [mRNA stability notebook](Code/Stage1_stability.ipynb) as the first worked analysis. Read each question, its code and saved output, and the interpretation that follows. Links beside the analysis open the relevant method explanation.
3. Read the [GTEx transcript-expression notebook](Code/Stage1_gtex.ipynb) to compare the findings when the DNA input includes genomic context and the transcript includes regions absent from the protein. GTEx stands for [Genotype-Tissue Expression](https://www.genome.gov/Funded-Programs-Projects/Genotype-Tissue-Expression-Project); this experiment uses a small sample of IsoFormer's processed data with expression targets for 30 human tissues.

The saved notebooks can be studied without running their cells. When ready to execute an experiment, follow [Setup](#setup), select the notebook's Python environment, and use a fresh kernel with `Code/` as its working directory. Each experiment runs independently; reading stability first is a learning route, not an execution dependency.

For a specific question, go directly to the [method guides](docs/README.md#understand-the-methods), [Stage 1 results](#stage-1-results), [dataset descriptions](#data), or [Stage 2 alignment definition](#stage-2-defining-alignment). The [documentation index](docs/README.md#find-plans-and-literature) links the research canvas, schedule, bibliography, and literature review.

## Project background

The aim is to develop methods for choosing which biological modalities and pretrained models to combine, and for deciding when and how fusion is useful for a target task. The longer-term goal set with Mina Basirat is to fuse several pretrained encoders, with more than one model per modality (for example, several DNA, RNA, and protein models), and to compare those combinations. Those choices should be grounded in the biological question and the wet-lab work the predictions could inform. The [meeting slides](docs/notes/Mina_Meeting_Slides_2026-09-23.pptx), particularly slide 3, frame the contribution as a method for deciding whether and how to fuse.

Stage 1 compares twelve pretrained encoders, four each for DNA, RNA, and protein. An encoder turns a biological sequence into numerical representations called embeddings; the notebooks summarize these as one vector per input sequence. Frozen means the encoder weights are not updated during these analyses. We fit simple predictive models to those vectors and compare how the encoders represent corresponding inputs. The [overview's model table](docs/stage1-overview.md#models-and-input-settings) identifies the encoders, their pretraining data, and why they are included.

The two experiments supply different biological context. In the stability experiment, the DNA and RNA inputs are letter conversions of the same coding sequence, and the protein is its translation, which loses synonymous codon distinctions. In the GTEx experiment, the inputs include genomic DNA around the transcription start site, the transcript or its coding sequence, and the protein. Comparing these settings helps us examine whether a combination adds useful information for the prediction task. The choice of future tasks, model combinations, and evaluation criteria remains part of the research.

### Architecture priorities

| Priority | Architecture |
| --- | --- |
| 1 | Coupled Mamba |
| 2 | Isoformer-style cross-attention |
| Fallbacks | Cross-Mamba / BiMamba; mixture-of-experts fusion |

Coupled Mamba's priority was agreed at the September 23, 2026 online meeting. The [meeting slides](docs/notes/Mina_Meeting_Slides_2026-09-23.pptx), slide 6, list the primary and fallback architectures. Concatenation + MLP is the reference baseline in the research design.

## Repo structure

| Location | Contents |
| --- | --- |
| [Code/](Code/) | Analysis notebooks and the script that builds the GTEx pilot; local runs also produce intermediate outputs here. |
| [datasets/](datasets/) | The three supplied CSV inputs; their sources and contents are described in [Data](#data). |
| [Code/mbf/](Code/mbf/) | Reusable Python functions called by both notebooks: load datasets and encoders, prepare sequences, assign evaluation folds, extract embeddings, and calculate analyses. Follow notebook source links when you want to inspect a calculation. |
| [docs/](docs/README.md) | The Stage 1 overview, shared method explanations, research plans, bibliography, literature review, and selected sources. |
| [Code/build_gtex_pilot.py](Code/build_gtex_pilot.py) | Rebuilds the GTEx pilot CSV from external source data. The CSV is already included; this script is only needed to rebuild it. See [Data](#data) for its source and contents. |
| [tests/](tests/README.md) | Automated software checks for notebook setup, dataset paths, links to function definitions, and shared calculations using controlled examples. They help detect code breakage; they do not validate biological conclusions. |
| [docs/notes/Reviews/](docs/notes/Reviews/) | Individual paper reviews and interpretations, organized by contributor. |
| [docs/notes/](docs/notes/) | Working notes, meeting records, and discussion slides. |
| [Papers/](Papers/README.md) | Catalog of core and supporting papers, including retained PDFs, versions, and source links. |

Keep individual paper notes in contributor folders, such as [docs/notes/Reviews/Julian/](docs/notes/Reviews/Julian/) and `docs/notes/Reviews/Chase/`, so interpretations remain attributable. Obsidian supports personal study; explanations needed to understand the project belong in the shared documentation.

Agents working in this repository should start with [AGENTS.md](AGENTS.md).

## Stage 1 analysis

Each experiment keeps its settings, analysis calls, outputs, and interpretations together. The [method guides](docs/README.md#understand-the-methods) provide the longer derivations and biological explanations; links beside the relevant notebook cells connect them to the analysis. Reusable calculations live in [Code/mbf/](Code/mbf/). See [Reading implementation code](#reading-implementation-code) to inspect a function from the notebook.

The saved outputs were retained from the combined notebook. The [overview](docs/stage1-overview.md#summary) identifies the samples, evaluation protocol, and findings; follow [Setup](#setup) for a new run.

### Stage 1 results

Codon-level RNA and protein encoders carry the strongest linearly accessible stability signal. Sequence composition explains much of the raw agreement between encoders, and simple transcript-length features remain a strong GTEx baseline. Read the overview's [interpretation](docs/stage1-overview.md#summary) and [result tables](docs/stage1-overview.md#result-tables) alongside the notebook evidence before choosing a fusion experiment.

### Published comparisons and distinct inputs

The overview's [published comparisons](docs/stage1-overview.md#published-comparisons-and-distinct-inputs) place the frozen probes on BioLangFusion's and IsoFormer's published splits and explain the limits of comparing them with trained fusion models.

## Stage 2: defining alignment

Stage 2 is a decision rather than a computation, so it has no notebook. It is recorded in the Overleaf section "Stage 2: Defining Alignment" and in the [project notes](docs/notes/Mina_Project_Notes.docx), and the [October 7, 2026 meeting slides](docs/notes/Mina_Meeting_Slides_2026-10-07.pptx) present it. Any code it calls for is built and tested in Stage 3.

- **Definition.** Two encoders are aligned to the extent that their representations of corresponding inputs agree on held-out examples after the composition control. Agreement that letter, codon, and amino-acid frequencies reproduce is reported but not counted as alignment, and alignment is reported separately from predictive complementarity (the concatenation gain).
- **No explicit alignment loss in the first fusion models.** Per-sequence alignment does not track prediction error on either dataset, and most cross-modal agreement is composition, so the fusion architecture learns any useful interaction. An alignment-loss ablation on CaLM with a protein encoder, the one cross-modal pairing whose agreement survives the control, stays a conditional Stage 3 experiment.
- **Positional correspondence only where the biology supplies it.** On the stability data, DNA, RNA, and protein tokens can share one codon grid, the step grid Coupled Mamba needs. On GTEx, the DNA window, transcript, and protein have no position-by-position correspondence, so fusion there uses cross-attention or pooled representations.
- **One encoder per redundant family.** One ESM-2 model represents that family; ProtBERT is a separate candidate.
- **First pairings.** A codon-level RNA encoder (mRNA-FM or CaLM) with a protein encoder (ESM-2 150M or ProtBERT) on the stability data, and RNA with protein, adding the DNA window, on GTEx, where every comparison includes the length baseline.

### Next steps: Stage 3 fusion experiments

Stage 3 trains fusion models on the encoder pairings above and compares each with the best single encoder, the concatenation + MLP reference baseline, and, on GTEx, the length baseline. It needs token-level embeddings rather than the pooled vectors Stage 1 uses, and Coupled Mamba remains the priority architecture. Each Stage 3 fusion result is read against the Stage 1 alignment and complementarity measurements for the same pair.

## Data

The input datasets are included as CSV tables in `datasets/`. The notebooks load these tables; you can use the existing files without running a data-building script:

- [mRFP_Expression.csv](datasets/mRFP_Expression.csv): 1,459 rows containing 1,455 distinct sequence strings, from synonymous codon randomization of one gene.
- [mRNA_Stability.csv](datasets/mRNA_Stability.csv): 65,356 rows containing 29,949 distinct sequence strings.
- [GTEx_pilot.csv](datasets/GTEx_pilot.csv): 2,000 protein-coding human transcripts from [IsoFormer's GTEx transcript-expression table](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression), 1,000 from its published train split and 1,000 from its test split, each with its expression in 30 tissues. Each row holds the transcript, its untranslated regions and coding sequence, its protein, and 6,000 nucleotides of GRCh38 DNA centered on its transcription start site. [build_gtex_pilot.py](Code/build_gtex_pilot.py) produced it: it downloads IsoFormer's table, samples the transcripts, and reads the DNA windows from the [Ensembl REST service](https://rest.ensembl.org/documentation/info/sequence_region_post).

These counts describe the included CSVs before notebook filtering or subsampling.

**Rebuilding the GTEx pilot.** The `build_gtex_pilot.py` script prepares this input table rather than running the representation analyses. Run `python Code/build_gtex_pilot.py` from the repository root when a rebuild is needed. It downloads IsoFormer's larger table into the repository's `.data-cache/` if it is not already cached, selects transcripts from the published splits, obtains their genomic DNA windows from Ensembl, and writes `datasets/GTEx_pilot.csv`. Rebuilding is a separate preparation step that requires network access and replaces that CSV; it is not part of running the notebook on the supplied pilot. The [generated-file inventory](#generated-files) distinguishes the full source-table download from the supplied pilot and the embeddings saved during analysis.

The builder checks required source columns and the ordered 30-tissue block before use. Downloads and pilot CSVs are published only after a complete temporary-file write and validation; failure before replacement preserves the previous file. Invalid cached CSVs stop for inspection rather than silently downloading again. Preserve questionable files and their evidence before deliberately selecting a fresh source cache. A valid schema cannot establish whether an older download was complete. Sequence requests follow the [Ensembl batch and rate-limit contract](docs/sources.md#ensembl-sequence-requests-and-rate-limits): transient failures have at most five attempts, each network open uses a 120-second timeout, and a requested retry delay above 120 seconds stops with an explanation. Malformed or incomplete response batches fail without retry. These are software recovery checks; they do not validate the pilot's biological design.

The [dataset audit in the stability notebook](Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) records the following properties of `mRNA_Stability.csv`, which matter for comparisons with published results:

- The `Split` column assigns 45,749 rows to train, 9,803 to validation, and 9,804 to test. Of the 8,283 distinct test sequences, 5,392 also occur in training.
- 12,844 sequences occur more than once, and 12,775 of those carry differing `Value` labels; the median standard deviation of labels within one sequence is 0.49.
- 37% of rows are at most 1,000 nucleotides, so single-nucleotide encoders with a limit near 1,000 tokens see truncated input for most sequences.
- [BioLangFusion](Papers/core/BioLangFusion.pdf), Appendix A.2, reports 41,123 raw and 23,929 used mRNA stability sequences with the CodonBERT splits. Whether this file is the same version, and which filtering produced 23,929, is unresolved.

Comparisons with published numbers should therefore use the official split, while leakage-free estimates need a deduplicated, sequence-disjoint split. The CSVs were sourced from the fine-tuning benchmark data in [Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT/tree/master/benchmarks/CodonBERT/data/fine-tune). Their inclusion supplies the notebook inputs; interpreting a prediction still requires checking the source assay, labels, and retained sequence regions.

## Generated files

Keep routine model caches and regenerable intermediate outputs local and ignore their specific paths. Retain source datasets and notebooks in Git. Share selected results deliberately, with a [run record](docs/run-records.md) linking their producing code, inputs, model revisions, settings, environment, and retained outputs. Avoid blanket ignore rules for scientific file formats. [Sandve et al.'s reproducibility rules](docs/sources.md#sandve-et-al-ten-simple-rules-for-reproducible-computational-research) explain why result-producing steps, versions, intermediate data, and seeds matter; the inventory below is only part of that record.

When code adds or changes an output, update this inventory and its handling. Add or adjust specific paths in [.gitignore](.gitignore) for outputs kept local. Preserve an identified copy of any result needed as evidence before rerunning code that overwrites it. Paths below assume the working directory specified in Setup.

| Generated file | Producer | Purpose | Handling |
| --- | --- | --- | --- |
| `Code/stage1_embeddings/<dataset>_n<rows>_seed<seed>/` | [Stability notebook](Code/Stage1_stability.ipynb) | One embedding matrix per encoder for one dataset sample, saved with its cache version, checkpoint, requested revision, token limit, row count, and effective-input fingerprint. Only valid matching caches are reused. | Local and ignored; preserve old evidence and use a fresh directory for deliberate recomputation. See [cache recovery](docs/methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse). |
| `Code/stage1_embeddings/<dataset>_published_train<rows>_seed<seed>/` and `..._published_test<rows>_seed<seed>/` | [Stability notebook](Code/Stage1_stability.ipynb#Published-split-comparison) | Embeddings of the rows sampled from the published train and test splits for the published-split comparison, saved and reused as above. | Local and ignored. |
| `Code/stage1_embeddings/gtex_pilot_train/` and `gtex_pilot_test/` | [GTEx notebook](Code/Stage1_gtex.ipynb) | Embeddings of the two halves of `GTEx_pilot.csv`, one matrix per encoder from that encoder's own input, saved and reused as above. | Local and ignored. |
| `.data-cache/GTEx_final.csv` | `Code/build_gtex_pilot.py` | IsoFormer's full GTEx table, about 645 MB, downloaded to sample the pilot. | Local and ignored; only needed to rebuild `GTEx_pilot.csv`. |

## Setup

### Notebook dependencies

Use your existing Python environment for the project. Install PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) for your operating system and available compute backend. Install the remaining notebook dependencies in that environment:

```sh
python -m pip install numpy pandas scipy scikit-learn umap-learn matplotlib biopython "transformers==5.15.1" multimolecule einops ipykernel
```

Open notebooks with VS Code's Python and Jupyter extensions and [select the environment containing these packages as the kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management). Use `Code/` as the kernel's working directory for local package imports and embedding-cache paths. Dataset loaders resolve the supplied CSVs from the repository's `datasets/` folder independently of that working directory.

The notebooks select an available backend through `encoders.select_device`: CUDA, then MPS, then CPU. This is a runtime choice for each contributor's environment. Record the device and software versions associated with shared results in [run records](docs/run-records.md), including the origin of reused embeddings.

`multimolecule` provides RNA-FM, RiNALMo, mRNA-FM, CaLM, and HyenaDNA. The installation command currently pins `transformers` to 5.15.1; the [earlier compatibility report](docs/run-records.md#legacy-observations-without-complete-run-identities) records why. If an import fails, inspect its traceback and the installed package versions before choosing a dependency change.

Nucleotide Transformer v2 and DNABERT-2 load custom model code written for `transformers` 4; the encoder registry, [encoders.py](Code/mbf/encoders.py), applies the small compatibility adjustments they need under `transformers` 5 and pins both to a fixed checkpoint revision.

`PYTORCH_ENABLE_MPS_FALLBACK` must be configured before PyTorch is imported for the MPS fallback to take effect. Each experiment notebook's Imports cell uses `os.environ.setdefault` before `import torch`, setting the value to `1` only when it is absent; an existing value is preserved.

When an encoder is needed, either experiment's loading or embedding cells download its checkpoint unless it is already cached. The default twelve encoders total about 8 GB; [encoders.py](Code/mbf/encoders.py) lists their Hugging Face identifiers. DNABERT-2's model code requires `einops`. Hugging Face normally stores these downloads in the [user's cache](https://huggingface.co/docs/transformers/installation#cache-directory); the notebooks do not configure the repository's `.model-cache/` directory.

The experiment notebooks save embedding matrices separately under `Code/stage1_embeddings/`. See the [generated-file inventory](#generated-files) for their paths and the policy for preserving results before recomputation, and [embedding matrices and cache reuse](docs/methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse) for what the cache checks before reusing a matrix.

This dependency list covers the imports in both Stage 1 experiment notebooks and the `mbf` package. A fresh-environment run has not been verified. The repository does not yet pin package versions other than `transformers`. Nucleotide Transformer v2, DNABERT-2, HyenaDNA, RiNALMo, CaLM, and ProtBERT have pinned model revisions; the other six encoders load the current revision of their checkpoint.

### Reading implementation code

The [stability](Code/Stage1_stability.ipynb#Implementation-displays) and [GTEx](Code/Stage1_gtex.ipynb#Implementation-displays) notebooks use `SHOW_IMPLEMENTATION = False` to keep implementation displays compact: each `show_source` cell lists links to the functions or classes in `Code/mbf/`, including their source lines. Both notebooks still show their analysis calls, settings, and results.

To expand the implementation while studying a method, set `SHOW_IMPLEMENTATION = True`, run that setting, and rerun the relevant `show_source` cell. These display cells inspect source code; they do not load encoders or rerun analyses. Set the flag back to `False` and rerun a display cell to return to its links. Source links and expanded listings describe the code currently on disk; they do not independently verify which implementation produced a saved research result.

If the source-display helper changes while a kernel is open, reload `mbf.notebook` and reimport `show_source`, or restart the kernel and rerun Setup, before using the updated helper.

The [presentation helper](Code/mbf/notebook.py) keeps full listings as the default for calls without a `full` argument, preserving the behavior of other notebooks. Relative source links assume the notebook is in `Code/`. A viewer may open the file without jumping to its line anchor; the displayed function name and line number locate the definition.

### Interpreting loading messages

A Hugging Face warning about unavailable Windows symlinks means downloaded files can still be cached, but storing multiple revisions can require more disk space. The warning alone does not show that a download failed. See the [Hub cache limitations](https://huggingface.co/docs/huggingface_hub/v2.1.1/guides/manage-cache#limitations), checked in the version 2.1.1 documentation on October 5, 2026; this documentation version does not identify the library installed in a notebook environment.

Loading through the Transformers auto classes can report checkpoint weights that do not match the instantiated model. For Nucleotide Transformer 500M and the three ESM-2 models, the reports typically include:

| Parameter names in the report | Status | Meaning for this notebook |
| --- | --- | --- |
| `lm_head.*` | `UNEXPECTED` | The checkpoint contains language-modeling prediction-head weights that the loaded base encoder does not use. |
| `pooler.dense.weight` and `pooler.dense.bias` | `MISSING` | These pooler parameters are absent from the checkpoint and were newly initialized. |

Unused checkpoint weights and newly initialized parameters are different cases; see the Transformers [loading-warning definitions](https://huggingface.co/docs/transformers/main_classes/model). `embeddings.embed` averages `last_hidden_state` itself. It uses neither the separate `pooler_output` nor the language-modeling head, so these particular entries do not identify missing weights in the hidden-state path used here. See the [ESM output definitions](https://huggingface.co/docs/transformers/main/en/model_doc/esm).

RNA-FM reports the same pooler entries and unused `lm_head.*` and `ss_head.*` weights; the latter belong to a secondary-structure prediction head. mRNA-FM, RiNALMo, CaLM, and HyenaDNA load through the same `multimolecule` package, and the same component-level interpretation applies where their reports list these entries. ProtBERT's unexpected `cls.*` weights belong to its masked-token and next-sentence prediction heads. These listed heads and poolers are not used by `embed`; a different missing or unexpected parameter requires checking which component it belongs to.

Nucleotide Transformer v2 and DNABERT-2 use the compatibility loaders in [encoders.py](Code/mbf/encoders.py), which build their models and compare checkpoint weights directly. They do not print the auto-class report above: the loader raises an error if anything other than the unused pooler is missing or unexpected. A completed load confirms the weight comparison only; embedding extraction and downstream analyses require their own checks.

Past environment observations and their evidence are listed in [run records](docs/run-records.md#legacy-observations-without-complete-run-identities). Warnings associated with a particular analysis remain beside its notebook results.

### Checking notebook and shared-code changes

The [notebook checks](tests/README.md) explain how to check independent setup, saved source links, and shared concatenation calculations. They use synthetic inputs for computation checks and do not run the research analyses or download models. Their passing results do not establish scientific reproducibility. Follow the linked instructions from the repository root using the notebook environment.

The [research software readings](docs/sources.md#organizing-and-checking-research-software) explain the modularity, testing, and provenance practices relevant to this organization, with their local uses and limits.

### Viewing project files in VS Code

These optional extensions provide convenient access to the documents and datasets in this repository.

| Files | Extension | Use |
| --- | --- | --- |
| `.pptx`, `.docx`, `.xlsx` | [Office Viewer for VS Code](https://marketplace.visualstudio.com/items?itemName=imfing.office-viewer-vscode) | Preview slides, documents, and spreadsheets. |
| `.pdf` | [PDF Viewer](https://marketplace.visualstudio.com/items?itemName=mathematic.vscode-pdf) | Read papers inside VS Code. |
| `.csv` | [Spreadsheet Viewer](https://marketplace.visualstudio.com/items?itemName=GrapeCity.gc-excelviewer) | Browse data in a grid, sort columns, and filter rows. |

For CSV files, choose **Open Preview** from the file's context menu. To display numeric values as stored, set `csv-preview.formatValues` to `never`; the default preview formats numbers to two significant digits.

## Contributors

- Julian McKinley ([@jtmckinley4](https://github.com/jtmckinley4))
- Chase Di Maria Breisinger ([@Lqvy](https://github.com/Lqvy))
