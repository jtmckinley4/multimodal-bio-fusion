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

Start with the [overview](docs/stage1-overview.md) for the models, research questions, and cross-dataset summary. Read the stability notebook for the first method walkthrough, then use GTEx to compare the findings when DNA, transcript, and protein are distinct inputs.

| File | What to read or run |
| --- | --- |
| [Stage 1 overview](docs/stage1-overview.md) | Model context, research questions, and cross-dataset summary in a Markdown document. |
| [Stage1_stability.ipynb](Code/Stage1_stability.ipynb) | Independent experiment: sequence checks, dataset audit, embeddings, stability and biological-target probes, representation comparisons, synonymous-recoding control, attention diagnostics, and the BioLangFusion published-split comparison. |
| [Stage1_gtex.ipynb](Code/Stage1_gtex.ipynb) | Independent experiment: GTEx inputs, expression probes and baselines, concatenation, the IsoFormer published-split comparison, representation comparisons, and attention diagnostics. |

Each experiment keeps its settings, analysis calls, outputs, and interpretations together. The [method guides](docs/README.md#understand-the-methods) provide the longer derivations and biological explanations; links beside the relevant notebook cells connect them to the analysis. Reusable calculations live in [Code/mbf/](Code/mbf/). See [Reading implementation code](#reading-implementation-code) to inspect a function from the notebook.

Each experiment has its own Setup section and can run without executing the other. For a new run, use a fresh kernel with `Code/` as its working directory and follow that experiment from Setup onward; see [Setup](#setup). The saved outputs and execution counts were retained from the combined notebook, not produced by a new run of the separated files. They cover twelve encoders on 981 retained mRNA stability sequences and the GTEx pilot's 999 train and 996 test transcripts. Keep these results associated with the producing code version and settings.

Probes use ridge regression over five folds that keep identical sequences together. A hash of each sequence pins the fold assignment, so it does not depend on the machine, and each probe is repeated over ten further fold assignments to show how much the partition moves the result. The concatenated-embedding probe uses a linear model, distinct from the concatenation + MLP reference baseline. These analyses can inform experiment design; their outputs alone do not establish which fusion method to use or validate a wet-lab application.

### Stage 1 results

The main results retained in the [stability](Code/Stage1_stability.ipynb) and [GTEx](Code/Stage1_gtex.ipynb) notebooks cover 981 retained stability sequences and the GTEx pilot's 999 train transcripts for cross-validation. Probe scores are mean $R^2$ over the five pinned folds, with the fold standard deviation, and the mean over ten further fold assignments. Each experiment gives the full tables and interpretations beside its outputs; the overview's [Summary](docs/stage1-overview.md#summary) draws them together.

| Encoder | Modality | Stability $R^2$, pinned (fold SD) | Stability $R^2$, ten further assignments | GC3 $R^2$ | Share of embedding variance explained by composition | GTEx expression $R^2$, pinned (fold SD) |
| --- | --- | --- | --- | --- | --- | --- |
| Nucleotide Transformer 500M human-ref | DNA | 0.032 (0.043) | 0.046 | 0.968 | 0.618 | 0.034 (0.025) |
| Nucleotide Transformer v2 100M multi-species | DNA | 0.080 (0.032) | 0.074 | 0.986 | 0.804 | 0.036 (0.030) |
| DNABERT-2 | DNA | 0.054 (0.031) | 0.057 | 0.948 | 0.610 | 0.033 (0.028) |
| HyenaDNA large | DNA | 0.026 (0.017) | 0.019 | 0.896 | 0.924 | 0.031 (0.028) |
| RNA-FM | RNA | 0.052 (0.032) | 0.046 | 0.917 | 0.677 | 0.149 (0.037) |
| RiNALMo 150M | RNA | 0.078 (0.021) | 0.079 | 0.949 | 0.648 | 0.149 (0.047) |
| mRNA-FM | RNA | 0.127 (0.032) | 0.110 | 0.946 | 0.296 | 0.172 (0.044) |
| CaLM | RNA | 0.127 (0.058) | 0.132 | 0.982 | 0.472 | 0.120 (0.032) |
| ESM-2 8M | Protein | 0.127 (0.044) | 0.131 | 0.379 | 0.479 | 0.091 (0.022) |
| ESM-2 35M | Protein | 0.114 (0.049) | 0.123 | 0.387 | 0.446 | 0.104 (0.031) |
| ESM-2 150M | Protein | 0.144 (0.035) | 0.142 | 0.436 | 0.437 | 0.130 (0.043) |
| ProtBERT | Protein | 0.143 (0.041) | 0.135 | 0.440 | 0.415 | 0.163 (0.040) |

On GTEx, four transcript-length features alone reach an expression $R^2$ of 0.242 (0.036).

| Encoder pairs, stability data | Linear CKA | CKA after composition control | CCA Recall@1 (chance 0.0034) | Recall@1 after composition control |
| --- | --- | --- | --- | --- |
| Within ESM-2 | 0.553 to 0.744 | 0.376 to 0.643 | 0.942 to 0.955 | 0.701 to 0.869 |
| ProtBERT with ESM-2 | 0.199 to 0.312 | 0.093 to 0.162 | 0.818 to 0.873 | 0.481 to 0.622 |
| Within DNA | 0.270 to 0.736 | 0.038 to 0.113 | 0.591 to 0.773 | 0.103 to 0.162 |
| RNA-FM with RiNALMo | 0.809 | 0.378 | 0.931 | 0.704 |
| mRNA-FM with CaLM | 0.163 | 0.084 | 0.619 | 0.124 |
| DNA with RNA-FM or RiNALMo | 0.171 to 0.613 | 0.044 to 0.164 | 0.491 to 0.677 | 0.052 to 0.244 |
| DNA with mRNA-FM or CaLM | 0.030 to 0.517 | 0.012 to 0.075 | 0.271 to 0.632 | 0.027 to 0.086 |
| DNA with protein | 0.058 to 0.163 | 0.010 to 0.072 | 0.168 to 0.540 | 0.021 to 0.082 |
| RNA-FM or RiNALMo with protein | 0.116 to 0.339 | 0.035 to 0.161 | 0.196 to 0.509 | 0.021 to 0.223 |
| mRNA-FM with protein | 0.037 to 0.295 | 0.021 to 0.261 | 0.405 to 0.464 | 0.069 to 0.113 |
| CaLM with protein | 0.309 to 0.448 | 0.246 to 0.401 | 0.794 to 0.866 | 0.581 to 0.687 |

- The protein encoders and the two codon-level RNA encoders carry the most linearly accessible stability signal ($R^2$ about 0.11 to 0.14). The encoders that read single nucleotides, six-nucleotide tokens, or byte-pair tokens reach 0.03 to 0.08, and sequence length alone predicts almost nothing ($R^2$ 0.003).
- Every nucleotide encoder retains synonymous codon information that the protein encoders cannot see, shown by the GC3 probes.
- On the stability data, every pairing of a codon-level RNA encoder with a protein encoder improves on the better of the two under all ten further fold assignments, by about 0.01 to 0.02. Larger concatenations do not improve on the best single encoder.
- Agreement follows tokenization and training more than the modality label: RNA-FM and RiNALMo agree most of any pair, and encoders reading single letters or short tokens of the same sequence agree whether they are labeled DNA or RNA. mRNA-FM agrees globally with almost no encoder.
- Most agreement involving a DNA encoder comes from shared sequence composition: removing letter, codon, and amino-acid frequencies cuts CKA by 52% to 90% for every such pair. Three groups keep strong agreement after the control: the protein encoders, RNA-FM with RiNALMo, and CaLM with the protein encoders, which still retrieves its protein partner 58% to 69% of the time, about 170 to 200 times chance.
- Position never explains the attention associations at candidate stability patterns, and codon context explains most of them. CaLM's attention to AU-rich pentamers survives both permutation nulls, a lead rather than a finding.
- On GTEx, where each modality is a different molecule, RNA and protein encoders predict expression (0.09 to 0.18) far better than DNA encoders reading the region around the start site (0.03 to 0.05), but four length features beat every single encoder. 47 of the 66 pairs gain under every fold assignment, and all twelve encoders together reach 0.239, the level of the length features. The DNA window shares almost no retrievable correspondence with the transcript or protein (Recall@1 at most 0.041).
- Per-sequence alignment does not track prediction error for any of the 66 pairs on either dataset after correction for multiple tests.

### Published comparisons and distinct inputs

The frozen encoders are compared with published fusion studies in two settings: [Setting A in the stability notebook](Code/Stage1_stability.ipynb#Published-split-comparison), fitting probes on rows sampled from the published train split and scoring its test split, and [Setting B in the GTEx notebook](Code/Stage1_gtex.ipynb#GTEx-published-split-comparison), whose rows carry separate DNA, transcript, and protein sequences. The Overleaf sections on the published split and on GTEx give the full comparison.

| Setting | Dataset | How DNA enters | Published comparison | Frozen encoders here |
| --- | --- | --- | --- | --- |
| A: derived modalities | CodonBERT mRNA stability (the current CSV) | A DNA encoder reads the coding sequence in DNA letters, an RNA encoder reads it in RNA letters, and a protein encoder reads its translation. | [BioLangFusion](Papers/core/BioLangFusion.pdf), Table 1: best fusion Spearman 0.563 versus 0.553 for the best single encoder. | BioLangFusion's three encoders concatenated reach Spearman 0.364 on 981 published test rows; ProtBERT alone reaches 0.396. |
| B: distinct modalities | [IsoFormer GTEx transcript expression](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression) | Genomic DNA centered on the transcription start site, alongside the full transcript and the protein. | [IsoFormer](Papers/core/Multi-Modal-Transfer-Learning.pdf), Table 2: three modalities reach $R^2$ 0.43 versus 0.36 for RNA alone. | The three-modality sets reach $R^2$ 0.216 to 0.249 on 996 published test transcripts, with the modalities in IsoFormer's order. |

In Setting A all three inputs derive from one coding sequence, so differences between encoders come from pretraining corpora and tokenization rather than new biological information. The stability CSV has no gene or transcript identifiers, so genomic context around each gene is not available without a separate mapping step. Setting B supplies DNA that carries promoter and regulatory context absent from the protein. The frozen-probe values fall short of the published ones because those studies train their heads on token-level embeddings with far more data; the comparison places the frozen encoders on the published splits rather than reproducing the published models.

The twelve encoders include BioLangFusion's three, Nucleotide Transformer v2 100M multi-species, RNA-FM, and ESM-2 8M, and IsoFormer's protein encoder, ESM-2 150M. Nucleotide Transformer v2 and DNABERT-2 load custom model code written for `transformers` 4; the encoder registry, [encoders.py](Code/mbf/encoders.py), applies the small compatibility adjustments they need under `transformers` 5 and pins both to a fixed checkpoint revision. Trained fusion architectures, including BioLangFusion's fusion heads and IsoFormer's cross-attention, need token-level embeddings and belong to Stage 3.

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

The [dataset audit in the stability notebook](Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) records the following properties of `mRNA_Stability.csv`, which matter for comparisons with published results:

- The `Split` column assigns 45,749 rows to train, 9,803 to validation, and 9,804 to test. Of the 8,283 distinct test sequences, 5,392 also occur in training.
- 12,844 sequences occur more than once, and 12,775 of those carry differing `Value` labels; the median standard deviation of labels within one sequence is 0.49.
- 37% of rows are at most 1,000 nucleotides, so single-nucleotide encoders with a limit near 1,000 tokens see truncated input for most sequences.
- [BioLangFusion](Papers/core/BioLangFusion.pdf), Appendix A.2, reports 41,123 raw and 23,929 used mRNA stability sequences with the CodonBERT splits. Whether this file is the same version, and which filtering produced 23,929, is unresolved.

Comparisons with published numbers should therefore use the official split, while leakage-free estimates need a deduplicated, sequence-disjoint split. The CSVs were sourced from the fine-tuning benchmark data in [Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT/tree/master/benchmarks/CodonBERT/data/fine-tune). Their inclusion supplies the notebook inputs; interpreting a prediction still requires checking the source assay, labels, and retained sequence regions.

## Generated files

Keep routine model caches and regenerable intermediate outputs local and ignore their specific paths. Retain source datasets and notebooks in Git. Share selected results deliberately, with the producing notebook or code version, input identities, model revisions, and relevant settings so that others can interpret them. Avoid blanket ignore rules for scientific file formats.

When code adds or changes an output, update this inventory and its handling. Add or adjust specific paths in [.gitignore](.gitignore) for outputs kept local. Preserve an identified copy of any result needed as evidence before rerunning code that overwrites it. Paths below assume the working directory specified in Setup.

| Generated file | Producer | Purpose | Handling |
| --- | --- | --- | --- |
| `Code/stage1_embeddings/<dataset>_n<rows>_seed<seed>/` | [Stability notebook](Code/Stage1_stability.ipynb) | One embedding matrix per encoder for one dataset sample, saved with its checkpoint, revision, token limit, and a fingerprint of the ordered input sequences. The notebook reuses a matrix only when all of these match. | Local and ignored; delete an encoder's file to recompute it. |
| `Code/stage1_embeddings/<dataset>_published_train<rows>_seed<seed>/` and `..._published_test<rows>_seed<seed>/` | [Stability notebook](Code/Stage1_stability.ipynb#Published-split-comparison) | Embeddings of the rows sampled from the published train and test splits for the published-split comparison, saved and reused as above. | Local and ignored. |
| `Code/stage1_embeddings/gtex_pilot_train/` and `gtex_pilot_test/` | [GTEx notebook](Code/Stage1_gtex.ipynb) | Embeddings of the two halves of `GTEx_pilot.csv`, one matrix per encoder from that encoder's own input, saved and reused as above. | Local and ignored. |
| `.data-cache/GTEx_final.csv` | `Code/build_gtex_pilot.py` | IsoFormer's full GTEx table, about 645 MB, downloaded to sample the pilot. | Local and ignored; only needed to rebuild `GTEx_pilot.csv`. |

## Setup

### Notebook dependencies

Use your existing Python environment for the project. Install PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) for your operating system and CPU or CUDA configuration. Install the remaining notebook dependencies in that environment:

```sh
python -m pip install numpy pandas scipy scikit-learn umap-learn matplotlib biopython "transformers==5.15.1" multimolecule einops ipykernel
```

Open notebooks with VS Code's Python and Jupyter extensions and [select the environment containing these packages as the kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management). Use `Code/` as the kernel's working directory for local package imports and embedding-cache paths. Dataset loaders resolve the supplied CSVs from the repository's `datasets/` folder independently of that working directory.

The Stage 1 experiment notebooks use a CUDA GPU when available, then an Apple Silicon GPU through PyTorch's MPS backend, and otherwise the CPU; on MPS they let operations the backend lacks fall back to the CPU. `multimolecule` provides RNA-FM, RiNALMo, mRNA-FM, CaLM, and HyenaDNA. Version 0.2.1 imports with `transformers` 5.14.1 and 5.15.1 but not 5.16 or later, which is why `transformers` is pinned. If `import multimolecule` fails in an Anaconda environment with an older `datasets` or `huggingface_hub`, upgrade `datasets` and `fsspec` and reinstall `huggingface_hub`.

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

Loading through the Transformers auto classes can report checkpoint weights that do not match the instantiated model. For Nucleotide Transformer 500M and the three ESM-2 models, the reports typically include:

| Parameter names in the report | Status | Meaning for this notebook |
| --- | --- | --- |
| `lm_head.*` | `UNEXPECTED` | The checkpoint contains language-modeling prediction-head weights that the loaded base encoder does not use. |
| `pooler.dense.weight` and `pooler.dense.bias` | `MISSING` | These pooler parameters are absent from the checkpoint and were newly initialized. |

Unused checkpoint weights and newly initialized parameters are different cases; see the Transformers [loading-warning definitions](https://huggingface.co/docs/transformers/main_classes/model). `embeddings.embed` averages `last_hidden_state` itself. It uses neither the separate `pooler_output` nor the language-modeling head, so these particular entries do not identify missing weights in the hidden-state path used here. See the [ESM output definitions](https://huggingface.co/docs/transformers/main/en/model_doc/esm).

RNA-FM reports the same pooler entries and unused `lm_head.*` and `ss_head.*` weights; the latter belong to a secondary-structure prediction head. mRNA-FM, RiNALMo, CaLM, and HyenaDNA load through the same `multimolecule` package, and the same component-level interpretation applies where their reports list these entries. ProtBERT's unexpected `cls.*` weights belong to its masked-token and next-sentence prediction heads. These listed heads and poolers are not used by `embed`; a different missing or unexpected parameter requires checking which component it belongs to.

Nucleotide Transformer v2 and DNABERT-2 use the compatibility loaders in [encoders.py](Code/mbf/encoders.py), which build their models and compare checkpoint weights directly. They do not print the auto-class report above: the loader raises an error if anything other than the unused pooler is missing or unexpected. A completed load confirms the weight comparison only; embedding extraction and downstream analyses require their own checks.

The saved HyenaDNA CPU/MPS comparison remains in the [notebook's loading notes](Code/Stage1_stability.ipynb#Interpreting-the-loading-messages), and its [CCA results](Code/Stage1_stability.ipynb#Canonical-correlation-and-cross-modal-retrieval) identify the pairs with convergence warnings.

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
