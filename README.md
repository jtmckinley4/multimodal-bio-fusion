# multimodal-bio-fusion

A research project in the [Complex Adaptive Systems Laboratory](https://complexity.cecs.ucf.edu/directors-welcome/) at the University of Central Florida, studying how to choose and combine pretrained DNA, RNA, and protein foundation models. The team's manuscript is the Overleaf project [multimodal-bio-fusion](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f).

## Project background

We're investigating how models trained on DNA, RNA, and protein sequences can work together. These models have learned from different amounts and kinds of biological data. Some were trained across many species, while others focus on particular organisms or sequence types. Their training data can also overlap, and their model designs and training methods differ. An **encoder** turns a sequence into a numerical representation that a predictive model can use. Even when two encoders receive the same sequence, their representations may make different patterns useful for prediction.

That gives us a reason to compare the encoders before deciding how to combine them. If one already provides what we need for a prediction, adding another may contribute little. A second encoder may instead make useful information available that the first does not readily provide. We want to understand when combining representations, or **fusion**, improves prediction beyond a suitable single encoder and simpler features calculated from the sequence.

The longer-term goal set with the advisor is to compare combinations with more than one model per modality, such as several DNA, RNA, and protein encoders. The September 23 meeting frames our contribution as a method for deciding whether and how to fuse. We are exploring that question through biological prediction benchmarks and comparisons with methods from the papers. These give us concrete outcomes to predict while we develop the methodology; no particular wet-lab application.

**Project direction:** [September 23 meeting slides, especially slide 3](docs/notes/Slides/Mina_Meeting_Slides_2026-09-23.pptx) · [Research plan and Gantt reading route](docs/README.md#follow-the-research-plan).

## Research progression

We began by examining the encoders' existing representations in Stage 1, then used those observations to define what we mean by alignment in Stage 2. Stage 3 builds and checks the models needed to test our choices, and Stage 4 compares their predictive performance. This gives the project a reading order, but the work moves back and forth: the current plan takes the Stage 3 baseline implementations into Stage 4 comparisons, then returns to Stage 3 to develop the proposed fusion designs before evaluating them in Stage 4. Later findings can also lead us to revisit an earlier interpretation. When that happens, we retain the definition and evaluation procedure used for the earlier result so its meaning remains clear.

### Stage 1 analysis

Before choosing how to combine these models, we first examined what their existing representations could support. This follows the initial direction in the [project notes](docs/notes/Mina_Project_Notes.docx) to inspect frozen representations before designing an alignment method. We compare what can be predicted from each encoder, how their representations relate, and whether simply joining two representations already helps. That gives later experiments a starting point: which individual models to compare against, which limitations to investigate, and which combinations may be worth trying. It does not settle the best fusion architecture in advance. The initial stability pilot expanded from two encoders to twelve, and we extended the investigation to transcript-expression prediction across human tissues using GTEx.

The **mRNA stability experiment** uses coding sequences paired with numerical stability scores. Stability concerns how long an mRNA persists before degradation. Stability/expression prediction was already proposed in the [initial downstream-task draft](docs/notes/DNA_RNA_Protein_Models_and_Downstream_Task%5B95%5D.docx). DNA and RNA encoders receive the same coding sequence in their respective letter alphabets; protein encoders receive its translation into amino acids. This follows BioLangFusion's input construction and provides a common sequence origin for the first comparison.

Reading stability first introduces the common analyses with inputs derived from one coding sequence. GTEx then shows what changes when the inputs cover different biological regions and the prediction targets span multiple tissues.

**Stage 1 reading route:** [Stage 1 folder guide](Code/stage1/README.md) · [mRNA stability notebook](Code/stage1/Stage1_stability.ipynb) · [mRNA stability dataset: sequences, scores, and source](#mrna-stability).

The **GTEx transcript-expression experiment** extends that investigation to inputs containing different biological regions. It uses protein-coding human transcripts and expression targets for 30 tissues, including brain, liver, and kidney. DNA encoders read a genomic window around the transcription start site; RNA encoders read the transcript or its coding region, depending on the model; protein encoders read the corresponding amino-acid sequence. The transcript includes untranslated regions absent from the protein. This pilot uses IsoFormer's processed Genotype-Tissue Expression data. Its target and input regions differ from stability, so differences in results cannot be attributed to the input construction alone.

**Continue:** [GTEx notebook](Code/stage1/Stage1_gtex.ipynb) · [GTEx pilot: inputs, tissue targets, and preparation](#gtex-transcript-expression) · [GTEx project](https://www.genome.gov/Funded-Programs-Projects/Genotype-Tissue-Expression-Project).

Each experiment compares twelve encoders, four per modality. We summarize each encoder's output as a vector of numbers called a **sequence embedding**. The encoders remain **frozen**, meaning their learned weights are not updated using these datasets. We then fit the same kind of smaller predictive model, a standardized linear ridge-regression probe, to each encoder's embeddings. This lets us ask which representations make the target easier for that predictor to recover.

The dataset gives us sequences and outcomes, but it cannot tell us in advance how well a particular representation will support prediction. We examine all twelve encoders individually and compare all 66 distinct encoder pairs, including pairs within and across modalities. These comparisons describe what the chosen analyses reveal about the frozen representations; other predictors or fine-tuning could produce different results.

To understand the comparisons, we examine how the representations organize the same examples, how much agreement can be explained by sequence composition, and whether joining embeddings improves prediction. We also inspect attention in selected compatible nucleotide encoders; that diagnostic does not cover every model equally. The stability notebook includes an additional **mRFP control**: different nucleotide sequences encode the same red fluorescent protein. It checks that nucleotide representations can vary while a protein encoder receives an unchanged amino-acid sequence. Stage 4 later tests prediction of the expression scores in this control dataset.

**Comparison scope:** [Encoder coverage and the 66-pair calculation](Code/stage1/README.md#comparing-the-encoders) · [Stage 1 encoder table](docs/stage1-overview.md#models-and-input-settings) · [mRFP control dataset](#mrfp-expression-control).

The saved notebooks can be read without running their cells. Neither experiment requires the other to execute. Each keeps its question, settings, analysis calls, outputs, and interpretation together; the [run records](docs/run-records.md#recorded-runs) identify the retained executions. For a new run, follow [Setup](#setup).

#### Stage 1 results

Codon-level RNA and protein encoders carry the strongest linearly accessible stability signal. Sequence composition explains much of the raw agreement between encoders, and simple transcript-length features remain a strong GTEx baseline. Read the overview's [interpretation](docs/stage1-overview.md#summary) and [result tables](docs/stage1-overview.md#result-tables) alongside the notebook evidence before choosing a fusion experiment.

#### Published comparisons and distinct inputs

The overview's [published comparisons](docs/stage1-overview.md#published-comparisons-and-distinct-inputs) place the frozen probes on BioLangFusion's and IsoFormer's published splits and explain the limits of comparing them with trained fusion models.

### Stage 2: defining alignment

Stage 1 gives us several ways to compare representations, but their scores do not all mean the same thing. Stage 2 makes that meaning explicit before we use “alignment” to guide model design. We distinguish **biological correspondence**, such as a coding sequence and its translated protein, from **representation agreement**, meaning relationships between the encoders' numerical outputs. **Predictive complementarity** is a further question: does using both representations improve prediction over the stronger individual one?

The four main summaries in the Stage 1 pair tables examine different aspects of agreement:

| Summary | What it asks |
| --- | --- |
| Linear centered kernel alignment (CKA) | Do the complete representations organize the same examples similarly? |
| Neighborhood overlap | Are the same examples nearby in both representation spaces? |
| First canonical correlation | After fitting corresponding linear directions with CCA, how strongly do the first paired directions vary together on held-out rows? |
| Retrieval Recall@1 | After that fitted transformation, how often is an example's paired representation its nearest match among the test candidates? |

These are separate measurements, not components of one combined alignment score. A pair can agree on broad structure without reliably retrieving individual partners. Neither outcome establishes that combining the pair will improve the biological prediction. That is why the concatenation probes and later trained-fusion comparisons remain separate evidence.

Our **working definition** counts agreement between representations of corresponding inputs on held-out examples after the specified composition control. Agreement reproduced by the selected letter, codon, amino-acid, and related sequence features is reported separately. The control subtracts their fitted linear contribution; it does not establish that the remainder is purely biological information or independent of every composition effect.

This definition describes the intended evidence standard, not a claim that every Stage 1 calculation already meets it. CKA and neighborhood overlap describe the supplied analysis rows. PCA/CCA are fitted on training rows and evaluated on held-out rows. In the composition-controlled comparison, however, the earlier composition fit uses all analysis rows. The [Stage 1 guide](Code/stage1/README.md#why-the-analyses-have-three-tracks) explains that boundary and the unresolved differences among saved composition-controlled values. In the current code, CKA and retrieval are the comparisons repeated after that control.

The current design choices follow from this evidence, with room to revise them as later comparisons develop:

- **Begin without an explicit alignment loss.** An alignment loss would add a training objective encouraging selected representations to agree, alongside the prediction objective. The initial designs let the fusion model learn useful interactions through the prediction task. The exploratory alignment/error diagnostic did not establish an association under its tested conditions, but it did not test whether adding such a loss would help. CaLM with a protein encoder remains a candidate for the conditional alignment-loss ablation.
- **Use positional correspondence where the inputs supply it.** On coding-sequence data, nucleotide positions can be grouped into codons corresponding to amino-acid positions. That supports the common step grid considered for Coupled Mamba. GTEx's genomic window, transcript, and protein do not share one direct position-by-position grid across all three inputs, so their comparison uses cross-attention or pooled representations.
- **Avoid treating every encoder as a distinct source of useful information.** One ESM-2 model represents that family in the initial design; ProtBERT remains a separate candidate. This is a design choice informed by the observed redundancy, not a claim that the models are interchangeable on every task.
- **Carry candidate pairings forward for testing.** Codon-level RNA encoders (mRNA-FM or CaLM) with a protein encoder (ESM-2 150M or ProtBERT) are initial stability candidates. GTEx examines RNA and protein, then the addition of genomic DNA, with the length baseline alongside. These choices nominate comparisons; Stage 4 tests whether they improve prediction under the chosen protocol.

Stage 2 interprets the Stage 1 evidence without introducing another dataset or a separate executable notebook. Its output is the working definition and the reasoning behind these choices, recorded here and in the manuscript. Later evidence can refine them while preserving the definition used for each earlier result.

**Definition and evidence:** [Representation-comparison methods](docs/methods/representation-comparison.md) · [Project notes](docs/notes/Mina_Project_Notes.docx) · [October 7 alignment meeting slides](docs/notes/Slides/Mina_Meeting_Slides_2026-10-07.pptx). The manuscript records the decision in its “Stage 2: Defining Alignment” section. Gantt `Details` task 7 specifies the objects, metrics, controls, and permitted conclusions; task 23 keeps the alignment-loss comparison conditional.

### Stage 3: building and checking models

Stage 3 turns the planned comparisons into working implementations: BioLangFusion's fusion methods, LucaOne's pretrained gene/protein representations, and an IsoFormer-style aggregator. We need to check how our code prepares the inputs and carries out each method before interpreting its scores. For example, two arrays can have matching dimensions while their positions refer to different codons; a model can still train and produce a score without exposing that mismatch. Checking sequence correspondence, row identity, and controlled examples helps us establish that we are evaluating the intended operation. Stage 4 then tests its predictive value. An initial fit can help with both, so the distinction is the question we are answering, not simply whether training takes place.

The first notebook, `01_blf_datasets.ipynb`, **audits the existing benchmark CSVs**. It reuses Stage 1's stability dataset and introduces fungal expression, E. coli protein expression, and CoV-Vac degradation to establish the BioLangFusion comparison across four available benchmarks. The paper's fifth benchmark, Ab1, is unavailable in the release used here. The notebook reads and filters these files, compares row counts with the paper, and checks sequence properties and split overlap. The following notebooks extract token embeddings and assemble the models; the IsoFormer-style comparison later reuses the GTEx pilot.

Token-level fusion retains one representation per token instead of only the pooled sequence vector used by the main Stage 1 probes. **Token resolution** means how much sequence a token represents: a nucleotide-model token can cover several bases, while a protein-model token represents an amino acid. Tokenization already runs during embedding extraction. BioLangFusion-style grid operations then map the resulting representations to corresponding codon positions; those operations do not retrain the tokenizer.

**Implementation route:** [Stage 3 folder guide](Code/stage3/README.md) · [Existing benchmark audit](Code/stage3/01_blf_datasets.ipynb) · [Token extraction](Code/stage3/02_token_embeddings.ipynb) · [BioLangFusion build and comparison](Code/stage3/03_blf_build_check.ipynb) · [LucaOne and IsoFormer checks](Code/stage3/04_lucaone_isoformer_check.ipynb).

These four existing notebooks establish the baseline implementations. The folder guide separately lists the planned `05_coupled_mamba_check.ipynb` and `06_cross_attention_check.ipynb` for the project's proposed designs. The latter extends the cross-attention investigation with encoder choices informed by Stage 1; it is distinct from the IsoFormer-style baseline already examined in notebook 04. The filenames do not by themselves settle execution priority; the research plan and its dependencies do.

#### Architecture priorities

| Priority | Architecture |
| --- | --- |
| 1 | Coupled Mamba |
| 2 | Isoformer-style cross-attention |
| Fallbacks | Cross-Mamba / BiMamba; mixture-of-experts fusion |

Coupled Mamba's priority was agreed at the September 23, 2026 online meeting. The [meeting slides](docs/notes/Slides/Mina_Meeting_Slides_2026-09-23.pptx), slide 6, list the primary and fallback architectures. Concatenation + MLP is the reference baseline in the research design.

The advisor's [research plan](<docs/Multimodal_Research Plan .docx>) comes first. Its baselines (BioLangFusion reproduced exactly, then its encoders swapped; LucaOne; IsoFormer reproduced on GTEx beside them) and its datasets (full-transcript half-life, translation efficiency, the mRFP control) are in place before Coupled Mamba and cross-attention are trained, from January 2027. The [Gantt chart](docs/Research_Project_Plan_Gantt.xlsx) gives the dates.

### Stage 4: controlled comparisons

With those implementations in place, Stage 4 asks what accounts for an improvement in prediction. We first compare the baseline methods, encoder choices, and input contexts; later experiments will compare our proposed fusion designs against these baselines. The BioLangFusion experiments evaluate single encoders and fusion methods under matched conditions. Encoder swaps then change one encoder while retaining the surrounding method. Stage 1 helps us choose substitutions worth testing, and Stage 4 asks whether they help once the prediction head is trained. LucaOne and IsoFormer provide additional comparisons where their inputs and evaluation conditions are compatible. Repeating experiments with different seeds shows how much results vary with stochastic fitting or partitioning under the specified protocol; it does not resolve problems with the split or establish biological validity.

The **mRFP expression control** isolates synonymous sequence variation: different codons can specify the same amino acid, allowing nucleotide sequences to differ while the protein remains unchanged. Stage 1 checks that this input construction behaves as expected. Stage 4 asks whether nucleotide embeddings predict the expression differences beyond simple composition and structure features. This is a diagnostic of the representations, not a comparison of fusion architectures.

The later **full-transcript half-life** task concerns RNA stability too, but uses different data that include untranslated regions as well as coding sequence. It asks whether information outside the coding region adds predictive value. Translation efficiency is a separate target in the same notebook. The dataset and split checks precede interpretation; these are not interchangeable with the earlier coding-sequence stability scores.

**Experiments:** [Stage 4 folder guide](Code/stage4/README.md) · [mRFP expression control](Code/stage4/tier1_mrfp_control.ipynb) · [BioLangFusion comparisons](Code/stage4/tier2_blf_phaseA.ipynb) · [Encoder swaps](Code/stage4/tier2_encoder_swaps.ipynb) · [LucaOne and IsoFormer comparisons](Code/stage4/tier2_lucaone_isoformer.ipynb) · [Full-transcript half-life and translation efficiency](Code/stage4/tier3_halflife_te.ipynb).

**Execution coordination:** [Stage 4 runbook](Code/stage4/RUNBOOK.md) · [Run records](docs/run-records.md). Gantt `Details` tasks 17, 24, 41, and 44 describe the model comparisons; tasks 18 and 42 distinguish the control and full-transcript tasks.

**Status, October 22.** Stage 3's first round is built and run, and every item of the advisor's priority list except the remaining encoder swaps now has results. Notebooks 01 to 04 in [Code/stage3/](Code/stage3/README.md) build and check the fusion models; [Code/stage4/](Code/stage4/README.md) runs the experiments, with [RUNBOOK.md](Code/stage4/RUNBOOK.md) saying who runs which. Run so far: the mRFP control, BioLangFusion Phase A on four benchmarks, four of the eight Phase B encoder swaps, LucaOne frozen on mRNA stability, and the human half-life and translation-efficiency probes behind their gate. The single encoders and cross-attention reproduce the paper and the gated head does not. The largest effect in Stage 4 is not a fusion head: swapping one frozen encoder raised E. coli by eight points and dissolved the fusion gain measured there. Fusion still wins where the inputs are physically different molecules, on GTEx and on full transcripts, and one jointly pretrained model beats three fused ones on stability. Each result and its caveats are in the [run records](docs/run-records.md).

## Repo structure

The stage links above follow the research through [Code/](Code/). This root README connects the stages and explains why we move between them. Within a stage, the folder README develops that part of the story: why its experiments belong together, how their findings inform the next decisions, and which notebook contains the evidence. Each notebook then connects its own inputs and choices to the code, outputs, and interpretation.

The supporting folders serve those analyses. [datasets/](datasets/) supplies the included inputs. [Code/shared_code/](Code/shared_code/) contains reusable implementations, including the [fusion models](Code/shared_code/fusion/) imported by later stages. [tests/](tests/README.md) checks software behavior on controlled examples; those checks help verify an implementation but do not establish a biological conclusion.

| Form of explanation | Purpose |
| --- | --- |
| Root README | Tell the overall research story: the problem, why each stage is needed, how the stages inform one another, and where to continue. |
| Folder README | Develop the purpose and reasoning for that part of the project, connect its files, and explain the relevant reading or execution order. A stage guide also connects its findings to later decisions. |
| Dataset description | Identify the biological source, meaning of a row and its columns, preparation, and interpretation limits. |
| Notebook Markdown | Connect the biological question, inputs, method choices, outputs, and interpretation. Retain enough context to follow the analysis locally. |
| Notebook comments | Explain a local implementation choice or caution beside the statement it affects. |
| Shared-code docstrings and comments | Describe reusable operations, inputs, outputs, assumptions, and non-obvious implementation choices for all callers. |
| Method guides | Develop reusable derivations and examples; a notebook retains the explanation needed to understand its own operation and claim. |

A short explanation can appear at more than one level when it helps the reader follow the story. For example, this README gives the scope of the encoder comparison, while the Stage 1 guide explains how twelve encoders give 66 pairs. A link should add detail or identify evidence without requiring the reader to open another file to understand every paragraph. These roles guide the ongoing revision; the existing files do not all follow them yet. The [implementation-reading instructions](#reading-implementation-code) explain how notebook source links expose the maintained functions.

Shared research explanations, plans, and readings are indexed in [docs/](docs/README.md). The [paper catalog](Papers/README.md) identifies retained source versions. [Selected sources](docs/sources.md) records why references are useful, what was read, and how they relate to the project; it supports optional evidence lookup. [Meeting notes and slides](docs/notes/) and [individual paper reviews](docs/notes/Reviews/) retain their authorship; contributor folders such as [Julian's reviews](docs/notes/Reviews/Julian/) and `docs/notes/Reviews/Chase/` keep individual interpretations attributable. Obsidian supports personal study; explanations needed to understand the project belong in the shared documentation.

**Working conventions:** [AGENTS.md](AGENTS.md#shared-documentation-and-sources) is the entry point to the maintained documentation, Markdown, and engineering workflows. It routes agents to those rules rather than duplicating their full contents here. [Agent context](docs/agent-context.md) retains project-specific evidence cautions; [run records](docs/run-records.md) connect results to their producing inputs, code, settings, and environments.

## Data

The six supplied CSV tables in [datasets/](datasets/) can be read without rebuilding them. Stage 1 uses stability and GTEx for its main analyses, plus mRFP as a control. Stage 3 adds fungal expression, E. coli, and CoV-Vac for the BioLangFusion comparisons. Stage 4 reuses these inputs and introduces separately downloaded full-transcript tables. Counts below describe the supplied files before notebook filtering or subsampling; using the same source table does not mean that two experiments retain the same rows or use the same evaluation procedure.

### mRNA stability

[mRNA_Stability.csv](datasets/mRNA_Stability.csv): 65,356 rows containing 29,949 distinct sequence strings. Each row is a record whose `Sequence` column contains a coding sequence and whose `Value` column contains the numerical stability score used as a prediction target. `Dataset` identifies the benchmark and `Split` records its supplied partition. We have not established how the original measurements were converted into the stored scores or what units, if any, those numbers have; they should not be described as hours of half-life.

The [dataset audit in the stability notebook](Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) records the following properties of `mRNA_Stability.csv`, which matter for comparisons with published results:

- The `Split` column assigns 45,749 rows to train, 9,803 to validation, and 9,804 to test. Of the 8,283 distinct test sequences, 5,392 also occur in training.
- 12,844 sequences occur more than once, and 12,775 of those carry differing values in the `Value` column; the median standard deviation of scores within one sequence is 0.49.
- 37% of rows are at most 1,000 nucleotides, so single-nucleotide encoders with a limit near 1,000 tokens see truncated input for most sequences.
- [BioLangFusion](Papers/core/BioLangFusion.pdf), Appendix A.2, reports 41,123 raw and 23,929 used mRNA stability sequences with the CodonBERT splits. [The Stage 3 dataset step](Code/stage3/01_blf_datasets.ipynb) resolves the filtering: keeping sequences of at most 999 nucleotides reproduces 23,929 here, and the same rule reproduces the used counts of the three other benchmarks. The raw counts still disagree (41,123 reported against 65,356 here), so this file is a different release of the benchmark; that part is unresolved.

Comparisons with published numbers should therefore use the official split, while leakage-free estimates need a deduplicated, sequence-disjoint split. The CSVs were sourced from the fine-tuning benchmark data in [Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT/tree/master/benchmarks/CodonBERT/data/fine-tune). Their inclusion supplies the notebook inputs; interpreting a prediction still requires checking the source assay, labels, and retained sequence regions.

### mRFP expression control

[mRFP_Expression.csv](datasets/mRFP_Expression.csv): 1,459 rows containing 1,455 distinct sequence strings, from synonymous codon randomization of one gene. mRFP is the red fluorescent protein encoded by these variants. Synonymous codon changes preserve its amino-acid sequence while allowing the nucleotide sequence to vary. Stage 1 checks the resulting representation behavior; Stage 4 uses the recorded expression scores to test what the nucleotide representations predict beyond simpler features.

**Control explanation:** [Synonymous recoding](docs/methods/biological-targets-and-controls.md#pipeline-control-synonymous-recoding).

### GTEx transcript expression

[GTEx_pilot.csv](datasets/GTEx_pilot.csv): 2,000 protein-coding human transcripts from [IsoFormer's GTEx transcript-expression table](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression), 1,000 from its published train split and 1,000 from its test split, each with its expression in 30 tissues. Each row holds the transcript, its untranslated regions and coding sequence, its protein, and 6,000 nucleotides of GRCh38 DNA centered on its transcription start site. [build_gtex_pilot.py](Code/stage1/build_gtex_pilot.py) produced it: it downloads IsoFormer's table, samples the transcripts, and reads the DNA windows from the [Ensembl REST service](https://rest.ensembl.org/documentation/info/sequence_region_post).

**Rebuilding the GTEx pilot.** The `build_gtex_pilot.py` script prepares this input table rather than running the representation analyses. Run `python Code/stage1/build_gtex_pilot.py` from the repository root when a rebuild is needed. It downloads IsoFormer's larger table into the repository's `.data-cache/` if it is not already cached, selects transcripts from the published splits, obtains their genomic DNA windows from Ensembl, and writes `datasets/GTEx_pilot.csv`. Rebuilding is a separate preparation step that requires network access and replaces that CSV; it is not part of running the notebook on the supplied pilot. The [results and cache inventory](#results-caches-and-intermediate-files) distinguishes the full source-table download from the supplied pilot and the embeddings saved during analysis.

The builder checks required source columns and the ordered 30-tissue block before use. Downloads and pilot CSVs are published only after a complete temporary-file write and validation; failure before replacement preserves the previous file. Invalid cached CSVs stop for inspection rather than silently downloading again. Preserve questionable files and their evidence before deliberately selecting a fresh source cache. A valid schema cannot establish whether an older download was complete. Sequence requests follow the [Ensembl batch and rate-limit contract](docs/sources.md#ensembl-sequence-requests-and-rate-limits): transient failures have at most five attempts, each network open uses a 120-second timeout, and a requested retry delay above 120 seconds stops with an explanation. Malformed or incomplete response batches fail without retry. These are software recovery checks; they do not validate the pilot's biological design.

### Other BioLangFusion benchmarks

These three datasets enter through the [Stage 3 audit](Code/stage3/01_blf_datasets.ipynb). Together with stability, they form the four available BioLangFusion benchmarks. GTEx and mRFP serve other comparisons and are not members of that five-benchmark suite; Ab1 is the missing member.

- [Fungal_expression.csv](datasets/Fungal_expression.csv): 7,089 sequence records paired with expression scores. The cited source uses RNA sequencing; the local score transformation and retained species subset remain unresolved. It supplies the initial trained-model comparison in Stage 3.
- [E.Coli_proteins.csv](datasets/E.Coli_proteins.csv): 6,348 sequence records with three stored protein-expression categories. E. coli is the expression host, not necessarily the source organism of each gene. The task is described as low, medium, or high expression, but the mapping from the local numeric classes to those meanings remains unverified. It supplies Stage 4's classification comparison.
- [CoV_Vaccine_Degradation.csv](datasets/CoV_Vaccine_Degradation.csv): 2,400 fixed-length, 81-nucleotide RNA vaccine-construct fragments paired with degradation scores. It supplies another Stage 4 regression comparison; its fragment inputs require an explicit protein-input convention.

**Assay context:** [Fungal and E. coli targets](docs/methods/biological-targets-and-controls.md#fungal-and-e-coli-benchmark-targets) distinguishes the source assays from what the local files establish.

### Full-transcript half-life and translation efficiency

The Stage 4 [half-life and translation-efficiency notebook](Code/stage4/tier3_halflife_te.ipynb) loads additional tables distributed through mRNABench into a local download cache. These are new source datasets, not expanded versions of `mRNA_Stability.csv`, and are separate from the six included CSVs. Their transcript-region information supports comparisons of coding sequence, untranslated regions, and full transcripts. The [Stage 4 guide](Code/stage4/README.md#full-transcript-half-life-and-translation-efficiency) distinguishes six-table feasibility checks, composition baselines on eligible tables, and the two primary human encoder comparisons; the [dataset catalog](Code/shared_code/mrnabench.py) records each table's source, species, and target.


## Setup

### Notebook dependencies

Use your existing Python environment for the project. Install PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) for your operating system and available compute backend. Install the remaining notebook dependencies in that environment:

```sh
python -m pip install numpy pandas scipy scikit-learn umap-learn matplotlib biopython "transformers==5.15.1" multimolecule einops ipykernel
```

Open notebooks with VS Code's Python and Jupyter extensions and [select the environment containing these packages as the kernel](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management). For the Stage 1 notebooks, use the notebook's folder, `Code/stage1/`, as the kernel's working directory (the default in VS Code and Jupyter): the Imports cell finds `Code/shared_code/` from there, and the embedding-cache paths are relative to it. Dataset loaders resolve the supplied CSVs from the repository's `datasets/` folder independently of that working directory.

The Stage 1 notebooks select an available backend through `encoders.select_device`: CUDA, then MPS, then CPU. This is a runtime choice for each contributor's environment. Record the device and software versions associated with shared results in [run records](docs/run-records.md), including the origin of reused embeddings.

`multimolecule` provides RNA-FM, RiNALMo, mRNA-FM, CaLM, and HyenaDNA. The installation command currently pins `transformers` to 5.15.1; the [earlier compatibility report](docs/run-records.md#legacy-observations-without-complete-run-identities) records why. If an import fails, inspect its traceback and the installed package versions before choosing a dependency change.

Nucleotide Transformer v2 and DNABERT-2 load custom model code written for `transformers` 4; the encoder registry, [encoders.py](Code/shared_code/encoders.py), applies the small compatibility adjustments they need under `transformers` 5 and pins both to a fixed checkpoint revision.

`PYTORCH_ENABLE_MPS_FALLBACK` must be configured before PyTorch is imported for the MPS fallback to take effect. Each experiment notebook's Imports cell uses `os.environ.setdefault` before `import torch`, setting the value to `1` only when it is absent; an existing value is preserved.

When an encoder is needed, either experiment's loading or embedding cells download its checkpoint unless it is already cached. The default twelve encoders total about 8 GB; [encoders.py](Code/shared_code/encoders.py) lists their Hugging Face identifiers. DNABERT-2's model code requires `einops`. Hugging Face normally stores these downloads in the [user's cache](https://huggingface.co/docs/transformers/installation#cache-directory); the notebooks do not configure the repository's `.model-cache/` directory.

The experiment notebooks save embedding matrices separately under `Code/stage1/embeddings/`. See the [results and cache inventory](#results-caches-and-intermediate-files) for their paths and the policy for preserving results before recomputation, and [embedding matrices and cache reuse](docs/methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse) for what the cache checks before reusing a matrix.

This dependency list covers the imports in both Stage 1 experiment notebooks and the `shared_code` package. A fresh-environment run has not been verified. The repository does not yet pin package versions other than `transformers`. Nucleotide Transformer v2, DNABERT-2, HyenaDNA, RiNALMo, CaLM, and ProtBERT have pinned model revisions; the other six encoders load the current revision of their checkpoint.

### Additional dependency for the Stage 4 mRFP control

The [mRFP control notebook](Code/stage4/tier1_mrfp_control.ipynb) also needs ViennaRNA for its RNA-folding calculations. Install it in the same Python environment selected as the notebook kernel:

```sh
python -m pip install ViennaRNA
```

The package is imported as `RNA`. The notebook imports it without running an installation command. It also needs the pooled embeddings produced by the control section of [Stage 1's stability notebook](Code/stage1/Stage1_stability.ipynb); its loading cell reads those saved inputs without running pretrained encoders.

### Reading implementation code

The [stability](Code/stage1/Stage1_stability.ipynb#Implementation-displays) and [GTEx](Code/stage1/Stage1_gtex.ipynb#Implementation-displays) notebooks use `SHOW_IMPLEMENTATION = False` to keep implementation displays compact: each `show_source` cell lists links to the functions or classes in `Code/shared_code/`, including their source lines. Both notebooks still show their analysis calls, settings, and results.

To expand the implementation while studying a method, set `SHOW_IMPLEMENTATION = True`, run that setting, and rerun the relevant `show_source` cell. These display cells inspect source code; they do not load encoders or rerun analyses. Set the flag back to `False` and rerun a display cell to return to its links. Source links and expanded listings describe the code currently on disk; they do not independently verify which implementation produced a saved research result.

If the source-display helper changes while a kernel is open, reload `shared_code.notebook` and reimport `show_source`, or restart the kernel and rerun Setup, before using the updated helper.

The [presentation helper](Code/shared_code/notebook.py) keeps full listings as the default for calls without a `full` argument, preserving the behavior of other notebooks. Relative source links assume the notebook is in a stage folder such as `Code/stage1/`. A viewer may open the file without jumping to its line anchor; the displayed function name and line number locate the definition.

### Interpreting loading messages

A Hugging Face warning about unavailable Windows symlinks means downloaded files can still be cached, but storing multiple revisions can require more disk space. The warning alone does not show that a download failed. See the [Hub cache limitations](https://huggingface.co/docs/huggingface_hub/v2.1.1/guides/manage-cache#limitations), checked in the version 2.1.1 documentation on October 5, 2026; this documentation version does not identify the library installed in a notebook environment.

Loading through the Transformers auto classes can report checkpoint weights that do not match the instantiated model. For Nucleotide Transformer 500M and the three ESM-2 models, the reports typically include:

| Parameter names in the report | Status | Meaning for this notebook |
| --- | --- | --- |
| `lm_head.*` | `UNEXPECTED` | The checkpoint contains language-modeling prediction-head weights that the loaded base encoder does not use. |
| `pooler.dense.weight` and `pooler.dense.bias` | `MISSING` | These pooler parameters are absent from the checkpoint and were newly initialized. |

Unused checkpoint weights and newly initialized parameters are different cases; see the Transformers [loading-warning definitions](https://huggingface.co/docs/transformers/main_classes/model). `embeddings.embed` averages `last_hidden_state` itself. It uses neither the separate `pooler_output` nor the language-modeling head, so these particular entries do not identify missing weights in the hidden-state path used here. See the [ESM output definitions](https://huggingface.co/docs/transformers/main/en/model_doc/esm).

RNA-FM reports the same pooler entries and unused `lm_head.*` and `ss_head.*` weights; the latter belong to a secondary-structure prediction head. mRNA-FM, RiNALMo, CaLM, and HyenaDNA load through the same `multimolecule` package, and the same component-level interpretation applies where their reports list these entries. ProtBERT's unexpected `cls.*` weights belong to its masked-token and next-sentence prediction heads. These listed heads and poolers are not used by `embed`; a different missing or unexpected parameter requires checking which component it belongs to.

Nucleotide Transformer v2 and DNABERT-2 use the compatibility loaders in [encoders.py](Code/shared_code/encoders.py), which build their models and compare checkpoint weights directly. They do not print the auto-class report above: the loader raises an error if anything other than the unused pooler is missing or unexpected. A completed load confirms the weight comparison only; embedding extraction and downstream analyses require their own checks.

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

## Results, caches, and intermediate files

Keep routine model caches and regenerable intermediate outputs local and ignore their specific paths. Retain source datasets and notebooks in Git. Share selected results deliberately, with a [run record](docs/run-records.md) linking their producing code, inputs, model revisions, settings, environment, and retained outputs. Avoid blanket ignore rules for scientific file formats. [Sandve et al.'s reproducibility rules](docs/sources.md#sandve-et-al-ten-simple-rules-for-reproducible-computational-research) explain why result-producing steps, versions, intermediate data, and seeds matter; the inventory below is only part of that record.

When code adds or changes an output, update this inventory and its handling. Add or adjust specific paths in [.gitignore](.gitignore) for outputs kept local. Preserve an identified copy of any result needed as evidence before rerunning code that overwrites it. Paths below assume the working directory specified in Setup.

| File or directory | Producer | Purpose | Handling |
| --- | --- | --- | --- |
| `Code/stage1/embeddings/<dataset>_n<rows>_seed<seed>/` | [Stability notebook](Code/stage1/Stage1_stability.ipynb) | One embedding matrix per encoder for one dataset sample, saved with its cache version, checkpoint, requested revision, token limit, row count, and effective-input fingerprint. Only valid matching caches are reused. | Local and ignored; preserve old evidence and use a fresh directory for deliberate recomputation. See [cache recovery](docs/methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse). |
| `Code/stage1/embeddings/<dataset>_published_train<rows>_seed<seed>/` and `..._published_test<rows>_seed<seed>/` | [Stability notebook](Code/stage1/Stage1_stability.ipynb#Published-split-comparison) | Embeddings of the rows sampled from the published train and test splits for the published-split comparison, saved and reused as above. | Local and ignored. |
| `Code/stage1/embeddings/gtex_pilot_train/` and `gtex_pilot_test/` | [GTEx notebook](Code/stage1/Stage1_gtex.ipynb) | Embeddings of the two halves of `GTEx_pilot.csv`, one matrix per encoder from that encoder's own input, saved and reused as above. | Local and ignored. |
| `Code/stage1/embeddings_legacy/` | Earlier runs of the Stage 1 notebooks, before October 6 | Older embedding caches and `stage1_main_embeddings.npz`, kept for reference; the current notebooks do not read them. | Local and ignored. |
| `.data-cache/GTEx_final.csv` | `Code/stage1/build_gtex_pilot.py` | IsoFormer's full GTEx table, about 645 MB, downloaded to sample the pilot. | Local and ignored; only needed to rebuild `GTEx_pilot.csv`. |
| `docs/figures/stability/` and `docs/figures/gtex/` | Tagged figure cells in each notebook; [model-free exporter](Code/export_figures.py) | SVG/PNG, numeric CSVs, completed-result JSON and manifests. Regenerate from the saved snapshot without loading encoders or rerunning analyses. | Deliberately shared in Git; see the [figure guide](docs/figures/README.md) for commands and result-state limits. Writing these files does not commit or push them. |
| `.data-cache/mrnabench/` | [`Code/stage4/tier3_halflife_te.ipynb`](Code/stage4/tier3_halflife_te.ipynb) | The six mRNABench parquet files the half-life and translation-efficiency gate reads, about 150 MB. | Ignored, like the other download caches. Delete it to force a fresh download. |
| `Code/stage3/results/<benchmark>/` and `Code/stage4/results/<notebook>/` | The Stage 3 and Stage 4 notebooks | One small JSON per training run with its settings, score and per-row test predictions; CSV tables for the probe notebooks. New shared-trainer runs also retain best-validation weights as `*.weights.pt`, linked by filename, checksum and size in the JSON. Existing results without weights are reused without retraining to fill that gap. | JSON and CSV records are tracked; `*.weights.pt`, recovery checkpoints, interrupted publications and everything under `Code/*/embeddings/` are ignored. |
| `Code/stage4/results/blf_phaseA/ecoli_proteins/` and `Code/stage3/results/fungal_expression/reproducible/` | Chase's Phase A runner and his Step 3 reruns, Windows, October 19 to 20 | His appendix-convolution E. coli cohort with its provenance records, and his seed-0 fungal fits; see the [run records](docs/run-records.md). | Tracked. Their runner and verifier scripts were written against a trainer refactor that is not on this branch. |

## Contributors

- Julian McKinley ([@jtmckinley4](https://github.com/jtmckinley4))
- Chase Di Maria Breisinger ([@Lqvy](https://github.com/Lqvy))
