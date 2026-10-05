# Research documentation

This directory brings together the team's research plans, bibliography, and explanations connecting biological questions, analyses, and evidence for DNA, RNA, and protein foundation models. It supports code development, experiment planning, and manuscript writing in the [multimodal-bio-fusion Overleaf project](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f).

## Read the Stage 1 analyses

Start with the biological question and model choices in the overview. Follow the stability experiment for the first method walkthrough, then compare it with GTEx. Each experiment links its methods, implementation, and interpretations at the relevant cells.

| Question | Starting point |
| --- | --- |
| What biological question motivates the analysis? | The [project background](../README.md#project-background) and [Stage 1 overview](stage1-overview.md), which holds the model context, research questions, and cross-dataset Summary. |
| Where do I follow the mRNA stability experiment? | The [stability notebook](../Code/Stage1_stability.ipynb): input preparation, embeddings, prediction, geometry, recoding control, attention, and the BioLangFusion comparison. |
| Where do I compare distinct DNA, transcript, and protein inputs? | The [GTEx notebook](../Code/Stage1_gtex.ipynb): expression across 30 tissues, length and composition baselines, representation comparisons, attention, and the IsoFormer comparison. |
| What do the sequences and labels represent? | The [dataset overview](../README.md#data), [stability data preparation](../Code/Stage1_stability.ipynb#Data), [stability dataset audit](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons), and [GTEx data and expression labels](../Code/Stage1_gtex.ipynb#GTEx-data). Check what a row and its label mean, how the label was obtained, and which examples enter the analysis. |
| What did Stage 1 find? | The overview's [Summary](stage1-overview.md#summary) and [result tables](stage1-overview.md#result-tables), with the producing outputs and local interpretations in each experiment notebook. |
| How do the frozen encoders compare with published fusion studies? | The [comparison overview](stage1-overview.md#published-comparisons-and-distinct-inputs), [BioLangFusion published-split comparison](../Code/Stage1_stability.ipynb#Published-split-comparison), and [IsoFormer published-split comparison](../Code/Stage1_gtex.ipynb#GTEx-published-split-comparison). |
| Where are methods and results written up for the manuscript? | The [Overleaf project](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f), including its Stage 1 metric definitions and published-split comparisons. |
| What does alignment mean here, and what comes next? | [Stage 2: defining alignment](../README.md#stage-2-defining-alignment), [Stage 3 fusion experiments](../README.md#next-steps-stage-3-fusion-experiments), and the Stage 2 section of the Overleaf project. |

The [project notes](notes/Mina_Project_Notes.docx) and the October 3 comparison entry in the [Gantt workbook](Research_Project_Plan_Gantt.xlsx) retain `Stage1_analysis.ipynb` as the historical source of saved results. Slide 2 of the [October 7 meeting deck](notes/Mina_Meeting_Slides_2026-10-07.pptx) also names that retired notebook. Use the overview and two experiment notebooks linked above for current navigation.

## Understand the methods

The guides develop the longer explanations and derivations. The notebooks retain the experiment-specific choices, calls, and interpretations; guide links return to the relevant analysis.

| Question | Guide |
| --- | --- |
| How do sequences become matched embedding vectors? | [From sequences to embeddings](methods/inputs-and-embeddings.md): filtering, translation, token limits, pooling, and cache reuse. |
| How does a linear probe work, step by step? | [Linear probing](methods/linear-probing.md): standardization, ridge regression, evaluation, and the internal-validation limitation. |
| What do the biological targets and recoding control measure? | [Biological targets and controls](methods/biological-targets-and-controls.md): length, GC, GC3, the Nussinov proxy, and synonymous recoding. |
| How do representation comparisons differ? | [Comparing representations](methods/representation-comparison.md): CKA, neighborhoods, CCA/retrieval, composition, PCA bands, and alignment diagnostics. |
| How are attention scores compared with candidate RNA patterns? | [Attention and candidate motifs](methods/attention-and-motifs.md): extraction, nucleotide mapping, comparisons, and permutation controls. |

## Inspect or run the implementation

Both experiment notebooks use the shared [mbf package](../Code/mbf/), with their own Setup sections. See [environment setup](../README.md#setup) before a new run; neither experiment requires executing the other. The overview is a Markdown document for reading the shared context and cross-dataset summary.

Use [Reading implementation code](../README.md#reading-implementation-code) to follow a compact source link or display a function in full with `SHOW_IMPLEMENTATION`. See the [notebook checks](../tests/README.md) for setup, source-display, and synthetic calculation checks after an edit. These checks do not reproduce research results.

When interpreting a result, identify its producing notebook version, settings, inputs, outputs, baselines, and evaluation split. Check whether those observations support the claim and retain unresolved discrepancies between versions.

## Find plans and literature

| Question | Starting point |
| --- | --- |
| What are we proposing to investigate? | The [Research Canvas](Research_Canvas.pptx). |
| What is the project schedule? | The [Gantt presentation](Research_Project_Plan_Gantt.pptx) and [editable Gantt workbook](Research_Project_Plan_Gantt.xlsx). |
| Where is the annotated bibliography? | The [team bibliography](Annotated_Bibliography.docx). |
| Where is the team's literature review? | [Literature review (PDF)](<Literature Review - Biological Foundation Models.pdf>) and [editable Word document](<Literature Review - Biological Foundation Models.docx>). |
| Which readings are worth returning to? | [Selected sources and readings](sources.md), with reasons to retain them, reading scope, and local relevance. The [research software section](sources.md#organizing-and-checking-research-software) explains practices behind the notebook and shared-code organization. |
| Where are the papers and contributor interpretations? | The [paper catalog](../Papers/README.md) identifies core and supporting papers, retained versions, and source links. The [individual reviews](notes/Reviews/) include Julian's [BioLangFusion review](notes/Reviews/Julian/BioLangFusion.md) and [alignment paper review](notes/Reviews/Julian/Alignment_Theory_Paper_Review.md). Read each interpretation alongside its source paper. |
