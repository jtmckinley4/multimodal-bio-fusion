# Research documentation

This directory brings together the team's research plans, bibliography, and explanations connecting biological questions, analyses, and evidence for DNA, RNA, and protein foundation models. It supports code development, experiment planning, and manuscript writing in the [multimodal-bio-fusion Overleaf project](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f).

## How these documents work together

Choose the level of detail that answers the question you are reading for. A brief local reminder can accompany a notebook or summary; the linked guide or record holds the fuller explanation and evidence.

| Location | Purpose |
| --- | --- |
| [Project README](../README.md) and stage READMEs | The project README connects the overall investigation, setup and layout. Each stage README connects that stage's questions, experiments, findings and unresolved issues. |
| Experiment notebooks | Keep a specific research question, inputs, choices, executable steps, saved outputs and interpretation together. The stage READMEs locate them. |
| [Shared research guides](#understand-the-methods) | Explain dataset meaning and reusable biological, mathematical and computational procedures, with worked examples and their limits. |
| [Shared code](../Code/shared_code/) and [software checks](../tests/README.md) | Implement shared calculations and check software behavior. Dataset files, caches and saved results supply the actual inputs and numerical artifacts; see the [artifact policy](../README.md#results-caches-and-intermediate-files). |
| [Run records](run-records.md) | Identify what was inspected or executed, its code/input versions and settings, retained outputs and unresolved evidence. These records support the explanations without replacing them. |
| [Research plan](#follow-the-research-plan) and [Stage 4 runbook](../Code/stage4/RUNBOOK.md) | The Gantt records planned work and dependencies; the runbook provides execution and recovery guidance. Recorded status is dated evidence. |
| [Selected sources](sources.md), [paper catalog](../Papers/README.md) and [individual reviews](notes/Reviews/) | Record reading scope and relevance, locate retained papers, and preserve attributed interpretations. |
| [Repository instructions](../AGENTS.md), project skills and [agent context](agent-context.md) | Guide AI work and maintenance practices and point to unresolved project questions. |

## Read the Stage 1 analyses

Start with the biological question and model choices in the overview. Follow the stability experiment for the first method walkthrough, then compare it with GTEx. Each experiment links its methods, implementation, and interpretations at the relevant cells.

| Question | Starting point |
| --- | --- |
| What biological question motivates the analysis? | The [project background](../README.md#project-background) and [Stage 1 overview](stage1-overview.md), which holds the model context, research questions, and cross-dataset Summary. |
| Where do I follow the mRNA stability experiment? | The [stability notebook](../Code/stage1/Stage1_stability.ipynb): input preparation, embeddings, prediction, geometry, recoding control, attention, and the BioLangFusion comparison. |
| Where do I compare distinct DNA, transcript, and protein inputs? | The [GTEx notebook](../Code/stage1/Stage1_gtex.ipynb): expression across 30 tissues, length and composition baselines, representation comparisons, attention, and the IsoFormer comparison. |
| What do the sequences and labels represent? | The [dataset overview](../README.md#data), [stability data preparation](../Code/stage1/Stage1_stability.ipynb#Data), [stability dataset audit](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons), and [GTEx data and expression labels](../Code/stage1/Stage1_gtex.ipynb#GTEx-data). Check what a row and its label mean, how the label was obtained, and which examples enter the analysis. |
| What did Stage 1 find? | The overview's [Summary](stage1-overview.md#summary) and [result tables](stage1-overview.md#result-tables), with the producing outputs and local interpretations in each experiment notebook. |
| Where are the saved figures and their numeric inputs? | The [figure guide](figures/README.md), with stability/GTEx panels, result snapshots and commands that regenerate figures without rerunning models. |
| How do the frozen encoders compare with published fusion studies? | The [comparison overview](stage1-overview.md#published-comparisons-and-distinct-inputs), [BioLangFusion published-split comparison](../Code/stage1/Stage1_stability.ipynb#Published-split-comparison), and [IsoFormer published-split comparison](../Code/stage1/Stage1_gtex.ipynb#GTEx-published-split-comparison). |
| Where are methods and results written up for the manuscript? | The [Overleaf project](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f), including its Stage 1 metric definitions and published-split comparisons. |
| What does alignment mean here, and what comes next? | [Stage 2: defining alignment](../README.md#stage-2-defining-alignment), [Stage 3 fusion experiments](../README.md#stage-3-building-and-checking-models), and the Stage 2 section of the Overleaf project. |

The October 3 comparison entry in the [Gantt workbook](Research_Project_Plan_Gantt.xlsx) names `Stage1_analysis.ipynb`, the single notebook the two experiment notebooks replaced. Use the overview and the two experiment notebooks linked above for current navigation.

## Understand the methods

The three shared research guides follow a reading route: establish what the data measure, follow how sequences become predictions, then examine what the representations and attention patterns reveal. They retain equations, worked examples, implementation links and interpretation limits used across notebooks. Each guide states its scope; the Stage 1 pooled-vector procedures do not automatically describe Stage 4 token-grid or fusion training.

| Reader question | Guide |
| --- | --- |
| What does each dataset row and target mean? | [Data and biological targets](methods/data-and-targets.md): assays, label transformations, source-to-local-data verification, examples and limits on interpreting scores. |
| How do sequences become representations and predictions? | [From sequences to predictions](methods/sequences-to-predictions.md): input preparation, tokens, pooling, cache reuse, linear probes and evaluation; [calculated targets and controls](methods/sequences-to-predictions.md#calculated-targets-and-controls) cover length, GC, GC3, folding and synonymous recoding. |
| How do encoders differ, and what might those differences mean? | [Comparing and interpreting representations](methods/representation-analysis.md): representation geometry, composition controls, retrieval and prediction-error comparisons, followed by frozen-encoder attention, candidate motifs and permutation controls. |

Notebooks retain enough local explanation to follow their experiment, together with the actual settings, calls, saved outputs and interpretation. Guide links return to the relevant analyses for those choices and results.

## Inspect or run the implementation

Both experiment notebooks use the shared [shared_code package](../Code/shared_code/), with their own Setup sections. See [environment setup](../README.md#setup) before a new run; neither experiment requires executing the other. The overview is a Markdown document for reading the shared context and cross-dataset summary.

Use [Reading implementation code](../README.md#reading-implementation-code) to follow a compact source link or display a function in full with `SHOW_IMPLEMENTATION`. See the [notebook checks](../tests/README.md) for setup, source-display, and synthetic calculation checks after an edit. These checks do not reproduce research results.

Use [run records](run-records.md) to connect a shared result to its producing notebook version, settings, inputs, outputs, environment, and evaluation split. Compare it with the relevant baselines. Check whether those observations support the claim and retain unresolved discrepancies between versions.

The [shared engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md) guides executable-code work for both contributors' agents. The [coding review and implementation plan](shared-coding-proposal.md) records its rationale, bounded batches, and verification status.

## Follow the research plan

Open the [Gantt workbook](Research_Project_Plan_Gantt.xlsx) to follow the team's research sequence. In `Gantt`, locate the task ID, recorded status and scheduled window; then find the same ID in `Details` for its prerequisites and expected output or review condition. The [Gantt presentation](Research_Project_Plan_Gantt.pptx) provides an overview for meetings. Cite the task ID when explaining how a notebook, literature finding or manuscript addition relates to the plan.

The plan distinguishes initial input mapping and published-split probes from later trained fusion comparisons: see `Details` task 9, task 17 for BioLangFusion-style comparisons, and conditional task 25 for IsoFormer-style fusion. Read the plan's as-of date and check current evidence before relying on a recorded status. The [agent planning instructions](../AGENTS.md#research-planning) explain how agents follow the plan and surface proposed changes.

## Find plans and literature

| Question | Starting point |
| --- | --- |
| What are we proposing to investigate? | The [Research Canvas](Research_Canvas.pptx). |
| What is the project schedule? | Start with [Follow the research plan](#follow-the-research-plan), then use the [Gantt presentation](Research_Project_Plan_Gantt.pptx) or [editable Gantt workbook](Research_Project_Plan_Gantt.xlsx). |
| Where is the annotated bibliography? | The [team bibliography](Annotated_Bibliography.docx). |
| Where is the comparative literature review? | [Comparative draft v5 (PDF)](<Literature Review - Biological Foundation Models - Comparative Draft v5.pdf>) and its [editable Word source](<Literature Review - Biological Foundation Models - Comparative Draft v5.docx>), dated October 14, 2026, by Chase Di Maria Breisinger and Julian McKinley. It connects biological inputs, model choices, and evaluation; unresolved draft notes remain. |
| Where is the earlier annotated review? | [Annotated October 12 review (PDF)](<annotated-Literature Review - Biological Foundation Models-2.pdf>), retained with its existing annotations. |
| How do we develop a review beyond individual paper summaries? | The [review-writing guidance](sources.md#pautasso-and-purdue-owl-writing-a-literature-review) explains synthesis, critical comparison, and open questions. |
| Which readings are worth returning to? | [Selected sources and readings](sources.md), with reasons to retain them, reading scope, and local relevance. The [research software section](sources.md#organizing-and-checking-research-software) explains practices behind the notebook and shared-code organization. |
| Where are the meeting slides, and what does each slide mean? | [docs/notes/Slides/](notes/Slides/) holds the meeting decks. The [October 22 slide guide](notes/Slides/Mina_Progress_Slides_2026-10-22_Guide.md) explains every slide of the Stage 4 progress deck: why each experiment was run, how, what the numbers mean, and the advisor's questions with their answers; the deck's speaker notes carry the same text. |
| Where are the papers and contributor interpretations? | The [paper catalog](../Papers/README.md) identifies core and supporting papers, retained versions, and source links. The [individual reviews](notes/Reviews/) include Julian's [BioLangFusion review](notes/Reviews/Julian/BioLangFusion.md) and [alignment paper review](notes/Reviews/Julian/Alignment_Theory_Paper_Review.md). Read each interpretation alongside its source paper. |
