# Research documentation

Use this page to find the explanation, evidence or instructions you need. The [project README](../README.md) connects the investigation; the [Overleaf manuscript](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f) develops its written argument.

## How these documents work together

| Reader question | Starting point |
| --- | --- |
| How do the stages connect? | [Research progression](../README.md#research-progression), including Stage 2's alignment definition. |
| Where are reference models established? | [Stage 3](../Code/stage3/README.md): dataset checks, token caches, method implementations, predictive validation and repeats. |
| Where are comparative and explanatory experiments? | [Stage 4](../Code/stage4/README.md): encoder swaps and input-region comparisons, alongside prepared, unrun tests of fusion behavior. |
| What fusion hypotheses will the prepared experiments test? | [Stage 4 controlled experiments](../Code/stage4/README.md#proposed-tests-of-fusion-explanations): correction, gates, attention and region correspondence; molecular-partner extensions remain conditional. |
| What produced a saved result? | [Run records](run-records.md), the producing notebook and [artifact policy](../README.md#results-caches-and-intermediate-files). |
| Where are agent instructions? | [AGENTS.md](../AGENTS.md) routes to maintained workflows; [agent context](agent-context.md) points to research cautions and unresolved questions. |

## Read the Stage 1 analyses

Start with the [Stage 1 guide](../Code/stage1/README.md), then read [stability](../Code/stage1/Stage1_stability.ipynb) before [GTEx](../Code/stage1/Stage1_gtex.ipynb). Stability introduces the shared methods; GTEx changes the input regions and tissue targets. This reading order is not an execution dependency.

| Find | Location |
| --- | --- |
| Encoder settings and findings | [Encoder panel](../Code/stage1/README.md#comparing-the-encoders) and [Stage 1 findings](../Code/stage1/README.md#what-carries-into-the-next-stages). The notebooks hold the saved tables. |
| Row/target meaning and retained examples | [Data overview](../README.md#data), stability's [dataset audit](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons), and [GTEx data](../Code/stage1/Stage1_gtex.ipynb#GTEx-data). |
| Comparisons with published studies | The [stability](../Code/stage1/README.md#how-the-stability-data-are-evaluated) and [GTEx](../Code/stage1/README.md#how-gtex-is-evaluated) evaluation sections explain the scope and link to the notebook comparisons. |
| Figures and their numeric inputs | [Figure guide](figures/README.md): saved snapshots and figure regeneration without model reruns. |

Older planning entries name `Stage1_analysis.ipynb`, which the two linked experiment notebooks replaced.

The [mRFP control](../Code/stage1/mrfp_representation_control.ipynb) extends the fixed-protein representation check with calculated controls and residual prediction. The [LucaOne comparison](../Code/stage1/lucaone_representation_comparison.ipynb) evaluates another frozen representation family with matched ridge probes. The separate [GTEx frozen fusion reference](../Code/stage3/gtex_frozen_fusion_reference.ipynb) belongs to Stage 3; it is not a LucaOne–IsoFormer model.

## Understand the methods

| Question | Guide |
| --- | --- |
| What does each biological target measure? | [Data and biological targets](methods/data-and-targets.md): assays, transformations, source checks and interpretation limits. |
| How do sequences become predictions? | [From sequences to predictions](methods/sequences-to-predictions.md): preparation, tokens, pooling, cache reuse, probes and evaluation, including [calculated controls](methods/sequences-to-predictions.md#calculated-targets-and-controls). |
| How are representations and attention examined? | [Stage 1 representation analysis](../Code/stage1/representation-analysis.md): geometry, composition control, retrieval and exploratory diagnostics. |

Guides hold reusable explanations and examples; notebooks hold the actual experiment settings, outputs and local interpretation. Stage 1 pooled-vector procedures do not automatically describe the token-level fusion used by Stage 3 references and Stage 4 experiments.

## Inspect or run the implementation

Follow [environment setup](../README.md#setup) and the selected notebook's own settings before execution. [Reading implementation code](../README.md#reading-implementation-code) explains source links and `SHOW_IMPLEMENTATION`; [shared code](../Code/shared_code/) contains the reused calculations.

Use [software checks](../tests/README.md) after edits; synthetic checks do not reproduce research results. [Run records](run-records.md) identify producing versions, inputs, evaluation settings and retained evidence. The [Stage 4 runbook](../Code/stage4/RUNBOOK.md) owns execution and recovery routes.

For cluster work, the [common ARCC guide](../execution/arcc/README.md) covers workspace separation, environment/GPU preflight and transfer/evidence handling. Choose the recipe for [encoder swaps](../execution/arcc/encoder_swaps/README.md), [grouped stability](../execution/arcc/grouped_stability/README.md) or [controlled fusion studies](../execution/arcc/fusion_studies/README.md). The [returned-result route](../Code/stage4/RUNBOOK.md#select-a-returned-result-set) covers reading the distinct Newton cohort with canonical dataset files; the [November 16 record](run-records.md#2026-11-16-arcc-recipe-layout-and-returned-input-deduplication) retains verified batch history and the returned-input mapping.

For code changes, consult the [shared engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md). The [documentation retirement record](run-records.md#2026-11-20-retire-the-stage-1-overview-and-coding-proposal) identifies the Git revision with the original coding review and its verification history.

## Follow the research plan

Read the [Research Canvas](Research_Canvas.pptx) for project framing and [the advisor's revised protocol](Biological_Fusion_Plan.docx) for proposed comparisons. [Stage 4](../Code/stage4/README.md#organize-the-preparation-notebooks-around-questions) connects those comparisons to existing work and planning task IDs. Earlier records may call these proposals Stage 5; their current home is the combined comparative and explanatory investigation.

In the [Gantt workbook](Research_Project_Plan_Gantt.xlsx), find a task's ID, recorded status and window on `Gantt`, then match that ID in `Details` for prerequisites and expected outputs or review conditions. Cite the task ID when connecting work to the plan. Recorded status is dated; check evidence before treating it as completion.

The dated workbook retains its historical stage/file names, task IDs, statuses and dates. Use the current stage READMEs to locate their present homes, including the former Stage 5 proposals now in Stage 4; the reorganization does not revise experiment priorities or schedules.

The [research execution and experiment batch plan](research-execution-plan.md) connects Newton result integration, reading-first notebooks and the advisor's experimental direction, with team decision points and completion checks. It distinguishes implemented B1–B4 safeguards, B5's completed 504-fit cohort and four reports, and the prepared but unrun controlled comparisons. CPU input inspections are separate from trained findings; workload and resource limits are agreed before execution.

The [Gantt presentation](Research_Project_Plan_Gantt.pptx) is an earlier overview, not updated for the revised protocol. The [planning instructions](../AGENTS.md#research-planning) explain how agents follow the workbook and surface proposed changes.

## Find literature and meeting material

| Material | Location and role |
| --- | --- |
| Bibliography and retained papers | [Team bibliography](Annotated_Bibliography.docx), [paper catalog](../Papers/README.md) and [selected sources](sources.md), which record reading scope and relevance. |
| Comparative literature review | [Draft v5 PDF](<Literature Review - Biological Foundation Models - Comparative Draft v5.pdf>) and [Word source](<Literature Review - Biological Foundation Models - Comparative Draft v5.docx>), October 14, 2026, by Chase Di Maria Breisinger and Julian McKinley. Unresolved draft notes remain. |
| Earlier annotated review | [October 12 PDF](<annotated-Literature Review - Biological Foundation Models-2.pdf>), preserved with its annotations. |
| Review-writing guidance | [Synthesis, critical comparison and open questions](sources.md#pautasso-and-purdue-owl-writing-a-literature-review). |
| Meeting material | [Slides](notes/Slides/) and the [progress-deck slide guide](notes/Slides/Mina_Progress_Slides_2026-10-22_Guide.md). Historical presentations are not live result inventories. |
| Contributor interpretations | [Individual reviews](notes/Reviews/), including Julian's [BioLangFusion](notes/Reviews/Julian/BioLangFusion.md) and [alignment-paper](notes/Reviews/Julian/Alignment_Theory_Paper_Review.md) reviews. Read attributed interpretations alongside their source papers. |
