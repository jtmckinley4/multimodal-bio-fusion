# Research documentation

Use this page to find the explanation, evidence or instructions you need. The [project README](../README.md) connects the investigation; the [Overleaf manuscript](https://www.overleaf.com/project/6ab8535fb63c8540bed7e56f) develops its written argument.

## How these documents work together

| Reader question | Starting point |
| --- | --- |
| How do the stages connect? | [Research progression](../README.md#research-progression), including Stage 2's alignment definition. |
| Where are the models built and checked? | [Stage 3](../Code/stage3/README.md): dataset checks, token caches and reference implementations. |
| Where are the prediction comparisons? | [Stage 4](../Code/stage4/README.md): controls, encoder swaps, fusion and input-region comparisons. |
| What fusion hypotheses are proposed for testing? | [Stage 5](../Code/stage5/README.md): controlled correction, gate, attention and region-correspondence experiments. Existing-comparison follow-ups remain in Stage 4. |
| What produced a saved result? | [Run records](run-records.md), the producing notebook and [artifact policy](../README.md#results-caches-and-intermediate-files). |
| Where are agent instructions? | [AGENTS.md](../AGENTS.md) routes to maintained workflows; [agent context](agent-context.md) points to research cautions and unresolved questions. |

## Read the Stage 1 analyses

Start with the [Stage 1 guide](../Code/stage1/README.md), then read [stability](../Code/stage1/Stage1_stability.ipynb) before [GTEx](../Code/stage1/Stage1_gtex.ipynb). Stability introduces the shared methods; GTEx changes the input regions and tissue targets. This reading order is not an execution dependency.

| Find | Location |
| --- | --- |
| Encoder settings and consolidated findings | [Stage 1 overview](stage1-overview.md), [result tables](stage1-overview.md#result-tables) and [summary](stage1-overview.md#summary). |
| Row/target meaning and retained examples | [Data overview](../README.md#data), stability's [dataset audit](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons), and [GTEx data](../Code/stage1/Stage1_gtex.ipynb#GTEx-data). |
| Comparisons with published studies | [Scope and limitations](stage1-overview.md#published-comparisons-and-distinct-inputs), with the producing notebook sections linked there. |
| Figures and their numeric inputs | [Figure guide](figures/README.md): saved snapshots and figure regeneration without model reruns. |

Older planning entries name `Stage1_analysis.ipynb`, which the two linked experiment notebooks replaced.

## Understand the methods

| Question | Guide |
| --- | --- |
| What does each biological target measure? | [Data and biological targets](methods/data-and-targets.md): assays, transformations, source checks and interpretation limits. |
| How do sequences become predictions? | [From sequences to predictions](methods/sequences-to-predictions.md): preparation, tokens, pooling, cache reuse, probes and evaluation, including [calculated controls](methods/sequences-to-predictions.md#calculated-targets-and-controls). |
| How are representations and attention examined? | [Stage 1 representation analysis](../Code/stage1/representation-analysis.md): geometry, composition control, retrieval and exploratory diagnostics. |

Guides hold reusable explanations and examples; notebooks hold the actual experiment settings, outputs and local interpretation. Stage 1 pooled-vector procedures do not automatically describe Stage 4 token-level fusion.

## Inspect or run the implementation

Follow [environment setup](../README.md#setup) and the selected notebook's own settings before execution. [Reading implementation code](../README.md#reading-implementation-code) explains source links and `SHOW_IMPLEMENTATION`; [shared code](../Code/shared_code/) contains the reused calculations.

Use [software checks](../tests/README.md) after edits; synthetic checks do not reproduce research results. [Run records](run-records.md) identify producing versions, inputs, evaluation settings and retained evidence. The [Stage 4 runbook](../Code/stage4/RUNBOOK.md) owns execution and recovery routes.

For code changes, consult the [shared engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md). The [coding review and implementation plan](shared-coding-proposal.md) retains its rationale and verification history.

## Follow the research plan

Read the [Research Canvas](Research_Canvas.pptx) for project framing and [Mina's revised protocol](Biological_Fusion_Plan.docx) for proposed comparisons. [Stage 5](../Code/stage5/README.md#organize-the-proposed-notebooks-around-questions) connects those comparisons to existing work and planning task IDs.

In the [Gantt workbook](Research_Project_Plan_Gantt.xlsx), find a task's ID, recorded status and window on `Gantt`, then match that ID in `Details` for prerequisites and expected outputs or review conditions. Cite the task ID when connecting work to the plan. Recorded status is dated; check evidence before treating it as completion.

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
