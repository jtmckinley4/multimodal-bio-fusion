# Research context for agents

Use this page for interpretation pitfalls and unresolved evidence questions. [AGENTS.md](../AGENTS.md) owns working instructions; the [documentation index](README.md) locates shared explanations, literature, and plans.

## Notebook-study context

Chase is studying the Stage 1 pipeline and its relationship to the literature. This learning focus does not define the team's research scope. Start with the [Stage 1 overview](stage1-overview.md) for the model choices, research questions, and cross-dataset synthesis, then follow the relevant experiment in its opening table. The [project background](../README.md#project-background) and [Stage 2 definition](../README.md#stage-2-defining-alignment) describe the wider direction.

Saved notebook outputs predate the file separation. Associate each result with its producing data, code, model revisions, and settings; a link to today's implementation does not identify the code that produced an older output. [Run records](run-records.md) hold execution-specific environments and evidence gaps; shared setup guidance should remain applicable to contributors using different computers.

## Evidence and unresolved questions

- **Dataset version:** Does the supplied stability CSV match BioLangFusion's version and filtering? The [dataset overview](../README.md#data) records the differing counts and overlap between train and test; inspect the [notebook audit](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) before a comparison.
- **Modality definition:** When DNA, RNA, and protein inputs derive from one coding sequence, what does an encoder difference measure? The overview's [input settings](stage1-overview.md#models-and-input-settings) distinguish that construction from GTEx's distinct inputs.
- **Dataset meaning:** What does `Value` in [mRNA_Stability.csv](../datasets/mRNA_Stability.csv) represent, how was it measured, and which sequence regions are included? The CSV header alone does not establish assay or label meaning.
- **Evaluation:** Which rows are retained, how are training and test examples selected, and what generalization claim does the split support? Check the producing code and settings rather than inferring the evaluation from a dataset column.
- **Result provenance:** If prose, outputs, reviews, or slides disagree, identify their producing versions and retain the disagreement until evidence resolves it.
- **Meaning of alignment:** Distinguish biological sequence correspondence, representation similarity, and predictive complementarity. Use the [Stage 2 definition](../README.md#stage-2-defining-alignment) and the overview's [composition-control findings](stage1-overview.md#summary); compare contributor interpretations with their [source papers](../Papers/README.md).
- **Further experiments:** What comparison would distinguish the proposed explanation from alternatives? Use the [architecture priorities](../README.md#architecture-priorities) and [Stage 3 plan](../README.md#next-steps-stage-3-fusion-experiments). Keep literature ideas, meeting preferences, and implemented methods distinct.

## Implementation and verification routes

Inspect the relevant helper and its callers in [Code/mbf/](../Code/mbf/) before changing shared behavior: both experiments may use it. The [implementation-reading instructions](../README.md#reading-implementation-code) explain source displays, and [tests/README.md](../tests/README.md) owns verification procedures. Scientific conclusions require checking the producing evidence beyond those software checks.

## Shared context across sessions

Maintain shared decisions, rationale, sources, and unresolved qualifications in the repository so collaborators can retrieve them without private chats or agent memory. Keep task assignments, pending reviews, and next actions in the relevant conversation or its task record. Update this page when the interpretation pitfalls or unresolved questions change.
