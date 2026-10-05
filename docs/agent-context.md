# Research context for agents

Use this page to locate the research material and questions relevant to a task. The [project README](../README.md) owns the project overview and repository map; the [documentation index](README.md) provides a reading route. Working instructions are in [AGENTS.md](../AGENTS.md). [Selected sources](sources.md) retains useful readings and explains what was consulted and why; it complements the papers and contributor reviews.

## Notebook-study context

Chase's current study focus is understanding the Stage 1 pipeline, its results, and its relationship to the literature. This individual learning focus does not define the scope or sequence of the whole project; the [project background](../README.md#project-background) describes the broader research aim. Select the file that owns the requested explanation or analysis:

| File | Role |
| --- | --- |
| [Stage1_analysis.ipynb](../Code/Stage1_analysis.ipynb) | Shared models, research questions, role of Stage 1, and cross-dataset Summary. It contains Markdown only. |
| [Stage1_stability.ipynb](../Code/Stage1_stability.ipynb) | mRNA stability experiment, dataset audit, biological-target probes, representation comparisons, synonymous-recoding control, attention diagnostics, and BioLangFusion published-split comparison. |
| [Stage1_gtex.ipynb](../Code/Stage1_gtex.ipynb) | GTEx transcript-expression experiment, baselines, representation comparisons, attention diagnostics, and IsoFormer published-split comparison. |

Each experiment has its own Setup section and can run without executing the other. The [README reading route](../README.md#stage-1-analysis) explains where to start; [environment setup](../README.md#setup) owns the kernel, working-directory, and dependency instructions.

The experiment notebooks retain outputs and execution counts from the combined Stage 1 notebook; these are not evidence of fresh runs of the separated files. Associate a result with the data, code version, model revisions, and settings that produced it. Current source links show the implementation on disk, which may differ from the implementation that produced a saved result.

The saved run uses twelve frozen encoders, four per modality: four DNA encoders read each coding sequence in DNA letters, four RNA encoders (RNA-FM and RiNALMo one token per nucleotide, mRNA-FM and CaLM one token per codon) read it in RNA letters, and four protein encoders read its translation. The overview's [research questions](../Code/Stage1_analysis.ipynb#Research-questions) describe three investigations: what simple predictive models can recover from frozen embeddings, how the representation spaces of every encoder pair organize the same examples, and how a selected attention summary of the nucleotide encoders relates to candidate sequence patterns. On the stability data all three inputs are derived from one sequence, and this constructed relationship matters when interpreting the representations; the [GTEx experiment](../Code/Stage1_gtex.ipynb) repeats the main analyses where genomic DNA, the transcript, and the protein are distinct inputs.

## Project scope: DNA and multiple encoders

Mina Basirat set the longer-term goal of fusing several pretrained encoders, with more than one model per modality across DNA, RNA, and protein. The [Stage 1 overview](../Code/Stage1_analysis.ipynb) describes DNA, RNA, and protein with four encoders per modality. The [stability experiment](../Code/Stage1_stability.ipynb#Published-split-comparison) compares against BioLangFusion on the CodonBERT stability data, where all three modality inputs derive from one coding sequence. The [GTEx experiment](../Code/Stage1_gtex.ipynb#GTEx-published-split-comparison) compares against IsoFormer on its transcript-expression data, where genomic DNA is a distinct input; the [published comparisons](../README.md#published-comparisons-and-distinct-inputs) summarize both. Stage 2's decision on what alignment means, recorded in the [project README](../README.md#stage-2-defining-alignment) and the Overleaf project "multimodal-bio-fusion", sets up the Stage 3 fusion experiments, which are planned, not implemented.

## Evidence and unresolved questions

Use the following questions to identify what a task needs to establish, rather than assuming that the documentation has settled them.

- **Dataset version:** Is [mRNA_Stability.csv](../Code/mRNA_Stability.csv) the same version BioLangFusion used? Its 65,356 rows and 29,949 distinct sequences differ from the paper's 41,123 raw and 23,929 used sequences, and the supplied `Split` places 5,392 distinct test sequences in training as well. See the [dataset overview](../README.md#data) and the [stability dataset audit](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons).
- **Modality definition:** When every modality input derives from one coding sequence, what does a difference between DNA, RNA, and protein encoders measure? Compare with a dataset where DNA is a distinct input before generalizing.
- **Dataset meaning:** What does `Value` in [mRNA_Stability.csv](../Code/mRNA_Stability.csv) represent, how was it obtained, and which biological sequence regions are included? The [dataset overview](../README.md#data) identifies the reported source; the CSV header alone does not establish the assay or label interpretation.
- **Evaluation:** Which rows are retained, how are training and test examples selected, and what generalization claim does that split support? Check the producing code and settings rather than inferring the split from a dataset column.
- **Result provenance:** Which data, notebook, model revisions, settings, and saved outputs support a reported result? If notebook prose, outputs, reviews, or slides disagree, identify the versions involved and retain the disagreement until it is resolved.
- **Meaning of alignment:** Distinguish biological sequence correspondence, similarity of representation spaces, and combining representations for prediction. Stage 1's composition control shows that most measured agreement involving the frozen DNA embeddings is reproduced by sequence composition alone, while agreement among the protein encoders, between RNA-FM and RiNALMo, and between CaLM and the protein encoders largely survives it. Stage 2 defines alignment as agreement on held-out examples after that control. Julian's [alignment review](../Notes/Reviews/Julian/Alignment_Theory_Paper_Review.md) and [BioLangFusion review](../Notes/Reviews/Julian/BioLangFusion.md) are starting points for reading the [source papers](../Papers/).
- **Further experiments:** What comparison would distinguish the proposed explanation from alternatives? The [September 23 meeting slides](../Notes/Mina_Meeting_Slides_2026-09-23.pptx), especially slides 3 and 6, record the contribution and architecture candidates, and the [October 7 slides](../Notes/Mina_Meeting_Slides_2026-10-07.pptx) present the Stage 1 results and the Stage 2 decision. The [architecture priorities](../README.md#architecture-priorities) reflect the meeting's decision to prioritize Coupled Mamba. Distinguish literature ideas, meeting preferences, and implemented methods. The Stage 1 experiments' exploratory findings alone do not select a fusion method.

## Implementation and verification routes

The experiment notebooks own dataset and encoder choices, analysis calls, output presentation, and experiment-specific interpretations. The [shared method guides](README.md#understand-the-methods) own the longer derivations and biological explanations. Reusable computation lives in [Code/mbf/](../Code/mbf/); inspect the relevant function and its callers before changing shared behavior. A change to a shared helper can affect both experiments.

Use [Reading implementation code](../README.md#reading-implementation-code) for compact source links and expanded listings. The [notebook checks](../tests/README.md) cover independent setup namespaces, source displays, and shared concatenation calculations with synthetic inputs. The same guide owns the check command and source-link refresh procedure. Passing these checks does not reproduce scientific results or verify notebook rendering.

For a task that moves or clarifies an explanation, follow its incoming links from the notebooks and method guides so readers still reach the relevant evidence. Preserve the scientific choices and saved results within that task's scope. A reorganization does not resolve the evidence questions above.

## Shared context across sessions

GitHub shares the files and versions contributed to the repository. The shared source index and linked explanations preserve selected learning and evidence across contributors' sessions; private agent memory is not their maintained owner. Separate AI conversations are not part of that shared record unless their relevant content is explicitly recorded here. Carry forward consequential decisions with their rationale, source links, and unresolved qualifications; avoid requiring collaborators to reconstruct an entire chat.

This page holds research context rather than the status of every active task. Keep assignments, pending reviews, and next actions in the relevant task conversation or an explicitly maintained task record. Update this page when the research focus or shared assumptions change.

Navigation and file roles checked against the repository on October 5, 2026. Scientific interpretations were not reassessed by that navigation check.
