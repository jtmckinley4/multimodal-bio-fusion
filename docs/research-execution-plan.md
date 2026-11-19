# Research execution and experiment batch plan

This plan connects the completed Newton comparisons, notebook execution safeguards and Mina's proposed experiments. It is written for Chase and Julian to review with Mina where scientific direction is involved. The aim is to make the work repeatable and partly automatable while keeping consequential choices visible to the researchers.

**Preparation checkpoint (superseded by the [November 19 execution record](run-records.md#2026-11-19-newton-e-coli-encoder-extension-and-controlled-studies)).** **B1–B4 implemented and checked offline; B5's 504-fit Newton cohort and four three-seed reports reviewed; B6a's grouped-stability protocol approved for implementation and synthetic checks, not research execution.** The [returned-evidence record](run-records.md#2026-11-16-arcc-recipe-layout-and-returned-input-deduplication) identifies source/input verification, and the [report interpretation record](run-records.md#2026-11-16-newton-three-seed-report-interpretation-without-recomputation) identifies the completed comparisons. The agreed A.2–A.6 pipelines now have four preparation notebooks, tested helpers and a source-only Newton recipe; research execution remains separate. Preserve existing results, caches and saved outputs. New research analyses, transfers, model downloads, training, job submission, commits and pushes require their own scope; completed work and approvals remain valid within their original scope. Agreement with the overall direction does not authorize every future run.

Use the [Stage 4 README](../Code/stage4/README.md) for the connected scientific investigation, [runbook](../Code/stage4/RUNBOOK.md) for execution instructions, [run records](run-records.md) for observed work, and [Gantt workbook](Research_Project_Plan_Gantt.xlsx) for task IDs and recorded dependencies. This document is a reviewable implementation sequence, not a replacement schedule or a second provenance system. Batch numbers below are local to this plan; they are not new Gantt task IDs.

## Read and approve the plan

Read [the current control-repair checkpoint](#control-repairs-before-newton), [the boundaries](#agreed-direction-and-open-decisions), [the batch map](#batch-map) and [human review](#human-review-and-semi-automated-work) first. The detailed batches explain the permitted work and expected evidence. The returned Newton reports inform the next protocols; they do not complete the separately grouped stability evaluation or establish scientific acceptance.

Chase authorized **B1–B4: implementation and synthetic verification of the reading/import/execution workflow, without research runs or transfers**; those batches are complete. B5 is the separately identified return-and-analysis batch. B6 settles the first scientific protocol; B7 implements it; B8 measures and executes only the agreed workload. Initial preparation through A.6 and the control repairs below are implemented following Chase's review and approval. Their synthetic verification and package evidence are recorded separately from research execution. B8 and subsequent research execution remain conditional on interpretation and resources. There is no need to approve all batches at once.

## Focused E. coli encoder additions

The advisor's Table 25 follow-up belongs to `Gantt`/`Details` task 24 and extends the completed encoder-swap investigation. The prepared `ecoli_ernie` mode reuses the E. coli cohort, NT-v2 100M/ESM-2 8M inputs and fixed BLF head recipe, replacing RiNALMo with RNAErnie and ERNIE-RNA. It requests four fits per training seed, with both gated fits before concatenation. Original references and cross-attention are not refitted; new single controls remain outside this panel, so it reports head scores without claiming a gain over a newly selected single encoder.

Chase approved using the authors' newer 2,048-token RNAErnie checkpoint to preserve the retained sequence lengths; this differs from the older short-context port linked in the request. The [source review](sources.md#rnaernie-and-ernie-rna-checkpoint-implementations) identifies the chosen versions and implementation limits. The inherited MIL penalty remains the historical original seed-0 selection, rather than a new search. The [Newton recipe](../execution/arcc/encoder_swaps/README.md#focused-e-coli-rna-extension) separates extraction of only the two new RNA caches from fitting, and preserves the completed 504-fit cohort. Newton subsequently completed extraction and the three requested seeds. [Execution and reported scores](run-records.md#2026-11-19-newton-e-coli-encoder-extension-and-controlled-studies).

## Control repairs before Newton

**Current checkpoint.** Chase reviewed this plan and authorized the required gate, attention and A.6 repairs, synthetic tests, updated Newton packages and optional-additions DOCX. The three notebooks and shared helpers implement the controls below; the optional experiments remain document-only. The [implementation record](run-records.md#2026-11-17-required-fusion-controls-before-newton) records checks and the replacement source package. No training, extraction, statistical analysis, transfer or job submission is authorized by this preparation. Existing results, caches and saved notebook outputs are preserved.

The requirements come from [Mina's protocol, A.3–A.6 and its final decision rule](Biological_Fusion_Plan.docx): compare sequence and position gates, control trainable capacity, retain the strongest single-encoder baseline and distinguish region availability from adaptive weighting. The exact controls and counts below are project implementations of those requirements, not numbers or architectures prescribed by Mina. They advance Gantt/Details tasks **46, 25 and 42**, without changing the separately prepared 36-fit grouped-stability panel or auxiliary correction.

### Match the gate and attention comparisons

**A.4: add a joint-input sequence gate.** In [gate diagnostics](../Code/stage4/gate_diagnostics.ipynb), retain Mina's requested BLF-style sequence gates and the existing joint-input position gate. Add a sequence comparator that applies the same affine sigmoid scorers to the masked mean of the joint projected DNA/RNA/protein features, then broadcasts its weights over positions. Both joint-input gates have 5,403 scoring parameters at width 600. Keep projections, prediction head, masks, training budget and initialization policy consistent. Their direct contrast tests where weights vary; the older sequence-versus-position contrast also changes scoring functions and remains separately labeled. Retain the approved full-width nongated A.4 control. This adds 12 fits: two benchmarks × two encoder combinations × three training seeds.

**A.5: add an active nonattention capacity control.** In [anchor cross-attention](../Code/stage4/anchor_cross_attention.ipynb), add a two-layer nonlinear mixer of the three projected streams at each position, with an anchor skip and the existing TextCNN head. Retain the selected single, projected-anchor-only and simple concatenation baselines. Derive the new mixer's hidden width from the attention budget before seeing outcomes; GELU is the selected project activation. At projection width 600, hidden width 1,203 gives 2,889,003 mixing parameters against 2,888,402 for the two auxiliary attention modules plus joint gates, a difference below 0.03%. Verify constructed active counts rather than padding with unused parameters. Compare this control with anchor-skip and gated attention while keeping their input grid and training budget fixed. Similar parameter counts do not establish equal expressivity, optimization difficulty or runtime. This adds 12 fits across the same benchmark/encoder/seed panel.

### Complete the A.6 baselines and gate controls

Retain the eight existing A.6 arms in [region-aware fusion](../Code/stage4/region_aware_fusion.ipynb), including native versus pooled RNA and the true/permuted masks for C, D and E. Add the following controls; the four new fusion controls use true masks only. Do not automatically multiply them by permuted-mask variants.

| Addition | Comparison it enables | Added fits across half-life and TE |
| --- | --- | ---: |
| Native ESM-2 protein-only predictor | Fusion versus the strongest declared single, rather than RNA alone | 6 |
| D with sequence-wide joint-input gates | Sequence versus position with the same scorers and C's readout | 24 |
| D's full-width nongated positional control | Adaptive weighting versus a comparable-count static transformation | 24 |
| E with ungated, CDS-restricted attention updates | Gates versus unit update weights with identical permitted update positions | 24 |
| E with sequence-wide, CDS-restricted gates | Sequence versus position within the same attention operation | 24 |

Each 24-fit addition covers two datasets × two RNA encoders × two grids × three training seeds. Fit the protein-only predictor on native residues once per dataset/seed, using the same eligible rows, targets and splits; it must not consume RNA features or require duplicate fits for each RNA encoder/grid. Select the strongest declared single on seed-0 validation among the four RNA-only encoder/grid configurations and native protein-only, then hold the selected configuration across seeds 0/1/2. Retain within-encoder/grid RNA-only contrasts alongside that strongest-single reference. RNA remains E's specified anchor; a protein-only winner qualifies that design rather than silently reversing its attention direction.

**D's readout and nongated control.** Retain the documented gated-concatenation adaptation: D applies joint affine sigmoid gates before C's concatenation/mask mixer and prediction head. This keeps C→D focused on adding gates; it is not Mina A.4's literal weighted-sum readout. The new sequence gate uses masked means with the same two affine scorers, giving 2,402 gate parameters at width 600. For the nongated comparator, retain the full 1,200-wide concatenated feature path at half scale and add a rank-one affine update before the unchanged mixer. An input map with bias and output map without bias give 2,401 update parameters; initialize the input normally and the output to zero. Reapply the RNA validity and protein-availability masks to their respective streams. Only the learned adjustment is rank constrained, not the whole representation. This replaces the earlier optional bottleneck control in the intended panel under a distinct method identity.

**E's update controls.** Preserve the same projections, attention, update normalization, dropout, RNA skip and CDS-intersected true protein availability in all three E variants. Change only the update weighting: one, sequence sigmoid or position sigmoid. The two gated variants have the same 1,201 gate parameters; the ungated model omits those parameters rather than registering unused weights. Replace the earlier optional unrestricted-query control in the intended panel with the CDS-restricted ungated variant; the old key must not silently acquire new behavior. E−D remains a comparison of complete designs because their readouts and mixing operations differ. These controls isolate gate effects within D or E, not attention alone across E−D.

**Policies preserved.** Keep the approved complete-transcript cohort/exclusions, terminal-stop handling, mapped homology groups and unmapped-gene fallback, fixed split and seed roles, RNA encoder pair, protein encoder, chunking, masks and training budgets. Chunking still loses encoder context across windows; the new controls do not resolve that limitation or unknown homology. No new embeddings are needed solely because of these model controls; A.6 still needs the already planned token extraction. Half-life precedes a separate review before expanding to TE.

### Review the workload and optional scope

| Study | Earlier prepared fits | Repaired panel | Additional fits |
| --- | ---: | ---: | ---: |
| Gates, A.3–A.4 | 72 | 84 | 12 |
| Attention, A.5 | 60 | 72 | 12 |
| Regions, A.6 across both datasets | 192 | 294 | 102 |
| Total for these three studies | 324 | 450 | 126 |

A.6 becomes 147 half-life fits followed, only after review, by 147 TE fits. Counts refer to training fits, not scheduler jobs or GPU-hour estimates; they exclude extraction, diagnostic inference and saved-prediction statistics. Preparing the panel does not authorize running all of it. Existing compatible reference fits remain reused.

The two stronger-claim experiments stay outside code, launch inventories and these counts: an A.5 capacity-matched anchor-only model (12 additional fits if later selected), and an A.6 local nonlinear update matched to E's skip/masks/gate budget (24 additional fits across both datasets). They are retained in the [optional-additions proposal](Biological_Fusion_Optional_Additions.docx) as future options. B12 owns their later selection, rather than treating them as mandatory repairs.

### Verify and rebuild the execution package

The implementation follows the reviewed order: A.4 comparator, A.5 capacity control, A.6 baselines/controls, then reports and packaging. Definitions and inspection remain separate from training calls. The verification contract is:

- Add synthetic checks for scorer/input equivalence across sequence/position pairs, masks and padding, gradients through active control parameters, zero-update behavior and actual parameter budgets.
- Verify identical CDS availability across the E variants, no protein update in UTRs or terminal-stop positions, and the unchanged D mixer/head contract.
- Verify that protein-only uses native residues without RNA inputs, is deduplicated across RNA comparisons, and participates in validation-only selection fixed across seeds.
- Update fit inventories, paired contrasts, coverage checks and result identities; incomplete panels must not silently select from an available subset. Preserve prior artifacts and refuse incompatible reuse.
- Update the three notebooks, affected shared helpers/tests, maintained Newton recipe/checklists and optional-additions DOCX. Regenerate and verify the fusion execution snapshot after its sources are final; use the replacement `ARCC Fusion Studies 2026-11-17 Controls` package, not the earlier preparation. The grouped-stability package needs no scientific change, but its snapshot must still pass its own compatibility checks if shared-source packaging is refreshed.

**Review boundary.** The agent owns routine implementation choices and synthetic verification within this selected scope, with consequential choices documented rather than silently introduced. Chase approved implementation after this written checkpoint. Subsequent Newton transfer, resource limits, training and TE expansion remain separate decisions; neither the selected repair scope nor passing tests authorizes them. Completion of this batch requires code, tests, documentation and package manifests to agree, not a claim of scientific validity or completed research.

## Agreed direction and open decisions

The current direction is controlled experiments with frozen encoders and trainable prediction/fusion components. Reference methods belong in Stage 3; experiments using them belong in Stage 4. Full published-training reproduction is not a prerequisite for the new correction/gate/attention investigation. The retained IsoFormer preparation and historical execution framework remain deferred. Existing BLF, IsoFormer and Mamba operations must be described according to their actual implementation and adaptations.

Desktop work covers implementation, synthetic checks, reading, plots and authorized analyses of saved evidence. Newton is the intended environment for the final neural-training comparisons. Local pilot fits remain possible, explicitly scoped and stored separately. CPU execution is not synonymous with reading: a ridge probe, statistical resampling or figure export can produce new research evidence even if it is inexpensive.

Keep data partitions, preprocessing, encoder/cache identities, validation choices, training settings and seed roles explicit. The fresh Newton swap batch retains its historical seed-0 validation choices and supplied partitions. It must remain distinct from desktop/Mac fits and from a future stability evaluation without exact-sequence overlap. Training heads on Newton does not change the producer of reused embeddings.

| Decision | Present standing | Who resolves anything still open? |
| --- | --- | --- |
| Preserve completed results, caches and historical evidence | Required; no overwrite, relabeling or automatic replacement fits | Already agreed |
| Fresh Newton swaps: seeds 0, 1 and 2 with historical validation choices | 504 returned fits and four completed three-seed reports reviewed; supplied stability partitions retain overlap | Researchers assess the interpretation; new analyses remain separately scoped |
| Reading-first maintained notebooks, separate CPU analysis and fitting | Implemented and checked offline through B1–B4; no research execution or transfer | Chase reviews the implemented route before a later run |
| Grouped stability benchmark and panel | B6a approved: two original/mRNA-FM encoder combinations, six unique single-input controls and six fusion fits per seed; seeds 0/1/2 give 36 fits | Implementation and synthetic checks authorized; research execution and resource limits remain separate |
| Grouped stability evaluation | Exact normalized nucleotide-sequence groups; fixed 70/15/15 hash assignment with split seed 42; preserve observations and targets | Near-identical, family, species or condition holdouts remain separate choices |
| Anchor-correction benchmark panel | CoV-Vac, fungal expression and separately grouped stability; original and mRNA-FM encoder combinations | Approved for local implementation; existing test observations make this exploratory |
| Correction families and folds | Separate pooled regularized-linear and token-level TextCNN corrections; five group-aware OOF folds with fold-local stopping; final anchor trained on training rows only | Approved; remaining control and tuning details must be visible in the prepared protocol |
| Anchor selection across repeats | Choose the configuration on seed-0 validation, then hold it fixed across training seeds; OOF predictions are conditional on this selected configuration | Approved; do not describe this as fully nested encoder selection |
| Region-aware preparation | Complete A.6 pipeline for human half-life then TE; separate RNA-FM and RiNALMo arms, each with ESM-2 8M | Approved for implementation, including explicit chunked token extraction; no extraction run authorized |
| Region cohort | Preserve full transcripts; terminal-stop nucleotides have no protein update; exclude ambiguous/internal-stop or inconsistent CDS/protein rows equally from comparisons and record reasons | Approved; retain exclusion reasons and the resulting population limits |
| Region split and mask control | Mapped human homology groups; unmapped rows grouped by recorded gene identity; fixed 70/15/15 split seed 42. Within-partition donor masks use relative CDS coordinates mapped to recipient length | Approved; the map remains required. Unmapped-gene grouping and this mask control are explicit project adaptations; unrecorded homology is not resolved |
| Gate equations and capacity controls | Joint-input affine gates, matched sequence and attention-capacity controls, and the expanded A.6 panel are implemented; research execution remains outstanding | The [control-repair checkpoint](#control-repairs-before-newton) records the approved choices; preserve the existing A.4 control and A.6 input policies. Routine dimensions/tests do not need separate scientific approval; optional stronger-claim experiments remain outside the panel |
| Compute limits and expansion criteria | No new GPU-hour, storage, concurrency or numerical improvement threshold agreed | Chase within allocation limits, with Mina for research priorities |
| Biological interpretation and manuscript argument | Agent can assess alternatives and draft evidence-linked claims; scientific acceptance remains human | Chase/Mina, with Julian's attributed contributions |

The [Gantt and Details sheets](Research_Project_Plan_Gantt.xlsx) connect evaluation/baselines to tasks 15/16, swaps and reliability to 24/28, correction to 45, gates to 46, attention to 25, and transcript regions to 42. Mamba tasks 20/21 and partner tasks 47/48 are conditional. Recorded dates and older Stage 5 names are historical; they do not establish current completion or a new deadline.

## Notebook and artifact organization

Keep one notebook per research question with its related controls, rather than one notebook for the whole protocol or one for every ablation. Each should explain its inputs, what previous evidence motivates it, what it computes, how to read the results and what remains uncertain. Notebooks consume named saved artifacts; they should not execute one another as a hidden dependency.

| Notebook | Scope and connection | Gantt/Details |
| --- | --- | --- |
| Existing [encoder swaps](../Code/stage4/encoder_swaps.ipynb) | Representation replacement, complementarity and fusion-method comparisons; imports and summarizes the fresh Newton cohort | 24, 28 |
| [Grouped stability](../Code/stage4/grouped_stability.ipynb) | Approved exact-sequence-grouped evaluation of the original and mRNA-FM combinations; implemented separately from the supplied-split results, not yet run | 15, 16, 28 |
| [anchor_correction.ipynb](../Code/stage4/anchor_correction.ipynb) | Auxiliary error correction, separate/joint auxiliaries and null controls | 45 |
| [gate_diagnostics.ipynb](../Code/stage4/gate_diagnostics.ipynb) | Sequence-level gate variants, then position-level gates with explicit capacity-control review; related ablations stay together | 46 |
| [anchor_cross_attention.ipynb](../Code/stage4/anchor_cross_attention.ipynb) | Ordinary versus anchor-skip and gated attention, with matched simple baselines | 25 |
| Existing [half-life/TE](../Code/stage4/halflife_translation_efficiency.ipynb), then [region-aware fusion](../Code/stage4/region_aware_fusion.ipynb) | Preserve pooled evidence; separately test token-level RNA/CDS/protein correspondence | 42 |
| Conditional partner notebooks | Choose actual datasets and evaluation units before naming or creating pipelines | 47, 48 |

The preparation notebooks and [source-only Newton recipe](../execution/arcc/fusion_studies/README.md) implement separate explicit operations; none has produced research results in this preparation pass. Keep scientific settings and interpretable experiment flow visible in the notebook. Share only helpers with a real common contract or a meaningful independent test; avoid a universal experiment engine. Preserve purposeful local functions and readable cell boundaries.

**Expanded local preparation.** Chase subsequently requested that the experiments be prepared together before transfer to Newton, including the complete A.6 pipeline. This authorizes implementation and synthetic verification of correction, gates, anchor attention and region-aware inputs/models. It does not authorize research calculations, model downloads, transfers or jobs. The two original/mRNA-FM combinations are the current A.2–A.5 scope; extending to the wider encoder panel remains a later decision. A.2 includes all three existing BLF regression tasks, with stability using its new grouped partition. Classification correction, partner datasets, feature-reconstruction residuals and further Mamba work remain outside this first correction/preparation panel unless separately selected. Code readiness, input readiness, an agreed experiment matrix and resource authorization are separate checks; finishing one must not mark the others complete.

**Decision-support pass.** The four notebooks put source-versus-project choices and optional supporting inspections beside their models: fold/OOF/control feasibility in A.2, parameter/rank and gate/contribution diagnostics in A.3–A.5, and cohort/window/effective-mask checks in A.6. The [completed CPU inspections](run-records.md#2026-11-17-cpu-decision-inspections-before-newton) informed Chase's subsequent approval of a frequency-preserving correction shuffle and a full-width nongated control with a small linear adjustment. These are project control choices, not newly specified equations from Mina. A.6 retains its agreed cohort, mask, window and model policies, including gene grouping for unmapped rows. The inspections produced no trained comparisons; resource trials and Newton execution remain separate authorization boundaries.

**Three operations, independent of location.** Existing switches can remain where they express these boundaries clearly; uniform semantics matter more than identical variable names.

| Operation | Allowed behavior | Must not happen implicitly |
| --- | --- | --- |
| Read results, saved default | Inspect selected saved records, coverage and existing analysis reports; refresh displays and draw in-memory figures from them | Downloads, model construction, extraction, fitting, bootstrap, recovery, directory creation or artifact publication |
| Analyze saved evidence, explicit | Run named statistical calculations on verified inputs; export a requested report/figure to its own analysis destination | Training/extraction, silent split or baseline changes, replacement of the source evidence |
| Run an experiment, explicit | Execute an exact declared workload, reuse compatible completions and retain compatible recovery state | Inferring scope from GPU availability, widening the grid, replacing incompatible files or promoting a pilot into final evidence |

An absent report in reading mode must be reported as absent, not generated. Older notebooks without reusable numeric summaries can display their saved figures/outputs or explain the unavailable reading route; that limitation does not authorize rebuilding statistics. Existing saved cell outputs remain historical until an authorized execution refreshes them. A new default setting alone must not relabel those outputs as Newton results.

“Reading” is an execution boundary, not a filesystem lock: the notebook remains editable and runnable. Jupyter or VS Code can still save displayed cell outputs through normal user/editor saving behavior. The promise is that notebook code does not silently fit models or publish research artifacts; implementation batches separately preserve the existing saved outputs and counts.

**Named result sets.** Keep historical locations unchanged. The [returned Newton root](../Code/stage4/results/newton_h100_seeds012/README.md) preserves `outputs/`, `local_analyses/` and `provenance/`. Its unchanged preparation snapshot and launch records identify archived execution sources and hash-verified canonical inputs/selection records; duplicate historical JSONs and CSVs need not remain in that root. The [deduplication record](run-records.md#2026-11-17-newton-evidence-deduplication-and-git-byte-preservation) documents the mapping without rewriting producer manifests. Follow the existing [artifact policy](../README.md#results-caches-and-intermediate-files) for shared evidence versus local large artifacts. One visible result-set choice resolves scoring, historical selection and analysis roots together. Never choose a result set merely because it is the newest directory. The label aids navigation; recorded identities and hashes establish the evidence.

The maintained repository owns scientific source. The sibling `ARCC Execution` folder remains an explicit execution snapshot; it and the uploaded Newton copy do not synchronize automatically. Local experiments use a separate pilot destination. Compact results, predictions and provenance can return for reading; unchanged embedding caches need not be transferred back when a verified copy already exists. Final weights and unfinished recovery checkpoints are retained according to their actual reuse/recovery need and existing policy, not committed merely because they exist.

## Batch map

B1–B4 are implemented with synthetic verification. B5 has returned evidence and completed three-seed reports. B6a has an approved grouped-stability protocol and implementation scope; no grouped-stability results exist yet. The correction, gate, attention and region pipelines are authorized for implementation and synthetic checks; their research runs remain unexecuted. The checkpoints below distinguish implementation, observed evidence and remaining authorization.

| Batch | Current checkpoint |
| --- | --- |
| B1 | Path contract implemented; four synthetic tests verify unchanged historical paths, separate imported scoring/selection/report roots, no directory creation or fallback, and overlap/import-write refusal. No files imported. |
| B2 | Implemented: explicit result-set selection, frozen historical selection adapters and existing-report reading. 119 focused synthetic adapter/report/notebook tests passed; saved outputs/counts/metadata across all 14 active notebooks and 329 protected results/source files remain unchanged. No real report recalculated. |
| B3 | Implemented: all 14 active notebooks default to saved-evidence reading. Synthetic mode/reader tests, source parsing and protected-state checks passed; no missing report was reconstructed. |
| B4 | Implemented: maintained [encoder-swap recipe](../execution/arcc/encoder_swaps/README.md), explicit operation/device/destinations, unchanged notebook source plus injected run settings, and complete-cache-only fitting. Package/runtime/mock-shell tests passed; the native Windows symlink fixture was unavailable. B4 inspected an inventory without materializing or transferring a real version-2 package or submitting a job; the completed historical version-1 batch is distinct. |
| B5 | Returned cohort verified: 504 fits and four reports, each with 72 complete three-seed contrasts. The [interpretation record](run-records.md#2026-11-16-newton-three-seed-report-interpretation-without-recomputation) supersedes the earlier pending-report status for these reports only. Grouped-stability work and broader scientific acceptance remain outstanding; task 28 is not completed by the repeats alone. |
| B6a | Approved protocol: fixed exact-sequence-grouped 70/15/15 split, split seed 42, two encoder combinations, 12 unique fits per training seed and seeds 0/1/2. Code, notebook and Newton recipe preparation are authorized, along with synthetic checks; no real split, fit, bootstrap, transfer or job submission is authorized in this implementation pass. |
| B7, B9–B11 preparation | Four connected notebooks, tested input/model/execution helpers and a source-only Newton recipe prepare A.2–A.6 together. This advances implementation, not the research-run or scientific-acceptance checkpoints. A.6 map and encoder weights are locally available, and the [unmapped-gene policy is implemented](run-records.md#2026-11-17-a6-grouping-for-unmapped-genes). Newton staging/runtime checks, new token caches, reference-weight byte verification and resource limits remain pre-run checks. See the [input preparation record](run-records.md#2026-11-17-controlled-fusion-input-preparation). |

Fresh grouped-stability and controlled-fusion workspaces are now assembled and locally verified; their [preparation record](run-records.md#2026-11-17-newton-experiment-package-preparation) distinguishes those copies from the recipes and older execution archives. Remote input checks, transfer, allocated preflight and resource-trial decisions remain separate. No new scientific run is established by package preparation.

Those verified workspaces predate the [selected control repairs](#control-repairs-before-newton). Local integrity does not establish that the fusion package contains the revised panel; regenerate its snapshot after implementation and synthetic verification. This review checkpoint does not modify either existing package.

| Batch | Deliverable | Dependency and location | Human checkpoint |
| --- | --- | --- | --- |
| B1 | Separate result-set and return-package contract | Current source inspection; desktop, no research calculation | Approve preparation scope and proposed storage layout |
| B2 | Encoder-swap import/read/report support | B1; desktop and synthetic fixtures | Review visible reading behavior and retained selection policy |
| B3 | Remaining active notebook execution guards | B2 pattern; desktop, one notebook at a time | Raise only scientific or scope changes; report each completed group |
| B4 | Reproducible Newton packaging and launch preparation | B1–B3; completed after B3 on the desktop | Approval needed before a real transfer or compute launch |
| B5 | Verified Newton import and three-seed interpretation | Completed returned cohort and four reports available | Review scientific interpretation; further calculations need their own scope |
| B6 | First experiment protocol and candidate shortlist | B5 for evidence-led choices; protocol preparation can start earlier | Chase/Mina settle unresolved scientific choices |
| B6a | Separate grouped-stability evaluation | Protocol and 36-fit panel approved; implementation can overlap B6/B7 | Review implementation and resource limits before research execution |
| B7 | Anchor-correction notebook and tested helpers | B6 choices affecting interfaces; desktop synthetic checks | Review method implementation before research execution |
| B8 | Bounded resource pilot, then approved correction runs | B7; pilot host specified in approval; final neural comparisons on Newton | Separate measured pilot and production/expansion approvals |
| B9 | Gate diagnostics and position-wise controls | B8 interpretation or an explicitly justified diagnostic scope | Approve variants, controls and run matrix |
| B10 | Anchor-preserving attention comparisons | B9 and matched-baseline evidence | Approve architecture adaptations and run matrix |
| B11 | Transcript-region investigation | B9/B10 methods and input audit; new token work possible | Approve biological cohort, mapping, context and compute |
| B12 | Conditional extensions and remaining reference work | Only when the core evidence identifies a question/resource need | Chase/Mina select, defer or reject each extension |
| B13 | Reconciled findings, figures and meeting/manuscript handoff | Repeated after completed research batches; final integrated pass | Human acceptance of claims; publication remains separate |

B5's completed reports do not require a new version-2 preparation or repeated fits. B6/B7's method-independent preparation can proceed within its authorized scope, but no open scientific parameter should acquire an undocumented default. B6a remains an explicit reliability outcome even if correction starts on another dataset. Later work needs the relevant evidence and accepted protocol, rather than completion of every historical notebook.

## Detailed batches

### B1 — Establish the result and preservation boundaries

**Work.** Inspect current Git changes, notebook sources, readers, result writers and execution-copy differences. Preserve Chase's pending Stage 1 repairs and any Julian changes. Define the named Newton import root, historical selection roots, local pilot root and analysis destinations. Capture a bounded before-change comparison of affected code/notebook metadata and completed artifacts; use existing Git and run-record mechanisms rather than a new tracking service.

**Deliverable.** A small result-set/path contract and exact return checklist using the existing runbook. The checklist includes fresh original and replacement fits, launch/environment records, execution-source identity and corresponding source files, executed notebooks, selection evidence, and relevant model/recovery files. Unknown historical producer fields stay unknown.

**Check and exit.** No path maps a Newton write onto historical results or caches. Selection evidence is distinguished from fitted results. The layout supports a teammate who has imported results but has no contributor-specific ARCC workspace. No result import or analysis occurs in this batch. Supports tasks 24/28.

### B2 — Make encoder swaps the working example

**Work.** Add an explicit result-set choice and separate existing-report reading from report calculation. Review and integrate the narrow ARCC adaptations for historical seed-0 MIL and single-encoder selection. Use existing identity, coverage and reuse helpers. Provide a bounded mapping for imported Newton paths without rewriting returned report identities or source JSON bytes.

**Deliverable.** The maintained notebook can read either historical evidence or a selected returned Newton cohort. It labels the selected source, available seeds and missing/incompatible records. The same scientific reporting definitions apply on desktop and Newton; the execution copy does not become a separate maintained scientific fork.

**Check and exit.** Synthetic fixtures cover absent caches/datasets, complete/incomplete/duplicate/malformed records, path relocation, fixed selection, conflicting modes, no cross-cohort seed fallback, and byte-preserving import/reuse. Reading a missing paired report does not calculate one. Mean-of-seed metrics is not silently changed into prediction ensembling. Preserve all current saved scientific outputs and execution counts during implementation. No real predictions are rescored. Supports tasks 24/28.

### B3 — Apply safe execution boundaries to remaining notebooks

**Work.** Read, plan and adjust one notebook at a time. Use the established meaning of reading versus analysis versus fitting without forcing a general wrapper around every calculation. Imports and source displays must not start downloads or model construction in the reading path. Keep code/comment readability and useful scientific Markdown intact.

**Order.** First review already guarded experiment readers: half-life/TE, LucaOne and mRFP, BLF reference/appendix, and Coupled Mamba. Explicitly separate Mamba's bootstrap from reading. Then address the BLF build and frozen GTEx main bodies, token-extraction and dataset-preparation routes, both main Stage 1 notebooks, and the GTEx audit's saved mode. This is an implementation order, not a new scientific sequence. Preparation notebooks may reasonably show a metadata inventory and explain how to enable their work rather than inventing a results dashboard.

**Check and exit.** Use isolated notebook-branch fixtures with producing entry points blocked, including end-to-end reading control flow where feasible. Existing tests of a few helpers do not establish safety of a whole notebook. Cover missing resources without silently recomputing; preserve output/metadata hashes and existing experimental defaults apart from the explicitly authorized execution-mode changes. If safe reading requires a new substantive analysis artifact, report that gap and propose its scope rather than manufacturing it. Supports the execution routes of tasks 15/16/17/24/28/42.

### B4 — Prepare repeatable execution snapshots

**Work.** Consolidate the needed notebook, helper, dependency, launcher and reporting adaptations into a reproducible preparation route for the sibling ARCC workspace. Retain the current batch's original source snapshot and its later recorded execution changes. Parameterize environment paths, kernel, selected workload, result/cache roots and expected device instead of hand-editing scientific code in the uploaded copy.

**Deliverable.** An inspectable package inventory and commands for the exact proposed job, with separate environment/source preflight, fitting and saved-result reporting. Reuse the existing launcher and Slurm controls, live epoch logs, per-run source/environment capture, locks and compatible checkpoint handling where suitable. Verify dependency versions before deciding whether any changes are necessary; do not silently upgrade the environment used for comparisons.

**Check and exit.** Offline tests cover missing source files, startup/environment failures, accidental CPU execution for a GPU-designated production run, duplicate writers, separated destinations, existing completions and incomplete recovery state. Package preparation does not copy huge caches, transfer files, mutate a currently running job or submit Slurm work. Real cluster preflight is a later, explicitly scoped compute action. Supports tasks 24/28 and later execution batches.

### B5 — Return, verify and interpret the Newton swap batch

**Current checkpoint.** The historical batch has returned and its source/input hashes have been verified, including the mapping from four duplicate CSV copies to canonical repository datasets. The [returned-evidence record](run-records.md#2026-11-16-arcc-recipe-layout-and-returned-input-deduplication) owns exact counts and evidence. Four completed three-seed reports have since been reviewed for scoring/selection bindings and interpretation; the [report record](run-records.md#2026-11-16-newton-three-seed-report-interpretation-without-recomputation) identifies their scope. The maintained encoder-swap notebook can read that evidence without recalculation. The procedure below remains the contract for further authorized analyses, not a request to repeat completed work. `Gantt`/`Details` tasks 24/28 retain their recorded in-progress status; the workbook's dated absence of repeat results does not describe the returned evidence, and grouped stability remains outstanding.

**Work.** Import only the authorized returned package into its separate root; compare transfer hashes and fit coverage. The prepared grid expects 504 fits across four benchmarks and three seeds, but file counts alone do not establish valid completion. Validate configurations, row/target/prediction correspondence, seed membership, launch links and the fixed historical selection rule. Inspect actual GPU evidence in logs/notebooks; an `expected_gpu` setting or generic `device="cuda"` is not an observation of H100 use.

**Remaining execution.** Default reading refreshes saved scores and coverage without new statistics. After reviewing exclusions and authorizing calculation, keep `READ_SAVED_RESULTS = True`, set only `RUN_SAVED_COMPARISONS = True`, leave `RUN_BOOTSTRAP = False`, and use `COMPARISON_SEEDS = (0, 1, 2)`. A single notebook execution covers all configured benchmarks and requested seeds; `SEED` controls individual displays, not this reporting set. The [runbook](../Code/stage4/RUNBOOK.md#run-focused-reports-and-cpu-probes) gives the canonical-input and separate report destinations. No new fits, embeddings or transfers are required for this step.

**Deliverable.** A coverage/exception inventory followed, when authorized, by the three-seed saved-prediction report and interpretation: replacement versus original, fusion versus selected single, and attention/gates versus same-input concatenation. Preserve returned reports as immutable producer artifacts and reuse a compatible completed report where available. A new local calculation receives its own analysis identity/destination; a missing Newton report never falls back to historical desktop/Mac intervals. Report seed variation separately from paired test-group bootstrap. Preserve null, negative and contradictory findings.

**Check and exit.** Refuse ambiguous aggregation and preserve damaged/incompatible inputs for review; no replacement training. A partial cohort can be displayed but cannot masquerade as the full requested mean. Historical and new numerical comparisons must identify changed environments and any other changed settings. Update tasks 24/28, README findings and run records only to the extent verified. Supplied-split stability results remain distinct from the still-unrun overlap-free evaluation. Statistical reporting changes produce a new identified analysis, not an overwrite of historical intervals.

### B6 — Agree the first scientific protocol

**Work.** Turn the verified results and Mina's questions into a short candidate shortlist. For each candidate, name the target's biological meaning, complete anchor predictor, auxiliary representations, expected benefit and plausible competing explanations. Separate an exploratory test on a familiar benchmark from a stronger prospective generalization claim: previous test observations have already influenced hypotheses.

**Human decisions.** Choose the first benchmark/panel; whether regression alone is the first deliverable; evaluation unit and exact grouping/split policy; admissible anchor candidates and validation selection; number of OOF folds; early-stopping/tuning procedure; correction family/capacity; controls; metrics; practical improvement criterion; and resource limits. Explicitly decide whether the new experiment holds the selected anchor configuration fixed across training seeds or reselects it by a declared validation rule; these estimate different sources of variation. Neither option changes the existing Newton batch's frozen historical selection. Recommend options with consequences rather than asking Chase to design helper functions. No arbitrary fold count, epoch cap, patience or truncation is presented as paper-derived.

**Evaluation work.** If stability is selected for an unseen-sequence claim, prepare a separately identified group-disjoint train/validation/test split and matched baselines. Exact duplicates, related sequence clusters and species/condition holdouts answer different questions. Cluster thresholds and metadata availability need justification; they are not interchangeable cleanup steps. New partitions require new head fits, even if compatible frozen matrices can be reused. Historical half-life split seeds are not independent neural-initialization repeats.

**Check and exit.** A concrete protocol names both what is fixed and what will be selected on validation. The final test is not used to select architecture, tuning or which runs to report. The permitted final-training rows are explicit; do not silently switch between train-only and train-plus-validation refits. Tasks 15/16/28/45 supply the prerequisites.

**Sequential investigation.** Decide which evaluations are exploratory and which constitute independent confirmation. If B8's held-out results guide B9–B11's architectures, repeated evaluation on that same holdout is an adaptive exploratory investigation, not a succession of untouched final tests. For a prospective claim, reserve an appropriate independent evaluation until the relevant choices are fixed. Creating a new seed or renaming the split does not erase prior exposure to its examples. Record this distinction before promising confirmatory interpretation.

### B6a — Retain the separate grouped-stability experiment

**Agreed protocol.** The [grouped-stability notebook](../Code/stage4/grouped_stability.ipynb) owns the scientific settings and [Newton recipe](../execution/arcc/grouped_stability/README.md) owns packaging and execution. Preserve the existing ≤999-nucleotide cohort, row order and supplied targets. Group the complete normalized nucleotide sequence, treating T/U as equivalent for group identity without rewriting cached encoder inputs. A deterministic 70/15/15 group assignment with split seed 42 is shared by all models and training seeds 0/1/2; actual row proportions are approximate because groups stay intact. Do not average repeated observations, stratify on targets or search split seeds for a favorable result. Retain the supplied split as historical evidence.

**Panel and selection.** Compare the original NT-v2 100M / RNA-FM / ESM-2 8M combination with its mRNA-FM replacement. Six unique single-input controls (the four native encoders plus NT-v2 and RNA-FM after codon-grid transformation) and concatenation/MIL/cross-attention for each combination give 12 fits per seed, 36 overall. Shared single controls are not duplicated. Keep the existing training recipe and MIL penalty 0.01; refit prediction heads from scratch. Select eligible single-input baselines using the new grouped seed-0 validation results, retaining that selection across the reported seeds. Report comparisons within each encoder combination and against the strongest selected single across the declared panel.

**Source and scope.** Mina's section A.1 supplies exact-sequence grouping, fresh fits, three-seed reliability and separate group-bootstrap uncertainty. The 70/15/15 allocation, split seed, bounded panel and additional DNA-grid control are the project's agreed implementation choices. They test unseen exact nucleotide strings, not unseen protein sequences, homologous families, species or experimental conditions. Similarity clustering or reconstruction of omitted source-profile metadata could support later experiments, but is not silently added here. Historical test observations informed this panel, so it remains exploratory.

**Human gate.** Implementation, synthetic tests and Newton run-file preparation are authorized; real partition generation, fits, statistical reports, transfers and submissions have not been performed or authorized by that implementation pass. Review the prepared workload and resource limits before execution. Three seed jobs can each run 12 sequential fits, using B8's pilot/production procedure where resource measurement is needed. No automatic repeat of all eight replacements is planned.

**Check and exit.** Deterministic checks establish group exclusion and retained row/target correspondence. New baseline fits and authorized reporting establish only the agreed new evaluation. Reuse frozen caches where row/input identity permits, retain the original results, and keep exact-sequence independence distinct from gene-family or species generalization. If correction later uses this split, its OOF groups and selection must respect the same outer boundary. Tasks 15/16/28.

### B7 — Implement anchor correction and verify its mechanics

**Agreed implementation settings.** Use the three regression benchmarks and the original/mRNA-FM bundles. The six-candidate single-input panel contains native NT-v2, RNA-FM, mRNA-FM and ESM-2 plus NT-v2/RNA-FM codon-grid controls. Select within each bundle using seed-0 validation and hold the winner across seeds 0/1/2. Five OOF folds use only outer-training groups; inner stopping uses 20% of each fold's fitting groups. Twenty percent is our explicit reconstruction choice, not a number supplied by Mina. The final anchor uses outer training with outer-validation stopping; it is not refitted on training plus validation. Pooled corrections use the existing ridge penalty grid; TextCNN corrections retain the BLF training budget. Separate and joint auxiliaries receive real, frequency-preserving within-partition shuffled, and training-scaled random-feature controls. Those details and their source limits remain visible in the notebook.

The approved shuffle uses each donor row once within its permitted partition/stratum and excludes donors from the receiver's exact-sequence group. Joint auxiliaries share one donor and carry its lengths with the features. This preserves empirical input frequencies, including repeated observations, while disrupting their pairing with fixed receiver targets and anchors. Repeated receivers may receive different donors; this is row-level randomization, not the earlier whole-group donor assignment. Infeasible constraints must stop rather than drop rows or silently retain same-group donors. The historical shuffle evidence stays separate from the new control identity.

**Work.** Implement `anchor_correction.ipynb` and only the tested helpers it actually needs. Each training row receives an out-of-fold prediction from an anchor fitted without its whole group. Build correction inputs from the agreed auxiliary streams separately and jointly. Preserve compatible cached representations and store fold/anchor/correction identities separately from the original swap results.

**Scientific contract.** For regression, learn residuals from OOF predictions and add the correction to the final anchor prediction. For classification, retain the class order and OOF logits, then use additive logit corrections with the agreed loss; predicted class IDs are insufficient. Existing test predictions or a checkpoint trained on all training rows cannot reconstruct honest OOF training predictions.

Keep the fitted anchor fixed while learning the correction. The corrected predictor and its anchor-only comparator must use the same final anchor artifact. The agreed first implementation keeps outer validation rows out of correction training and uses them for selection. Expanding only the anchor's final fit does not supply honest correction targets for newly included rows. If expanded correction refitting is later requested, it needs a separately agreed resampling and selection procedure: recomputing folds with settings previously tuned on those same held-out labels does not establish the strict OOF exclusion required here.

The fold-held-out labels must not choose that fold's preprocessing, epochs or hyperparameters. Define the agreed inner stopping/tuning procedure using only permitted rows; outer validation/test rows remain outside OOF training. Shuffled-input and random-feature controls need declared random seeds, feature dimensions/scales, training capacity and a meaningful permutation scheme. A shuffle within an unchanging duplicate group may produce no actual perturbation and is not automatically a valid control.

**Check and exit.** Synthetic cases establish group exclusion, one OOF prediction per eligible row, row/class-axis alignment, training-only transforms, zero-correction recovery of the anchor, exact correction arithmetic, reproducible controls, incompatible reuse refusal and partial-failure preservation. Clearly mark any classification path excluded from the first batch. Tests establish software behavior, not biological usefulness. No embeddings or research fitting are run. Task 45.

### B8 — Measure a small workload, then run the agreed correction comparison

**Pilot approval.** Present the exact source snapshot, inputs, variants, seeds/folds, maximum jobs/concurrency, per-job and total resource limits, output location and stop conditions. A shortened or sampled pilot is labeled accordingly and cannot substitute for the final protocol. Its tuning choices cannot be selected on final test performance.

**Measurement.** Record elapsed training time, data-loading/verification overhead, peak memory where measurable, throughput, storage and actual device/environment. For each anchor/seed combination admitted by B6's selection policy, the basic K-fold plan needs K fold fits plus a final permitted-data refit, before correction models, tuning candidates and controls. Multiply by the actual approved matrix, not the entire former swap grid. Report estimates with their measured basis and uncertainty; no unmeasured completion-time promise.

**Production approval and work.** Present the measured budget and finalized run list. Run only those jobs after approval. Reuse exact compatible completed artifacts, respect one-writer boundaries, and preserve recoverable failures. Repeat comparisons over the agreed seeds, including the already intended three-seed reliability goal. A changed batch size, precision, sequence limit, stopping rule or CPU fallback is a protocol change, not an automatic OOM fix.

**Check and exit.** Validate completion and produce matched reports, resource accounting and competing interpretations. Include failed and exploratory work in cost reporting. Null gains do not establish that an auxiliary has no useful information. Whether to widen, revise, stop or proceed to gates is a human interpretation/resource decision under pre-agreed criteria. Tasks 28/45.

### B9 — Diagnose gates before expanding attention

**Work.** Implement `gate_diagnostics.ipynb` in two reviewable increments: sequence-level variants first, then position-wise gates and a comparable-capacity nongated positional control. Include the verified mean-summary softmax baseline, entropy ablation, independent sigmoid and mean-plus-max summaries. Mina specifies no entropy regularization for the new sigmoid gates; adding it would be a separate explicit ablation. A downstream-head-derived gate is optional, not silently added to the mandatory matrix.

The approved A.4 nongated control retains a fixed half-sum of the projected modality features and adds a rank-2 affine update from their concatenation. Initialize the update's output map to zero and its input map normally, retaining the full-width additive signal at startup. The update has 5,402 trainable parameters at three 600-wide streams, versus 5,403 for the joint sigmoid gates. This is a matched-count static comparator, not equal expressive capacity or a two-dimensional bottleneck on the whole representation. It replaces the earlier rank-150 primary control under a new method identity without adding fits. The [pending repair](#match-the-gate-and-attention-comparisons) retains this control and adds a same-scorer sequence comparator. A.6's separate control revision is described in the [A.6 repair scope](#complete-the-a6-baselines-and-gate-controls).

**Human gate.** Agree variants, capacity/tuning budget, feature-scale treatment, retained encoder/alignment choices and the selected benchmark panel. Gate position or prediction-head placement changes must be named explicitly and compared against the existing operation. Saved per-example and position gates are diagnostics; biological claims need perturbation or other supporting evidence.

**Check and exit.** Synthetic tests cover masks/padding, shapes, gate ranges/normalization, gradients, parameter counts and the intended operation. Softmax-versus-sigmoid changes can alter feature scale; a gain cannot automatically be assigned to less modality competition. Follow B8's measured pilot and bounded production process, with separately approved runs. Task 46.

### B10 — Test anchor-preserving attention under matched controls

**Work.** Implement `anchor_cross_attention.ipynb` with anchor-only, same-input simple fusion, ordinary attention, anchor-skip attention and agreed gated updates. The [selected capacity-control repair](#match-the-gate-and-attention-comparisons) adds a similarly sized nonattention mixer without removing those baselines. Keep projection, normalization, residual paths, masks and capacity differences inspectable. Preserve the anchor predictor baseline; a representation skip does not guarantee preservation of its score.

**Human gate.** Agree which operation is being tested, any adaptation from a published method, context/grid handling and the matched comparison budget. Generic anchor-query attention is not labeled an IsoFormer reproduction. Full Enformer/NT/Borzoi training reconstruction is not revived by this batch.

**Check and exit.** Hand-calculated small attention cases and masking/gradient tests verify the implemented operation; matched skips and nongated controls support attribution. Run only the agreed panel after resource measurement. Report when a method improves on a single but not on concatenation, or improves only at greater cost. Task 25.

### B11 — Test transcript-region correspondence

**Work.** Preserve the existing pooled half-life/TE analyses and implement `region_aware_fusion.ipynb` only for the selected biological question. Audit full-transcript inputs, protein/CDS coordinates, availability, sequence lengths and grouping. Native-resolution token features may require new extraction; pooled vectors cannot recover local sequence positions.

**Human gate.** Confirm the dataset and target interpretation, RNA context, transcript/protein correspondence rules, missing/overlength handling and resource allocation. Mina's order is human half-life then TE; expanding to mouse or other populations is a separate choice. Do not confuse stability scores with hours or ribosome occupancy with direct protein production.

**Comparisons and exit.** Stage RNA-only, protein mapped to CDS with zero-filled UTR positions, explicit region availability, local gates and CDS-restricted attention with matched skip controls. The [selected A.6 repairs](#complete-the-a6-baselines-and-gate-controls) add the missing protein-only baseline and sequence/nongated controls, preserving the agreed cohort and masks. Include Mina's native-resolution versus codon-pooled RNA comparison and true versus transcript-permuted valid region masks. Define the admissible permutation and preserved mask properties before running; the shuffled mask is a negative control, not a biologically valid alternative. A mask supplied as an input feature does not enforce exclusion; the protein update must respect the agreed mask. Verify coordinate/frame/mask contracts synthetically, then follow bounded extraction and training approvals. Interpret region information separately from adaptive weighting, and do not attribute a simultaneous attention-and-skip change solely to attention. Task 42.

### B12 — Review optional additions and unfinished reference work

This is a decision batch, not an instruction to execute every remaining idea. The [optional-additions proposal](Biological_Fusion_Optional_Additions.docx) remains distinct from Mina's core protocol.

The next documentation update should add the two stronger-claim controls from the [repair review](#review-the-workload-and-optional-scope) to that DOCX, preserving its existing concise proposal style. They remain unimplemented and excluded from the Newton workload unless separately selected: A.5 asks whether auxiliary information helps beyond enlarging the anchor alone; A.6 asks whether attention beats a similarly sized local update under matching skip/mask/gate conditions. Neither option changes the approved input policies or authorizes additional runs.

| Candidate | What earns further work | Boundary |
| --- | --- | --- |
| Coupled Mamba, tasks 20/21 | Measured long-sequence time/memory bottleneck and a matched comparison worth testing | Verify the reference operation/adaptation; do not cross every architecture with every dataset automatically |
| IsoFormer operations | A specific attention/control question needing those operations | Reuse verified components as appropriate; full published-cohort reproduction remains deferred |
| LucaOne or other encoder families | A representation gap motivated by the results and an input-compatible task | Separate encoder extraction/probe costs from fusion-method questions; ridge split repeats are not neural seeds |
| Existing matched LucaOne and half-life/TE CPU probes | A relevant unanswered baseline or complementarity question | These fit models and need execution scope despite using CPU; verify whether results already exist before rerunning |
| Prediction ensembles, pooled-versus-positional controls, auxiliary feature residuals | A confound left unresolved by the core comparisons | Explicitly distinguish target-error correction from feature reconstruction residuals |
| Fungal/yeast/plant or other held-out evaluations | A defined biological transfer question and usable species/condition/family metadata | Dataset availability and grouping determine the claim; taxonomic difference alone is not a controlled experiment |
| RNA–protein binding, task 47 | Suitable partner-pair labels, negatives and entity identities | Design RNA-disjoint, protein-disjoint or dual-disjoint evaluation and partner-reassignment controls |
| Protein–ligand affinity, task 48 | Compatible assay endpoint, molecular inputs and entity metadata | Define cold-protein/ligand/both evaluation; avoid silently pooling incompatible affinity targets |
| Missing historical predictions/repeats | Needed for a named retained comparison | Investigate recoverability first; unknown provenance or missing arrays do not authorize replacement training |

**Exit.** Chase/Mina select a bounded next question or defer it. Julian's ownership and existing implementation should be checked before duplicate work; this plan does not send him assignments. Each accepted extension receives its own protocol and budget through the same gates.

### B13 — Reconcile the argument and hand off the evidence

**Work.** After each research batch, update the producing notebook's interpretation, concise Stage 4 findings, execution record and relevant Gantt status. At an agreed milestone, assemble the cross-experiment table/figures and meeting or manuscript narrative. Preserve the path from biological target to input, comparison, metric, uncertainty and justified claim.

**Human gate and exit.** Review contradictory findings, practical effect sizes, costs, assumptions and limitations. Distinguish exploratory findings from independent confirmation and learned gate patterns from biological mechanisms. Do not prune completed negative results to strengthen the story. Human acceptance of the argument is separate from software checks and result integrity. Preparing a commit, pushing, sending a document or publishing remains separately authorized. Tasks 19/28 and the contributing experiment tasks.

## Human review and semi-automated work

### What the agent may decide inside an approved batch

The agent can choose readable helper names/cell boundaries, reuse existing utilities, resolve ordinary implementation details, add focused synthetic tests, check links and preserve bytes, and fix defects that do not change the approved scientific contract. It should report consequential implementation choices and any remaining limits without interrupting Chase for every reversible detail. Independent agents may review bounded work, but their agreement is not scientific validation or human approval.

Within authorized research execution, deterministic code should inventory inputs, check identities, assign approved folds, compute outputs and record evidence. The LLM proposes and explains choices; it should not estimate dataset counts by reading examples, invent a missing producer, or decide that a result is acceptable merely because it looks plausible.

### Where work pauses for a decision

| Trigger | Concrete question/evidence to return | Work that may continue |
| --- | --- | --- |
| Conflicting or absent source requirements | Sources, exact discrepancy, proposed setting and consequences of alternatives | Independent inspections and synthetic scaffolding |
| Scientific departure | Changed cohort/split, targets, model operation, tuning/refit rule, precision, sequence limit or budget | Existing approved branches unaffected by that choice |
| Input/result mismatch | Exact identities, affected files and dependent comparisons; preserve original files | Valid independent result sets, without filling the gap from another cohort |
| Real-data compute or transfer not yet authorized | Exact matrix, destinations, source snapshot, measurements/estimate, limits and recovery behavior | Offline checks and package preparation |
| Failed run or resource limit | Failure evidence, recoverable state and bounded options; no silent method downgrade | Independent authorized jobs unless the failure indicates a shared defect |
| Unexpected/contradictory result | Verified observations, competing explanations and the smallest diagnostic that distinguishes them | Evidence review; no automatic additional test-set tuning |
| Expansion or final claims | Effect size/uncertainty, validation evidence, resource cost, limitations and specific decision | Documentation of established observations |

When a structured question interface is available, use a pop-up; otherwise ask clearly in the chat. Each question should contain one consequential decision, a recommendation, two or three meaningful alternatives where useful, what each changes and what is blocked. It should link the concrete diff/protocol/result being reviewed. For example: “The pilot exceeds the agreed memory budget. Keep the protocol and request a larger allocation, or approve a smaller batch as a new configuration?” This is an example, not an observed failure.

An unanswered question remains pending. A preselected option, elapsed time, old permission, passing tests or a completed earlier batch is not approval of a new dependent action. If a question is merely a preference within already authorized reversible work, the agent can state a reasonable assumption and proceed after allowing a response; this exception does not apply to scientific choices or resource authorization. Once a decision is explicit, carry it forward rather than asking again.

### Approval examples and batch completion

These examples are proposed wording for later approvals, not instructions to execute now:

- “Approve B1–B4 implementation and synthetic tests. Preserve research artifacts and saved outputs. No transfers, real-data analyses, downloads or cluster jobs.”
- “Approve B5 import from this supplied directory and the specified three-seed saved-prediction report. No fitting or replacement results.”
- “Approve the B6 protocol and B7 implementation. Leave the unresolved settings disabled; no research runs yet.”
- “Approve the exact B8 pilot matrix and stated limits.” Production runs follow once the measured budget is reviewed, unless they were already explicitly included with sufficiently concrete limits.

For an approved sequence, finish each batch's checks and continue into the next authorized independent batch. Return a concise checkpoint: changed files/behavior, checks and limits, scientific decisions applied, artifacts produced or preserved, resource use if any, and the next permitted action. Ask only when a listed gate is reached. Capture durable decisions in the owning notebook/protocol and actual execution in existing run records; do not create a second approval database.

At resumption, inspect current files and job/artifact evidence before relying on the last summary. No plan promises unattended execution after the chat stops or access to a disconnected cluster. If access or a decision is unavailable, report the specific boundary and continue only independent authorized work. Transient retries must remain within the approved protocol and retry/resource budget; repeated failure is not permission to launch unlimited replacement jobs.

## Checks and completion criteria

**Preparation is complete** when the selected result sets cannot mix silently; default reading paths cannot produce research artifacts; explicit analysis and fitting remain usable; the Newton package can be prepared from identified maintained source; protected outputs/metadata/results remain unchanged; and affected links/instructions agree with the implemented routes. An unavailable reading artifact is a reported limitation, not a reason to fabricate a new result.

**An experiment is complete** only for its declared protocol and matrix: accepted inputs and partitions, checked implementations, authorized executions, verified coverage, appropriately paired reporting, resource evidence and a human-reviewed interpretation. A successful Slurm state, number of JSON files, green test suite or attractive figure alone is insufficient. A justified null result can complete the experiment; a positive result does not authorize expansion automatically.

Use the [existing synthetic and notebook check guide](../tests/README.md) and [shared engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md). Preserve the pre-existing dirty work rather than resetting the checkout. Separate implementation-display updates from scientific saved outputs and counts. New research outputs are written only in the batches that explicitly authorize their production.

## Basis, provenance and limits of this draft

Prepared by Codex for Chase's review on November 16, 2026. Sources inspected were the current [advisor protocol](Biological_Fusion_Plan.docx), [Stage 4 guide](../Code/stage4/README.md), [runbook](../Code/stage4/RUNBOOK.md), Gantt/Details entries cited above, active notebook settings/readers, the local ARCC preparation and existing tests. At initial preparation, the last-two-fits status was user-reported. The later [returned-evidence verification](run-records.md#2026-11-16-arcc-recipe-layout-and-returned-input-deduplication) supersedes that status for transfer, source/input identity and basic completion; the [report interpretation](run-records.md#2026-11-16-newton-three-seed-report-interpretation-without-recomputation) supersedes the pending-statistics status for the four Newton reports. B6a's approved implementation scope is distinct from running its new evaluation. Local execution preparation is not itself Newton execution evidence.

Three external sources informed the review process, not the biological experiment settings:

- [NIST AI RMF Core, GOVERN 3.2](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) calls for distinguishing human/AI responsibilities and oversight. The relevant core entry was consulted for explicit roles and decision boundaries; this plan is not a claim of formal compliance.
- [NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist), particularly claims/limitations, reproducibility and compute-resource reporting, informed the evidence and cost handoff. The relevant guidance was consulted; it does not determine this project's venue or require every candidate experiment.
- [Google Research's AI co-scientist overview](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/) describes researcher-specified goals, feedback and expert-guided evaluation. The public overview was consulted as an example of human-directed AI research assistance; it does not validate Codex's scientific judgment or our proposed results. No additional biological paper or system dependency is introduced.

Reading and agreeing this draft can establish the workflow. Execution authority, scientific parameter choices, measured resource feasibility and acceptance of findings remain separate decisions as described above.
