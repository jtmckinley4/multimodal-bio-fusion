# Shared coding: review and implementation

The existing combination of experiment notebooks, numerical functions, and small dataset/model records is a reasonable foundation. The implemented improvements expose scientific choices and make failures easier to recover from. There is no demonstrated need for an experiment framework or an object-oriented rewrite.

Prepared with Codex for Chase and Julian on October 5, 2026, against commit `1cd528a`; implementation was then authorized. The findings below describe the pre-change review, not the current state after implementation. The active rules live in the [shared engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md). See [selected engineering sources](sources.md#choosing-code-boundaries) for reading scopes and attribution.

## Implementation status

| Batch | Status and evidence |
| --- | --- |
| 1. Shared workflow | Installed after two isolated behavioral evaluations: ridge extraction, invalid-null rejection, and interrupted-download recovery; new-split and Markdown-only scenarios respected the boundary. The original 20-test baseline passed; local Codex discovery confirmed the repository skill enabled. The Claude wrapper resolves to the same owner. |
| 2. Validation and cleanup | Implemented. Eight validation tests preserve supported scalar/null behavior and reject unsupported settings before work; five resource tests cover normal release and simultaneous inference/cleanup failure across all three model owners. |
| 3. Shared ridge definition | Implemented. Five tests compare original scalar/30-output/classification behavior and independent fitted state; a fresh private ridge constructor serves both entry points. |
| 4. Seeds and displayed direction | Implemented. Both notebooks retain 42 while distinguishing `PROBE_SEED` from `SEED`; six seed/direction checks cover default and nondefault routing, repeated assignments, and mirrored asymmetric callbacks. |
| 5. Cache identity and publication | Implemented in separate identity and publication passes. Ten cache tests cover actual prepared inputs, archive closure, incompatible/corrupt files, valid mismatches and interrupted replacement. Cleanup review identified and corrected primary-error masking; team caches were untouched. |
| 6. GTEx coordinates | Implemented. Eight coordinate tests include independent plus/minus, even/odd-window examples and comparison with the prior formulas. Filtering, sampling, 6,000-base windows and boundary behavior remain unchanged. |
| 7. GTEx external operations | Implemented. Twenty-three offline input/output tests plus the eight coordinate checks pass. Downloads and CSVs publish after validation; retries distinguish recoverable failures, honor bounded service delays, retain the final cause and identify the failed batch. Independent review found no outstanding defect. |
| 8. Integrated audit | Complete for this implementation scope. All 85 automated tests pass; 613 local Markdown references and 65 compact implementation links pass their checks. Independent review verified the executable-change boundary and preserved saved results. Existing guides/routes are reconciled, and the shared skill clarifies primary-error versus cleanup-error handling. |

The isolated skill evaluations and the combined suite used small fixtures and substitute operations. Tests passed in the staged checkout before installation. Installation checked each destination against its prior hash to detect concurrent edits and verified the installed bytes. The suite does not establish biological correctness, real-device performance, original-result provenance, or research reproducibility. Julian's application discovery and human readability remain separate checks.

Both notebooks preserve every scientific output, saved execution count, cell ID, and unrelated metadata. The permitted display refresh changed 22 stability and two GTEx compact outputs, using guarded isolated Setup and source display only. Executable edits add the explicit pinned seed and pass it to probes, or expose the shared ridge constructor; all effective scientific defaults remain unchanged. The [test guide](../tests/README.md) describes the maintained checks and their limits.

Legacy embedding caches now require deliberate recovery through the [cache instructions](methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse); no existing team cache was migrated or deleted. GTEx source-schema checks cannot determine whether a schema-valid old cache was historically complete. The coordinate extraction retains the existing boundary behavior. Model downloads, scientific notebook bodies, pilot rebuilds, commits and pushes were not performed in this implementation.

## What the shared skill should accomplish

An agent should help a reader follow the path from biological question to inputs, calculation, evaluation, and interpretation. It should explain why an extraction or new abstraction makes that path easier to follow, identify what behavior must remain, and verify the affected contract. Shorter files and more classes are not success measures.

Chase's personal Shared Engineering skill supplies the starting criteria: KISS and YAGNI, one owner for shared knowledge, meaningful functions, explicit interfaces, predictable failures, and proportionate verification. The shared version should contain the applicable criteria itself, so Julian's agent does not depend on Chase's personal installation or filesystem. The project-specific additions are biological row correspondence, split and preprocessing rules, model/input identity, scientific change boundaries, and the existing notebook reading route.

[Fowler's YAGNI discussion](sources.md#fowler-yagni) allows work that makes present code easier to change while opposing speculative capabilities. [The Pragmatic Programmer's DRY discussion](sources.md#thomas-and-hunt-duplication-of-knowledge) distinguishes a shared rule from coincidentally similar code. [Martin's SOLID discussion](sources.md#martin-solid-as-responsibility-and-interface-guidance) informs responsibility and interface questions; applying those questions to this Python code does not require class hierarchies.

## Choosing between inline code, a function, and a class

These are proposed decision criteria, not a size quota. A longer function can be easier to understand than a chain of helpers that hide every intermediate step. The [C++ Core Guidelines](sources.md#c-core-guidelines-meaningful-functions-and-related-data) offer transferable questions about meaningful operations and related state, with language-specific details left out. [Google's review guidance](sources.md#google-reviewing-complexity-and-useful-tests) asks whether readers can understand and safely modify the code.

| Form | Use it when | Application here |
| --- | --- | --- |
| Inline notebook code | It expresses this experiment's settings, comparison, or short sequence of steps. | Keep dataset selection, encoder combinations, controls, displayed results, and interpretation visible. |
| Function | An operation has a useful name, a coherent input/output contract, reuse, or an independently checkable boundary. | Share ridge-pipeline construction; extract GTEx coordinate conversion from downloading and writing. A function can be useful with one caller. |
| Module | A coherent family has enough independent use or change pressure to improve navigation. | Existing sequence, split, dataset, encoder, and embedding modules make sense. File length alone does not require splitting `analysis.py`. |
| Dataclass or other simple record | Related named fields describe one item or result more clearly than positional arguments. | Keep `Encoder`, `DatasetSpec`, and `Dataset`. Their existence is not evidence of overengineering. |
| Stateful class | Several operations need the same persistent state or must maintain a lifecycle/invariant. | Consider a loaded-encoder session only if repeated use and cleanup across operations become an actual requirement. Ordinary `try/finally` is sufficient for the current cleanup gaps. |
| Simplify or inline a helper | Its name adds no meaning and following it costs more than reading the operation locally. | No current class clearly needs demotion. Do not create one wrapper per calculation or remove useful source-navigation helpers solely because they add code. |

## Findings from the pre-change implementation

### Recoverable files and model cleanup

In [the GTEx builder](../Code/build_gtex_pilot.py), `source_table` writes the download directly to its final filename and subsequently reuses it based on existence. An interrupted download can leave a partial file that is then treated as available; a truncated CSV may either fail parsing or contain an incomplete but parseable table. `main` similarly writes the final pilot directly. This is a concrete interruption hazard, not evidence that the retained dataset is corrupt.

Use a temporary sibling file, validate the required content, then replace the destination after success. Check the expected tissue column names/order rather than relying only on `df.columns[9:39]`. A tiny interrupted-download fixture can verify that the previous complete file survives and an incomplete file is not accepted. No research download is needed for that check.

`fetch_regions` retries every `Exception`, including parsing and schema failures, although its comment promises transient-error retries. Limit retries to identified recoverable failures, respect the service's rate-limit response, and report the failed batch with its cause. [AWS's retry article](sources.md#aws-retries-and-operation-identity) supplies the distinction between retryable operations and uncertain effects; it does not justify adding a distributed idempotency service to this script.

In [embedding extraction](../Code/shared_code/embeddings.py), `attention_profiles` already releases its model in `finally`, whereas `embed_with_cache` and `embed_layers` release it only after successful work. Apply the existing cleanup pattern to those operations. This reduces avoidable resource retention after errors; it cannot promise immediate release of every tensor still referenced by an exception or notebook state.

### Cache identity must describe the actual computation

`embed_with_cache` fingerprints the supplied strings before applying `encoder_input`, and its metadata does not include the `prepared` mode. The same supplied strings can mean different model inputs under the two supported modes while still matching the recorded cache identity. Input-conversion or pooling changes are also absent from that identity. This is an API-level weakness; it does not establish that either notebook's retained results used a wrong cache.

Prefer fingerprinting the actual prepared model-input strings and recording a small explicit embedding-format/preprocessing version when behavior changes. Preserve checkpoint revision and token-limit checks, and validate cached matrix shape. Close the loaded archive deterministically and publish replacement caches through a temporary sibling file so an interrupted write does not destroy the previous complete cache. Report damaged or incompatible caches clearly before an expensive authorized recomputation. Do not build a general cache registry. Keep cache evidence distinct from the [producing run record](run-records.md).

A controlled reproduction used the actual cache function with a substitute embedding operation: `ATGGCT` became the two-residue protein `MA` in conversion mode, but was a six-residue protein string in prepared mode. The second call reused the first result. Current stability and GTEx callers use different cache directories, so the demonstrated defect is not evidence of a collision in their retained outputs. Unpinned checkpoint revisions and unrecorded historical pooling changes remain separate limitations.

### Make seed roles visible

In [analysis.py](../Code/shared_code/analysis.py), `probe_scores` accepts a seed, but `probe_table` and `pairwise_concatenation_scores` rely on its pinned default `42`. Repeated-assignment means deliberately use seeds `0` through `n_assignments - 1`. Notebook `SEED` also controls other operations, including sampling and CCA. Changing that visible setting therefore does not change every partition.

Name these roles explicitly and pass the pinned probe seed through the relevant helpers, preserving today's defaults. Verify with synthetic scalar and multioutput targets that the default computation stays the same and a nondefault probe seed reaches both single and paired probes. Changing the actual partitions is a new experimental configuration; it should not slip into a readability refactor.

### Share the ridge definition without hiding the experiment

`probe_scores` and `fit_predict` separately construct the standardized `RidgeCV` pipeline, while `fit_predict` explicitly promises the same probe. A private function returning a fresh pipeline is a useful extraction because the two callers own the same modeling rule. Keep cross-validation, scoring, classification, and published-split evaluation in their respective callers. Preserve access to the construction through the existing implementation-display route.

Compare predictions and scores with the prior construction on small scalar and 30-output examples. Preserve the penalty grid and preprocessing exactly. The [documented inner-validation limitation](methods/linear-probing.md#evaluation-and-interpretation) requires its own methodological decision and evaluation; correcting it would be more than extraction.

### Fail clearly for unsupported scientific settings

`_strata` recognizes `"position"` and treats every other string as codon grouping. A typo can therefore select a different permutation null. Validate the two supported values at the public `motif_permutation_test` boundary before doing work. Check that valid seeded examples remain unchanged and an invalid value raises a useful error. Keep the permutation calculation together unless a named intermediate operation actually helps understanding.

In [datasets.py](../Code/shared_code/datasets.py), `audit` assumes a scalar `label_column`, whereas GTEx has multiple `label_columns` and an empty scalar name. Current notebooks avoid this path for GTEx. Narrow and enforce the audit's present scope rather than silently inventing a multioutput summary. Also correct the module introduction's future-tense description of the distinct-input setting, which already exists.

### Keep direction distinct from a symmetric display

`pairwise_matrix` evaluates each pair once and mirrors the result. This is natural for CKA; `cca_retrieval` queries the second representation with the first, so its recall is directional. The [method guide already states the direction and mirrored display](methods/representation-comparison.md#canonical-correlation-and-cross-modal-retrieval). The improvement is to expose this assumption at the helper and output too, not to claim the behavior is undocumented or the saved values are wrong.

Clarify the helper name/contract and retrieval labels. A tiny asymmetric callback can check which direction gets evaluated. Computing both directions or averaging them would change the reported experiment and needs a separate result-producing change.

### A useful boundary inside the GTEx builder

The strand-aware TSS/window conversion in `main` has an independently checkable meaning. Extract a function accepting coordinates, strand, and window size and returning the requested regions. Keep `main` readable as select rows, construct regions, fetch sequences, and save the pilot. Use hand-worked plus- and minus-strand examples to check coordinate conventions, window length, and TSS placement. Do not change filtering, sampling, boundary handling, or the biological window while moving the code.

### What should remain simple

Keep pure numerical functions, the encoder registry, dataset records, the short grouped-split functions, and the two experiment notebooks. Similar published-split evaluation code need not be unified when its metrics differ. Model compatibility code has current consumers; its complexity is not automatically waste. Existing synthetic regression checks and `show_source` support the reader's route and should not be removed just to reduce file count.

The largest risks found are incomplete contracts and failure handling, plus a few opportunities to share a real rule. A new class hierarchy, universal experiment runner, plugin system, or workflow engine would add structure without a demonstrated present need. Splitting plotting into another module can wait until navigation or independent reuse makes it worthwhile.

## What research software engineering adds

Research software uses ordinary engineering principles but must also preserve the relationship between code and scientific claims. [Wilson et al. (2014)](sources.md#wilson-et-al-best-practices-for-scientific-computing) supports incremental changes and checks; [Wilson et al. (2017)](sources.md#wilson-et-al-good-enough-practices-in-scientific-computing) helps keep their support cost proportionate.

| Concern | Proposed practice here |
| --- | --- |
| Biological correspondence | Make clear what a row represents and preserve its identity across sequences, embeddings, labels, groups, and filters. Equal shapes do not prove matching examples. Start with boundary checks or existing IDs rather than a new data framework. |
| Evaluation independence | Specify the relevant unit of separation: sequence, gene, transcript, or another justified grouping. Preserve learned preprocessing and model-selection boundaries. [DOME](sources.md#dome-biological-machine-learning-validation) supplies questions, not one universal split. |
| No known exact answer | Combine tiny known-answer examples with mathematically required relationships. For nondegenerate matrices, jointly permuting matching rows should preserve linear CKA within numerical tolerance. This proposed check follows the inspected formula; it is an application of [metamorphic testing](sources.md#xie-et-al-testing-when-the-exact-answer-is-unknown), not a result established for this repo by that paper. |
| Reproducibility | Record inputs, code state, model versions, settings, environment, cache origin and observed outputs in the existing [run record](run-records.md). A seed alone is insufficient. Repeating folds on the same data measures sensitivity, not independent biological replication. [Sandve](sources.md#sandve-et-al-ten-simple-rules-for-reproducible-computational-research) and [the National Academies terminology](sources.md#national-academies-reproducibility-and-replicability) help distinguish these claims. |
| A readable scientific argument | Keep question, consequential choices, output, and interpretation in the notebook; share stable computations and link fuller explanations. [Rule et al.](sources.md#rule-et-al-writing-and-sharing-jupyter-analyses) supports this division without requiring extraction of every cell. |
| Preservation versus validation | Comparing old and new code detects changes; an independent example or property can detect an error both versions share. Neither proves a biological interpretation. Define numerical tolerances according to the operation and conditions rather than treating all floating-point differences as bugs. |

## Shared skill and how both agents use it

The maintained [engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md) contains the shared criteria and project contracts. A [thin Claude entry point](../.claude/skills/multimodal-bio-shared-engineering/SKILL.md) points to it. [AGENTS.md](../AGENTS.md#shared-engineering) routes executable-code tasks; `CLAUDE.md` imports that file. The [Codex](sources.md#codex-repository-skills) and [Claude](sources.md#claude-code-skills-and-name-conflicts) sources explain discovery, which still depends on the contributor's application and checkout access.

Apply it when designing, implementing, debugging, or reviewing executable research code. Markdown-only edits use the existing documentation and formatting workflows. It can be selected for relevant tasks or explicitly requested by repository path. It is not a scheduled process and does not authorize experiments or publication. Its behavioral evaluation is recorded above; keep future rule changes in the maintained skill rather than duplicating them here.

## Authorized implementation batches

The following records the authorized sequence and its original finish conditions. Current completion and evidence are in the status table above. Preserve the existing proposal/source changes and any later teammate edits. Each batch includes its affected tests, explanation, and source-display links; the final audit checks their integration rather than postponing those obligations.

### Batch 1: Establish the shared coding workflow

Recheck the checkout, current diff, affected callers, and test baseline before editing. Preserve the current notebook research outputs and scientific defaults as comparison evidence. Use the existing test runner; record any pre-existing failures separately from regressions introduced by this work.

Turn the proposed skill content into a compact candidate with no personal-machine dependencies. Evaluate it in an isolated workspace against a meaningful function extraction, invalid scientific option, interrupted download, requested biological split change, and Markdown-only edit. The evaluation should assess the agent's actual choices and scope, not merely whether it repeats the principles. Where useful, compare with the same task without the candidate to see what decision value it adds.

After correcting demonstrated problems, install the maintained `.agents/skills/multimodal-bio-shared-engineering/SKILL.md`, the thin Claude entry point, and one AGENTS route. Follow the existing import arrangement rather than duplicating instructions in CLAUDE.md. Keep sources in the existing index and move the current workflow out of proposal status once adopted; avoid retaining two competing maintained copies.

**Finish condition:** The skill has valid metadata and working repository links, representative evaluation supports its decisions, and local discovery is checked. Julian's fresh-session discovery remains explicitly unverified until his application reports the loaded path; that external check need not block the subsequent local batches. This is an instruction workflow, not scheduled automation.

### Batch 2: Correct small input and resource-handling gaps

First reject unsupported permutation-null names before computation, give the scalar-label dataset audit an explicit supported scope and clear error for multioutput use, and correct its outdated distinct-input description. Verify those changes before a separate cleanup pass applying the existing `try/finally` ownership pattern to the two embedding operations that currently skip cleanup after failure. Keep the two passes independently reviewable because they address different failure conditions.

**Finish condition:** Valid position/codon examples and scalar audit results are preserved, unsupported inputs fail clearly, and controlled inference errors trigger cleanup while preserving the original error. Check that no completed cache is published after failure. These tests use tiny fixtures and substitutes, not downloaded models; they do not claim all device memory is immediately freed.

### Batch 3: Give the ridge probe one maintained definition

Extract a private constructor returning a fresh standardized ridge pipeline for `probe_scores` and `fit_predict`. Preserve the penalty grid, scoring, cross-validation, classification branch, and each caller's role. Keep the implementation accessible through the existing source-display route.

**Finish condition:** Small scalar and 30-output examples produce equivalent predictions/fold scores within justified tolerances; separately constructed pipelines do not share fitted state. No inner-validation redesign is included.

### Batch 4: Make configuration and displayed meaning explicit

Distinguish sampling/CCA seeds, the pinned probe seed, and repeated-assignment seeds. Thread the pinned seed through single and paired probe callers while preserving today's effective values. A nondefault-seed fixture should prove that both sides use the requested partition. Preserve the documented assignment sequence and baseline correspondence.

Clarify that the pairwise helper mirrors one evaluated pair and that retrieval has a query direction. Prefer a clear contract and nearby labels over a public rename unless the rename demonstrably improves navigation. Update the affected notebook explanation and source-display cells. Keep saved scientific outputs and their historical settings; do not make them appear to come from a rerun.

**Finish condition:** Existing-default comparisons pass, a changed probe seed reaches all relevant callers, and an asymmetric callback establishes the helper's mirroring behavior. No reversed/averaged retrieval result is invented or calculated in this batch.

### Batch 5: Make embedding caches identifiable and recoverable

First establish and check the identity/recovery policy: fingerprint actual prepared model-input strings and introduce only the minimal explicit format/preprocessing version needed to distinguish changed computation. Retain checkpoint/revision/token-limit checks and validate required metadata and matrix dimensions. Then implement deterministic archive closure and temporary-sibling NPZ publication as a separately checked pass, so identity changes and interruption recovery can be reviewed independently.

Define the caller-visible outcomes for a valid hit, ordinary input mismatch, legacy format, and corrupt cache before implementation. Preserve the existing public return shape where practical. A legacy or damaged cache should produce a clear recovery instruction rather than silently initiating expensive work. Implement and test the policy without deleting or rewriting the team's real caches. Unresolved checkpoint versions remain an explicit limit; this batch cannot reconstruct historical model identity.

**Finish condition:** Fixtures cover valid reuse, changed effective input, the prepared-mode collision, legacy/missing metadata, corrupt archives, wrong row count, and interrupted replacement. A failed replacement preserves the previous complete cache. No model download or embedding recomputation occurs during these checks.

### Batch 6: Isolate and explain GTEx coordinate conversion

Extract the strand-aware TSS/window calculation into a function with explicit coordinate conventions. Keep the script readable as selecting rows, constructing regions, fetching sequences, and saving the pilot. Preserve filtering, sampling, column order, window size, and current chromosome-boundary behavior.

**Finish condition:** Hand-worked plus- and minus-strand examples verify region endpoints, requested length, and TSS placement; representative fixtures match the prior expressions. If a boundary case reveals a biological-method ambiguity, report it separately rather than silently changing the selected windows.

### Batch 7: Make GTEx preparation recover cleanly

Use temporary sibling files and replacement after success for source downloads and final CSV publication. Check required fields and expected tissue labels/order. Add an explicit network timeout, retry only defined transient conditions, validate response batches before merging them, and retain the final failure cause. Check the current service contract when selecting retry/rate-limit behavior.

**Finish condition:** Controlled cases cover interruption, temporary failure followed by success, permanent failure, missing regions, invalid response content, and failed output replacement. The previous complete file remains intact, incomplete data is not accepted as a completed download, and exhausted retries do not add another unnecessary delay. No live pilot rebuild or remote data request is part of verification.

### Batch 8: Audit the integration and refine the skill

Run the applicable existing and new checks together after the last executable change. Review notebook setup, configuration routes, input/target/group order, imports, and source-display destinations. For link refreshes, use an isolated setup namespace and regenerate only the affected `show_source` output content; do not save newly generated device/setup outputs. Enumerate those permitted display changes and confirm that scientific outputs, saved execution counts, unrelated metadata, and effective scientific defaults stayed intact. Use an independent review for the consequential failure and cache paths.

Refine the shared skill only where the actual batches revealed a transferable decision problem. Reconcile the affected README, method explanations, test guide, source index, AGENTS route, and Claude entry point with the final implementation. Remove superseded duplicate instructions where appropriate; keep useful proposal history clearly historical rather than as a second policy.

**Finish condition:** Relevant checks pass or each unresolved failure is clearly identified; local links and scope checks pass; the changed code remains traceable from its notebooks; and the final report distinguishes implemented, checked, unverified, and deferred work. Human readability and Julian's actual discovery are separate from automated checks. A live scientific run, if later requested, receives its own run record.

### Implementation boundaries

Work through the authorized batches sequentially. Delegate independent review or isolated fixture work where useful, with one owner for each edited file. Finish and inspect each batch before starting the next; do not create a commit or report document for every small change. Keep a concise completion status in this existing plan and explain meaningful outcomes in chat.

Routine implementation choices and passing checks do not require another approval after every batch. Stop the affected work if requirements conflict, a proposed correction would change the scientific method, or completion requires a real experiment, consequential overwrite, or other action outside the request. Continue independent authorized work while that issue is resolved. Commits, pushes, and publication follow the user's explicit scope.

Bidirectional retrieval, altered pooling, redesigned inner validation, and new biological split policies remain separate research decisions. Adding class hierarchies, an experiment framework, a cache service, or a broad folder reorganization is outside this plan unless later evidence establishes a concrete need.

This review used current source, documentation, and selected primary/author sources. Three isolated checks used extracted functions and controlled substitutes to demonstrate the prepared-mode cache collision, skipped cleanup after an inference error, and reuse of a partial source download. They loaded no models and made no network requests. No research notebooks were rerun, retained scientific results reproduced, or compatibility across devices established. That initial research pass changed only this proposal, the selected-source record, and documentation navigation; the implementation status above records subsequent work.
