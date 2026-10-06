---
name: multimodal-bio-shared-engineering
description: Design, implement, debug, or review executable research code in this repository with readable boundaries, explicit scientific assumptions, and proportionate checks. Use for Python and notebook code; leave Markdown-only changes to the documentation workflows.
---

# Shared research engineering

Make the path from biological question to inputs, calculation, evaluation, and interpretation easy to follow. Work within [AGENTS.md](../../../AGENTS.md); this workflow does not authorize experiments, model downloads, commits, or publication. This file owns the workflow; the Claude skill is an entry point and requires no personal skill installation.

## Establish the affected behavior

Read the current code, callers, tests, and relevant [method explanation](../../../docs/README.md#understand-the-methods). Distinguish a refactor from a correction to failure behavior, a changed experimental configuration, or a changed scientific method. Identify the observable finish condition and what must remain true. Keep those changes reviewable separately even when they share a file.

For relevant data boundaries, state what a row represents, how identities/order stay matched across modalities and targets, array dimensions, sequence/coordinate conventions, transformations, split groups, token limits, pooling, scoring, and seed roles. Apply only the assumptions involved in the task. Equal shapes do not establish biological correspondence. Reject an unsupported scientific option rather than silently selecting another method.

## Choose useful boundaries

Keep dataset/model choices, comparison settings, outputs, and interpretation visible in the experiment notebook. Extract a function when it names a coherent operation, shares a rule, or enables a meaningful independent check; one caller can be sufficient. Use a record for related named data and a class when persistent state or lifecycle justifies it. Avoid arbitrary line limits, one wrapper per action, and speculative frameworks.

Share code when consumers must follow the same rule, not merely because their text resembles each other. Preserve deliberate differences between datasets, controls, metrics, and published evaluation protocols. Construct fresh estimators when callers need independent fits; do not share fitted state accidentally. Judge simplification by reader effort and maintenance, not file/function counts. [Selected design sources](../../../docs/sources.md#choosing-code-boundaries) explain KISS/YAGNI, DRY, function boundaries, and SOLID's responsibility/interface questions.

## Handle resources and recoverable work

Separate calculation from external input/output or presentation when that boundary helps a current consumer or check. Release owned models and files on success and failure. Preserve the operation's original exception if cleanup also fails; otherwise report a cleanup failure. Reuse a cache only when its actual inputs and computation identity match; a producer's evidence differs from the environment consuming the cache. Report incompatible or damaged caches before expensive recovery.

For affected file producers, validate content before publishing a complete replacement and preserve the last completed artifact on failure. Retry only known recoverable operations with bounded attempts/time and a retained failure cause. Add coordination only for actual competing writers. A hash or file's existence alone does not prove completeness, correctness, or historical provenance.

## Verify the claim and retain the reading route

Use the [existing check procedures](../../../tests/README.md) and the smallest meaningful additions: independent small answers, necessary mathematical properties, comparisons with prior behavior, or controlled failure fixtures. Comparing old and new code can preserve an old defect; synthetic checks do not establish biological validity. Select tolerances from the operation and tested conditions. The [research-software readings](../../../docs/sources.md#organizing-and-checking-research-software) explain these distinctions.

During a non-result-producing notebook refactor, preserve scientific outputs, saved execution counts, and unrelated metadata. Refresh affected implementation-display content only, using an isolated setup namespace without saving new device/setup output. Do not run research bodies merely to update source links. Explain any newly exposed setting without implying old outputs used a new configuration.

For an authorized new scientific result, use the existing [run-record procedure](../../../docs/run-records.md). Update affected explanations, source influences, and navigation through the [shared-documentation workflow](../multimodal-bio-shared-documentation/SKILL.md), rather than creating another reporting system. State what changed, what was checked, and remaining limits. Resolve scientific or scope-changing uncertainty before dependent work; continue independent authorized work where possible.
