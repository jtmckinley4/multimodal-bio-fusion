---
name: multimodal-bio-shared-documentation
description: Maintain shared multimodal-bio-fusion project documents and retain reusable research sources. Use for shared documentation or source capture, not individual notes, reviews, or notebook/code changes.
---

# Shared research documentation

Use this workflow within the scope defined by [AGENTS.md](../../../AGENTS.md). Maintain shared explanations and useful source context so collaborators can continue without reconstructing a private chat. This repository file owns the workflow; the Claude skill is only an entry point.

## Choose the document and scope

Read the requested files and the relevant existing explanation. The [root README](../../../README.md) owns the project overview and setup; [docs/README.md](../../../docs/README.md) owns navigation; the [Stage 1 overview](../../../docs/stage1-overview.md) owns the shared model context, research questions, and cross-dataset synthesis; [agent-context.md](../../../docs/agent-context.md) points to research context and unresolved questions; [sources.md](../../../docs/sources.md) collects selected readings; the [paper catalog](../../../Papers/README.md) identifies retained files, versions, and source links. Use the [method guides](../../../docs/README.md#understand-the-methods) for shared explanations and [tests/README.md](../../../tests/README.md) for check procedures, linking to their maintained content rather than copying it into entry pages.

Use the repository's [Markdown-formatting workflow](../multimodal-bio-markdown-formatting/SKILL.md) for layout, equations, and citation placement. [AGENTS.md](../../../AGENTS.md#shared-documentation-and-sources) defines shared-file scope and the exception for individual notes and reviews. When drawing on those materials, preserve attribution and qualify the interpretation without rewriting the original. These shared workflows add no personal-study requirements.

## Write for the reader

- Lead with purpose and concrete meaning. README files provide orientation and navigation. Keep temporary assignments, approval discussion, and review summaries in the task conversation or an explicitly requested task record. Introduce the reading route before an extensive file inventory or results table. Explain unfamiliar file roles by what the reader uses them for, including whether a preparation step is needed to use an already supplied input.
- Match the form to the need: explanation develops understanding; reference supports lookup; instructions support an action. Do not force every page into one template.
- Keep one maintained owner for each shared explanation or rule and link to it. Brief definitions and reminders can stay where readers need them; an earlier mention or a link does not establish understanding. Explain what a concept means in the current analysis without copying the full guide.
- Place qualifications beside their claims. Separate published findings, contributor interpretations, notebook outputs, and accepted project decisions. A source can influence a proposal without proving it or making it an adopted requirement.

## Select sources worth retaining

Retain a source when it supports or challenges a consequential shared claim or choice, offers an explanation likely to be useful again, or is a promising lead with a specific unanswered question. Keep relevant contrary evidence. Do not equate frequency of retrieval, popularity, or model confidence with importance.

Skip duplicate links, incidental lookups, and search results with no identified future use. Prefer the original paper or official documentation for technical claims; an accessible tutorial may separately be worth keeping for learning. Label an unread lead instead of treating it as assessed evidence.

Use the source index for reusable readings that need a route beyond one paragraph; do not duplicate a complete record in multiple files. Update an existing entry when the same source is consulted again.

## Leave enough context to reuse the source

An entry normally needs a descriptive title and exact URL or section, why it is useful, what was actually read, and where it informed the work if applicable. Add author, revision or access date, and important limitations when needed to identify or interpret it. Keep reading status, the source's claim, and the team's decision distinct.

For example, a documentation primer may explain the distinction between reference and explanation. That can inform how a shared page is written; it does not establish a biological claim or require a particular directory structure.

Distinguish a source that supports an existing choice from one that actually informed it. Link the affected artifact or decision when relevant, and identify which recommendations are implemented and which remain options. Do not infer historical influence or team adoption from a later citation.

Keep a dated reading record separate from current project use. If implementation advances after a source review, preserve what was read and unresolved then, and link the current artifact separately. Updating that link does not establish a fresh reading of the external source or resolve its earlier uncertainties.

For a changed or rejected interpretation, preserve the relevant source and briefly record what changed and why. A URL and access date do not preserve the page itself. Link to an existing retained copy where available; do not bulk-copy articles or archive chats as a substitute for a useful annotation.

Do not convert an author's recommendation into a team decision or a contributor's private notes into shared consensus.

## Finish the authorized work

During authorized project research, capture qualifying sources and their annotations before finishing, even if the discussion began in chat. For read-only work or private notes, report suitable shared additions without writing them, following the scope boundaries in AGENTS.md.

After a reorganization, check entry pages and descriptions against the new file roles. A link may resolve while sending readers to an overview where an executable experiment used to be. Update affected reading and execution routes, result locations, and producer labels after their destinations exist.

Check supported claims and follow the [documentation checks and reporting guidance](../../../AGENTS.md#checks-and-handoff). A small edit needs only the relevant checks, not a separate report.
