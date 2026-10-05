---
name: multimodal-bio-markdown-formatting
description: Format shared multimodal-bio-fusion project Markdown, repository skill prose, and notebook Markdown cells using defined headings, separators, lists, citations, and LaTeX math. Use for presentation changes; exclude executable code and individual notes or reviews.
---

# Shared Markdown formatting

Make contributions fit the shared document's structure using the rules below and the scope in [AGENTS.md](../../../AGENTS.md). This repository file owns presentation conventions, not a contributor's research process or learning style.

## Select the applicable treatment

Read the requested file and the surrounding material before editing. Identify the reader's question, the parent section of the proposed contribution, and the task's change boundary. Choose the file format for its role: a standalone overview or synthesis belongs in Markdown; explanations accompanying executable analysis belong beside that analysis in the notebook. During an authorized conversion, preserve the explanation and repair its incoming links. Length or mathematical notation alone does not justify another file.

| Target | Applicable treatment |
| --- | --- |
| Shared documentation | Apply the heading, emphasis, list, table, and citation rules below. Let the document's purpose determine its outline: a README provides orientation and routes to material; a reference organizes lookup; an explanation develops a question. |
| Notebook Markdown cells or explanations moved from a notebook | Also read [Notebook presentation](references/notebooks.md) for computational units, notebook/guide links, separators, cell boundaries, and mathematics. Apply its reading-dependency guidance to a moved explanation, while letting the destination guide choose its own outline. Do not impose an equation-and-example sequence on every document. |
| Repository instructions and skill prose | Apply the common Markdown rules, preserving functional metadata, imports, paths, and examples. Formatting alone does not authorize changing triggers, scope, or behavior. |
| Executable code | Outside this skill. Code may be read to understand an explanation; code formatting and implementation are separate work. No external coding skill is a prerequisite. |

Individual notes and reviews retain their author's format under the [scope rules](../../../AGENTS.md#shared-documentation-and-sources). Use the [shared-documentation workflow](../multimodal-bio-shared-documentation/SKILL.md) for purpose, tone, attribution, and source retention; a small formatting correction needs only the applicable presentation rules.

## Define the outline before choosing heading levels

Heading levels express containment, not a preferred font size. Establish the document's top-level reader questions or workflow parts, then place each smaller topic beneath the question or part it develops.

| Role | Decision rule |
| --- | --- |
| Title: `#` | Names the subject and scope of the entire document or notebook. Use one title; every section belongs beneath it. |
| Section: `##` | Answers one top-level reader question or groups one workflow part identified in the outline. For example, a README's Setup section answers how to prepare the project. |
| Subsection: `###` | Addresses a named component of its parent section. For example, Imports belongs within notebook Setup. |
| Subdivision: `####` | Separates a further component readers need to locate independently within a subsection. For example, a particular baseline belongs within an analysis track. |
| Local label: bold text | Names the role of the following paragraph or short block within the same topic, such as **Interpretation.** It does not create another navigable topic. |

Choose the parent first and descend one level; do not skip levels to obtain a visual size. The same topic can be a title in its own file and a subdivision in a larger notebook. Additional depths follow the same containment rule only when the content has another actual grouping. Paragraph length alone does not justify a new heading.

Use a bold label when introducing a notation key, example, interpretation, or implementation connection within one explanation. Within one procedure or calculation, distinguish dependent intermediate operations with ordered steps or bold numbered labels. A new equation or operation alone does not justify a heading. Keep or add a heading when it identifies a distinct analysis or reference topic, or when the requested outline or an existing incoming link establishes a separate navigation target. Do not flatten useful headings merely because their topics depend on one another.

A heading's scope extends through the following content until the next heading of the same or a higher level, including across notebook cells. Check that subsequent explanation, examples, code, and outputs belong under it. A new cell or horizontal rule does not close that scope. Avoid a narrow heading that incorrectly places a whole procedure's example or implementation beneath only its final step.

## Use explicit layout rules

- Write each prose paragraph on one physical source line and let the viewer wrap it. Preserve blank lines between paragraphs and around headings, lists, tables, fenced blocks, and horizontal rules. Do not indent prose merely to adjust its appearance.
- Use ordinary Markdown. Do not add raw HTML or viewer-specific widgets for layout. Literal HTML being documented belongs in inline code or a fenced example.
- Use `-` bullets for unordered peers, such as symbols or assumptions. Use numbered items for steps whose order matters. Nest an item only when it is a detail of that specific parent; put explanations applying to the whole list outside it. Indent the child marker to the column where the parent's text begins.
- Use tables when rows are comparable instances and columns describe the same attributes, such as location/role, operation/result/shape, or alternative/limitation. Explain sequence or mechanism in prose or ordered steps instead of forcing it into a table.
- Use backticks for code identifiers, paths, and literal syntax. Use language-tagged code fences for multi-line source examples. Content inside a fence is an example, not part of the document's heading or divider structure.

Headings normally provide sufficient separation in ordinary documents. Add `---` only at a boundary between distinct workflow phases or analysis families identified in the outline. A phase changes what the reader is doing, such as preparing inputs versus interpreting results; an analysis family groups investigations of the same research question. Do not add a rule beneath every heading, around local labels, or at an editing-batch boundary. Put a visual rule on its own line with blank lines around it; the notebook reference defines its cell placement.

An opening YAML frontmatter block in a skill can also use `---`. Preserve its delimiters and fields as metadata; do not treat them as visual rules. Similarly, preserve literal headings, dollar signs, and separators inside syntax examples.

## Place citations beside their claims

Place a descriptive source link immediately after the supported sentence. A paragraph-end citation is suitable when it clearly supports the whole paragraph. When sentences use different sources or include our interpretation, attach the references to their particular claims and distinguish the interpretation.

For a sourced equation, identify the reference in the introducing or immediately following prose, including a page, section, figure, or equation number when available. Do not invent missing locators. Label learning links as further reading when they do not support the adjacent claim. Use the shared-documentation workflow to decide what belongs in the reusable source index; do not repeat its whole entry at each citation.

Use relative repository links for shared files and descriptive labels for external links. After changing a heading or its destination, check affected navigation against the intended concept or subsection, not only the existence of a target.

## Check the requested change

Existing formatting helps locate structure but does not replace the definitions above. Do not normalize untouched material merely because it differs; follow AGENTS.md for task exceptions and conflicts.

Check the outline, list nesting, and literal examples, plus frontmatter and routing when editing a skill. Follow the [documentation checks and reporting guidance](../../../AGENTS.md#checks-and-handoff) for links, change scope, and the handoff. When rendering matters, inspect the intended viewer if available and state any unverified behavior; source checks do not establish visual correctness.
