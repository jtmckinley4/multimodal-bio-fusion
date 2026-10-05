# Notebook presentation

Use this reference when creating or revising Markdown cells in a shared project notebook, or moving an existing notebook explanation into a shared method guide. The [core formatting skill](../SKILL.md) owns common Markdown rules. This reference defines how explanations relate to computation and how mathematics is presented.

## Organize around computational units

A computational unit is one operation or analysis, potentially spanning several code cells, with identifiable inputs and an output or effect: loading an encoder, producing embeddings, fitting a probe, or calculating a comparison. Inspect the relevant code to identify that unit; a single code cell can contain more than one operation.

Place purpose, inputs, prerequisites, and the mathematics or explanation links needed to understand an operation before its code. Place interpretation of the observed or saved result after the relevant output when one exists. Introduce reused concepts near their first use and link to their maintained explanation from later uses. Overview and reference cells can stand separately when their links make the connection to computation clear.

A reading dependency is information needed to understand the next explanation or expression: a biological term, the objects being measured, or the result of an earlier calculation. A code dependency is a variable, function, or resource required for execution. Explain the relevant reading dependencies before using them; do not infer reader comprehension merely because the code can execute.

Use the core heading definitions across the entire notebook, not separately inside each cell. The [stability notebook](../../../../Code/Stage1_stability.ipynb#Representation-analysis) groups `## Representation analysis`, then `### Track A: Decodability`, then `#### Sequence-length baseline`. The [GTEx notebook](../../../../Code/Stage1_gtex.ipynb#GTEx-Track-A:-Expression-decodability) instead uses `## GTEx Track A: Expression decodability` with `### Length and composition baselines` beneath it. Both express the topic's relationship to its parent. Notation and interpretation can be local labels within an analysis. These are examples of containment, not a required set of tracks or heading depths.

For chained calculations within one analysis, apply the core skill's distinction between local steps and navigation topics. For example, centroid, per-row distance, and mean distance are steps of one spread calculation; numbered local labels keep its example and implementation under the same analysis. A reference organized to look up those concepts separately can retain their headings and incoming links.

## Connect notebooks and method guides

Use the [documentation index](../../../../docs/README.md) to find an explanation's maintained home. During an authorized extraction, keep the local question, inputs, experiment-specific choices, key assumptions, analysis call, and interpretation with the notebook. A short equation or definition can remain when readers need it to understand the operation. Move a long derivation, reusable worked example, or repeated background explanation into the relevant guide when that improves the reading route. Neither mathematical notation nor a position before Track A is by itself a reason to move material.

Link the relevant guide section beside each computational unit that needs it. An introductory guide link does not replace these links at the point of use. Repeated uses can have a short reminder and a direct link without repeating the derivation. For example, GTEx probes can link directly to the [linear-probing guide](../../../../docs/methods/linear-probing.md); readers need not pass through the stability notebook to find the shared method. Label guide return links with the experiment or section they open when more than one notebook uses the method.

When moving an explanation, keep its definitions, assumptions, worked examples, and supporting citations together so it remains intelligible at the destination. Preserve links needed for reading dependencies, and update affected notebook, guide, and index links within the authorized scope. Choose guide headings for lookup and understanding; they need not reproduce notebook heading depths. Current implementation displays and their compact/full modes are explained in [Reading implementation code](../../../../README.md#reading-implementation-code). Link to maintained code instead of copying function bodies into Markdown; extracting or changing executable code is outside this formatting workflow.

## Place separators and Markdown cell boundaries

Use a horizontal rule when the outline changes between workflow phases or analysis families. In the experiment notebooks, examples include Setup to data preparation, data preparation to encoders and embeddings, producing embeddings to analyzing them, and decodability to cross-modal geometry. A transition from method preparation to running that analysis can also warrant a rule. Moving from sequence-length prediction to GC-content prediction stays within one family and does not require a rule. Follow the actual workflow; the shared helper functions live in `Code/mbf/` and do not require an in-notebook definitions phase.

Place `---` at the beginning of the Markdown cell introducing the new phase or family, followed by a blank line and its heading. Do not place it between an equation and its explanation or between output and its result interpretation. Editing batches and visual boundaries are different decisions.

Begin a new explanatory unit when its question, operation, or result changes. Keep the equation, its notation definitions, and its interpretation in the same Markdown cell. Keep the introduction adjacent to the code it explains and the result interpretation adjacent to the relevant output. One explanatory unit may span Markdown, code, output, and another Markdown cell; adjacency does not mean merging those cell types.

For newly created or explicitly reorganized Markdown, split a long cell at a named subtopic that readers need to navigate to or edit separately, not at an arbitrary word count. Keep its prerequisites available through a local introduction or an exact earlier link. Put the heading with its opening explanation where practical. A cell boundary does not itself require a heading or rule.

Preserve executable cells, their order, saved outputs, and execution counts during formatting work. Preserve existing cell identities and metadata; split, merge, or move Markdown cells only when that restructuring is within the request. Do not run the notebook to validate prose or reconstruct outputs. A formatting task does not authorize changing scientific methods, comments/docstrings inside executable cells, or results.

## Write LaTeX mathematics in Markdown

Use LaTeX mathematical syntax with the notebook's existing dollar delimiters. Use `$...$` for symbols or short expressions within a sentence. Use displayed mathematics for a formula receiving its own explanation, derivation, or worked example: opening and closing `$$` each occupy a separate line, with blank lines separating the block from surrounding prose.

For example, the following fenced block shows the source syntax. In a notebook Markdown cell intended to render the formula, omit the code fence:

```markdown
$$
\mathrm{GC}(s) = \frac{N_G(s) + N_C(s)}{L(s)}.
$$
```

Do not replace these delimiters with alternative math environments solely for style. Backticks display literal source; they do not replace rendered mathematical notation. After programmatic edits, inspect the literal math source: an incorrectly escaped `\times` can become a tab followed by `imes`. Valid notebook JSON does not establish valid LaTeX. See the [math-formatting sources](../../../../docs/sources.md#mathematics-in-notebook-markdown) for the syntax support and its limits.

## Order mathematics by reading dependencies

For a calculation that needs explanation, use the following reading order. Labels identify the role of material within the same topic; they are not mandatory new headings.

1. State the purpose and inputs in prose before the equation: what question is being answered and which objects are measured.
2. Present the equation.
3. Follow it with **In this notation:** and bullets for newly introduced symbols and operators. Identify scalar/vector/matrix roles, dimensions, index meanings or ranges, and relevant earlier definitions when needed to read the expression. A brief reminder or exact link can stand in for repeating an established definition.
4. Add **Interpretation.** of the equation after the notation definitions. Explain what the operation does and what its result means in this analysis, with the assumptions or limitations needed to interpret it.
5. Add **Worked example.** when a concrete calculation improves understanding. Identify illustrative values as examples rather than observed notebook results.
6. Add **Connection to the code.** Identify the exact implementing operation or function. Distinguish a direct calculation, a library's computation or fitting objective, and a conceptual formula that is not executed here.

Use the parts that the calculation requires; do not invent an equation or example for a simple operation that prose already explains. When equations depend on one another, explain the earlier calculation before presenting the dependent one. For example, introduce and interpret the centroid before a distance calculation that uses it. Side-by-side expressions are suitable for comparing already explained alternatives, not for concealing a dependency chain.

Use notation bullets for symbol definitions. Use an operation/result/shape table when several code operations must be mapped to mathematical quantities; use prose for a single operation and a numbered list for an ordered procedure. Keep code identifiers in backticks and mathematical objects in math delimiters, explicitly connecting the two.

Preserving mathematical meaning means retaining the operation, assumptions, indices, dimensions, and relationship to the implemented calculation. It is separate from the placement rule: the plain-language interpretation follows the notation. If prose and code disagree, identify the discrepancy; do not silently change a formula, computation, or saved result as a formatting correction.

## Check the reading route and preserved state

Check both directions of affected notebook/guide links and confirm that each anchor opens the intended explanation, not merely an existing section. Read the local purpose, linked method, code call, output, and interpretation as a sequence. Check relocated mathematics for unchanged meaning and prerequisite definitions. Compare executable cells, saved outputs, execution counts, cell identities and order, and metadata against the pre-edit notebook, allowing only restructuring included in the request.

Use the [notebook checks](../../../../tests/README.md) when setup, implementation displays, or shared calculations change. Follow the core skill's [presentation checks](../SKILL.md#check-the-requested-change) for viewer inspection. Preserved outputs do not establish a fresh run after a notebook split or reorganization.
