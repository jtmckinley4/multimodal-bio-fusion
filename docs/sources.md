# Selected sources and readings

Use these readings to revisit useful explanations, evidence, and guidance for the project. Start with the [paper catalog](../Papers/README.md) and [individual reviews](notes/Reviews/) for the biological literature. The catalog identifies retained files, versions, source links, and availability; this page records reading scope and relevance to the project. Add useful research readings here as they arise; group them by the question they help answer.

A citation that already serves a specific claim can stay in its owning page. This index gives reusable readings a route beyond that single use. Selection and annotation guidance lives in the [shared-documentation workflow](../.agents/skills/multimodal-bio-shared-documentation/SKILL.md).

## Writing for different reader needs

### Diataxis: Start here

[Diataxis in five minutes](https://diataxis.fr/start-here/), by Daniele Procida, distinguishes tutorials, how-to guides, reference, and explanation.

**Why retain it:** It helps decide whether a page should teach a concept, answer a lookup question, or guide an action. It informed the separation of README navigation from fuller explanations in the shared workflow.

**Reading scope and limits:** The introductory page was read. It offers a way to choose a document's purpose; it does not mandate this repository's folders, Markdown formatting, or scientific conclusions.

## Benchmarks and encoders for multimodal comparison

### IsoFormer dataset and model cards

[InstaDeepAI/multi_omics_transcript_expression](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression) and [InstaDeepAI/isoformer](https://huggingface.co/InstaDeepAI/isoformer) on Hugging Face.

**Why retain them:** The dataset supplies GTEx transcript expression with a genomic DNA sequence, the transcript, and the protein for each example, and a fixed gene-level split. It was retained as a candidate for Setting B, where DNA is a distinct input rather than a rewriting of the coding sequence. The current [GTEx experiment](../Code/Stage1_gtex.ipynb) uses a pilot of this dataset; the [dataset overview](../README.md#data) identifies the local file and its producer.

**Reading scope and limits:** The dataset and model cards were read through a summarizing fetch on September 29, 2026, which reported 82,205 train, 4,718 test, and 4,318 evaluation transcripts and protein-coding genes only. That source review did not read the cards in full, download data, or identify a license; it is not a description of the later pilot implementation. The current experiment records its fields and retained rows, but the earlier review did not resolve licensing.

### RNA-FM model card

[multimolecule/rnafm](https://huggingface.co/multimolecule/rnafm) on Hugging Face.

**Why retain it:** RNA-FM is BioLangFusion's RNA encoder. The card states that it was pretrained on non-coding RNA from RNAcentral and has 640-dimensional embeddings, which matters when it reads coding sequences in the stability benchmark. The same collection lists an mRNA-trained variant, mRNA-FM, as a candidate additional RNA encoder.

**Reading scope and limits:** The card was read through a summarizing fetch on September 29, 2026; maximum input length was not identified there. At that review, loading through `multimolecule` had not been tested. For the current loader and environment requirements, see the [encoder registry](../Code/mbf/encoders.py) and [notebook setup](../README.md#notebook-dependencies).

### CodonBERT repository

[Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT), including its [embedding-extraction script](https://github.com/Sanofi-Public/CodonBERT/blob/master/benchmarks/CodonBERT/extract_embed.py).

**Why retain it:** The repository is the source of the stability and mRFP CSVs and hosts the codon-level CodonBERT model, a candidate mRNA encoder. Its benchmark files are needed to resolve the dataset-version question recorded in the [stability dataset audit](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons).

**Reading scope and limits:** Only the search listing was seen on September 29, 2026. How weights are distributed and which data version BioLangFusion used remain unread leads.

## Organizing and checking research software

These sources explain practices relevant to separating experiment narratives from reusable computation and checking changes in small steps. The implemented structure is described in the [Stage 1 reading route](../README.md#stage-1-analysis), [shared package](../Code/mbf/), and [notebook checks](../tests/README.md). The readings support evaluating those choices; they do not establish that every recommendation was adopted or that a source caused an earlier decision.

For an introduction, read Wilson et al. (2017), then Wilson et al. (2014) for more detailed practices, and Sandve et al. for reproducibility. The nf-core and ROOT examples help assess future infrastructure as the project grows.

### Wilson et al.: Best Practices for Scientific Computing

[Wilson et al. (2014), Best Practices for Scientific Computing](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745), especially Box 1 and the sections on incremental changes, duplication, testing, and documentation.

**Why retain it:** It supports small refactoring steps, reusable computational functions, automated regression checks, and explanations of purpose. These practices help explain the current division between experiment notebooks, [shared calculations](../Code/mbf/analysis.py), [method guides](README.md#understand-the-methods), and [tests](../tests/README.md).

**Reading scope and limits:** The sections named above were read on October 5, 2026. This is supporting rationale for the present organization, not evidence that passing tests validates a biological interpretation or reproduces saved research results.

### Wilson et al.: Good enough practices in scientific computing

[Wilson et al. (2017), Good enough practices in scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510), especially Software, Collaboration, Project organization, and What we left out.

**Why retain it:** It supports reusable functions, explicit dependencies, useful project overviews, and simple example checks. Its discussion of adoption costs helps assess how much structure this project needs: continuous integration and coverage tools can burden newcomers before their benefits justify the work.

**Reading scope and limits:** The sections named above were read on October 5, 2026. They support keeping the [reading route](../README.md#stage-1-analysis) and [checks](../tests/README.md) proportionate. The paper does not prescribe this repository's exact folders or notebook boundaries, or require adding CI now.

### Sandve et al.: Ten Simple Rules for Reproducible Computational Research

[Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285), Rules 1–6.

**Why retain it:** These rules address result-producing steps, executable processing, software versions, script history, intermediate results, and random seeds. They provide a starting point for designing records that connect a result to its inputs and producing code. The [generated-file policy](../README.md#generated-files) and [embedding-cache explanation](methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse) describe relevant existing practices.

**Reading scope and limits:** Rules 1–6 were read on October 5, 2026. Complete run provenance remains a future design consideration; preserved outputs and regression checks do not establish that it exists. Recording a seed alone does not establish deterministic behavior across hardware and libraries.

### nf-core: Shared components and collaborative review

[nf-core, Contributing overview](https://nf-co.re/docs/contributing/overview), especially Pipelines, Components, Configs, Documentation, and Reviewing pull requests.

**Why retain it:** This is a concrete bioinformatics example of shared modules and subworkflows, environment-specific configuration, and collaborative review. It gives context for the use of reusable functions in [Code/mbf/](../Code/mbf/) while the notebooks retain their experiment-specific choices and interpretations.

**Reading scope and limits:** The sections named above were read on October 5, 2026. They provide an example, not a requirement to adopt Nextflow, nf-core governance, or its full pipeline architecture here.

### nf-core: Continuous integration testing

[nf-core, Continuous integration testing](https://nf-co.re/docs/specifications/pipelines/requirements/ci_testing).

**Why retain it:** The requirement describes automated CI using a small test dataset that exercises as much of a pipeline as possible. It is a useful reference if the team later decides to run the [existing local checks](../tests/README.md) automatically on proposed changes.

**Reading scope and limits:** The complete requirement text was read on October 5, 2026. These are nf-core's requirements, not this repository's policy. This notebook reorganization does not introduce CI, and small or synthetic tests do not validate scientific conclusions.

### nf-core: Docker Support

[nf-core, Docker Support](https://nf-co.re/docs/specifications/pipelines/requirements/docker).

**Why retain it:** The requirement describes bundled software and stable, pinned container versions rather than moving labels such as `latest`. It is a useful example when considering a portable replacement for the current [notebook environment setup](../README.md#notebook-dependencies).

**Reading scope and limits:** The complete requirement text and packaging note were read on October 5, 2026. Containerization is a future option, not an adopted part of this reorganization. A versioned container would not by itself identify datasets, model revisions, seeds, or the producing code state.

### ROOT: GitHub-based continuous integration

[Piparo et al. (2024), Thoroughly testing and integrating hundreds of Pull Requests per month: ROOT's new Cost-efficient and Feature Rich GitHub-based CI](https://indico.cern.ch/event/1338689/contributions/6010399/), CHEP 2024 contribution description.

**Why retain it:** The description presents ROOT's move from Jenkins to GitHub Actions and testing across operating systems. It answers the question of whether large research projects use established software engineering practices and offers a later comparison point for testing infrastructure.

**Reading scope and limits:** Only the contribution description was read on October 5, 2026; presentation slides were not read. It is an example, not an independently evaluated benchmark or a reason to transfer ROOT's infrastructure requirements to this project.

### Society of Research Software Engineering: History

[Society of Research Software Engineering, History](https://society-rse.org/about/history/).

**Why retain it:** The account traces the research software engineering community to a 2012 workshop, a UK association in 2013, and the Society in 2019. It provides background for the established role of software engineering in research.

**Reading scope and limits:** The History section was read on October 5, 2026. It offers historical context, not evidence for a particular modularization choice. Membership and group counts on the page should not be assumed current.

The remote pages in this section were checked on October 5, 2026 and were not archived.

## Mathematics in notebook Markdown

### Jupyter and MathJax: LaTeX syntax and delimiters

[Jupyter Notebook 7.0.2: Markdown cells, LaTeX equations](https://jupyter-notebook.readthedocs.io/en/v7.0.2/examples/Notebook/Working%20With%20Markdown%20Cells.html#latex-equations) describes mathematical expressions in Markdown cells, including single-dollar inline syntax. [MathJax 4.0: TeX and LaTeX math delimiters](https://docs.mathjax.org/en/latest/input/tex/delimiters.html) describes double-dollar display delimiters and the configuration dependence of single-dollar inline delimiters.

**Why retain them:** They explain the distinction between mathematical syntax and the delimiters used by a viewer. They support the syntax discussion in the [notebook presentation reference](../.agents/skills/multimodal-bio-markdown-formatting/references/notebooks.md). The choice to retain `$...$` inline and separate-line `$$` blocks follows the notebook's existing source and Chase's requested convention. The heading definitions, notation/interpretation sequence, and cell-placement rules were developed through project feedback; the vendor documentation does not establish those editorial choices.

**Reading scope and limits:** Jupyter's LaTeX-equations section and MathJax's delimiter page were read on September 25, 2026. Jupyter documents a configured application; single-dollar inline math is not a default of standalone MathJax. These references establish syntax capabilities, not identical rendering in every editor or proof that this checkout was visually tested. The remote pages were not archived.

## Working with repository-aware agents

### Claude Code: Repository instructions and memory

[How Claude remembers your project](https://code.claude.com/docs/en/memory), especially [file loading](https://code.claude.com/docs/en/memory#how-claude-md-files-load) and [sharing one instruction file](https://code.claude.com/docs/en/memory#share-one-file-with-other-coding-tools).

**Why retain it:** It explains startup and nested instruction loading, imports, and the distinction between project instructions and local auto memory. The import mechanism supports this repository's [CLAUDE.md](../CLAUDE.md), which loads [AGENTS.md](../AGENTS.md).

**Reading scope and limits:** The relevant loading, AGENTS.md, import, and memory sections were read. Direct AGENTS.md discovery depends on version and configuration; the explicit import avoids relying on that fallback. Windows symlink limitations also informed the choice of an ordinary file. Instructions guide behavior rather than enforce permissions.

### Claude Code: Skills and name conflicts

[Extend Claude with skills](https://code.claude.com/docs/en/skills), especially [skill locations](https://code.claude.com/docs/en/skills#choose-where-skills-load) and [same-name resolution](https://code.claude.com/docs/en/skills#resolve-skills-that-share-a-name).

**Why retain it:** It documents project skills under `.claude/skills/`, selection through descriptions, explicit invocation, and supporting-file links. It informed the small [Claude entry point](../.claude/skills/multimodal-bio-shared-documentation/SKILL.md) that reads the maintained repository workflow.

**Reading scope and limits:** Discovery, naming, invocation, and supporting-file sections were read. A personal skill can take precedence over a project skill with the same name, which motivates the distinctive project name and explicit path. The two-entry-point arrangement is this project's design, not a vendor-provided integration recipe.

### Claude Code: Repository access and the Desktop Code surface

[How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works#what-claude-can-access) and [Desktop shared configuration](https://code.claude.com/docs/en/desktop#shared-configuration).

**Why retain them:** They distinguish the model from the application that can search and read the checkout. They help a collaborator determine whether their Claude session can use the repository instructions and skills.

**Reading scope and limits:** Repository-access and shared-configuration sections were read. The Desktop Code surface shares configuration with the CLI; this does not establish equivalent discovery in an ordinary Claude chat.

### Codex: Repository instructions

[Custom instructions with AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md), especially instruction discovery and project instructions.

**Why retain it:** It explains how global and repository guidance is assembled. It supports using the root AGENTS.md as the shared starting point while keeping this project's writing conventions explicitly scoped.

**Reading scope and limits:** Discovery and project-layering sections were read. This is Codex behavior, not a universal precedence rule for every agent application.

### Codex: Repository skills

[Build skills](https://learn.chatgpt.com/docs/build-skills), especially local skill locations and invocation.

**Why retain it:** It documents `.agents/skills/` discovery and loading the full skill only when relevant. It informed the location of the [maintained workflow](../.agents/skills/multimodal-bio-shared-documentation/SKILL.md).

**Reading scope and limits:** Discovery, invocation, and local locations were read. Same-named skills are not merged, so a personal installation should not be assumed to replace or reproduce the repository workflow.

## Checking a teammate's setup

In a fresh session opened on this checkout, ask the agent to identify the repository instructions and the source path of the skill relevant to the task: shared documentation or Markdown formatting. In Claude Code, `/context` lists loaded memory files; `/multimodal-bio-shared-documentation` and `/multimodal-bio-markdown-formatting` explicitly invoke their respective entry points. If discovery differs, either workflow can still be read through its repository path when the application has file access.

The repository-agent references above were checked on September 25, 2026; the remote pages were not archived. File and link checks can validate this repository's structure. Actual skill discovery and behavior must also be checked in the teammate's application.
