# Selected sources and readings

Use these readings to revisit useful explanations, evidence, and guidance for the project. Start with the [paper catalog](../Papers/README.md) and [individual reviews](notes/Reviews/) for the biological literature. The catalog identifies retained files, versions, source links, and availability; this page records reading scope and relevance to the project. Add useful research readings here as they arise; group them by the question they help answer.

A citation that already serves a specific claim can stay in its owning page. This index gives reusable readings a route beyond that single use. Selection and annotation guidance lives in the [shared-documentation workflow](../.agents/skills/multimodal-bio-shared-documentation/SKILL.md).

## Writing for different reader needs

### Diataxis: Start here

[Diataxis in five minutes](https://diataxis.fr/start-here/), by Daniele Procida, distinguishes tutorials, how-to guides, reference, and explanation.

**Why retain it:** It helps decide whether a page should teach a concept, answer a lookup question, or guide an action. It informed the separation of README navigation from fuller explanations in the [shared workflow](../.agents/skills/multimodal-bio-shared-documentation/SKILL.md#write-for-the-reader).

**Reading scope and limits:** The introductory page was read. It offers a way to choose a document's purpose; it does not mandate this repository's folders, Markdown formatting, or scientific conclusions.

### Pautasso and Purdue OWL: Writing a literature review

[Pautasso (2013), Ten Simple Rules for Writing a Literature Review](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003149), especially Rule 6, and [Purdue OWL, Writing a Literature Review](https://owl.purdue.edu/owl/research_and_citation/conducting_research/writing_a_literature_review.html).

**Why retain them:** They help move from explaining individual papers to comparing findings, methods, disagreements, and unanswered questions. Use them when developing the [continuing literature review](README.md#find-plans-and-literature).

**Reading scope and limits:** Pautasso's Rules 4–7 and Purdue's main guidance were checked during the October 5, 2026 review. Their use informs writing; it does not establish an adopted systematic-review protocol or replace evaluation of each scientific source.

### Google: Tone and tables

[Google's Voice and tone](https://developers.google.com/style/tone) and [Tables](https://developers.google.com/style/tables) provide technical-writing guidance.

**Why retain them:** They support clear, direct explanations and using tables for comparable information. They inform the [reader-focused workflow](../.agents/skills/multimodal-bio-shared-documentation/SKILL.md#write-for-the-reader) and its companion formatting guidance.

**Reading scope and limits:** Both pages were checked on October 5, 2026 after their use in September 25 documentation discussions. The project's tone, heading, and equation-order decisions also came from user feedback; these pages do not prescribe them all.

### MLA: Describing AI assistance

[MLA, Beyond Citation: Describing AI Use in Your Work](https://style.mla.org/describing-ai-use/).

**Why retain it:** It informed the distinction between citing underlying evidence and explaining an assistant's role in the [literature review](<Literature Review - Biological Foundation Models.docx>), under Preparation and AI Assistance.

**Reading scope and limits:** The editorial guidance was checked on October 5, 2026. It is writing guidance, not biological evidence or a statement of the professor's policy. Its relevance belongs beside the disclosure, separate from the scientific Works Cited.

## Planning research work

### Research proposal and Gantt examples

[Mélot-Chesnel's research proposal, section 4 and Gantt figure](https://journals.sagepub.com/doi/10.3233/FAIA260562), and [UQ's AGRC7617 proposal assignment](https://course-profiles.uq.edu.au/course-profiles/AGRC7617-62151-7660).

**Why retain them:** These examples informed the September 27 discussion of scheduling research, interpretation, and revision. Their local use is explained in the [Gantt workbook's Details planning notes](Research_Project_Plan_Gantt.xlsx) and the [presentation's speaker notes](Research_Project_Plan_Gantt.pptx).

**Reading scope and limits:** Section 4 and the figure caption, plus UQ's proposal/Gantt and AI-feedback passage, were rechecked on October 5, 2026. These examples do not establish CAP6942 deadlines or measured AI time savings; the team's schedule remains its own plan.

## Benchmarks and encoders for multimodal comparison

### IsoFormer dataset and model cards

[InstaDeepAI/multi_omics_transcript_expression](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression) and [InstaDeepAI/isoformer](https://huggingface.co/InstaDeepAI/isoformer) on Hugging Face.

**Why retain them:** The dataset supplies GTEx transcript expression with a genomic DNA sequence, the transcript, and the protein for each example, and a fixed gene-level split. It was retained as a candidate for Setting B, where DNA is a distinct input rather than a rewriting of the coding sequence. The current [GTEx experiment](../Code/Stage1_gtex.ipynb) uses a pilot of this dataset; the [dataset overview](../README.md#data) identifies the local file and its producer.

**Reading scope and limits:** The dataset and model cards were read through a summarizing fetch on September 29, 2026, which reported 82,205 train, 4,718 test, and 4,318 evaluation transcripts and protein-coding genes only. That source review did not read the cards in full, download data, or identify a license; it is not a description of the later pilot implementation. The current experiment records its fields and retained rows, but the earlier review did not resolve licensing.

### Ensembl sequence requests and rate limits

[POST sequence/region documentation](https://rest.ensembl.org/documentation/info/sequence_region_post) and [Ensembl's rate-limit guidance](https://github.com/Ensembl/ensembl-rest/wiki/Rate-Limits).

**Why retain them:** They define the service contract used by [the GTEx pilot builder](../Code/build_gtex_pilot.py). The endpoint accepts at most 50 regions per POST. A 429 response supplies `Retry-After` as a floating-point number of seconds. These support batch sizing and respecting the requested delay; the builder's attempt limit and decision to stop for a delay above 120 seconds are local recovery choices, not service requirements. See [rebuilding the pilot](../README.md#data) for its role.

**Reading scope and limits:** On October 5, 2026, read the endpoint parameters, request examples and resource limits, and the wiki's normal/exhausted-rate-limit response sections. No live sequence requests or dataset rebuilds were performed. These references do not verify the historical pilot's completeness, assembly, or biological correctness; they do not pin future service contents.

### RNA-FM model card

[multimolecule/rnafm](https://huggingface.co/multimolecule/rnafm) on Hugging Face.

**Why retain it:** RNA-FM is BioLangFusion's RNA encoder. The card states that it was pretrained on non-coding RNA from RNAcentral and has 640-dimensional embeddings, which matters when it reads coding sequences in the stability benchmark. The same collection lists an mRNA-trained variant, mRNA-FM, as a candidate additional RNA encoder.

**Reading scope and limits:** The card was read through a summarizing fetch on September 29, 2026; maximum input length was not identified there. At that review, loading through `multimolecule` had not been tested. For the current loader and environment requirements, see the [encoder registry](../Code/mbf/encoders.py) and [notebook setup](../README.md#notebook-dependencies).

### CodonBERT repository

[Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT), including its [embedding-extraction script](https://github.com/Sanofi-Public/CodonBERT/blob/master/benchmarks/CodonBERT/extract_embed.py).

**Why retain it:** The repository is the source of the stability and mRFP CSVs and hosts the codon-level CodonBERT model, a candidate mRNA encoder. Its benchmark files are needed to resolve the dataset-version question recorded in the [stability dataset audit](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons).

**Reading scope and limits:** Only the search listing was seen on September 29, 2026. How weights are distributed and which data version BioLangFusion used remain unread leads.

### Medina-Munoz et al.: Stability assay provenance lead

[Medina-Muñoz et al. (2021), Crosstalk between codon optimality and cis-regulatory elements dictates mRNA stability](https://link.springer.com/article/10.1186/s13059-020-02251-5), Methods, especially Estimation of mRNA stability and Data allocation.

**Why retain it:** The iCodon citation provides a lead for tracing the assay behind the stability benchmark. Use it alongside CodonBERT and the [dataset audit](../Code/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) to investigate how a measured stability value became a local label.

**Reading scope and limits:** Selected Methods and the iCodon citation were checked in the October 5, 2026 source review. The local CSV's units, normalization, and row mapping remain unresolved. The publication year is 2021 despite the DOI's 2020 component; this is distinct from Agarwal and Kelley (2022).

## Evaluating biological predictions

### DOME: Biological machine-learning validation

[Walsh et al. (2021), DOME recommendations for supervised machine learning validation in biology](https://www.nature.com/articles/s41592-021-01205-4), with its [author correction](https://www.nature.com/articles/s41592-021-01304-2).

**Why retain it:** Box 1 and Table 1 help examine data independence, preprocessing, baselines, model selection, and reporting in the [linear-probing evaluation](methods/linear-probing.md#evaluation-and-interpretation). This is evaluation guidance, not a claim of DOME compliance.

**Reading scope and limits:** The October 5, 2026 review read the introduction, data/split discussion, Box 1, and Table 1 in the [author-hosted PDF](https://www.biofold.org/pages/documents/papers/walsh_nmeth2021.pdf), which carries a placeholder publication date. The publisher's correction notice identifies a corrected specificity equation in Figure 2; that notice was checked through search when direct retrieval failed. Consult the corrected publication before reusing that equation.

**Coding-review follow-up:** The October 5 coding review additionally read the Optimization discussion for the [proposed data and evaluation contracts](shared-coding-proposal.md#what-research-software-engineering-adds). No corrected specificity equation was used.

## Organizing and checking research software

These sources explain practices relevant to separating experiment narratives from reusable computation and checking changes in small steps. The implemented structure is described in the [Stage 1 reading route](../README.md#stage-1-analysis), [shared package](../Code/mbf/), and [notebook checks](../tests/README.md). The readings support evaluating those choices; they do not establish that every recommendation was adopted or that a source caused an earlier decision.

For an introduction, read Wilson et al. (2017), then Wilson et al. (2014) for more detailed practices, and Sandve et al. for reproducibility. The nf-core and ROOT examples help assess future infrastructure as the project grows.

### Wilson et al.: Best Practices for Scientific Computing

[Wilson et al. (2014), Best Practices for Scientific Computing](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745), especially Box 1 and the sections on incremental changes, duplication, testing, and documentation.

**Why retain it:** It supports small refactoring steps, reusable computational functions, automated regression checks, and explanations of purpose. These practices help explain the current division between experiment notebooks, [shared calculations](../Code/mbf/analysis.py), [method guides](README.md#understand-the-methods), and [tests](../tests/README.md).

**Reading scope and limits:** The sections named above were read on October 5, 2026. This is supporting rationale for the present organization, not evidence that passing tests validates a biological interpretation or reproduces saved research results.

**Coding-review follow-up:** The October 5 coding review also read Make Incremental Changes, Plan for Mistakes, Optimize Software Only after It Works Correctly, and Document Design and Purpose, Not Mechanics for the [coding proposal](shared-coding-proposal.md#what-research-software-engineering-adds).

### Wilson et al.: Good enough practices in scientific computing

[Wilson et al. (2017), Good enough practices in scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510), especially Software, Collaboration, Project organization, and What we left out.

**Why retain it:** It supports reusable functions, explicit dependencies, useful project overviews, and simple example checks. Its discussion of adoption costs helps assess how much structure this project needs: continuous integration and coverage tools can burden newcomers before their benefits justify the work.

**Reading scope and limits:** The sections named above were read on October 5, 2026. They support keeping the [reading route](../README.md#stage-1-analysis) and [checks](../tests/README.md) proportionate. The paper does not prescribe this repository's exact folders or notebook boundaries, or require adding CI now.

**Coding-review follow-up:** The October 5 coding review revisited Software (especially 2g–2i), Collaboration, and What we left out for the [proposed proportionate coding workflow](shared-coding-proposal.md#shared-skill-and-how-both-agents-use-it).

### Sandve et al.: Ten Simple Rules for Reproducible Computational Research

[Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285), Rules 1–6.

**Why retain it:** These rules address result-producing steps, executable processing, software versions, script history, intermediate results, and random seeds. They provide a starting point for designing records that connect a result to its inputs and producing code. The [generated-file policy](../README.md#generated-files) and [embedding-cache explanation](methods/inputs-and-embeddings.md#embedding-matrices-and-cache-reuse) describe relevant existing practices.

**Reading scope and limits:** Rules 1–6 were read on October 5, 2026. Complete run provenance remains a future design consideration; preserved outputs and regression checks do not establish that it exists. Recording a seed alone does not establish deterministic behavior across hardware and libraries.

**Coding-review follow-up:** The later October 5 coding review read Rules 1–3 and 5–9 for the [coding proposal](shared-coding-proposal.md#what-research-software-engineering-adds). The current [manual run-record procedure](run-records.md) is now available; its existence does not establish complete historical provenance.

### Rule et al.: Writing and sharing Jupyter analyses

[Rule et al. (2019), Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007007), Rules 1–3 and 7.

**Why retain it:** Narrative, meaningful divisions, and reusable computation informed the September 25 notebook discussions. The [notebook presentation reference](../.agents/skills/multimodal-bio-markdown-formatting/references/notebooks.md#organize-around-computational-units) applies those ideas to purposes, inputs, outputs, and links to method explanations.

**Reading scope and limits:** The introduction and named rules were checked on October 5, 2026. They support evaluating the organization, not this repository's exact folder names, heading levels, or approval batches.

**Coding-review follow-up:** The October 5 coding review revisited the introduction and Rules 1–3 and 6–8 for the [proposed notebook/computation boundary](shared-coding-proposal.md#what-research-software-engineering-adds).

### Xie et al.: Testing when the exact answer is unknown

[Xie et al. (2011), Testing and validating machine learning classifiers by metamorphic testing](https://www.cs.columbia.edu/wp-content/uploads/sites/7/2016/08/jss2011.pdf), Journal of Systems and Software 84, 544–558; DOI 10.1016/j.jss.2010.11.920.

**Why retain it:** It helps distinguish required mathematical relationships from plausible statistical expectations when exact expected outputs are unavailable. It informs the [proposed scientific checks](shared-coding-proposal.md#what-research-software-engineering-adds); jointly permuting paired rows in CKA is our local application, not an experiment in this paper.

**Reading scope and limits:** On October 5, 2026, the research pass read the abstract/introduction, sections 2.3, 3.1–3.2, and 4.3.1; case-study details were sampled. The classifier study does not validate this repository's metrics. A surprising result is not automatically a coding fault unless the violated relation is a necessary property. The remote PDF was not archived.

### National Academies: Reproducibility and replicability

[National Academies of Sciences, Engineering, and Medicine (2019), Reproducibility and Replicability in Science, chapter 3](https://www.nationalacademies.org/read/25303/chapter/6), especially pages 43–47 and Conclusion 3-1.

**Why retain it:** Its convention distinguishes computational consistency with the same data/code/conditions from a new study using newly obtained data. This informs the [proposal's evidence distinctions](shared-coding-proposal.md#what-research-software-engineering-adds) and interpretation of [run records](run-records.md). Repeated folds are not new biological observations.

**Reading scope and limits:** The terminology discussion and Conclusion 3-1 were read on October 5, 2026. Terminology differs across communities; these definitions should be stated when needed, not assumed universal. Reproducing an output does not establish the method's validity. The remote chapter was not archived.


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

## Choosing code boundaries

These sources inform the [coding review](shared-coding-proposal.md) and the adopted [shared engineering workflow](../.agents/skills/multimodal-bio-shared-engineering/SKILL.md). The proposal adapts Chase's personal Shared Engineering criteria for this research repository without requiring that personal installation. The sources below were read on October 5, 2026; remote pages were not archived.

### Fowler: YAGNI

[Martin Fowler, Yagni](https://martinfowler.com/bliki/Yagni.html), May 26, 2015.

**Use and scope:** Read the article's discussion of speculative features, abstraction costs, refactoring, and inexpensive preparation. It informs the proposal to defer an experiment framework while allowing changes that improve current readability. It is a design argument, not proof that all preparation is wasteful or a reason to omit present reliability needs.

### Thomas and Hunt: Duplication of knowledge

[Dave Thomas and Andy Hunt, The Pragmatic Programmer: The Evils of Duplication](https://media.pragprog.com/titles/tpp20/dry.pdf), 20th Anniversary Edition excerpt.

**Use and scope:** Read “DRY is More Than Code” and “Not All Code Duplication is Knowledge Duplication,” PDF pages 7 and 9–10. These distinguish shared intent from matching text. They inform sharing the ridge definition while preserving different dataset evaluations in the [proposal](shared-coding-proposal.md#share-the-ridge-definition-without-hiding-the-experiment). They do not require eliminating independent test expectations.

### Martin: SOLID as responsibility and interface guidance

[Robert C. Martin, Solid Relevance](https://blog.cleancoder.com/uncle-bob/2020/10/18/Solid-Relevance.html), October 18, 2020.

**Use and scope:** Read the five principle discussions. Responsibility, substitution, and dependency boundaries inform the proposal's separation of calculations from external operations. Applying those questions to functions and modules is our project interpretation; the article does not prescribe Python class hierarchies or this repository's layout.

### C++ Core Guidelines: Meaningful functions and related data

[C++ Core Guidelines, F.1–F.2](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#f1-package-meaningful-operations-as-carefully-named-functions) and [C.1–C.2](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#c1-organize-related-data-into-structures-structs-or-classes).

**Use and scope:** Read those sections' rationales and examples. They inform the [function/record/class decision table](shared-coding-proposal.md#choosing-between-inline-code-a-function-and-a-class): name meaningful operations and represent genuinely related state. These are C++ guidelines; its class/struct distinction and language mechanics are not Python requirements. No numerical function-length limit is adopted.

### Google: Reviewing complexity and useful tests

[Google Engineering Practices, What to look for in a code review](https://google.github.io/eng-practices/review/reviewer/looking-for.html#complexity).

**Use and scope:** Read Design, Complexity, Tests, Naming, Comments, and Context. Reader comprehension, present needs, and tests that can detect realistic failures inform the [proposal's review criteria](shared-coding-proposal.md#what-the-shared-skill-should-accomplish). This is Google review guidance, not a mandate to import all its process or infrastructure.

### AWS: Retries and operation identity

[Malcolm Featonby, Making retries safe with idempotent APIs](https://aws.amazon.com/builders-library/making-retries-safe-with-idempotent-APIs/), AWS Builders' Library.

**Use and scope:** Read Retrying and side effects, Reducing client complexity, Late arriving requests, Same client request ID/different intent, and the conclusion. It informs the [builder recovery proposal](shared-coding-proposal.md#recoverable-files-and-model-cleanup), especially preserving operation intent and distinguishing retryable failures. The article concerns service contracts; it neither establishes Ensembl's API behavior nor calls for distributed coordination here. Publication date was not established.


## Mathematics in notebook Markdown

### Jupyter and MathJax: LaTeX syntax and delimiters

[Jupyter Notebook 7.0.2: Markdown cells, LaTeX equations](https://jupyter-notebook.readthedocs.io/en/v7.0.2/examples/Notebook/Working%20With%20Markdown%20Cells.html#latex-equations) describes mathematical expressions in Markdown cells, including single-dollar inline syntax. [MathJax 4.0: TeX and LaTeX math delimiters](https://docs.mathjax.org/en/latest/input/tex/delimiters.html) describes double-dollar display delimiters and the configuration dependence of single-dollar inline delimiters.

**Why retain them:** They explain the distinction between mathematical syntax and the delimiters used by a viewer. They support the syntax discussion in the [notebook presentation reference](../.agents/skills/multimodal-bio-markdown-formatting/references/notebooks.md). The choice to retain `$...$` inline and separate-line `$$` blocks follows the notebook's existing source and Chase's requested convention. The heading definitions, notation/interpretation sequence, and cell-placement rules were developed through project feedback; the vendor documentation does not establish those editorial choices.

**Reading scope and limits:** Jupyter's LaTeX-equations section and MathJax's delimiter page were read on September 25, 2026. Jupyter documents a configured application; single-dollar inline math is not a default of standalone MathJax. These references establish syntax capabilities, not identical rendering in every editor or proof that this checkout was visually tested. The remote pages were not archived.

### Knuth, Larrabee and Roberts: Mathematical Writing

[Mathematical Writing](https://cs.stanford.edu/~knuth/klr.html) is a reading route for explaining mathematics in prose.

**Why retain it:** It was raised in September 25 discussions of mathematical explanations. It offers further study alongside the [notebook reading-dependency guidance](../.agents/skills/multimodal-bio-markdown-formatting/references/notebooks.md#order-mathematics-by-reading-dependencies).

**Reading scope and limits:** The publication page was checked on October 5, 2026; the full text was not reread. The project's notation, interpretation, and worked-example sequence reflects user feedback and is not presented as a prescription from this book.

## Working with repository-aware agents

### Claude Code: Repository instructions and memory

[How Claude remembers your project](https://code.claude.com/docs/en/memory), especially [file loading](https://code.claude.com/docs/en/memory#how-claude-md-files-load) and [sharing one instruction file](https://code.claude.com/docs/en/memory#share-one-file-with-other-coding-tools).

**Why retain it:** It explains startup and nested instruction loading, imports, and the distinction between project instructions and local auto memory. The import mechanism supports this repository's [CLAUDE.md](../CLAUDE.md), which loads [AGENTS.md](../AGENTS.md).

**Reading scope and limits:** The relevant loading, AGENTS.md, import, and memory sections were read. Direct AGENTS.md discovery depends on version and configuration; the explicit import avoids relying on that fallback. Windows symlink limitations also informed the choice of an ordinary file. Instructions guide behavior rather than enforce permissions.

### Claude Code: Skills and name conflicts

[Extend Claude with skills](https://code.claude.com/docs/en/skills), especially [skill locations](https://code.claude.com/docs/en/skills#choose-where-skills-load) and [same-name resolution](https://code.claude.com/docs/en/skills#resolve-skills-that-share-a-name).

**Why retain it:** It documents project skills under `.claude/skills/`, selection through descriptions, explicit invocation, and supporting-file links. It informed the small [Claude entry point](../.claude/skills/multimodal-bio-shared-documentation/SKILL.md) that reads the maintained repository workflow.

**Reading scope and limits:** Discovery, naming, invocation, and supporting-file sections were read. A personal skill can take precedence over a project skill with the same name, which motivates the distinctive project name and explicit path. The two-entry-point arrangement is this project's design, not a vendor-provided integration recipe.

**Coding-review follow-up:** Project skill locations, description-based selection, explicit invocation, supporting files, and name precedence were rechecked on October 5, 2026 for the [shared coding proposal](shared-coding-proposal.md#shared-skill-and-how-both-agents-use-it). Julian's actual application discovery was not tested.

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

**Coding-review follow-up:** Discovery and explicit/implicit invocation were rechecked on October 5, 2026 for the [proposed shared coding skill](shared-coding-proposal.md#shared-skill-and-how-both-agents-use-it). That initial research pass did not install an entry point; the subsequently authorized implementation added the shared workflow and thin Claude route.

## Checking a teammate's setup

In a fresh session opened on this checkout, ask the agent to identify the repository instructions and the source path of the skill relevant to the task: shared engineering for executable code, shared documentation for purpose and evidence, or Markdown formatting for presentation. In Claude Code, `/context` lists loaded memory files; `/multimodal-bio-shared-engineering`, `/multimodal-bio-shared-documentation`, and `/multimodal-bio-markdown-formatting` explicitly invoke their respective entry points. If discovery differs, the relevant workflow can still be read through its repository path when the application has file access.

The repository-agent references above were checked on September 25, 2026; the remote pages were not archived. File and link checks can validate this repository's structure. Actual skill discovery and behavior must also be checked in the teammate's application.
