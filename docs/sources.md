# Selected sources and readings

Use these readings to revisit useful explanations, evidence, and guidance for the project. Start with the [paper catalog](../Papers/README.md) and [individual reviews](notes/Reviews/) for the biological literature. The catalog identifies retained files, versions, source links, and availability; this page records reading scope and relevance to the project. Add useful research readings here as they arise; group them by the question they help answer.

A citation that already serves a specific claim can stay in its owning page. This index gives reusable readings a route beyond that single use. Selection and annotation guidance lives in the [shared-documentation workflow](../.agents/skills/multimodal-bio-shared-documentation/SKILL.md).

## Understanding the biology behind the methods

These textbooks connect the biological processes to the quantities used in the [method guides](README.md#understand-the-methods): what sequence regions contain, how RNA and protein amounts arise, and how molecular structure and cellular context can affect regulation. They support the explanations; the original dataset papers, releases and verification records continue to establish assay procedures, label transformations and benchmark provenance. The publisher links identify the editions and access routes. The selected passages were read in privately accessed copies; textbook files and figures are not redistributed in this repository.

### Essential Cell Biology, 6th edition

[Essential Cell Biology, sixth edition](https://wwnorton.co.uk/books/9781324033394-essential-cell-biology-c7a43b27-8186-4ee2-b83f-e60dd511e7ac), Bruce Alberts and colleagues, W. W. Norton (2023).

**Use here:** The accessible account of transcription, translation and regulation supports the [gene-expression overview](methods/data-and-targets.md#connecting-the-targets-to-gene-expression), the distinction between letter conversion and cellular transcription, and the [GC3](methods/sequences-to-predictions.md#gc3-prediction) and [synonymous-recoding](methods/sequences-to-predictions.md#pipeline-control-synonymous-recoding) explanations. It explains why an unchanged amino-acid sequence need not preserve all nucleotide-level information, without assigning a mechanism to the observed encoder rankings.

**Reading scope and limits — November 2, 2026:** Selected passages in chapter 7, “From DNA to RNA” and “From RNA to Protein,” covering DNA/RNA chemistry, the genetic code, tRNAs and wobble decoding; and chapter 8, “Post-Transcriptional Controls,” covering translation and RNA regulation. This is a targeted reading, not a full-book review. Chapter and subsection names are the locators because electronic and print pagination differ.

### Molecular Biology of the Cell, 7th edition

[Molecular Biology of the Cell, seventh edition](https://wwnorton.co.uk/books/9780393884852-molecular-biology-of-the-cell), Bruce Alberts and colleagues, W. W. Norton (2022).

**Use here:** The account of RNA processing, gene regulation and degradation supports the [target overview and stability explanation](methods/data-and-targets.md#connecting-the-targets-to-gene-expression), [sequence-region distinctions](methods/sequences-to-predictions.md#from-sequences-to-embeddings), and [motif interpretation](../Code/stage1/representation-analysis.md#scanning-candidate-rna-stability-associated-patterns). The fluorescent-protein discussion connects [reporter synthesis, maturation and measured signal](methods/data-and-targets.md#from-engineered-bacteria-to-fluorescence-measurements). The GFP example explains a general reporter principle; the mRFP study remains the source for its constructs, measurements and normalization.

**Reading scope and limits — November 2, 2026:** Selected passages in chapter 6 on RNA structure (p. 324), processing and splicing (pp. 337–339), 3′-end formation and nuclear export (pp. 348–349), and codon decoding (p. 358); chapter 7 on cellular context and levels of gene control (pp. 397–401), untranslated regions and translation (p. 456), and “Changes in mRNA Stability Can Control Gene Expression” (pp. 459–460); and chapter 9, “Individual Proteins Can Be Fluorescently Tagged in Living Cells and Organisms” (pp. 573–574, including Fig. 9–16). Page numbers here are the printed page labels, not PDF viewer positions. Eukaryotic processing and decay examples are identified as such; they are not a universal assay model for every organism in the benchmarks. These passages do not establish that a scanned motif binds a protein, is chemically modified, or changes persistence in our samples.

### Lehninger Principles of Biochemistry, 8th edition

[Lehninger Principles of Biochemistry, eighth edition](https://www.macmillanlearning.com/college/us/product/Lehninger-Principles-of-Biochemistry/p/1319228003), David L. Nelson, Michael M. Cox and Aaron A. Hoskins, Macmillan Learning (2021).

**Use here:** The nucleic-acid and thermodynamic explanations support the [RNA-folding control](methods/sequences-to-predictions.md#secondary-structure-proxy-nussinov-pairing-fraction): an RNA strand can pair with itself, stems and loops are structural features, and free energy accounts for more than the number of permitted pairs. The guide distinguishes structural stability from persistence before degradation.

**Reading scope and limits — November 2, 2026:** Selected passages in §8.2, “Nucleic Acid Structure,” especially “Messenger RNAs Code for Polypeptide Chains” and “Many RNAs Have More Complex Three-Dimensional Structures”; and the opening of §13.1, “Bioenergetics and Thermodynamics,” including free energy, enthalpy, entropy and the relationship between standard free-energy change and equilibrium. Section names identify the passages across print and electronic layouts. These principles explain the simplification in the Nussinov pairing proxy; they do not validate its accuracy for intracellular RNA structures or establish a folding-energy-to-half-life conversion.

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

**Why retain it:** It informed the distinction between citing underlying evidence and explaining an assistant's role in the [literature review](<annotated-Literature Review - Biological Foundation Models-2.pdf>), under Preparation and AI Assistance.

**Reading scope and limits:** The editorial guidance was checked on October 5, 2026. It is writing guidance, not biological evidence or a statement of the professor's policy. Its relevance belongs beside the disclosure, separate from the scientific Works Cited.

## Planning research work

### Research proposal and Gantt examples

[Mélot-Chesnel's research proposal, section 4 and Gantt figure](https://journals.sagepub.com/doi/10.3233/FAIA260562), and [UQ's AGRC7617 proposal assignment](https://course-profiles.uq.edu.au/course-profiles/AGRC7617-62151-7660).

**Why retain them:** These examples informed the September 27 discussion of scheduling research, interpretation, and revision. Their local use is explained in the [Gantt workbook's Details planning notes](Research_Project_Plan_Gantt.xlsx) and the [presentation's speaker notes](Research_Project_Plan_Gantt.pptx).

**Reading scope and limits:** Section 4 and the figure caption, plus UQ's proposal/Gantt and AI-feedback passage, were rechecked on October 5, 2026. These examples do not establish CAP6942 deadlines or measured AI time savings; the team's schedule remains its own plan.

### UCF research computing: GPU access

UCF ARCC's [cluster documentation](https://arcc.ist.ucf.edu/docs/), [Newton description](https://rci.research.ucf.edu/resource/newton/), and [account access instructions](https://arcc.ist.ucf.edu/docs/access/).

**Why retain them:** They identify the institutional route for GPU resources needed by possible Stage 4 adaptation experiments. The current descriptions distinguish CPU-focused Stokes from GPU-equipped Newton, including H100 resources. This helps make a resource inquiry specific without assuming that a lab-owned GPU and the university cluster have the same access rules.

**Reading scope and limits, November 1, 2026:** Read resource descriptions and registration/orientation requirements. The pages do not establish this project's allocation, queue availability or permission to use a particular GPU. No account request or contact was submitted; availability and requirements need rechecking before execution.

## Papers raised in October 2026 (fusion evaluation, alignment, multi-omics benchmarks)

Eight papers came in by email on October 9 and 10 (three from Chase, five from Mina). Each is placed by role in the Overleaf related-work table and in Section 6.6 of the project notes; this index records why each is retained.

**Identity and availability check, 19 October 2026:** First pages and abstracts were checked for the seven locally held manuscripts below other than LucaOne. The [paper catalog](../Papers/README.md) records the exact versions, license evidence, five retained PDFs, and the source-only status of COMET and OmniGenome. This check does not replace the dated reading scopes below or constitute a full rereading.

### Rheude, Eils and Wild: Fusion or Confusion

[arXiv 2512.22991v3](https://arxiv.org/abs/2512.22991v3), [retained PDF](../Papers/supporting/Fusion_or_Confusion_arxiv_v3.pdf), and the [simple baseline repository](https://github.com/TillmannRheude/simple_mml_baseline).

**Why retain it:** Nineteen multimodal methods reimplemented on nine datasets under standardized tuning, initialization, subject-wise cross-validation and statistical tests; no method consistently beats a simple per-modality-projection baseline, and several fall below the best single modality. It supplies the Stage 4 evaluation rules (identical heads, tuned baselines, grouped cross-validation, mean and SD, a statistical comparison) and the reason the baseline hierarchy exists.

**Reading scope and limits:** Abstract and the methods and results summary were read on October 10, 2026. Its datasets are not biological sequences; it guides fair comparison, not biological conclusions.

### mRNABench

[bioRxiv 2025.07.05.662870v1](https://www.biorxiv.org/content/10.1101/2025.07.05.662870v1), [retained PDF](../Papers/supporting/mRNABench_biorxiv_v1.pdf), and the [repository](https://github.com/morrislab/mRNABench).

**Why retain it:** Ten mature-mRNA datasets and 59 subtasks with homology-aware splits, and evidence that random splits overestimate generalization. Supplies the half-life and translation-efficiency tasks (research plan items 3 and 4) and the grouped evaluation in the common rules.

**Reading scope and limits:** Abstract and task catalog read on October 10, 2026; the dataset gates (UTRs, coding-sequence coordinates, grouped splits) still have to be verified on the files before any run.

**Target-provenance reading, 28 October 2026:** Inspected dataset constructors at [mRNABench revision 778d691ab91c51c03addeb9219229f2ec81d195c](https://github.com/morrislab/mRNABench/tree/778d691ab91c51c03addeb9219229f2ec81d195c/mrna_bench/datasets), [Saluki Methods](https://doi.org/10.1186/s13059-022-02811-x) and [preprocessing code](https://github.com/vagarwal87/saluki_paper/tree/3aa4e56a19bbbf87ac9f5f0a251098cf749ad6bc), the [Orthrus intermediate release](https://zenodo.org/records/14708163), [RiboNN Methods](https://doi.org/10.1038/s41587-025-02712-x) and its human supplementary target table, and the original [HP5](https://doi.org/10.1038/s41594-022-00819-2) and [PERSIST-seq](https://doi.org/10.1038/s41467-022-28776-w) measurement descriptions. These sources establish why consensus half-life, abundance-adjusted ribosome occupancy, and mean ribosome load require separate interpretations. Every local human half-life label matches its immediate NPZ source; every human TE label matches supplementary `mean_te` after float32 conversion. The exact earlier half-life scaling cohort, raw-read TE processing, and source-row transformations for the secondary MRL/PERSIST tables remain unresolved. The study-specific sources support target definitions and interpretation; their numerical results do not validate our models or establish reproduction of their experiments.

### Krzakala et al.: MSAlign

[arXiv 2605.19752v2](https://arxiv.org/abs/2605.19752v2), [retained PDF](../Papers/supporting/MSAlign_arxiv_v2.pdf), and the [repository](https://github.com/KrzakalaPaul/MSAlign-NeurIPS2026).

**Why retain it:** MSAlign addresses metabolite identification by retrieving molecular structures from tandem mass spectra, aligning representations from frozen DreaMS and MolDeBERTa encoders through learned projections. Its relevance here is methodological: retrieval-based evaluation and the trade-off between leakage prevention and distribution shift when constructing splits. It does not directly validate DNA–RNA–protein alignment. Adapting these ideas to Stage 2 remains a proposed application; this source record does not establish implementation.

**Reading scope and limits:** Abstract read on October 10, 2026; the loss and metrics were not confirmed from the full text.

**Clarification, 19 October 2026:** The v2 first page and abstract identify the modalities and retrieval task above. The earlier wording that the diagnostic and split-shift measure "enter" the project is replaced with a proposed application to distinguish source relevance from adoption. This check does not resolve the earlier full-text limitations.

### Golkar et al.: MIMIC

[arXiv 2604.24506v1](https://arxiv.org/abs/2604.24506v1), [retained PDF](../Papers/supporting/MIMIC_arxiv_v1.pdf), [repository](https://github.com/PolymathicAI/MIMIC), and [model](https://huggingface.co/polymathic-ai/MIMIC).

**Why retain it:** A generative multimodal model conditioned across biomolecules, evaluated on mRNABench tasks. Motivates the full-transcript tasks; it is not evidence that fusing separately pretrained encoders helps.

**Reading scope and limits:** Abstract and author list read on October 10, 2026.

### Ren et al.: COMET

[arXiv 2412.10347v1](https://arxiv.org/abs/2412.10347v1). Repository availability is recorded in the [source-only catalog](../Papers/README.md#source-links-without-repository-pdfs).

**Why retain it:** Seventeen DNA, RNA, protein, cross-molecule and multi-molecule tasks, evaluated with single-omics models, their pairs, LucaOne and CaLM; multi-molecule tasks remain hard for all of them. Supplies distinct-molecule tasks with published combinations for Stage 4 tests 5 and 8.

**Reading scope and limits:** The full text was summarized on October 10, 2026; dataset sizes are in its Table 1 and were not re-verified.

### He et al.: LucaOne

[bioRxiv 2024.05.10.592927](https://www.biorxiv.org/content/10.1101/2024.05.10.592927v1), [model](https://github.com/LucaOne/LucaOne), [embedding app](https://github.com/LucaOne/LucaOneApp).

**Why retain it:** One encoder and one vocabulary for nucleic acid and protein; 1,280-token limit; inputs embedded separately. The jointly pretrained baseline (research plan item 2).

**Reading scope and limits:** Full text summarized on October 10, 2026; the input limit has to be checked against the stability and GTEx files before extraction.

**Availability follow-up, November 1, 2026:** The [published LucaOne paper](https://www.nature.com/articles/s42256-025-01044-4), Data and Code availability, and the [author repository](https://github.com/LucaOne/LucaOne) report released checkpoints/code and a pretraining-data deposit at CNSA, accession `CNP0007266`. Those statements were inspected; the deposited corpus was not downloaded or verified. This qualifies the proposed motivation for accessible-encoder fusion: public availability, usable input coverage and affordable compute are separate considerations. Neither inaccessible unified models nor their inevitable superiority should be assumed.

### Chen et al.: Structure-Aligned Protein Language Model

[arXiv 2505.16896v2](https://arxiv.org/abs/2505.16896v2) and [retained PDF](../Papers/supporting/Structure_Aligned_Protein_Language_Model_arxiv_v2.pdf).

**Why retain it:** Contrastive alignment of a protein language model's residue states to a structure encoder; a different sense of alignment (sequence to structure within one molecule), named in the Overleaf so it is not confused with alignment across molecules; a candidate structure feature for research plan item 6.

**Reading scope and limits:** Abstract read on October 10, 2026.

### Yang, Chen and Li: OmniGenome

[Bridging Sequence-Structure Alignment in RNA Foundation Models, AAAI 2025 proceedings](https://doi.org/10.1609/aaai.v39i20.35500), by Yang, Chen and Li, introduces OmniGenome. The [arXiv manuscript](https://arxiv.org/abs/2407.11242) is a separate version; the [source-only catalog](../Papers/README.md#source-links-without-repository-pdfs) identifies the proceedings copy checked on 19 October 2026.

**Why retain it:** Sequence-structure alignment in an RNA foundation model with bidirectional mappings; the same sense of alignment on the RNA side; a candidate RNA encoder and structure source for research plan item 6.

**Reading scope and limits:** Abstract read on October 10, 2026.

## Interpreting fusion designs and modality dominance

### BioLangFusion, Flamingo and Coupled Mamba: Fusion designs

[BioLangFusion v1, Section 2.2.2 and Appendix A.3/Table 4](https://arxiv.org/html/2506.08936v1), [Alayrac et al., Flamingo, Section 2.2 and Figure 4](https://arxiv.org/html/2204.14198), and [Li et al., Coupled Mamba, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/file/6e09c213ac18d6375704a4f3ea75c4f8-Paper-Conference.pdf).

**Why retain them:** BioLangFusion already evaluates token-level attention after codon alignment and selects its sequence-level gated method; codon-level gating alone therefore does not establish a new contribution. Flamingo combines cross-attention with learned layer-specific gates, showing that gating and cross-attention are compatible. Its scalar gates differ from our input-dependent modality weights. Coupled Mamba supplies a separate state-space design; its BRCA experiment uses measured expression, methylation and miRNA features, which differ from the sequence-derived streams here.

**Reading scope and limits, November 1, 2026:** Inspected the named architecture descriptions and ablations and Coupled Mamba's biological benchmark description. These readings qualify the call's proposed gating direction; they do not establish a biological benefit or an adopted architecture. The [paper catalog](../Papers/README.md) previously described Coupled Mamba as the architecture used by BioLangFusion. That description was corrected: BioLangFusion compares concatenation, gated MIL and cross-attention. This was a catalog correction, not a model change. No PDF copies were added.

### Peng et al.: Balanced multimodal learning

[Balanced Multimodal Learning via On-the-fly Gradient Modulation](https://arxiv.org/pdf/2203.15332), CVPR 2022, introduction, Figure 1 and method.

**Why retain it:** It supplies an optimization-based explanation for one modality dominating joint training and a method that responds to the imbalance. It is relevant to the hypothesis that a predictor favors an easier representation, but its trainable modality branches differ from our frozen encoders. In our [training objective](../Code/shared_code/fusion/train.py), a positive entropy coefficient explicitly encourages sharper gates; the zero-coefficient comparison omits that pressure. Gate concentration alone cannot identify its cause.

**Reading scope and limits, November 1, 2026:** Read the imbalance argument and gradient-modulation mechanism and inspected our gate computation and entropy-loss sign. No gradients or training trajectories were measured. The paper motivates a diagnostic question, not evidence that our encoders were under-trained or that rebalancing would help.

## Benchmarks and encoders for multimodal comparison

### IsoFormer dataset and model cards

[InstaDeepAI/multi_omics_transcript_expression](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression) and [InstaDeepAI/isoformer](https://huggingface.co/InstaDeepAI/isoformer) on Hugging Face.

**Why retain them:** The dataset supplies GTEx transcript expression with a genomic DNA sequence, the transcript, and the protein for each example, and a fixed gene-level split. It was retained as a candidate for Setting B, where DNA is a distinct input rather than a rewriting of the coding sequence. The current [GTEx experiment](../Code/stage1/Stage1_gtex.ipynb) uses a pilot of this dataset; the [dataset overview](../README.md#data) identifies the local file and its producer.

**Reading scope and limits:** The dataset and model cards were read through a summarizing fetch on September 29, 2026, which reported 82,205 train, 4,718 test, and 4,318 evaluation transcripts and protein-coding genes only. That source review did not read the cards in full, download data, or identify a license; it is not a description of the later pilot implementation. The current experiment records its fields and retained rows, but the earlier review did not resolve licensing.

**Target-provenance reading, 28 October 2026:** Inspected the [dataset card](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression/blob/330ab1795abca683e692c7ee138a2e89c0c882f1/README.md), [dataset loader](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression/blob/330ab1795abca683e692c7ee138a2e89c0c882f1/multi_omics_transcript_expression.py), and separate normalization table at revision `330ab1795abca683e692c7ee138a2e89c0c882f1`, plus the [IsoFormer paper](https://papers.nips.cc/paper_files/paper/2024/file/8f6b3692297e49e5d5c91ba00281379c-Paper-Conference.pdf). These distinguish tissue-averaged GTEx TPM, the stated `log(x+1)` transformation, and additional model-training normalization. The release loader returns tissue values unchanged and exposes normalization statistics separately; our pilot builder and target readers do not apply those statistics. The log base and original release-producing code were not established, and the raw GTEx measurements were not numerically reconstructed. This reading informs the target guide and label/metric explanations, not a change to the experiment.

**Adaptation comparison, November 1, 2026:** Read Sections 4.3–4.4, Tables 1–2 and Appendices B–C of the [published paper](https://papers.nips.cc/paper_files/paper/2024/file/8f6b3692297e49e5d5c91ba00281379c-Paper-Conference.pdf) and rechecked the model card. Our [IsoFormer-style notebook](../Code/stage4/tier2_lucaone_isoformer.ipynb) uses frozen encoders and a small GTEx pilot; the paper uses fine-tuning, different encoder configurations and a larger training cohort. Slide 13 of the October 22 progress deck compares these settings, so its statement attributing the score gap to freezing is too strong. The comparison does not isolate a fine-tuning effect. The deck was inspected but not changed; a matched adaptation study remains a proposal.

**Fuller reproduction specification, November 4, 2026.** The user selected fuller reproduction as the current aim. Read the final paper's NT comparison, length/training descriptions and modality tables, official model/configuration/tokenizer source, the dataset loader and the official repository file inventory. This informed the [Stage 4 specification and implementation prerequisites](../Code/stage4/RUNBOOK.md#isoformer-reproduction-specification). The NT-based Table 2 experiment is the proposed reference; the released checkpoint uses Enformer and does not by itself recover the NT training recipe.

| Inspected source | Pinned revision and purpose |
| --- | --- |
| [Dataset loader](https://huggingface.co/datasets/InstaDeepAI/multi_omics_transcript_expression/blob/330ab1795abca683e692c7ee138a2e89c0c882f1/multi_omics_transcript_expression.py) | `330ab1795abca683e692c7ee138a2e89c0c882f1` identifies split aliases, returned targets, normalization fields and sequence preparation. |
| [Released IsoFormer model](https://huggingface.co/InstaDeepAI/isoformer/blob/afac2a386c8f2241faf8fad3c5971a9307c13775/modeling_isoformer.py), [configuration](https://huggingface.co/InstaDeepAI/isoformer/blob/afac2a386c8f2241faf8fad3c5971a9307c13775/config.json) and [tokenizer](https://huggingface.co/InstaDeepAI/isoformer/blob/afac2a386c8f2241faf8fad3c5971a9307c13775/isoformer_tokenizer.py) | `afac2a386c8f2241faf8fad3c5971a9307c13775` identifies the Enformer-based inference implementation and context/pooling choices. |
| [Official repository documentation](https://github.com/instadeepai/nucleotide-transformer/blob/b8338effa8a07928c1565b778e0a06da431828c9/docs/isoformer.md) | `b8338effa8a07928c1565b778e0a06da431828c9` identifies the inspected documentation/tree. A complete NT training recipe was not located in that tree or model release. |
| [NT v2 250M checkpoint](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-250m-multi-species/tree/c0f0359229f36ff6bc3a021247eefe0a9c344bd1) | `c0f0359229f36ff6bc3a021247eefe0a9c344bd1` is a candidate reproducible initialization, not proof of the author's historical revision. |
| [ESM-2 150M checkpoint](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/a695f6045e2e32885fa60af20c13cb35398ce30c) | `a695f6045e2e32885fa60af20c13cb35398ce30c` is a candidate reproducible initialization under the same qualification. |

The dataset loader maps its named validation and test splits to the same underlying test rows, while the paper describes taking validation from training. This corrects any inference from the earlier card summary that the loader supplies an independent validation cohort. It does not establish that the authors selected models on test data. The released model has DNA attend to RNA and then protein, pools a configured DNA window and uses a two-layer softplus head. The local pilot's modality-pair attention and pooled concatenation differ. These source observations establish prerequisites for the new implementation, not completed repairs. The original NT architecture/training details, normalization use and token-limit conventions remain partly unresolved. No model weights, genome, dataset or new PDF copy was downloaded, and no research computation ran.

### Ensembl sequence requests and rate limits

[POST sequence/region documentation](https://rest.ensembl.org/documentation/info/sequence_region_post) and [Ensembl's rate-limit guidance](https://github.com/Ensembl/ensembl-rest/wiki/Rate-Limits).

**Why retain them:** They define the service contract used by [the GTEx pilot builder](../Code/stage1/build_gtex_pilot.py). The endpoint accepts at most 50 regions per POST. A 429 response supplies `Retry-After` as a floating-point number of seconds. These support batch sizing and respecting the requested delay; the builder's attempt limit and decision to stop for a delay above 120 seconds are local recovery choices, not service requirements. See [rebuilding the pilot](../README.md#data) for its role.

**Reading scope and limits:** On October 5, 2026, read the endpoint parameters, request examples and resource limits, and the wiki's normal/exhausted-rate-limit response sections. No live sequence requests or dataset rebuilds were performed. These references do not verify the historical pilot's completeness, assembly, or biological correctness; they do not pin future service contents.

### RNA-FM model card

[multimolecule/rnafm](https://huggingface.co/multimolecule/rnafm) on Hugging Face.

**Why retain it:** RNA-FM is BioLangFusion's RNA encoder. The card states that it was pretrained on non-coding RNA from RNAcentral and has 640-dimensional embeddings, which matters when it reads coding sequences in the stability benchmark. The same collection lists an mRNA-trained variant, mRNA-FM, as a candidate additional RNA encoder.

**Reading scope and limits:** The card was read through a summarizing fetch on September 29, 2026; maximum input length was not identified there. At that review, loading through `multimolecule` had not been tested. For the current loader and environment requirements, see the [encoder registry](../Code/shared_code/encoders.py) and [notebook setup](../README.md#notebook-dependencies).

**mRNA-FM follow-up, November 1, 2026:** Read the authors' [mRNA-FM documentation](https://github.com/ml4bio/RNA-FM#mrna-fm) and the [MultiMolecule mRNA-FM card/configuration](https://huggingface.co/multimolecule/mrnafm). The author repository describes codon-level pretraining on 45 million coding sequences; the port lists 12 layers, 1,280 hidden features and approximately 239 million parameters. Hidden width is not parameter count, and this model is smaller than ProtBERT. The port's general RNAcentral/noncoding-RNA prose is not used to establish mRNA-FM's pretraining corpus. These sources clarify the [swap comparison](../Code/stage4/tier2_encoder_swaps.ipynb) and a possible comparison of existing hidden layers; they do not establish which layer works best or authorize changing the pretrained network's depth.

**RNA-FM/mRNA-FM interpretation reading, November 1, 2026:** Inspected the [original RNA-FM manuscript, arXiv v5](https://arxiv.org/html/2204.00300v5), pretraining Methods and the structure/interaction/5′-UTR evaluations, plus the author repository's model descriptions, [tokenizer](https://github.com/ml4bio/RNA-FM/blob/main/fm/data.py) and [loader](https://github.com/ml4bio/RNA-FM/blob/main/fm/pretrained.py). The paper's downstream results belong to RNA-FM; it does not establish mRNA-FM's relative performance. The exact mRNA-FM training recipe and corpus composition remain unresolved. This reading informed the [four-part notebook interpretation](../Code/stage4/tier2_encoder_swaps.ipynb#RNA-FM-and-mRNA-FM-as-a-worked-encoder-comparison) and [Stage 4 synthesis](../Code/stage4/README.md#rna-fm-and-mrna-fm-connecting-pretraining-to-the-fusion-question), alongside existing saved scalar scores and interval tables. No model, notebook cell or metric recomputation was run, and no paper copy was added.

### ProtTrans and ESM-2: Interpreting protein encoder swaps

Elnaggar et al.'s [ProtTrans paper](https://arxiv.org/abs/2007.06225), the [Rostlab/prot_bert model card](https://huggingface.co/Rostlab/prot_bert), and its [configuration at the project's pinned revision](https://huggingface.co/Rostlab/prot_bert/blob/7a894481acdc12202f0a415dd567f6cfdb698908/config.json); Lin et al.'s [ESM-2 paper](https://doi.org/10.1126/science.ade2574) and the [official model and dataset tables](https://github.com/facebookresearch/esm#pre-trained-models). The [paper catalog](../Papers/README.md) already identifies both works; this entry adds their use in the swap comparison.

**Why retain them:** They identify consequential differences in the [protein encoder swaps](../Code/stage4/tier2_encoder_swaps.ipynb): approximately 420 million parameters and 1,024 features per residue for ProtBERT, compared with 8/35/150 million parameters and 320/480/640 features for the selected ESM-2 checkpoints. Both learn masked amino-acid prediction. Our checkpoint is the UniRef100 ProtBERT, not ProtBERT-BFD; ESM-2 uses UniRef50 cluster weighting with sequences sampled from UniRef90. Different corpus sizes and sampling rules do not establish greater fungal coverage or a biological cause of a downstream gain.

**Reading scope and limits, October 28, 2026:** Inspected the model cards/configurations, official ESM tables, ProtTrans architecture/pretraining descriptions, and the ESM-2 paper's training-data description. Compared these with the current encoder registry, token extraction, TextCNN construction and saved swap score/configuration fields, without recomputing predictions or running models. The current protein-only predictor consumes final-layer residue embeddings directly; wider inputs increase its trainable convolution parameters even with the same output-channel setting. These sources therefore inform hypotheses about encoder capacity, pretraining and compatibility with the predictor, not an isolated causal explanation. Neither model was trained on this benchmark's expression labels during pretraining. Historical result files lacking full producer identity are not retrospectively verified by inspecting today's code. No paper or weight files were added to the repository.

### PEER and Schmirler et al.: Task-dependent protein-model evaluation

[Xu et al., PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding](https://papers.neurips.cc/paper_files/paper/2022/file/e467582d42d9c13fa9603df16f31de6d-Paper-Datasets_and_Benchmarks.pdf), NeurIPS 2022, Sections 4.1 and 5.1 and Table 3; [Schmirler, Heinzinger and Rost, Fine-tuning protein language models boosts predictions across diverse tasks](https://doi.org/10.1038/s41467-024-51844-2), Nature Communications 15, 7407 (2024).

**Why retain them:** PEER provides contrary evidence to a universal encoder ranking: its frozen ProtBERT-BFD exceeds frozen ESM-1b on beta-lactamase activity, while ESM-1b leads on contact and secondary-structure prediction. These are different checkpoints, datasets and prediction heads from our swaps. Schmirler et al. compare frozen representations with task-specific adaptation, including ESM-2 sizes used here, and explain why the final layer optimized for pretraining need not be best for a downstream target. Neither study establishes why our ProtBERT fungal score is higher.

**Reading scope and limits, October 28, 2026:** Read the PEER model, predictor and repeated-run setup and relevant Table 3 rows; read Schmirler et al.'s abstract, Results and Discussion, model-training methods and concluding explanation. These readings support separating pretrained-model size, frozen-feature usefulness, predictor design and seed variation when interpreting the [Stage 4 comparison](../Code/stage4/tier2_encoder_swaps.ipynb). They do not authorize fine-tuning or change the experiment. Remote text was inspected without adding PDF copies.

### Yeast half-life: Codon usage, transcript regions and measured structure

[Presnyak et al., Codon optimality is a major determinant of mRNA stability](https://doi.org/10.1016/j.cell.2015.02.029), Cell 2015, and [GSE57385](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57385); [Cheng et al., Cis-regulatory elements explain most of the mRNA stability variation across genes in yeast](https://doi.org/10.1261/rna.062224.117), RNA 2017; [Rouskin et al., Genome-wide probing of RNA structure reveals active unfolding of mRNA structures in vivo](https://doi.org/10.1038/nature12894), Nature 2014, and [GSE45803](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45803).

**Why retain them:** Presnyak connects synonymous codon choice, translation and yeast mRNA decay, with public decay time courses. Cheng provides interpretable codon, UTR and sequence-feature baselines and shows why an apparent association should be tested after accounting for other features. Rouskin compares DMS accessibility in cells and in vitro, providing measured evidence relevant to the limits of calculated RNA folding. Together they motivate a proposed fungal question: do UTR or structure representations improve half-life prediction beyond codon-aware CDS features? This could inform the [full-transcript analysis](../Code/stage4/tier3_halflife_te.ipynb), but no new dataset or analysis was selected.

**Reading scope and limits, November 1, 2026:** Inspected the papers' abstracts and relevant results/methods descriptions, public accession designs and Cheng's supplement listing. Supplementary tables and raw reads were not downloaded or joined. Presnyak's transcription-shutoff design includes a temperature shift; strain, condition, transcript coordinates and coverage must be checked before combining any studies. DMS accessibility is not a complete measured base-pair map, and a folding estimate derived from the sequence is not an independent experimental modality. This proposal does not resolve the organism/assay mapping still missing from the current fungal-expression benchmark's upstream target history.

### Comparing encoders when different parts of a sequence vary

**Synonymous codons and RNA abundance.** Chen et al. measured mRNA levels for 3,556 synonymous GFP variants and 523 synonymous TDH3 variants in budding yeast. Within each gene the protein sequence stays fixed, so differences in RNA abundance give a codon encoder something to predict that a pooled protein sequence alone cannot distinguish. SynCodonLM restricts masked-codon prediction to synonyms and its authors evaluated the Chen libraries and the bacterial mRFP assay already used here. This offers a published comparison with the project's codon encoders under a fixed-protein control. [Chen et al., original yeast assay](https://doi.org/10.1093/molbev/msx229) · [SynCodonLM, model and benchmark methods](https://doi.org/10.1093/nar/gkag166).

**Transcript ends and yeast RNA decay.** Geisberg et al. report half-life estimates for 21,248 yeast RNA 3′ isoforms after nuclear RNA polymerase II depletion. The deposited isoform-level table offers a route to compare transcripts from the same gene that differ in their 3′ untranslated region. This would extend the [full-transcript half-life comparison](../Code/stage4/README.md#full-transcript-half-life-and-translation-efficiency) to variation within a gene. A comparison restricted to the same protein would require mapping the measured ends to the reference and verifying that the selected isoforms share a coding sequence. [Geisberg et al., original assay](https://doi.org/10.1016/j.cell.2013.12.026) · [GSE52286, half-lives by isoform](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE52286).

**Binding between an RNA and a separate protein.** Ray et al.'s RNAcompete assay measures how strongly RNA probes are recovered with a particular RNA-binding protein construct in vitro. Both partners can vary across the collection. Pairing the probe with the assayed protein fragment would test whether its sequence helps predict RNA preference for a held-out protein family. The probe intensities and the authors' summaries for seven-base motifs support different prediction tasks. [Ray et al., RNAcompete](https://doi.org/10.1038/nature12311) · [GSE41235, RNAcompete experiments](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41235).

**RNA and protein outcomes on the same bacterial constructs.** Cambray et al. designed 244,000 sequences in 56 factorial series and assayed reporter RNA abundance and decay, polysome profiles, protein production and growth in *E. coli*. The design varies codon and amino-acid properties. Comparing the same encoders on different outcomes from the same constructs could help distinguish a benefit specific to protein production from one shared with RNA abundance. Holding out whole design series would test transfer beyond related constructs. [Cambray et al., experimental design and assays](https://doi.org/10.1038/nbt.4238).

**Reading scope, November 4, 2026.** Reviewed the yeast assay descriptions, the SynCodonLM pretraining objective and benchmark description, the RNAcompete construct and scoring descriptions, the two GEO series listings, and the Cambray study abstract. These are candidate comparisons. Sequence reconstruction, cross-release overlap, per-assay usable sample counts and row-level quality checks remain open.

### Mushroom development: Non-yeast fungal task candidates

[Merényi et al., Gene age shapes the transcriptional landscape of sexual morphogenesis in mushroom-forming fungi](https://elifesciences.org/articles/71348), eLife 2022, and oyster-mushroom accession [GSE176179](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176179); [Krizsán et al., Transcriptomic atlas of mushroom development reveals conserved genes behind complex multicellularity in fungi](https://pmc.ncbi.nlm.nih.gov/articles/PMC6462078/), PNAS 2019, and Coprinopsis accession [GSE125184](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE125184); [Gehrmann et al., Nucleus-specific expression in the multinuclear mushroom-forming fungus Agaricus bisporus reveals different nuclear regulatory programs](https://pmc.ncbi.nlm.nih.gov/articles/PMC5924919/), PNAS 2018, with RNA-seq projects `PRJNA309475` and `PRJNA275107`.

**Why retain them:** These offer alternatives to yeast for the user's proposed fungal direction. Pleurotus ostreatus has public processed expression across developmental stages and tissues; Coprinopsis supplies a model-mushroom developmental atlas; Agaricus offers a more demanding nucleus-specific expression question. A candidate comparison would ask whether upstream regulatory sequence adds predictive value beyond CDS/protein representations for a defined expression profile or stage contrast. RNA-seq expression is the measured target in that proposal, not automatically another independent input modality.

**Reading scope and limits, November 1, 2026:** Inspected abstracts, relevant methods/data-availability passages and public accession/sample/file listings. No expression matrix, genome or checkpoint was downloaded or joined. These are shortlist candidates, not validated benchmarks or selected experiments. Genome versions, gene/transcript matching, regulatory-region availability, normalization, replicates and split units remain to be checked. A fixed gene sequence cannot explain changing conditions by itself: the target must be condition-specific, a fixed profile, or accompanied by condition inputs. Gene/allele and homolog grouping need explicit evaluation rules. Predicting expression would not by itself demonstrate a mechanism or improved mushroom yield.

### Sean D. Moore: E. coli data leads

[UCF faculty profile](https://med.ucf.edu/biomed/person/dr-sean-d-moore/); [Bose and Moore, Variable Region Sequences Influence 16S rRNA Performance](https://journals.asm.org/doi/full/10.1128/spectrum.01252-23), 2023, and its [analysis repository](https://github.com/nibose92/Relative-Entropy); [Smith et al., Ribosome collisions alter frameshifting at translational reprogramming motifs in bacterial mRNAs](https://pmc.ncbi.nlm.nih.gov/articles/PMC6815119/), 2019; [Moore, Thermal-bias PCR: generation of amplicon libraries without degenerate primer interference](https://pmc.ncbi.nlm.nih.gov/articles/PMC12558157/), 2025.

**Why retain them:** These identify concrete routes for a possible sequence-variant data inquiry: ribosome performance, frameshifting reporters and construct-library methods. The rRNA study has public sequence/analysis materials, but rRNA is noncoding and does not supply the translated protein stream used in our CDS experiments. The frameshifting work is a mechanistic data lead; the PCR paper concerns library preparation rather than a ready-made expression benchmark.

**Reading scope and limits, November 1, 2026:** Inspected the faculty research description, publication pages, relevant methods/data-availability statements and repository file roles. This is a targeted publication search, not an exhaustive bibliography. It did not establish access to a large labeled E. coli dataset or its suitability for fusion. An eventual inquiry would need full construct sequences, quantitative targets, assay conditions, replicates and sharing terms. No contact was made or data acquired.

### CodonBERT repository

[Sanofi-Public/CodonBERT](https://github.com/Sanofi-Public/CodonBERT), including its [embedding-extraction script](https://github.com/Sanofi-Public/CodonBERT/blob/master/benchmarks/CodonBERT/extract_embed.py).

**Why retain it:** The repository distributes the benchmark CSVs cited through CodonBERT, including fungal expression and E. coli proteins, and hosts the codon-level CodonBERT model. It provides a source-tracing route for the [benchmark target guide](methods/data-and-targets.md#fungal-and-e-coli-benchmark-targets) and the dataset-version question in the [stability dataset audit](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons). Release availability does not establish which bytes an earlier model run used.

**Reading scope and limits:** Only the search listing was seen on September 29, 2026. How weights are distributed and which data version BioLangFusion used remain unread leads.

**October 20, 2026 benchmark review:** Read the Dataset section of the [README at commit `451a1b1`](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/README.md) and the [fine-tuning loader](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/benchmarks/CodonBERT/finetune.py#L110-L148), alongside Li et al.'s [published CodonBERT article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11368176/), supervised-task description and Table 1, inspected through its [publisher-supplied XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11368176/fullTextXML). The paper describes fungal CDS expression regression, E. coli low/medium/high classification and 70/15/15 splits. The inspected loader reads `Value` and `Split`; it does not supply the upstream fungal target transformation or the E. coli numeric-label dictionary. This review informed the guide's source chain and qualifications. It did not match local rows against downloaded releases, reconstruct CSV generation or verify the historical BioLangFusion input version. No new PDF or dataset copy was added.

**October 28, 2026 mRFP target trace:** Read the [preparation notebook](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/benchmarks/notebooks/0_download_data.ipynb) and [released mRFP CSV](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/benchmarks/CodonBERT/data/fine-tune/mRFP_Expression.csv) at commit `451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4`. The code reads `S2 Validated Data` and sets `Value = np.log(mRFP Mean Corrected)`. The local CSV matches every released field and row, apart from file line endings; all 1,459 sequence-associated values also match the source workbook after that transformation. This resolves the mRFP target scale in the [target guide](methods/data-and-targets.md#mrfp-fluorescence-to-the-value-column). The exact intermediate-to-release row-order and split-generation script was not located; the [verification record](run-records.md#2026-10-28-mrfp-target-provenance) distinguishes code evidence, numeric matching and missing procedural history. This finding is specific to mRFP and does not resolve the other benchmark target questions above.

**October 28, 2026 other-target trace:** Verified the local mRNA stability and CoV-Vac files against this pinned release using their LF-normalized Git blob identities; fungal and E. coli files match every released field and row. Original-source comparisons resolve all mRNA stability labels as within-profile standardized stability slopes, all CoV-Vac labels as standardized nucleotide-profile means, fungal values as natural logs of the author's supplied `Exprs`, and the E. coli 0/1/2 class dictionary. The [target guide](methods/data-and-targets.md) and [verification record](run-records.md#2026-10-28-dataset-target-provenance) distinguish these numerical mappings from still-missing export/split-generation history. This extends the dated mRFP-only finding above; it does not prove which dataset bytes historical BioLangFusion runs consumed.

### Nieuwkoop et al.: mRFP measurement and target provenance

[Revealing determinants of translation efficiency via whole-gene codon randomization and machine learning](https://doi.org/10.1093/nar/gkad035), Nucleic Acids Research 51(5), 2363–2376 (2023), and its [supplementary archive](https://pmc.ncbi.nlm.nih.gov/articles/instance/10018363/bin/gkad035_supplemental_files.zip), containing `Supplementary Data.xlsx`.

**Why retain it:** This is the assay source for the mRFP synonymous control. The workbook exposes the GFPuv normalization formulas, making it possible to explain the target rather than calling it an unspecified expression score. It supports the [measurement-to-target guide](methods/data-and-targets.md#mrfp-fluorescence-to-the-value-column), the [root dataset description](../README.md#mrfp-expression-control), and the [Stage 4 notebook](../Code/stage4/tier1_mrfp_control.ipynb).

**Reading scope and limits:** On October 28, 2026, read the assay and validation methods, GFPuv-reference description and proteomics comparison; inspected S1/S2 workbook values and formulas and the supplementary Figure S3 quality-check explanation. Matched all validated sequence/target observations against the pinned CodonBERT release and local CSV, without fitting models or recomputing prediction metrics. Instrument event files and an executable gating procedure were not reconstructed; the check begins with published clone-level means. The normalization is a scaled red/green flow-cytometry ratio, while the OD-corrected plate measurements serve a separate quality check. The paper's ten-variant proteomics comparison supports fluorescence as an abundance proxy, not absolute molecule counts. Source files were inspected in a temporary directory outside the checkout; no additional paper or dataset copy was added to Git.

**Use in the normalization explanation:** The same study's plasmid-design methods document the constant GFPuv reference and weak expression intended to limit interference. The [target guide](methods/data-and-targets.md#gfpuv-normalization-and-the-natural-logarithm) distinguishes that design from the conditional cancellation of shared multiplicative effects, illustrated with hypothetical signals. Its log-scale examples explain the mathematics of the verified CodonBERT transformation; they do not establish the benchmark authors' undocumented rationale or add a new measured result.

### Wayment-Steele et al.: CoV-Vac chemical-degradation target

[Deep learning models for predicting RNA degradation via dual crowdsourcing](https://doi.org/10.1038/s42256-022-00571-8), Wayment-Steele et al. (2022), explains In-line-seq and the experimental conditions behind the benchmark. The [original RYOS1_MG50_0000 measurement record](https://rmdb.stanford.edu/detail/RYOS1_MG50_0000/) supplies the correct magnesium/50 °C profiles and their processing annotations. The [pinned CodonBERT README](https://github.com/Sanofi-Public/CodonBERT/blob/451a1b167c06028dfbf2ff7aa2cfdea46fbcc4f4/README.md) identifies averaging `deg_Mg_50C` as the sequence-level target.

**Read on October 28, 2026:** assay methods, original RDAT annotations and all matching measurement rows, released sequence/target pairs, and local loading/scoring code. Original 68-position means followed by population z-scoring across the 2,400 released rows reproduce every local `Value` within $5\times10^{-10}$. All input strings match original 107-nucleotide sequences cropped to `[5:86]`. These checks inform [target interpretation](methods/data-and-targets.md#cov-vac-chemical-degradation-of-designed-rna); they do not establish the undocumented generation script, crop rationale, or split-generation procedure. Raw reads were not reprocessed, and no experiment was run.

### Wint et al.: Fungal expression and tRNA context

[Wint, Salamov and Grigoriev (2022), Kingdom-Wide Analysis of Fungal Protein-Coding and tRNA Genes Reveals Conserved Patterns of Adaptive Evolution](https://pmc.ncbi.nlm.nih.gov/articles/PMC8826637/), especially the expression-linked codon-usage analysis and Methods: Genomic Data Acquisition, RNA sequencing, tRNA gene prediction, and Data Set Collection and Preprocessing.

**Why retain it:** It separates RNA-seq expression evidence from the study's tRNA-copy analysis, informing the [fungal target explanation](methods/data-and-targets.md#fungal-expression). Neither a broad paper title nor its species total establishes the contents of the downstream CSV.

**Reading scope and limits:** Those sections were read on October 20, 2026 through the [publisher-supplied XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8826637/fullTextXML). The [author's Codon2Vec preprocessing](https://github.com/rhondene/Codon2Vec/blob/1d3e5df41425753502076927b3bf2f0aa9e927a5/Codon2Vec/filterExprs.py#L42-L57) was also inspected: it selects expression extremes for a separate binary task, so its protocol does not identify the continuous CodonBERT labels. Exact local units, transformation, species composition and assay-to-row mapping remain unresolved. Remote full text was inspected without adding a repository copy.

**October 28, 2026 target trace:** Compared all 7,089 local observations with the [example expression table](https://github.com/rhondene/Codon2Vec/blob/1d3e5df41425753502076927b3bf2f0aa9e927a5/test_input/test_exprs.csv) and [coding sequences](https://github.com/rhondene/Codon2Vec/blob/1d3e5df41425753502076927b3bf2f0aa9e927a5/test_input/test_CDS.fasta) at Codon2Vec revision `1d3e5df41425753502076927b3bf2f0aa9e927a5`. Every local sequence–label observation matches a source observation after `Value = ln(Exprs)`, normalizing U/T for matching and preserving duplicate multiplicities; maximum absolute label difference is $5.0\times10^{-10}$. This numerical verification resolves the log transformation, not the undocumented preparation procedure. The source table contains 7,553 observations and does not define its original RNA-seq normalization or units. An embedded development history naming `Zyghet1_kog_exprs.csv` is a species-provenance lead, not a complete assay mapping. The [target guide](methods/data-and-targets.md#fungal-expression) now distinguishes the verified log relationship from unresolved source units, organism metadata and selection history. The wider study's 459-species scope must not be transferred to this example-derived benchmark.

### Jansen and Gerstein: Why protein sequence can correlate with RNA expression

[Analysis of the yeast transcriptome with structural and functional categories: characterizing highly expressed proteins](https://doi.org/10.1093/nar/28.6.1481), Nucleic Acids Research 28(6), 1481–1488 (2000); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC111042/).

**Why retain it:** The study relates mRNA abundance across ten yeast expression datasets to amino-acid composition, protein length, structural classes and functional categories. It supplies a plausible biological connection between protein-sequence inputs and the [fungal RNA-expression target](methods/data-and-targets.md#fungal-expression). That connection explains why protein inputs might be predictive; it does not show which features our predictor uses, establish the benchmark's species, or explain ProtBERT's advantage over ESM-2. Translation still removes synonymous codon identity.

**Reading scope and limits, October 28, 2026:** Read the abstract and relevant composition, structural/functional-category and expression-data discussion. Retained as explanatory background for the protein-swap investigation, not as a validation of the local dataset or a newly established biological mechanism. No PDF copy or experimental result was added.

### Boël and Ding: E. coli expression assay and label groups

[Boël et al. (2016), Codon influence on protein expression in E. coli correlates with mRNA levels](https://pmc.ncbi.nlm.nih.gov/articles/PMC5054687/), Large-scale protein-expression dataset and Methods on the dataset's proteins and scoring; and [Ding et al. (2022), MPEPE](https://pmc.ncbi.nlm.nih.gov/articles/PMC8913310/), Section 2.1 and Table 1.

**Why retain them:** Boël identifies the original host system and ordinal whole-cell gel assay. Ding explains the subsequent grouping used before CodonBERT's three-class task. They support the [E. coli target explanation](methods/data-and-targets.md#protein-expression-in-e-coli) and prevent conflating host organism, gene origin, soluble yield and native abundance.

**Reading scope and limits:** The named sections and Ding's evaluation description were read on October 20, 2026. Ding's prose reports 6,438 proteins, whereas its listed groups total 6,348, matching Boël and CodonBERT; this discrepancy is retained rather than silently selecting a new local count. No supplementary-row mapping was performed. The local `0/1/2` dictionary, filtered organism membership and construct-level correspondence remain unverified. Remote articles were inspected without adding repository copies.

**October 28, 2026 target trace:** Read Boël's scoring methods and matched all 6,348 local sequences to [MPEPE's original-score FASTA](https://github.com/BRITian/MPEPE/blob/fab98420306f1f1918b1b375a44b7086849ec45f/Dataset/exp_nature_gene.faa) at revision `fab98420306f1f1918b1b375a44b7086849ec45f`, using the repository's [dataset description](https://github.com/BRITian/MPEPE/blob/fab98420306f1f1918b1b375a44b7086849ec45f/Dataset/Readme) to interpret score numbering. Sequence-level matching verifies original scores 0–2 map to local class 0 (low), scores 3–4 to class 1 (medium), and score 5 to class 2 (high), with no unmatched observations. The methods specify whole-cell gel scores and retention of the maximum when transformant scores differed. This resolves the missing local dictionary and supports the [target guide](methods/data-and-targets.md#protein-expression-in-e-coli); original gels and replicate-level scoring were not reprocessed, organism identities were not reconstructed for the filtered subset, and the later release's split-generation procedure remains unverified.

### Medina-Munoz et al.: Stability assay provenance lead

[Medina-Muñoz et al. (2021), Crosstalk between codon optimality and cis-regulatory elements dictates mRNA stability](https://link.springer.com/article/10.1186/s13059-020-02251-5), Methods, especially Estimation of mRNA stability and Data allocation.

**Why retain it:** The iCodon citation provides a lead for tracing the assay behind the stability benchmark. Use it alongside CodonBERT and the [dataset audit](../Code/stage1/Stage1_stability.ipynb#Dataset-audit-for-published-comparisons) to investigate how a measured stability value became a local label.

**Reading scope and limits:** Selected Methods and the iCodon citation were checked in the October 5, 2026 source review. The local CSV's units, normalization, and row mapping remain unresolved. The publication year is 2021 despite the DOI's 2020 component; this is distinct from Agarwal and Kelley (2022).

**October 28, 2026 resolved target trace:** Read the stability-estimation and data-allocation methods, [iCodon's training-data description](https://doi.org/10.1038/s41598-022-15526-7), and its [training/testing data](https://github.com/santiago1234/iCodon/tree/f3afe2bcf36a6bc0298ef7ec0dc42b68835bcb62/data). The original [gathering code](https://github.com/santiago1234/MZT-rna-stability/blob/8b886d9b05cd1f304439e4735268bedbaf2d007a/data/19-04-29-AllDecayProfiles/gatherdata.R) harmonizes signs and converts supplied half-lives to signed decay slopes. Its [preprocessing](https://github.com/santiago1234/MZT-rna-stability/blob/8b886d9b05cd1f304439e4735268bedbaf2d007a/results/19-04-30-PredictiveModelDecayAllSpecies/19-04-30-EDA/EDAanalysos.Rmd) removes within-profile boxplot outliers and applies sample-SD z-scoring by species, cell type and technique. These sources explain why the local target has no hour units. All 65,356 local sequence/value observations match the standardized iCodon data after T/U normalization, with duplicate multiplicities retained and maximum absolute difference below $4.45\times10^{-16}$. The [guide](methods/data-and-targets.md#mrna-stability-standardized-decay-profiles) replaces the earlier uncertainty with this verified mapping and qualifies repeated-label variation as potentially biological context, not pure replicate noise. Raw read-to-decay estimation and the exact later benchmark export/split procedure were not reconstructed; source-row matching does not establish earlier run identities.

## Evaluating biological predictions

### DOME: Biological machine-learning validation

[Walsh et al. (2021), DOME recommendations for supervised machine learning validation in biology](https://www.nature.com/articles/s41592-021-01205-4), with its [author correction](https://www.nature.com/articles/s41592-021-01304-2).

**Why retain it:** Box 1 and Table 1 help examine data independence, preprocessing, baselines, model selection, and reporting in the [linear-probing evaluation](methods/sequences-to-predictions.md#evaluation-and-interpretation). This is evaluation guidance, not a claim of DOME compliance.

**Reading scope and limits:** The October 5, 2026 review read the introduction, data/split discussion, Box 1, and Table 1 in the [author-hosted PDF](https://www.biofold.org/pages/documents/papers/walsh_nmeth2021.pdf), which carries a placeholder publication date. The publisher's correction notice identifies a corrected specificity equation in Figure 2; that notice was checked through search when direct retrieval failed. Consult the corrected publication before reusing that equation.

**Coding-review follow-up:** The October 5 coding review additionally read the Optimization discussion for the [proposed data and evaluation contracts](shared-coding-proposal.md#what-research-software-engineering-adds). No corrected specificity equation was used.

## Organizing and checking research software

These sources explain practices relevant to separating experiment narratives from reusable computation and checking changes in small steps. The implemented structure is described in the [Stage 1 reading route](../README.md#stage-1-analysis), [shared package](../Code/shared_code/), and [notebook checks](../tests/README.md). The readings support evaluating those choices; they do not establish that every recommendation was adopted or that a source caused an earlier decision.

For an introduction, read Wilson et al. (2017), then Wilson et al. (2014) for more detailed practices, and Sandve et al. for reproducibility. The nf-core and ROOT examples help assess future infrastructure as the project grows.

### Wilson et al.: Best Practices for Scientific Computing

[Wilson et al. (2014), Best Practices for Scientific Computing](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1001745), especially Box 1 and the sections on incremental changes, duplication, testing, and documentation.

**Why retain it:** It supports small refactoring steps, reusable computational functions, automated regression checks, and explanations of purpose. These practices help explain the current division between experiment notebooks, [shared calculations](../Code/shared_code/analysis.py), [method guides](README.md#understand-the-methods), and [tests](../tests/README.md).

**Reading scope and limits:** The sections named above were read on October 5, 2026. This is supporting rationale for the present organization, not evidence that passing tests validates a biological interpretation or reproduces saved research results.

**Coding-review follow-up:** The October 5 coding review also read Make Incremental Changes, Plan for Mistakes, Optimize Software Only after It Works Correctly, and Document Design and Purpose, Not Mechanics for the [coding proposal](shared-coding-proposal.md#what-research-software-engineering-adds).

### Wilson et al.: Good enough practices in scientific computing

[Wilson et al. (2017), Good enough practices in scientific computing](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005510), especially Software, Collaboration, Project organization, and What we left out.

**Why retain it:** It supports reusable functions, explicit dependencies, useful project overviews, and simple example checks. Its discussion of adoption costs helps assess how much structure this project needs: continuous integration and coverage tools can burden newcomers before their benefits justify the work.

**Reading scope and limits:** The sections named above were read on October 5, 2026. They support keeping the [reading route](../README.md#stage-1-analysis) and [checks](../tests/README.md) proportionate. The paper does not prescribe this repository's exact folders or notebook boundaries, or require adding CI now.

**Coding-review follow-up:** The October 5 coding review revisited Software (especially 2g–2i), Collaboration, and What we left out for the [proposed proportionate coding workflow](shared-coding-proposal.md#shared-skill-and-how-both-agents-use-it).

### Sandve et al.: Ten Simple Rules for Reproducible Computational Research

[Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285), Rules 1–6.

**Why retain it:** These rules address result-producing steps, executable processing, software versions, script history, intermediate results, and random seeds. They provide a starting point for designing records that connect a result to its inputs and producing code. The [results and cache policy](../README.md#results-caches-and-intermediate-files) and [embedding-cache explanation](methods/sequences-to-predictions.md#embedding-matrices-and-cache-reuse) describe relevant existing practices.

**Reading scope and limits:** Rules 1–6 were read on October 5, 2026. Complete run provenance remains a future design consideration; preserved outputs and regression checks do not establish that it exists. Recording a seed alone does not establish deterministic behavior across hardware and libraries.

**Coding-review follow-up:** The later October 5 coding review read Rules 1–3 and 5–9 for the [coding proposal](shared-coding-proposal.md#what-research-software-engineering-adds). The current [manual run-record procedure](run-records.md) is now available; its existence does not establish complete historical provenance.

### Git and GitHub: Version history and plan-change review

[Pro Git, Viewing the Commit History](https://git-scm.com/book/en/v2/Git-Basics-Viewing-the-Commit-History), [Pro Git, Git Attributes: Diffing Binary Files](https://git-scm.com/book/en/v2/Customizing-Git-Git-Attributes#_diffing_binary_files), and [GitHub, Pull requests](https://docs.github.com/en/pull-requests/reference/pull-requests).

**Why retain them:** They support evaluating Git history and GitHub review as part of research-plan provenance: commits identify saved versions, while a written rationale can explain a change to the [Gantt plan](README.md#follow-the-research-plan). Office files can be versioned, but ordinary binary diffs do not explain their content changes. Existing [run records](run-records.md) separately describe producing executions.

**Reading scope and limits:** On October 10, 2026, read the commit-history examples and author/committer fields, the binary-diff and Word text-conversion example, and the pull-request conversation, commits, checks and changed-files descriptions. These informed the provenance discussion; a plan-change record and a review workflow remain proposals. No text-conversion driver or automatic tracking was configured. Git metadata does not by itself establish a run's producer, scientific verification or advisor approval.

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

**Why retain it:** This is a concrete bioinformatics example of shared modules and subworkflows, environment-specific configuration, and collaborative review. It gives context for the use of reusable functions in [Code/shared_code/](../Code/shared_code/) while the notebooks retain their experiment-specific choices and interpretations.

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


## Numerical reproducibility across computers

### PyTorch and NumPy: reproducibility limits and numerical diagnostics

[PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), [NumPy least squares and rank cutoff](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), and [NumPy runtime reporting](https://numpy.org/doc/stable/reference/generated/numpy.show_runtime.html).

**Why retain them:** They distinguish seed control from guarantees across platforms and explain rank thresholds and the BLAS/LAPACK environment. They inform the cross-run comparison and diagnostic limits in the [October 7 run record](run-records.md), rather than prescribing identical hardware for all contributors.

**Reading scope and limits:** Reproducibility/platform qualifications, least-squares `rcond`, rank and singular-value outputs, and `show_runtime` were checked on October 7, 2026. The local composition designs have a large gap at the current rank cutoff; that finding does not establish why an earlier result differs. No rank threshold, solver, precision or scientific calculation was changed by the figure-export work.
