# Stage 1 overview: DNA, RNA, and protein representations

Stage 1 investigates what pretrained DNA, RNA, and protein encoders represent and how their representations relate to one another. This overview holds the shared model context and cross-dataset summary. The analyses run independently in two notebooks, each with its own setup and saved results, using the `mbf.datasets` and `mbf.encoders` registries. The saved run uses twelve encoders, four per modality.

| Notebook | Inputs and target | Contents |
| --- | --- | --- |
| [mRNA stability](../Code/Stage1_stability.ipynb) | Coding sequences and stability labels from `mRNA_Stability.csv` | Input checks, embeddings, Tracks A–C, synonymous-recoding control, and the BioLangFusion published-split comparison |
| [GTEx transcript expression](../Code/Stage1_gtex.ipynb) | Genomic DNA, transcripts, proteins, and expression in 30 tissues from `GTEx_pilot.csv` | Independent setup, input checks, embeddings, Tracks A–C, and the IsoFormer published-split comparison |

Start with the stability notebook for the first method walkthrough. The GTEx notebook links to the same method guides and can run without executing stability. The [Summary](#summary) compares their retained results.

## Models and input settings

In the stability experiment, each retained coding sequence supplies one input per encoder: the sequence in DNA letters for a DNA encoder, the same sequence in RNA letters for an RNA encoder, and its translated amino acid sequence for a protein encoder. All encoders remain frozen during the analysis. Their parameters are not updated using the pilot dataset.

| Modality | Encoder | Pretraining data | Reason for inclusion |
|---|---|---|---|
| DNA | Nucleotide Transformer 500M human-ref | Human reference genome | Reference DNA encoder in `REFERENCE_KEYS` |
| DNA | Nucleotide Transformer v2 100M multi-species | Genomes of many species | DNA encoder in [BioLangFusion](../Papers/core/BioLangFusion.pdf), Table 3 |
| DNA | DNABERT-2 | Genomes of many species | Multi-species DNA encoder with byte-pair-encoding tokens |
| DNA | HyenaDNA large | Human reference genome | Long-context DNA encoder built from Hyena convolutions rather than attention; reads up to one million nucleotides, one token per nucleotide |
| RNA | RNA-FM | Non-coding RNA sequences from RNAcentral | Reference RNA encoder; RNA encoder in BioLangFusion |
| RNA | RiNALMo 150M | Non-coding RNA sequences from RNAcentral, Rfam, Ensembl, and Nucleotide | Second RNA encoder that reads one nucleotide per token, from a different group than RNA-FM |
| RNA | mRNA-FM | Messenger RNA coding sequences | RNA-FM variant pretrained on messenger RNA, with one token per codon |
| RNA | CaLM | Coding sequences from the European Nucleotide Archive | Codon-level encoder from a different group than mRNA-FM; grouped with the RNA encoders because it also reads the coding sequence one codon at a time |
| Protein | ESM-2 8M | UniRef50 protein sequences | Protein encoder in BioLangFusion, Table 3 |
| Protein | ESM-2 35M | UniRef50 protein sequences | Reference protein encoder |
| Protein | ESM-2 150M | UniRef50 protein sequences | Protein encoder in [IsoFormer](../Papers/core/Multi-Modal-Transfer-Learning.pdf) |
| Protein | ProtBERT | UniRef100 protein sequences | Protein encoder outside the ESM-2 family |

The stability inputs share one coding-sequence origin. DNA/RNA letter conversion preserves the nucleotide sequence, while translation loses synonymous codon distinctions, so the protein input does not retain all nucleotide information. The comparison therefore examines how pretrained encoders, trained on different corpora with different tokenizers and at different sizes, represent related biological information. This follows the modality definition used by [BioLangFusion](../Papers/core/BioLangFusion.pdf), Section 2.1, which gives the same coding sequence to DNA, RNA, and protein language models. The stability dataset carries no gene or transcript identifiers, so genomic DNA around each gene, such as promoters or introns, is not available here.

The registry records each dataset's setting. The stability notebook uses this derived setting. [GTEx transcript expression](../Code/Stage1_gtex.ipynb#Distinct-inputs:-GTEx-transcript-expression) repeats the main analyses in the distinct setting, on IsoFormer's GTEx data, where DNA encoders read genomic DNA around the transcription start site, the single-nucleotide RNA encoders read the transcript, the codon-level RNA encoders read its coding sequence, and protein encoders read the protein.

## Research questions

- **Track A: Decodability.** Which properties can simple predictive models recover from each encoder's embedding?
- **Track B: Cross-modal geometry.** How similarly do the embedding spaces organize the same examples, compared for every pair of encoders, including two encoders of the same modality, and how much of that agreement remains after removing sequence composition?
- **Track C: Attention and diagnostics.** How does the selected attention summary of the nucleotide encoders relate to matches for RNA stability-associated sequence patterns, and do sample-level alignment and representational similarity relate to one another and to prediction?
- **Distinct inputs.** Do the answers change when each modality is a different molecule, so that the DNA input holds promoter sequence and the transcript holds untranslated regions that the protein lacks?

## Role in the project

Stage 1 characterizes the representations available from the frozen encoders. Its observations inform the subsequent definition of alignment, development of fusion methods, and evaluation against single-modality baselines. In the derived stability setting, encoders of the same modality read the same input type, so their agreement is the reference against which agreement across modalities is read.

The analyses generate evidence and hypotheses for those later stages. Choosing an alignment objective or fusion architecture requires further controlled comparisons, such as the trained fusion architectures of BioLangFusion and IsoFormer on the same inputs.

**Reading and running.** Follow the questions, code, and results in the experiment notebook you choose. Use [From sequences to embeddings](methods/inputs-and-embeddings.md) for the preparation walkthrough and [Linear probing](methods/linear-probing.md) for probe mathematics; each experiment links to the relevant explanation where it is needed. See [Setup](../README.md#setup) for dependencies, the `Code/` working directory, and checkpoint downloads. The cells using `show_source` link to the maintained functions in `Code/mbf/`; [the display setting in stability](../Code/Stage1_stability.ipynb#Implementation-displays) or [GTEx](../Code/Stage1_gtex.ipynb#Implementation-displays) controls whether their full source is shown.

---

## Summary

The saved run covers twelve frozen encoders, four per modality, on 981 retained mRNA stability sequences, where every input derives from one coding sequence, and on the GTEx pilot, where DNA, transcript, and protein are distinct inputs (999 train and 996 test transcripts).

**What the frozen embeddings make accessible.**

- On stability, the protein encoders and the two codon-level RNA encoders carry the most linearly accessible signal ($R^2$ about 0.11 to 0.14). The encoders that read single nucleotides, six-nucleotide tokens, or byte-pair tokens reach 0.03 to 0.08, and sequence length alone predicts almost nothing ($R^2$ 0.003).
- Every nucleotide encoder retains synonymous codon usage (GC3 $R^2$ 0.90 to 0.98), which the protein encoders cannot see (0.38 to 0.44).
- On GTEx, RNA and protein encoders predict expression (0.09 to 0.18) far better than the DNA encoders reading the region around the start site (0.03 to 0.05), but four transcript-length features (0.24) beat every single encoder.
- Label noise bounds the stability scores: identical sequences in the full file differ by a median standard deviation of 0.49.

**How the representations relate.**

- Agreement follows tokenization and training more than the modality label. RNA-FM and RiNALMo agree most of any pair, the ESM-2 models next, and encoders reading single letters or short tokens of the same sequence agree whether labeled DNA or RNA. mRNA-FM agrees globally with almost no encoder.
- Most agreement involving a DNA encoder is composition: removing letter, codon, and amino-acid frequencies cuts CKA by 52% to 90% for every such pair.
- Three groups keep strong agreement after the control on both datasets: the protein encoders among themselves, RNA-FM with RiNALMo, and CaLM with the protein encoders. CaLM still retrieves its protein partner 58% to 69% of the time on stability, about 170 to 200 times chance.
- On GTEx, the DNA window and the gene's other molecules share almost no retrievable correspondence (Recall@1 at most 4%), as expected for different sequences.
- Composition, stability signal, and most agreement sit in each embedding's ten dominant principal components; the ESM-2 models and CaLM with the protein encoders also agree in the low-variance directions, and mRNA-FM's correspondence with the protein encoders lies mainly there.

**Combining representations.**

- On stability, every pairing of a codon-level encoder with a protein encoder improves on the better encoder under all ten further fold assignments, by about 0.01 to 0.02; larger concatenations do not help.
- On GTEx, 47 of 66 pairs gain under every assignment, every DNA encoder gains with every protein and codon-level encoder, and all twelve encoders together reach 0.24, the level of the length features.
- On the published splits, the frozen probes keep the cross-validated order. BioLangFusion's three encoders concatenated reach Spearman 0.364 against its published 0.539 to 0.563, and the GTEx three-modality sets reach $R^2$ 0.216 to 0.249 against IsoFormer's 0.43, in the same order of modalities.

**Diagnostics.**

- Per-sequence alignment does not predict prediction error for any of the 66 pairs on either dataset after correction.
- Position never explains the attention associations, and codon context explains most of them: mRNA-FM's DRACH shift and RiNALMo's ARE gap vanish under the codon-and-frame null. CaLM's attention to ARE pentamers survives both nulls, a lead rather than a finding. In 3' untranslated regions, RNA-FM shows no association and RiNALMo's ARE gap is explained by local three-nucleotide context.

**Implications for alignment and fusion.** Raw similarity between frozen pooled embeddings is a weak alignment target, because composition reproduces most of it; alignment and fusion comparisons should report agreement before and after a composition control. Per-sequence alignment does not track prediction error on either dataset, which is consistent with letting a fusion model learn interactions rather than imposing an explicit alignment loss, although these diagnostics do not test an alignment loss directly. Encoders of one modality are not interchangeable: mRNA-FM and CaLM both read codons and both complement the protein encoders, yet their geometries differ sharply. Where inputs are distinct, simple concatenation already adds signal across modalities, but transcript lengths match it, so any fusion comparison on GTEx should include length as a baseline. The pairing of a codon-level RNA encoder with a protein encoder is the clearest candidate for the trained fusion comparisons in Stage 3 on the stability data, and DNA with protein or codon-level encoders on GTEx.
