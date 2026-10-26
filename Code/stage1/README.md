# Stage 1: representation analysis

Stage 1 asks what the chosen pretrained encoders already offer before we decide how to combine them. We examine what a common predictor can recover from each representation, how the representations relate to one another, and whether simply joining them helps prediction. Together, these analyses give us baselines to compare against and hypotheses to carry into the alignment and fusion work. The notebooks use three datasets: stability and GTEx for the two main analyses, and mRFP expression for the synonymous-recoding control described below.

## Why stability and GTEx

We begin with [mRNA stability](Stage1_stability.ipynb): how well can the representations support prediction of the supplied stability score? Stability concerns how long an mRNA persists before degradation. Here the DNA and RNA inputs come from the same coding sequence, written in their respective letter alphabets, and the protein input is its amino-acid translation. Comparing these inputs lets us examine different pretrained representations of a common sequence origin. Translation removes synonymous codon distinctions, so the protein input does not preserve every nucleotide-level difference.

The [GTEx experiment](Stage1_gtex.ipynb) broadens the question to transcript expression across 30 human tissues. Its DNA encoders read a genomic window around the transcription start site; single-nucleotide RNA encoders read the transcript, codon-level RNA encoders read its coding sequence, and protein encoders read the corresponding protein. The genomic window and untranslated transcript regions provide sequence context absent from the protein input. This lets us examine another reason combining representations might help: the inputs themselves can supply different regions of biological sequence. GTEx also changes the target and dataset, so a difference from the stability result cannot be attributed to input regions alone.

Reading stability first introduces the common methods before those additional input and target differences. Neither notebook requires the other to run. The dataset descriptions for [stability](../../README.md#mrna-stability) and [GTEx](../../README.md#gtex-transcript-expression) explain the respective files; the [paper-input comparison](../../docs/stage1-overview.md#published-comparisons-and-distinct-inputs) connects their use to BioLangFusion and IsoFormer. Gantt `Details` tasks 8 and 9 distinguish mapping these inputs from running the frozen-probe comparisons.

The saved outputs record the Stage 1 analyses. The current [overview](../../docs/stage1-overview.md) brings together the encoder settings, result tables, and interpretations; the notebooks contain the calculations and their individual outputs.

## Comparing the encoders

Each main experiment examines twelve encoders, four per modality, and all 66 distinct encoder pairs. To count the pairs, each encoder has eleven possible partners. Doing this for all twelve counts each pair twice: A with B and B with A. We divide by two to count each pair once.

$$
\text{Number of pairs} = \frac{n(n-1)}{2} = \frac{12 \times 11}{2} = \frac{132}{2} = 66.
$$

In this notation:

- $n$ is the number of encoders: twelve here.
- $n-1$ is the number of partners available to each encoder, excluding itself.
- $n(n-1)$ counts both orders of each pair.
- Dividing by $2$ removes the duplicate pair listings. The result is exactly 66; no rounding is needed.

The count includes pairs within the same modality and across modalities. It describes which encoders are paired, not whether every comparison metric is independent of direction. The main pairwise analyses cover all pairs; attention analysis is a separate diagnostic using selected compatible nucleotide encoders.

**Model details:** [Encoder table and input settings](../../docs/stage1-overview.md#models-and-input-settings).

## From sequences to embeddings

The pretrained checkpoints supply the encoder architectures, learned weights, and associated tokenizers. A tokenizer converts the sequence letters into the tokens the model expects. Our pipeline calls those tokenizers; it does not train new ones. We choose how to prepare the sequences, which token limits to use for truncation, and how to summarize the outputs. For the main probes, we average the final-layer token representations into one vector per sequence, called a **sequence embedding**.

The encoders remain **frozen**: their learned weights are not updated using these datasets. We fit separate standardized linear ridge-regression probes to the embeddings instead. Ridge regression predicts a weighted combination of the embedding features while penalizing large coefficients, helping stabilize a fit when features are numerous or correlated. `RidgeCV` chooses the penalty strength through cross-validation. Using the same kind of predictor makes the encoder comparisons easier to interpret: it asks what this predictor can recover from each representation. It does not establish the best performance an encoder could achieve with a different predictor or fine-tuning.

**Method explanations:** [From sequences to embeddings](../../docs/methods/inputs-and-embeddings.md) · [Linear probing](../../docs/methods/linear-probing.md).

## Why the analyses have three tracks

**Track A asks what the embeddings make predictable.** Stability and tissue expression are the biological prediction targets. In the stability notebook, predicting GC, GC3, and a computed RNA-structure proxy from the embeddings asks what other sequence properties they retain. Those are different targets from stability. Length and composition baselines reverse the role of the simpler features: they use those features as inputs to predict the biological outcome, asking how much prediction is possible without an encoder. The Nussinov structure target is a calculated pairing proxy for a sequence prefix, not a laboratory measurement of RNA folding.

**Track B asks how the representations organize corresponding examples.** Linear centered kernel alignment (CKA) compares overall organization, neighborhood overlap compares nearby examples, and canonical correlation analysis (CCA) fits corresponding directions that can be tested through retrieval of matching examples. The composition control subtracts the component linearly explained by the specified sequence features before repeating the comparisons. This helps distinguish measured agreement from agreement those features can already reproduce; it does not remove every possible composition effect or identify a biological mechanism. The composition fit uses all analysis rows, including rows subsequently held out for PCA/CCA, so the post-control retrieval pipeline is not fully held out from preprocessing.

We also concatenate embeddings—append their feature columns—and fit the same ridge probe. The comparison is with the stronger individual encoder under the same folds. This is a direct test of complementary predictive information, distinct from geometric agreement. Stability places it at the start of Track B; GTEx places it with the expression probes. Its scientific purpose is the same in both notebooks.

**Track C examines selected model behavior and exploratory relationships.** For compatible nucleotide encoders, we summarize self-attention and compare it with candidate RNA-pattern matches, using position and local sequence-context controls. GTEx's attention analysis separately reads 3′ untranslated-region sequences; its three-base control blocks are not biological codons. Other diagnostics ask whether sample-level representation agreement relates to prediction error. These analyses can help formulate explanations to investigate; a motif match is not a validated regulatory site, and an attention association is not evidence of a causal regulatory mechanism.

The tracks therefore inform one another without forming a chain in which A proves B and B proves C. A pair can organize examples similarly yet add little predictive value when combined. A useful predictor can also have an attention summary that does not support the proposed biological explanation. We need those distinctions before deciding what alignment should mean or which fusion method to test.

**Methods:** [Representation comparisons](../../docs/methods/representation-comparison.md) · [Attention and candidate motifs](../../docs/methods/attention-and-motifs.md).

The stability notebook's **mRFP control** uses a [separate expression dataset](../../README.md#mrfp-expression-control), rather than a subset of the stability rows. Its nucleotide sequences differ while encoding the same red fluorescent protein. Protein encoders should receive identical strings while nucleotide encoders can respond to the codon differences. Stage 1 checks this input/embedding behavior; the [Stage 4 control](../stage4/README.md#mrfp-changing-the-nucleotide-sequence-while-keeping-the-protein-fixed) reuses the control to ask whether nucleotide representations help predict the expression differences.

## How the stability data are evaluated

In `mRNA_Stability.csv`, `Sequence` supplies the coding sequence, `Value` contains the numerical stability score used as the prediction target, and `Split` identifies the supplied training, validation, or test partition. The [dataset description](../../README.md#mrna-stability) explains their provenance and the unresolved meaning of the score's units. The notebook uses two evaluation procedures, which answer related questions under different splits.

The **main pilot** samples the data without using `Split` to assign evaluation partitions. Its five-fold evaluation keeps identical retained sequences together in each outer fold, so copies of the same sequence are not divided between that fold's training and evaluation rows. This lets us compare the encoders under a common pilot procedure.

The later **published-split comparison** separately samples the supplied training and test partitions. It does not use the supplied validation rows. Instead, `RidgeCV` chooses the ridge penalty through its default leave-one-out cross-validation within the training sample; test targets are used afterward to score predictions. There is internal validation, even though the file's separate validation partition is unused. This explains the current mechanism, but we have not established a recorded reason for choosing it over the supplied validation partition. It should not be described as reproducing the paper's complete training procedure.

The notebook also reports performance on test sequences absent from the entire supplied training partition because the supplied partitions overlap. Internal ridge selection has separate limitations: the scaler is fitted before the internal cross-validation, and that inner selection does not group duplicate sequences. The [linear-probing guide](../../docs/methods/linear-probing.md#current-evaluation-limitation) explains these boundaries. A result needs to be read with the procedure that produced it; the two evaluations are not interchangeable.

## How GTEx is evaluated

GTEx keeps the pilot's supplied training and test halves distinct. The main probes use five-fold evaluation within the training half, grouping transcripts by gene. They predict all 30 tissue targets, with a separate coefficient vector per tissue and one shared ridge penalty, and report the mean of the tissue-specific R-squared scores. The length and composition baselines use the same rows and folds, so their scores answer the same prediction question. Gene grouping applies to the outer folds; the shared ridge helper's internal penalty selection does not enforce gene groups and follows a scaler fitted on that outer training subset, as in the stability limitation above.

The later published-split comparison fits on the pilot's training half and evaluates its separate test half. Its genes are excluded from the training half. CCA retrieval uses a different held-out subset within the training half; it should not be confused with that supplied test partition. The [GTEx notebook](Stage1_gtex.ipynb#GTEx-published-split-comparison) explains why these small frozen-probe comparisons are not a reproduction of IsoFormer's end-to-end training and published score.

## What carries into the next stages

The retained [result tables](../../docs/stage1-overview.md#result-tables) provide the evidence behind the current interpretation. On stability, codon-level RNA and protein encoders are useful candidates under the linear probes. On GTEx, simple transcript-length features form a strong reference that later models need to be compared against. Composition explains a substantial part of the measured agreement. These observations give us candidate encoders and controls; they do not establish that a particular fusion architecture will improve prediction.

Some composition-controlled values differ between the saved Stage 1 sources imported by the [encoder-swap notebook](../stage4/tier2_encoder_swaps.ipynb). Those differences remain provisional. When an exact value is used to connect alignment with a later result, identify its source rather than treating the available summaries as interchangeable.

[Stage 2](../../README.md#stage-2-defining-alignment) records how we distinguish biological correspondence, representation agreement, and predictive complementarity. [Stage 3](../stage3/README.md) prepares stability inputs for the later BioLangFusion comparisons, adds other benchmarks, and reuses GTEx for the IsoFormer-style build; [Stage 4](../stage4/README.md) carries those comparisons further. Reusing the source tables does not make the scores interchangeable: later notebooks change filtering, representation handling, or the fitted predictor. A later result can also send us back to an earlier interpretation—for example, a stronger single encoder can change how we understand an apparent fusion gain. The [Gantt workbook](../../docs/Research_Project_Plan_Gantt.xlsx), `Details` tasks 7, 15, 16, and 19, records these definition, comparison, baseline, and interpretation responsibilities.

## Notebooks and supporting files

| File | Contents |
| --- | --- |
| [Stage1_stability.ipynb](Stage1_stability.ipynb) | Stability prediction from coding-sequence inputs and their protein translations, plus the mRFP synonymous-recoding control |
| [Stage1_gtex.ipynb](Stage1_gtex.ipynb) | Expression prediction across human tissues from the GTEx pilot's DNA windows, transcripts, and proteins |
| [build_gtex_pilot.py](build_gtex_pilot.py) | Rebuilds `datasets/GTEx_pilot.csv`; run from the repository root with `python Code/stage1/build_gtex_pilot.py` |
| `embeddings/` | Saved embedding caches, one folder per dataset sample; local and ignored by Git |
| `embeddings_legacy/` | Older caches from before the October 6 runs; local and ignored by Git |

Run a notebook with this folder, `Code/stage1/`, as the working directory (the default in VS Code and Jupyter). The Imports cell finds `Code/shared_code/` from here, and the embedding paths are relative to this folder. See [Setup](../../README.md#setup).
