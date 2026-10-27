# Guide to the October 22 progress slides

A slide-by-slide explanation of `Mina_Progress_Slides_2026-10-22.pptx`, written after the October 22 meeting with Mina. Every slide gets the same five parts: what the slide shows, why we ran that experiment, how we ran it, what the numbers mean, and the answer to what Mina asked about it. The same text is in the speaker notes of each slide. Julian and Chase: use this to put each slide in your own words, and change anything that does not match how you would say it.

Every number here is read from the result files under `Code/stage4/results/` and `Code/stage3/results/`, and from the run records in `docs/run-records.md`. Nothing is quoted from memory.

## The whole deck in five sentences

1. We rebuilt BioLangFusion (the paper that fuses a DNA, an RNA and a protein language model on a shared codon grid) from its text, and our copy gives the same scores as the paper for its single encoders (RNA-FM on fungal aside) and for its cross-attention head, so the build is right.
2. The paper's best fusion head, the gated attention head, does not reproduce, and its own attention weights show why: on CoV-Vac and E. coli it keeps mostly one modality instead of mixing three, and on CoV-Vac it keeps the wrong one. Concatenation also falls short of the paper on fungal and CoV-Vac.
3. The paper's central claim, that fusing three encoders beats the best single encoder, holds in two of six settings we can measure (E. coli and GTEx), and the bigger of those (E. coli, +6.9 accuracy points) is matched by the encoder alone once mRNA-FM or CaLM replaces the paper's weak RNA encoder: the better encoder alone gains 7 to 8 points and fusion then adds nothing resolved.
4. Fusion helps where the second input is a different molecule or region (a genomic DNA window, a transcript and a protein on GTEx, resolved; a full transcript with its protein on half-life and translation efficiency, not yet tested with a paired interval), and a single jointly pretrained model (LucaOne) beats three separately pretrained encoders fused on mRNA stability.
5. So the project's question is shifting from "does fusion help" to "when does fusion help, and which encoder should be in each slot", which is what the alignment work in Stages 1 and 2 was set up to answer.

## Vocabulary used on every slide

- **Encoder, foundation model (FM), language model.** A neural network pretrained on millions of biological sequences without labels. We use it only to turn a sequence into numbers.
- **Frozen.** The encoder's weights are never updated by us. Everything we train sits on top of the frozen encoder's output. This is the premise of the whole project: can pretrained models be combined without retraining them.
- **Embedding.** The numbers an encoder produces. A **token embedding** is one vector per token (one per 6-mer for Nucleotide Transformer, one per nucleotide for RNA-FM, one per amino acid for ESM-2, one per codon for mRNA-FM and CaLM). A **pooled embedding** is the average over tokens, one vector per sequence. Stage 1 used pooled vectors; the BioLangFusion models use token embeddings.
- **Codon grid.** The shared coordinate system BioLangFusion puts the three streams on: one position per codon (three nucleotides, one amino acid). Slide 3 explains it.
- **Head.** The small trained network on top of the frozen embeddings. A **fusion head** combines three streams into one; the **TextCNN prediction head** turns the combined stream into a prediction. The three fusion heads (concatenation, gated attention, cross-attention) are explained once below and again on slide 6.
- **Probe.** A ridge regression fit directly on pooled embeddings. Cheap, linear, no training loop. Used on slides 8, 11 and 12.
- **Spearman correlation.** The metric on the regression benchmarks, where the target is a continuous number (an expression level, a stability score, a degradation rate). It measures whether the model put the sequences in the right order: rank the true values 1 to N, rank the predictions 1 to N, and correlate the two lists of ranks. It runs from -1 to 1: 1 is a perfect ordering, 0 is an ordering unrelated to the truth, -1 is exactly backwards. Ranks rather than raw values because what matters is which genes are expressed more, not the exact number, and because ranks do not change if the target is rescaled (for example log-transformed) and are not thrown off by a few extreme values. On fungal these models score about 0.80, so a 0.02 difference between two of them is small.
- **Accuracy.** The metric on E. coli, the only classification benchmark: expression was binned into low, medium and high, and accuracy is the percentage of test sequences put in the right bin. With three roughly balanced classes, guessing scores about 33%; our models score 47.8% to 58.7%. Spearman and accuracy are different units, which is why E. coli always sits in its own column or panel.
- **Pearson r.** Like Spearman but correlating the values themselves rather than their ranks, so it measures linear agreement. The mRNABench half-life and translation-efficiency sets report it (slide 12), so we match them.
- **Published split.** The train / validation / test assignment the benchmark ships with. We use it so our numbers are comparable to the paper's.
- **Seed.** The random starting point of a training run. Different seeds give slightly different models; averaging over seeds shows how much of a difference is just training noise.
- **Paired bootstrap, interval, resolved.** The way we decide whether a difference between two models is real. Both models are scored on the same test rows. We resample the test rows with replacement 2,000 to 4,000 times, score both models on each resample, and take the difference each time. The middle 95% of those differences is the interval. If the interval does not contain zero, the difference is **resolved**; if it does, the two models are a **tie** on this test split. "Paired" matters: two models scored on the same rows make correlated errors, so the difference between them is measured far more precisely than either score alone.
- **Δfusion.** Mina's definition from the research plan: the fused model's score minus the best single encoder's score, on the same rows, with an interval. Positive and resolved means fusion added predictive value under that protocol.
- **Leakage.** Test sequences that also appear in training. A model can score well on them by memory rather than by understanding. Slide 7 is about this.

## The three fusion heads, explained once

All three start from the same thing: three streams on the codon grid, one vector per codon per modality. Each head produces one fused stream, one vector per codon, that goes into the same TextCNN prediction head. The heads differ only in how they combine the three vectors at each codon.

**Concatenation (the paper's Section 2.2.1).** At each codon, glue the three vectors end to end into one long vector (the DNA vector is first passed through a small projection), then one linear layer brings it back to a common width. Nothing is thrown away. The TextCNN that follows sees every feature from every modality at every codon and learns which ones to use. This is the simplest head and the reference baseline in our research design.

**Gated attention pooling, also called entropy-regularized MIL (Section 2.2.2).** First, each modality's stream is projected to a common width. Then each modality is summarized by one vector, the mean over all its codons. A small gate network turns each summary into a single score; a softmax over the three scores gives three weights that add to one, one weight per modality for the whole sequence. The fused stream is the weighted sum of the three projected streams. So this head can only decide how much of each modality to keep; it cannot pick different features from different modalities at different codons. "MIL" is multiple instance learning (Ilse et al. 2018): the three modalities are treated as three instances in one bag, and attention decides how much each instance contributes. The entropy term is added to the training loss and pushes the three weights toward sharp, one-sided values.

**Cross-modal multi-head attention (Section 2.2.3).** The three projected streams are laid end to end along the codon axis into one long context (three times the length). Each modality's stream then attends into that whole context: every DNA codon can look at every DNA, RNA and protein codon and pull information from any of them. The three updated streams are averaged, added to a learned combination, and layer-normalized. This is the only head where information can move across modalities and across positions at the same time.

Order of expressiveness: gated attention (three numbers per sequence) < concatenation (all features kept, position by position) < cross-attention (all features, any position to any position).

All three are drawn, panel by panel, on slide 6, and the same figure is in both manuscripts (`docs/notes/figures/figures_overleaf/fusion_heads.pdf` for the page version, `fusion_heads_slide.pdf` for the wide one on the slide).

---

## Slide 1. Title: progress since October 7

**What the slide shows.** The one-paragraph summary of the whole deck, and three chips: what has results (plan priorities 1, 2, 5, and 3 and 4), and what is left to run (seeds on the four surviving Phase B gains, and the sequence-grouped stability re-split). All eight encoder swaps have run (slides 9 and 10).

**Why this framing.** Mina's research plan of October 9 lists five priorities in order. The deck follows that order so that she can see each item as a line in her own plan: Priority 1 is the BioLangFusion reproduction (Phase A) and the encoder swaps (Phase B); Priority 2 is LucaOne; Priorities 3 and 4 are full-transcript half-life and translation efficiency; Priority 5 is the mRFP control.

**One sentence to say.** "Everything on your list has a first result; the reproduction holds, and the paper's fusion gains hold on fewer tasks than the paper suggests, because a better single encoder closes the biggest of them."

---

## Slide 2. Timeline: what happened between October 7 and October 27

**What the slide shows.** Eight dated steps, in the order they were run.

**Why this order.** The plan's own rule is: reproduce first, then change one thing at a time. So the order was: adopt the plan and place it in the Gantt chart (October 9 to 15); build the pipeline (October 15 to 19); check that the pipeline reproduces the paper on the cleanest benchmark, fungal expression (October 17 to 18); run the two things that do not need the BioLangFusion grid at all, IsoFormer on GTEx and the LucaOne feasibility check (October 18 to 19); run the cheap control, mRFP (October 19); run Phase A on the other three benchmarks (October 19 to 20, 27 hours on Julian's laptop); then the first Phase B swaps, LucaOne, and the half-life and translation-efficiency probes on Chase's GPU (October 21 to 22); then the other six Phase B swaps (October 25 to 27).

**Why fungal expression was the build check.** It is the only benchmark with clean published splits (no test sequence appears in training), published values for all six model variants, and no preprocessing choice of ours in the way. If the build reproduces anywhere, it has to reproduce there.

**One sentence to say.** "We did these in the plan's order: reproduce, then swap one encoder, then the jointly pretrained baseline, then the new datasets, with the cheap control in parallel."

---

## Slide 3. What exists in code: BioLangFusion rebuilt from the paper

**What the slide shows.** The pipeline figure (three encoders, the codon grid, three heads, one TextCNN), and the five things we had to decide because the paper does not fully specify them.

**Why we did this.** Priority 1 asks for a faithful reproduction before any encoder is swapped. There is no official code, so the paper's text, equations and appendix lists are the specification. Wherever the paper contradicts itself, we had to pick a reading and measure which one is right.

**How the pipeline works, step by step.**

1. Three frozen encoders read the same coding sequence in three alphabets. Nucleotide Transformer v2 (100M, multi-species) reads the DNA letters as 6-mer tokens (one token per six nucleotides). RNA-FM reads the RNA letters one nucleotide per token. ESM-2 (8M) reads the translated protein one amino acid per token.
2. The three token streams are placed on one codon grid. A codon is three nucleotides and codes one amino acid, so the protein stream is already on the grid (one token per codon). The RNA stream is averaged in groups of three (three nucleotide tokens become one codon vector). The DNA stream has one token per two codons, so it is upsampled by a factor of two with a learned transposed convolution (the paper's Equation 1). The result is three streams of the same length, one vector per codon per modality.
3. One of the three fusion heads combines the streams (explained above and on slide 6).
4. The TextCNN prediction head reads the fused stream: three parallel convolutions of width 3, 4 and 5 codons, ReLU, max over positions, dropout 0.2, one linear layer to the output. The same head is used for every variant so that only the fusion changes.
5. Training follows the paper's recipe: Adam at learning rate 3e-5, weight decay 1e-5, batch 32, up to 500 epochs, early stopping on validation, best validation epoch restored, mean squared error for regression and cross-entropy for classification.

**The two inconsistencies settled by measurement.**

- *999-nucleotide rule.* The paper says it truncates sequences longer than 1,000 tokens. But the row counts it reports (for example 23,929 mRNA stability sequences used, 3,138 fungal) only come out if sequences longer than 999 nucleotides are dropped, not truncated. Dropping reproduces all four published counts exactly, so that is what we do.
- *The transposed convolution.* The paper specifies the DNA upsampling twice: Equation 1 (kernel 2, stride 2) and the appendix lists (kernel 3, stride 2, padding 2). We ran an impulse test: feed a single token and see where its output lands. The appendix setting places the DNA stream two codons out of register with the protein frame and three codons short of it; Equation 1 lands exactly. Under Equation 1 the two heads that combine streams position by position score higher than under the appendix setting (concatenation +0.015, gated +0.008) and cross-attention is unchanged, which is exactly the signature of an alignment fault. Equation 1 is used everywhere.
- Also noted: the paper gives its "fixed" TextCNN two different widths (1,280 channels for the gated head, 100 for cross-attention). We use 100 for every head so that capacity is matched; the gated head at 1,280 is a diagnostic (slide 6).

**Error bars.** The paper reports one number per cell with no error bar. Every one of our runs saves its per-row test predictions, so any two models can be compared with a paired bootstrap. This is what lets us say "resolved" or "tie" on later slides.

**One sentence to say.** "The paper is the only spec, it contradicts itself twice, and we settled both contradictions by measurement rather than by guessing."

---

## Slide 4. Phase A reproduction: cross-attention and the single encoders reproduce, RNA-FM on fungal aside; concatenation and the gated head do not

**What the slide shows.** Our test score next to the paper's, for every model variant on the four benchmarks we have (Ab1, the fifth, is not public). Format: ours (paper). Spearman correlation everywhere except E. coli, which is accuracy in percent. Cells more than 0.01 below the paper are shaded: RNA-FM on fungal, concatenation on fungal and CoV-Vac, and the gated head on fungal and CoV-Vac with and without its entropy term and on stability with it. Our values are means over seeds where three seeds exist (the fusion heads, NT on fungal, and the single encoder validation chose on CoV-Vac and E. coli); the other rows and all of stability are one seed. Below the table, two things. On the left, the manuscript's version of the same numbers (`r34_reproduction.pdf`), one panel per benchmark, filled marker ours and open marker published, so the size of each agreement or gap is visible rather than read off. On the right, a second small table built from the paper's own Table 1: each of its three heads minus its own best single encoder.

| Paper's Table 1, head minus its best single | CoV-Vac | Fungal | E. coli | Stab. | Ab1 |
| --- | --- | --- | --- | --- | --- |
| Concatenation | -0.010 | +0.001 | -2.67 | -0.014 | +0.021 |
| MIL + entropy | +0.023 | +0.020 | -0.84 | +0.010 | +0.017 |
| Cross-attention | -0.013 | +0.008 | +0.98 | -0.003 | +0.022 |

**Why that second table is on the slide.** The paper's sentence is "fusion-based models consistently outperform the best single-modality baselines." Its own numbers give nine wins and six losses across the fifteen comparisons. Concatenation loses on three of five, cross-attention on two of five, and only MIL with entropy reaches four of five, losing E. coli. The claim holds only if you take the best head per task after seeing the test scores, and a different head wins on different tasks, so no method you could have committed to in advance wins everywhere. Add that there is no interval anywhere in the paper and that four of the five winning margins are 0.010 to 0.023, and the claim is weaker than the sentence. Two further details worth having ready: RNA-FM is the best single encoder on four of the five tasks, so "the best single baseline" is essentially RNA-FM throughout, and the head credited with beating it is MIL, the head that in our hands gives RNA almost no weight on CoV-Vac, E. coli and stability. And all three heads are ahead only on fungal (by +0.001 to +0.020, noise-sized) and on Ab1, the one task not in the public release.

**Why we did this.** This is the proof that the build is the paper's. If the single encoders alone give the paper's numbers, then the token caches, the splits, the codon grid, the TextCNN and the training loop are all right, because a single-encoder run uses all of them except the fusion head. If a fusion head then misses, the miss is in the head or in the paper's description of it, not in our plumbing.

**How to read the table.**

- Rows 1 to 4 are single encoders. "RNA-FM read on the codon grid" is RNA-FM with its nucleotide tokens averaged three to one before the TextCNN, which the paper does not report but which we need because that is how RNA-FM enters every fusion head.
- Rows 5 to 8 are the three fusion heads, with the gated head shown with and without its entropy term (the paper's Table 4 ablation).
- "Within 0.01" is what we count as reproduced: for a Spearman correlation on a few hundred test rows the 95% interval of the score itself is about plus or minus 0.03, so 0.01 is well inside noise. E. coli has 712 test rows and accuracy carries about plus or minus 3.7 points, so E. coli is not shaded: no E. coli score is that far below the paper, and two are that far above it (NT alone 47.8 against 41.3, concatenation 58.7 against 50.3).

**What the numbers mean.**

- Every single encoder matches the paper within 0.01 on CoV-Vac and stability, Nucleotide Transformer matches on fungal and ESM-2 lands 0.011 above it. The one single-encoder number we cannot place is RNA-FM on fungal. Read as the encoder writes it, one token per nucleotide, it is 0.722 against the paper's 0.767; pooled three to one onto the codon grid, the way every fusion head sees it, it is 0.794. The published value sits between our two readings, and both of ours are a single seed there. On the other three benchmarks the paper's RNA-FM numbers track the unpooled reading (CoV-Vac 0.841 against our 0.838, E. coli 52.9 against 50.7, stability 0.553 against 0.552), so the paper's baselines look unpooled and fungal is the outlier. Avoid saying the paper's baseline was codon-pooled: that would explain fungal and contradict CoV-Vac.

**What the RNA-FM-on-the-grid row is for, in one line of arithmetic.** Every fusion head does two things to RNA-FM: it averages the stream three to one onto codons, and it combines it with DNA and protein. When a fusion head beats RNA-FM alone, either step could be responsible. The grid row is RNA-FM with only the first step applied, so it splits the gap in two: fusion minus RNA-FM as written = (grid row minus RNA-FM as written) + (fusion minus grid row) = what the averaging did + what the fusion did. On the clean stability rows: 0.359 - 0.319 = 0.040 = (0.340 - 0.319) + (0.359 - 0.340) = 0.021 + 0.019. Half the apparent fusion gain is the averaging. Without the row, the averaging's effect gets credited to fusion.

**Why RNA-FM is the only encoder with two rows (worth having ready, since it looks arbitrary).** The codon grid exists to make three streams the same length so a head can combine them position by position. A single encoder has nothing to line up against, so an "alone" row is just that encoder's own tokens going into the TextCNN. ESM-2 writes one token per amino acid, which is already one per codon, so its native and gridded readings are the same object. Nucleotide Transformer writes one token per 6-mer, one per two codons, and putting it on the grid requires the learned transposed convolution that only exists inside a fusion model, so alone it is read natively. RNA-FM writes one token per nucleotide, three per codon, and pooling them three to one is a fixed average that needs no training, so both readings are available and they are genuinely different inputs: the TextCNN's kernels of 3, 4 and 5 span 3 to 5 nucleotides on the native stream and 3 to 5 codons, 9 to 15 nucleotides, on the gridded one. Same encoder, different receptive field.

- Pooling is not uniformly better, which is why it is reported as two rows and not as a correction. It gains 0.072 on fungal and costs a little everywhere else: CoV-Vac 0.823 against 0.838, E. coli 49.3 against 50.7, stability 0.547 against 0.552. The natural reading is that fungal expression is a codon-scale signal and CoV-Vac degradation is a nucleotide-scale one, but E. coli, which is also codon-scale, does not follow that, so treat it as an observation rather than a rule.
- Cross-attention reproduces everywhere: nowhere more than 0.01 below its published value, and above it on three benchmarks.
- The gated head, which the paper reports as its best head on three of five benchmarks, is the best head on none here. It is below the paper on fungal (0.801 against 0.824), far below on CoV-Vac (0.783 against 0.864), and 0.012 low on stability. On E. coli it is above the paper (54.5 against 52.1), but every E. coli difference in this table is inside the plus or minus 3.7 points of accuracy noise, so that cell decides nothing either way. So it misses clearly on fungal and CoV-Vac and by 0.012 on stability, and CoV-Vac is the largest miss in the whole reproduction. Slide 6 explains why.
- The same holds without the entropy term, which is the paper's own ablation (row 7, not a different attention mechanism): fungal 0.802 against 0.826 and CoV-Vac 0.792 against 0.854 still miss, and stability matches (0.558 against 0.556). Turning the entropy term off does not rescue the head, which is the point of showing both rows.
- Concatenation also falls short on fungal (0.791 against 0.805) and CoV-Vac (0.799 against 0.831), and is above the paper on E. coli (58.7 against 50.3) and stability (0.558 against 0.539).

**Mina's note: "another table."** The point of this table is only the comparison with the paper, not the comparison between heads. Read it as: do the single-encoder rows match (yes, RNA-FM on fungal aside), does cross-attention match (yes), do the other two heads match (no: both fall short on fungal and CoV-Vac). Slide 5 is the table for the fusion question.

**One sentence to say.** "Our build gives the paper's numbers for its single encoders, RNA-FM on fungal aside, and for cross-attention; concatenation and the paper's headline head, gated attention, fall short on fungal and CoV-Vac."

---

## Slide 5. The paper's central claim: does fusion beat the best single encoder?

**What the slide shows.** The manuscript's forest plot, Figure 13 in the short document. Every row is one fusion head minus the best single encoder on that benchmark, scored on the same test rows. The dot is the difference, the line is the 95% paired-bootstrap interval. A filled marker means the interval excludes zero, so the difference is resolved; an open marker means it contains zero, so the comparison is a tie at this sample size. The marker shape says which head it is. Panel (b) is E. coli on its own because it is a classification task, so its differences are in accuracy points rather than Spearman.

**The table under the figure.** The same comparisons as numbers, which is the manuscript's Table 15. Each cell is one head minus the best single encoder, with the 95% paired-bootstrap interval beside it. Shaded cells are the four that resolve.

| Head minus best single | Fungal | CoV-Vac | E. coli (points) | Stability, published |
| --- | --- | --- | --- | --- |
| Concatenation | -0.015 [-0.049, +0.017] | **-0.038 [-0.066, -0.012]** | **+6.9 [+4.2, +9.6]** | +0.007 [-0.006, +0.019] |
| Gated MIL | -0.005 [-0.041, +0.031] | **-0.055 [-0.093, -0.019]** | **+2.8 [+0.9, +4.8]** | -0.001 [-0.014, +0.012] |
| Cross-attention | +0.016 [-0.014, +0.048] | -0.002 [-0.026, +0.020] | +1.8 [-0.9, +4.5] | +0.001 [-0.012, +0.015] |

**What the interval is, if Mina or Chase asks.** We only ever get one test set, and the answer depends on which sequences happen to be in it. The bootstrap estimates how much that matters. Take the 430 fungal test rows, draw 430 of them at random with replacement (some appear twice, some not at all), score both models on exactly that resampled list, and write down the difference. Do that 4,000 times, sort the 4,000 differences, and trim 2.5% off each end. What is left is the interval. Both models are always scored on the same resampled rows, so difficulty that affects both cancels and the difference is measured far more precisely than either score alone. Repeated sequences are resampled together, which matters on stability where the same sequence recurs. Where there are three seeds they are averaged inside each draw. The width is driven almost entirely by how many test rows there are: 430 rows on fungal gives about plus or minus 0.03, 3,579 rows on stability about plus or minus 0.013, 712 rows on E. coli about plus or minus 2.7 accuracy points. One limit to state before she asks: this covers sampling of the test rows only, not how the benchmark was built or the splits drawn, so a resolved difference means "not explained by which sequences we tested on," which is a lower bound on the real uncertainty.

**Reading one row out loud, concatenation.** Fungal -0.015 with an interval from -0.049 to +0.017: the gap is a sixth of the interval's width, so the benchmark cannot settle it, and roughly one resample in five came out the other way. CoV-Vac -0.038 with an interval entirely below zero: concatenation is really worse than RNA-FM alone there. E. coli +6.9 entirely above zero: a real win, and the one a better RNA encoder alone matches in Phase B. Stability +0.007 straddling zero: a tie, on the split with 68% overlap. So concatenation is one for four, and the one does not survive a better encoder. The habit to teach: look at whether the interval crosses zero before reading the number. If it crosses, the number is not a result.

**Which rows are BioLangFusion and which are not.** The slide tests BioLangFusion's central claim on its four public benchmarks, and all four are there: fungal, CoV-Vac and stability in panel (a), E. coli in panel (b), separate only because it is scored in accuracy rather than Spearman. The table under the figure contains exactly those four and nothing else. Panel (a) carries two extra rows that are not BioLangFusion benchmarks, and they are the ones that confuse people. *Stability, unseen sequences* is not a fifth benchmark; it is the same stability models rescored on only the 1,135 test rows whose sequence never appears in training, because the published split puts copies of the same sequence on both sides (slide 7). *GTEx pilot* is not from BioLangFusion at all; it is IsoFormer's setup, where the three inputs are three different molecules, and it is on the slide as the contrast case, the one place fusion clearly wins. It is suggestive, not a controlled comparison: GTEx changes the inputs and the fusion method at the same time (BioLangFusion's heads need a codon grid, which a genomic window does not have), so the difference in outcome cannot be pinned on the inputs alone. Inside GTEx the comparison is fair, fusion against the best single modality with the same encoders, split and head. The controlled version of the cross-benchmark question is the project's IsoFormer-style cross-attention (Gantt 25, January) run on BioLangFusion's four benchmarks, which needs no grid, so it holds the method fixed and changes only the inputs. Read the nine rows above them as the test of the paper, and those two as context for it.

**How to read the figure in one pass.** Look at which markers are filled, and which side of the dashed zero line they sit on. Three are filled on the right: E. coli concatenation, E. coli gated, and GTEx. Two are filled on the left: CoV-Vac concatenation and CoV-Vac gated, where fusion is resolved *worse* than RNA-FM alone. The other nine sit on the line, which means this is a tie, not a small win. The E. coli pair is the win a better RNA encoder alone matches in Phase B, which is why the figure annotates it.

**Why the figure has fourteen rows and the table below has seven.** The figure asks the question per head: three heads times three Spearman benchmarks, plus stability on unseen sequences, plus GTEx, plus three heads on E. coli. The table collapses it to one row per benchmark by taking the head validation preferred, which is how the paper states its own claim. Both are in the manuscripts. Same answer either way.

| Benchmark | What the three encoders read | Best single encoder | Best fusion head | Fusion minus single, 95% interval | Verdict |
| --- | --- | --- | --- | --- | --- |
| Fungal expression | one coding sequence | NT, 0.807 | cross-attention, 0.823 | +0.016 [-0.014, +0.048] | tie |
| CoV-Vac | one coding sequence (81-nt fragments) | RNA-FM, 0.838 | cross-attention, 0.836 | -0.002 [-0.026, +0.020] | tie |
| mRNA stability, published split | one coding sequence | RNA-FM, 0.552 | concatenation, 0.558 | +0.007 [-0.006, +0.019] | tie, on a split with 68% train/test overlap |
| mRNA stability, unseen sequences | one coding sequence | RNA-FM, 0.319 | concatenation, 0.359 | +0.040 [-0.003, +0.083] | tie, leans fusion; one seed |
| E. coli, the paper's encoders | one coding sequence | ESM-2, 51.7% | concatenation, 58.7% | +6.9 [+4.2, +9.6] points | fusion wins |
| E. coli, mRNA-FM in the RNA slot | one coding sequence | mRNA-FM, 58.7% | concatenation, 59.0% | +0.3 [-2.9, +3.7] points | tie: the gain is gone |
| GTEx, IsoFormer's aggregator | three different molecules | RNA, 0.452 | DNA + RNA + protein, 0.499 | +0.047 [+0.027, +0.069] | fusion wins |

The figure on the slide is `docs/notes/figures/figures_overleaf/r34_fusion_forest.pdf`, the same file the manuscripts use, made by `docs/notes/figures/make_r34_figures.py`. The table above is drawn separately as `fusion_vs_single.pdf` and appears in the manuscripts beside it.

**Why we did this.** The paper's claim is not "fusion scores 0.82" but "fusion beats each encoder alone." That is a paired comparison on the same test rows, and the paper gives no error bar for it. We compute the paired bootstrap for the best head against the best single encoder, with seeds averaged inside each resample, so that each verdict is a statement with uncertainty attached.

**How we did it.** For each benchmark: the best single encoder is chosen on the validation split (not the test split, to avoid picking the winner after the fact). The best head is likewise the one validation prefers. Both are scored on the published test split; 4,000 paired resamples; identical sequences are resampled together so that a repeated sequence does not count as independent evidence; the three seeds are averaged inside each draw where three exist.

**What the verdicts mean, benchmark by benchmark, and why each came out that way.**

- *Fungal expression: tie (+0.016, interval -0.014 to +0.048).* All three streams are readings of one coding sequence. Nucleotide Transformer alone already carries what the task needs (0.807), and neither the RNA letters nor the translated protein add information it did not have. The paper's +0.020 sits inside our interval, so the paper's own gain here is the size of noise on a 430-row test split.
- *CoV-Vac: tie for cross-attention, and concatenation and the gated head are significantly worse than RNA-FM alone.* These are 81-nucleotide fragments scored for degradation. Degradation of a short RNA is about its structure (which positions are unpaired), which is what RNA-FM, a nucleotide-level model pretrained on structured non-coding RNAs, is good at. The "protein" of an 81-nucleotide fragment is a benchmark construction (a 27-residue peptide read in an arbitrary frame), and the DNA stream is the same letters at a coarser resolution. Adding them adds noise.
- *E. coli: fusion won in Phase A (+6.9 accuracy points, interval +4.2 to +9.6).* This was the one clear win, and slides 9 and 10 show it was fusion compensating for a weak RNA encoder rather than fusion adding information: with mRNA-FM in place of RNA-FM the single encoder is 8 points higher and the fusion gain is +0.3, a tie.
- *mRNA stability, published split: tie (+0.007).* But this split repeats a training sequence in 68% of its test rows, so every model's score here is mostly memory (slide 7).
- *mRNA stability, unseen sequences: leans fusion (+0.040, interval -0.003 to +0.083).* On the 1,135 test rows whose sequence never appears in training, the fusion heads sit 0.03 to 0.04 above RNA-FM as the encoder writes it, just short of resolving at one seed. Against RNA-FM on the codon grid (0.340 on these rows), the input every head actually receives, the gap is 0.007 to 0.019, so most of the lean is the pooling, not the fusion.
- *GTEx with IsoFormer's aggregator: fusion wins (+0.047, interval +0.027 to +0.069).* The only setting where the three inputs are physically different molecules (slide 13).

**The pattern, which is the interpretation Mina asked for.** Fusion of three readings of one coding sequence rarely beats the best single encoder once that encoder is a strong one, because the three encoders are seeing largely the same information in three alphabets (slide 10 has four single-seed exceptions). Fusion beats the best single encoder when the inputs carry information no single encoder could have seen (the genomic window around the transcription start site on GTEx; the untranslated regions on half-life and translation efficiency).

**Mina's note: "Interpret why these work better. For example: if someone wants to train a foundation model, we should train on task-specific datasets."** The evidence partly supports her reading. Among the paper's three encoders, RNA-FM (pretrained on non-coding RNA, structure-rich) wins CoV-Vac, a structure-driven degradation task on RNA fragments, and Nucleotide Transformer (pretrained on whole genomes of many species) wins fungal expression. With the Phase B swaps, mRNA-FM and CaLM (pretrained on coding sequences with codon tokens) are the best single encoders on both fungal and E. coli (slide 9), and mRNA-FM carries by far the most synonymous-codon information in the mRFP control (slide 8). But RiNALMo, pretrained on non-coding RNA at nucleotide level like RNA-FM, lifts fungal and E. coli almost as much, so pretraining domain is not the whole story. The practical rule: pick the strongest single encoder for the task first, then ask whether to fuse, and expect fusion to add something mainly when the second input is a different molecule or region.

**Mina's note: "Did it report 1,000 rows or 3,000 rows in BioLangFusion?"** This is about the stability rows and is answered on slide 7: the paper's number is on the full 3,579-row published test split.

**One sentence to say.** "Fusion beats the best single encoder on two of six settings, and the larger of those two is matched by a better single encoder."

---

## Slide 6. How the three fusion heads work, and why the gated one mostly keeps one modality

**What the slide shows.** Two things. On top, one figure with three panels, one per fusion head, all starting from the same three streams on the codon grid and ending in the same fixed TextCNN. Below it, the measured gated weights for every benchmark, which show that with the paper's encoders the head mostly selects rather than mixes.

**How to read the figure, panel by panel.** Panel a, concatenation: at each codon the three vectors are glued end to end into 1,560 numbers (DNA is first projected from 512 to 600) and one linear layer maps that to 600. Nothing is discarded, and the TextCNN decides which features of which modality to use. Panel b, gated attention: each stream is averaged over its codons into one summary, a gate scores the three summaries, a softmax turns the scores into three weights that add to one, and those same three weights multiply the full streams codon by codon. The weights drawn (DNA 0.11, RNA 0.00, protein 0.89) are the measured E. coli weights for seed 0, and the grayed-out RNA row is what a weight of zero does: the whole stream is gone from every position. Panel c, cross-modal attention: the three streams are laid end to end into one context of 3T positions, every position of every modality queries that context, and the updated streams are merged back; arrow width stands for attention weight. The same figure is Figure 12 in the short manuscript and Figure 31 in the long one, and the file is `docs/notes/figures/figures_overleaf/fusion_heads_slide.pdf`.

**The weights behind the footer** (entropy weight 0.01, averaged over the test split; three values are seeds 0, 1 and 2; stability is one seed).

| Benchmark | DNA | RNA | Protein |
| --- | --- | --- | --- |
| Fungal | 0.20 / 0.16 / 0.14 | 0.20 / 0.25 / 0.26 | 0.61 / 0.59 / 0.60 |
| CoV-Vac | 0.83 / 0.92 / 0.94 | 0.01 / 0.03 / 0.00 | 0.16 / 0.05 / 0.06 |
| E. coli | 0.11 / 0.18 / 0.10 | 0.00 | 0.89 / 0.82 / 0.90 |
| Stability | 0.28 | 0.00 | 0.71 |

At entropy weight 1.0, one modality takes 1.00 and the other two 0.00 on every benchmark; at 0.5 the same happens on CoV-Vac, E. coli and stability, while fungal keeps RNA 0.28 and protein 0.72. The two diagnostic numbers are the CoV-Vac miss (0.783 against 0.864 published) and the width test on fungal (0.801 at 100 channels, 0.812 at 1,280, against 0.824 published).

**What the gated head is, in detail (Mina's question "what is the gated head and how does it work").** After the three streams are projected to a common width, each stream is summarized by one vector, the mean of its codon vectors. Each summary goes through a gate: tanh(V h) multiplied elementwise by sigmoid(U h), then a linear score W. The three scores go through a softmax (we also learn a temperature, which the paper does not mention), giving three weights that add to one. The fused stream is weight_DNA times the DNA stream plus weight_RNA times the RNA stream plus weight_protein times the protein stream, codon by codon. The whole decision is three numbers per sequence. The paper calls these weights an interpretability output: "how much each modality is used."

**Yes, it is the MIL attention (Mina's question "is gated MIL attention related?").** The gate is the attention-based multiple instance learning pooling of Ilse, Tomczak and Welling (2018), where a bag of instances gets one attention weight per instance and the bag representation is the weighted sum. BioLangFusion uses the same equations with the three modalities as the three instances of one bag. That is why the paper calls it "MIL."

**The entropy term.** The loss is the task loss plus lambda times the negative entropy of the three weights. Lower entropy means sharper weights, so the term pushes the head toward putting most of the weight on one modality. The paper says this makes the weights "sharp and task-specific." Lambda is chosen on validation among 0.01, 0.5 and 1; validation chose 0.01 on every benchmark. At 1.0 one modality takes a weight of 1.00 and the others 0.00 on every benchmark, and at 0.5 on three of the four.

**What the weights actually show here.** Even at lambda 0.01 they are concentrated on CoV-Vac, E. coli and stability: RNA gets almost nothing on those three (CoV-Vac at most 0.03, E. coli 0.00, stability 0.00), DNA dominates CoV-Vac, protein dominates E. coli and leads on stability (0.71). Fungal is the one benchmark where all three streams keep a share. On E. coli the head keeps protein (0.89). On CoV-Vac it keeps DNA (0.83 to 0.94 across seeds) and scores 0.783, almost exactly the DNA-only score of 0.779, even though RNA-FM alone is the best single encoder at 0.838. So on CoV-Vac the head selected against the best encoder, and that one choice is the largest miss in the whole reproduction. The entropy term never helps: on all four benchmarks the head without it scores at or above the head with it.

**Why RNA goes to zero (Mina's question "E. coli, why is RNA 0").** Three things combine. First, the softmax plus the entropy penalty is a winner-take-all mechanism: once one modality's weight pulls ahead early in training, the other two get almost no gradient and cannot recover, which is the same collapse seen in gating networks generally. Second, the RNA-FM stream carries the least codon information of the nucleotide encoders: in the mRFP control (slide 8) RNA-FM has the smallest gain above all controls of any nucleotide encoder (+0.021), and on E. coli it is also redundant with the DNA stream, which is the same letters read as 6-mers. Third, the gate decides from a mean over all codons, which erases exactly the positional detail that makes a stream useful. So the head finds nothing in the RNA summary that the protein summary does not already give it, and it drops RNA. The clearest demonstration is in Phase B: with mRNA-FM in the RNA slot on E. coli, mRNA-FM alone is the best single encoder at 58.7%, and the gated head still gives RNA a weight of 0.0007 and scores 53.1%, five points below the encoder it threw away.

**How this relates to fusion and concatenation (Mina's question "how does it help with fusion / concatenation").** The gated head does not add information; it reweights. Because its weights are concentrated, it behaves like a single encoder plus a projection, and its score tracks the single-encoder score of whichever modality it kept. Concatenation keeps all three streams at every codon and lets the TextCNN pick features from any of them, so where the modalities are complementary (E. coli in Phase A) concatenation wins by seven points and the gated head by three. Cross-attention can move information across modalities and positions, so it is the only head that never loses to the best single encoder in Phase A. Cross-attention resolves above the gated head on fungal (+0.022, interval +0.009 to +0.036) and CoV-Vac (+0.052), the reverse of the paper's ordering of those two; concatenation is the best head on E. coli, resolved above both others; on stability the three heads tie.

**The width question.** The paper's lists give the gated head a 1,280-channel TextCNN and the other heads 100. Running the gated head at 1,280 on fungal recovers about half of its gap to the paper (0.801 to 0.812 against 0.824). Running it at 1,280 on the other three benchmarks is the last check before the verdict on the head is final.

**One sentence to say.** "With the paper's encoders the gated head is a three-number switch that mostly keeps one modality; it gives RNA almost nothing on CoV-Vac, E. coli and stability, and on CoV-Vac it drops the best encoder, which is the paper's largest unreproduced number."

---

## Slide 7. mRNA stability: the published split is mostly memorization

**What the slide shows.** Panel (b) of the manuscript's leakage figure (`r34_gated_leak.pdf`): the same eight trained models scored twice, once on the whole published test split (3,579 rows, filled marker) and once on the 1,135 of those rows whose sequence never appears in training (open marker). The number above each line is the drop. Every model loses about 0.2, and the shaded block at the bottom is the fusion heads, which lose as much as the single encoders do.

**Why we did this.** The Stage 3 dataset audit found that the published mRNA stability split repeats a training sequence in 2,444 of its 3,579 test rows (68%), usually with a different label (the same sequence measured again, median label spread 0.49). Validation leaks the same way (2,360 of 3,608 rows), so early stopping was also watching a leaky split. A score on this split is therefore mostly a test of whether the model remembers sequences it has seen, not whether it understands stability. We had to know how much of the published 0.55 is memory.

**How we did it.** No retraining. The models trained on the published split were scored on all 3,579 test rows (the published number) and then on only the 1,135 unseen-sequence rows. One seed.

**What the numbers mean.** Published-split scores cluster at 0.53 to 0.56 for every model, ours and the paper's, and the range is carried by the repeated rows. On unseen sequences the scores are 0.27 to 0.36. That is the honest generalization range for this benchmark, 0.2 to 0.27 below the published-split scores. On the unseen rows the three fusion heads (0.347 to 0.359) sit above RNA-FM as the encoder writes it (0.319) by 0.03 to 0.04, just short of resolving at one seed (concatenation +0.040, interval -0.003 to +0.083). But RNA-FM pooled onto the codon grid, the stream every fusion head actually receives, scores 0.340 on the same rows, so the part of that gap fusion itself adds is 0.007 to 0.019. Whether the leak hides a real fusion gain is open, and the sequence-grouped re-split with seeds is the run that settles it.

**Mina's question: "Did it report 1,000 rows or 3,000 rows in BioLangFusion?"** The 3,000. BioLangFusion reports its stability numbers on the full published test split, which is 3,579 rows after its own 999-nucleotide filter. The evidence is that our full-split numbers match the paper's within 0.01 for every single encoder (RNA-FM 0.552 against 0.553, Nucleotide Transformer 0.539 against 0.530, ESM-2 0.531 against 0.539). The 1,135 unseen rows are our construction; the paper never scores them, and no published number exists for them. This matters for how the paper's stability column should be read: it is the score on a split where two thirds of the test sequences were seen in training.

**One sentence to say.** "The paper's 0.55 on stability is on a split where 68% of test sequences are in training; on the sequences the models have never seen, everyone is at about 0.3, and fusion is slightly ahead."

---

## Slide 8. The mRFP synonymous control (plan item 5)

**What the slide shows.** The manuscript's mRFP figure (`r34_mrfp.pdf`), both panels. Panel (a) is what each encoder reads from the 1,000 synonymous recodings on its own, held-out R-squared, with the two control bars marked as dashed lines: the Track A controls alone reach 0.190 and all controls together 0.306. Panel (b) is the part that matters, the gain each encoder adds *above every control*, with the count of fold assignments in which that gain stayed above zero printed beside each bar (10/10 for almost all of them). mRNA-FM adds 0.18, the other nucleotide encoders 0.02 to 0.08, and the four protein encoders exactly 0.000, marked in the figure as the control holding.

**Why we did this.** Mina's plan item 5. The mRFP data set is one gene (a red fluorescent protein) recoded 1,000 ways with synonymous codons, so every sequence encodes the identical protein, and the measured expression differs only because of the codons chosen. This gives two clean tests. The protein encoders must be invariant: they see the same amino-acid string every time, so they must predict nothing (a negative control that checks the pipeline). And a nucleotide encoder is only interesting if it predicts expression beyond what simple codon statistics already predict; otherwise it has just memorized codon frequencies.

**How we did it.** Stage 1's pooled embeddings of the twelve encoders. Controls: GC content, GC3 (GC at the third codon position), the 64 codon frequencies, a Nussinov base-pairing proxy on the first 200 nucleotides, and ViennaRNA minimum free energy of the first 60 nucleotides and of the whole sequence. The controls together reach R-squared 0.306. For each encoder, a ridge probe is fit on the controls with inner cross-validation, the encoder is fit on the residual, and the gain in held-out R-squared is recorded over ten grouped fold assignments (identical sequences in one fold).

**What the numbers mean.** The protein control holds: 0.000 gain, in every fold assignment. Every nucleotide encoder beats the Track A controls (GC3, codon use, Nussinov) in 10 of 10 fold assignments, so all of them carry codon information beyond frequencies. mRNA-FM carries far more than the others (+0.18 above every control, +0.26 above the Track A set), and RNA-FM, BioLangFusion's own RNA encoder, carries the least (+0.02). Among the controls, the one structure feature that matters is the folding energy of the first 60 nucleotides after the start codon (R-squared 0.11 on its own; a weaker fold means higher expression), which is the classic Kudla et al. 2009 result for this kind of library. The Nussinov proxy carries nothing.

**Why mRNA-FM.** mRNA-FM and CaLM read coding sequences one codon per token and were pretrained on coding sequences, so their vocabulary is codon identity in context. RNA-FM reads single nucleotides and was pretrained on non-coding RNA, so synonymous codon choice is not something its pretraining ever rewarded. Nucleotide Transformer reads 6-mers across whole genomes, which contain codon structure but dilute it. That explains mRNA-FM's lead over RNA-FM, but pretraining domain does not order the rest: CaLM, also codon-level and pretrained on coding sequences, adds only +0.022 above all controls, less than NT 500M (+0.076) or RiNALMo (+0.049), and gains in 9 of 10 fold assignments rather than 10.

**Mina's note: "mRNA-FM performs best; maybe we want to do fusion against mRNA-FM, like DNA or RNA fusion with mRNA-FM; try different mRNA encoders as well."** Agreed, and this is already the direction Phase B took: mRNA-FM went into the RNA slot first because this control pointed to it (slide 9). The follow-up she is asking for is to make mRNA-FM the anchor: DNA + mRNA-FM, mRNA-FM + protein, and mRNA-FM beside the other RNA encoders already swapped in (CaLM, codon-level; RiNALMo, nucleotide-level), on the coding-sequence benchmarks. That goes into the plan as a Phase B extension.

**One sentence to say.** "Every nucleotide encoder knows something about codons beyond their frequencies, mRNA-FM knows by far the most, and the protein encoders know nothing, exactly as they should."

---

## Slide 9. Phase B, the encoder question: all eight swaps read alone

**What the slide shows.** One panel per benchmark, one row per swap. The open marker is the original encoder read alone, the filled marker is the encoder swapped in for it, colored by slot (blue DNA, green RNA, orange protein). A long line to the right means the swap is much better on its own. The table underneath is the same numbers, with the encoder it replaces in brackets. This slide is only about single encoders; slide 10 asks what fusion does on top of them.

**What ran.** Eight single-encoder swaps. DNA slot: NT 500M human-ref and HyenaDNA large in place of NT v2 100M. RNA slot: RiNALMo 150M, mRNA-FM and CaLM in place of RNA-FM. Protein slot: ESM-2 35M, ESM-2 150M and ProtBERT in place of ESM-2 8M. Each on fungal, CoV-Vac and E. coli, with mRNA-FM, CaLM and ESM-2 150M also on mRNA stability: 27 swap-benchmark pairs. By October 22 Chase had run two of the swaps, mRNA-FM and CaLM, on CoV-Vac and E. coli; the rest ran between October 25 and 27, seed 0, everything else at Phase A's settings. The bracketed reference values are the seed-0 runs the sweep compares against, so NT v2 on fungal reads 0.813 here against slide 4's seed mean of 0.807. DNABERT-2 is out of the sweep because its variable-length byte-pair tokens need a grid adapter. The figure is `r34_phaseB_alone_slide.pdf`, from `docs/notes/figures/make_phaseB_full_figure.py`.

**Why we did this.** Priority 1, Phase B: hold splits, preprocessing, heads and training fixed and change one encoder, so any difference is attributable to that encoder. The RNA slot went first because the mRFP control (slide 8) predicted mRNA-FM would carry the most codon information.

**What the numbers mean.**

- *The RNA slot is where the score moves.* On fungal, RiNALMo, mRNA-FM and CaLM alone score 0.844, 0.852 and 0.851 against RNA-FM's 0.722. That is the largest single effect in Stage 4, and it clears both NT v2 100M alone (0.813) and the paper's best fused model on fungal (0.824). On E. coli the same three lift the single encoder from 50.7% to 57.3%, 58.7% and 57.6%: mRNA-FM alone is +8.0 points over RNA-FM alone [+4.0, +11.8], CaLM +6.9 [+2.8, +10.8], both resolved.
- *The DNA slot moves the wrong way.* NT 500M and HyenaDNA are worse than NT v2 100M on fungal (0.740 and 0.737 against 0.813) and on CoV-Vac (0.727 and 0.755 against 0.779).
- *The protein slot moves less.* ESM-2 35M stays near ESM-2 8M everywhere; ESM-2 150M is better on fungal (0.729 against 0.698) and E. coli (53.9% against 50.1%); ProtBERT is better on fungal (0.761) but worse on CoV-Vac (0.776 against 0.816). None comes near the RNA swaps' gains.
- *CoV-Vac disagrees with the others about which RNA encoder is best,* and that is the point. There the RNA swaps do not help: mRNA-FM alone is 0.812 against RNA-FM's 0.833 and CaLM 0.844, both ties. CoV-Vac degradation is the folding of an 81-nucleotide fragment, which nucleotide-level RNA-FM, pretrained on structured non-coding RNA, already reads well. On fungal and E. coli, mRNA-FM and CaLM, pretrained on coding sequences with codon tokens, are the two best single encoders, but RiNALMo, pretrained on non-coding RNA like RNA-FM, gains almost as much, so pretraining domain is not the whole explanation.

**Mina's notes on this slide.**

- *"We want to see if fusion replaces RNA-FM and see if it works."* Replacing RNA-FM with a codon-level encoder is worth eight points on E. coli on its own. Whether fusion then adds anything is slide 10: with mRNA-FM or CaLM in the slot, no; with RiNALMo, two single-seed E. coli gains remain.
- *"Check RNA-FM."* RNA-FM is the weakest of the four RNA encoders on codon information in the mRFP control (+0.02 above the controls against +0.18 for mRNA-FM), and this slide is that prediction confirmed on trained models.

**One sentence to say.** "Changing the RNA encoder moves fungal by 0.13 and E. coli by up to eight points; no DNA or protein swap comes close, and the DNA swaps make fungal worse."

---

## Slide 10. Phase B, the fusion question: all eight swaps against their best single encoder

**What the slide shows.** A grid: each fusion head minus the best single encoder validation chose for that configuration, one seed, with the 95% paired interval deciding the shading. Blue means fusion is ahead and the interval excludes zero; orange means fusion is behind and the interval excludes zero; gray is a tie. The dashed top row is the original encoders at seed 0, for reference, so its E. coli cells are seed-0 values rather than slide 5's three-seed means. Underneath, the four gains that survive scrutiny, with their intervals.

**How to read the tally.** 81 comparisons: 15 resolve with fusion ahead, 14 with fusion behind, 52 are ties. The 15 have to be split, because two groups are artifacts of what fusion was compared against rather than fusion working.

| Group | Count | Why it does not count |
| --- | --- | --- |
| Fungal with ProtBERT (marked †) | 3 | Validation chose ProtBERT alone (0.761) as the reference although NT v2 100M alone scores 0.813 on test; against NT v2 these are ties |
| E. coli with the DNA and protein swaps | 8 | RNA-FM still fills the RNA slot in all of them, so fusion is compensating for a weak encoder, as in Phase A |
| **What survives** | **4** | **Not explained by a weak reference or a weak slot** |

The contrast that proves the second group: with mRNA-FM or CaLM in the RNA slot, no head gains on E. coli, and two are resolved worse (gated -5.6 with mRNA-FM, cross-attention -3.7 with CaLM).

**The four that survive.**

| Swap | Benchmark | Head | Over the best single encoder | Reaches |
| --- | --- | --- | --- | --- |
| HyenaDNA (DNA slot) | CoV-Vac | cross-attention | +0.039 [+0.012, +0.066] | 0.872, above the paper's 0.864 |
| RiNALMo (RNA slot) | CoV-Vac | cross-attention | +0.031 [+0.008, +0.054] | 0.854 |
| RiNALMo (RNA slot) | E. coli | gated | +6.5 points [+3.4, +9.6] | 62.5%, the project's best E. coli score |
| RiNALMo (RNA slot) | E. coli | concatenation | +3.9 points [+0.8, +7.2] | 60.0% |

Both E. coli gains are measured against the reference validation chose, RiNALMo pooled onto the codon grid (56.0%); RiNALMo read natively scores 57.3% on test, against which the point gains are +5.2 and +2.7 (no interval computed for that comparison). The gated one mixes RNA (0.42) and protein (0.50) rather than collapsing. Both CoV-Vac gains come from nucleotide-resolution encoders on a task governed by local RNA folding, which is the first sign in this project that letting nucleotide-level streams attend to one another helps a nucleotide-scale task. Each rests on one seed, so they are candidates rather than results.

**Two more patterns worth saying.** On CoV-Vac, concatenation and the gated head are resolved worse than the best single encoder in 12 of 16 configurations, while cross-attention is never resolved worse there. On stability all nine comparisons tie.

**The gated head with other encoders.** Slide 6's near-zero RNA weight belongs to the paper's encoder set on CoV-Vac, E. coli and stability. With a strong RNA encoder the gate gives RNA real weight (fungal 0.48 to 0.60 with the three RNA swaps, stability 0.79 with mRNA-FM, CoV-Vac 0.94 and E. coli 0.42 with RiNALMo), and even RNA-FM gets 0.52 on CoV-Vac once HyenaDNA replaces the DNA encoder. It still selects wrongly where it matters most, giving RNA 0.00 on E. coli with mRNA-FM, the best single encoder there, and scoring 53.1%.

**A side result: gridded DNA.** The sweep also read the swapped DNA encoders after the codon grid's learned upsampling. For NT 500M the grid helps (fungal 0.740 to 0.766, E. coli 47.5 to 49.6); for HyenaDNA, which already reads single nucleotides, it hurts (0.737 to 0.676, 50.4 to 46.2). Whether the grid helps a single encoder depends on the encoder, which is why the same check on the original NT v2 100M is still worth running.

**Caveats to have ready.** One seed per configuration, and the intervals measure which test rows were drawn, not training variation. The gated head's entropy weight is inherited from the original encoders, not retuned per swap. The single-encoder reference is chosen on validation, which went wrong once.

**What comes next.** Seeds on the four surviving gains; the mRNA-FM-anchored fusions from October 22; the plan's two-encoder replacements, chosen from these single swaps; DNABERT-2 once it has a grid adapter; the five swaps not yet run on stability.

**One sentence to say.** "Of 81 fusion comparisons only four gains survive once you remove the ones measured against a weak encoder, and two of those four are cross-attention on CoV-Vac, the first fusion results in this project that a better single encoder does not explain."

---

## Slide 11. LucaOne, one jointly pretrained model (Priority 2)

**What the slide shows.** The manuscript's LucaOne figure (`r34_lucaone.pdf`): ridge-probe Spearman on the published mRNA stability test split for a composition baseline, each BioLangFusion encoder's pooled vector, the three concatenated, and LucaOne's gene stream, protein stream and both together. The two bracketed annotations under the bars are the paired comparisons with their intervals, so the claim and its uncertainty are both on the figure.

**Why we did this.** Priority 2 asks whether one model pretrained jointly on nucleotide and protein sequences (LucaOne, 1.57 billion parameters as loaded, trained on both alphabets with one vocabulary) gives a stronger representation than three separately pretrained encoders and their fusion. If joint pretraining already captures the cross-modal structure, then fusing separately pretrained models is solving a problem that pretraining can solve better.

**How we did it.** LucaOne cannot be dropped into the BioLangFusion grid (its tokenizer and context are different), so the comparison is made at the level of pooled vectors with an identical ridge probe for every row: the same rows, the same split, the same probe. LucaOne reads the coding sequence as its gene stream and the translated protein as its protein stream; "both streams" is the two pooled vectors concatenated. The bootstrap groups identical test sequences together.

**What "BLF three fused" means (Mina's question).** It is the pooled vector of each of BioLangFusion's three encoders (Nucleotide Transformer v2 100M, 512 wide; RNA-FM, 640; ESM-2 8M, 320) concatenated into one 1,472-wide vector, with the same ridge probe fit on it. It is the probe-level equivalent of concatenation fusion, included so that LucaOne is compared against the three encoders combined and not only against each alone. It is not the trained TextCNN concatenation head from slides 4 and 5; those are trained on token streams, these are linear probes on pooled vectors, and the two are not interchangeable.

**What the numbers mean.** LucaOne's two streams reach 0.483, against 0.390 for the three BioLangFusion encoders concatenated: +0.092, interval +0.066 to +0.120, resolved. Against the best single BioLangFusion encoder (Nucleotide Transformer, 0.312) it is +0.170. Even LucaOne's gene stream alone (0.443) beats the three fused. So one jointly pretrained model carries more linearly accessible stability signal than three separately pretrained encoders combined.

**Two caveats that go with the number.** First, this is the published split with its 68% train/test sequence overlap; grouping repeated sequences in the bootstrap does not remove the overlap, so read the number beside slide 7. Second, LucaOne is 1.57B parameters as loaded, against about 205M for the three encoders combined (NT v2 100M about 98M, RNA-FM about 100M, ESM-2 8M about 8M), so this compares representations, not matched budgets.

**Mina's note: "Do LucaOne on the other datasets."** Agreed. E. coli, CoV-Vac and fungal all fit LucaOne's 1,280-token context after the 999-nucleotide rule, so the same probe comparison can be run on each. GTEx cannot: the 6,000-nucleotide DNA window does not fit and only 40% of transcripts do. This is added to the plan under Gantt 41.

**One sentence to say.** "On stability, one model pretrained on both alphabets at once beats three single-alphabet models fused, by a margin that resolves, with the leaky-split caveat attached."

---

## Slide 12. Full transcripts: half-life and translation efficiency (Priorities 3 and 4)

**What the slide shows.** Pearson r on the human mRNABench sets, for: counted composition features of the coding sequence and of the whole transcript; the RNA encoder on the 5' UTR, the coding sequence and the full transcript; the DNA encoder on the full transcript; the protein encoder on the coding sequence's protein; and two fused representations. Beside it is the manuscript's figure (`r34_halflife.pdf`), where the dashed line in each panel is the whole-transcript composition floor, the single number every encoder has to clear.

**Why we did this.** The stability benchmark in Phase A is coding sequence only. Real mRNA half-life and translation efficiency depend heavily on the untranslated regions (the 5' UTR for initiation, the 3' UTR for decay), which that benchmark cannot see. Priorities 3 and 4 ask whether richer transcript context changes the fusion question. The plan put a feasibility gate in front of the datasets: confirm that every row has both UTRs, a recoverable coding frame, and gene-grouped splits. The gate passed on all four primary sets (human and mouse half-life, human and mouse translation efficiency).

**How we did it.** mRNABench's human half-life (12,939 transcripts; the target is the first principal component over 39 half-life studies) and human translation efficiency (10,281 transcripts, from ribosome profiling). Frozen encoders (RNA-FM, Nucleotide Transformer v2, ESM-2 8M), windowed pooling for transcripts longer than an encoder's context, ridge probes, nine gene-grouped splits, mean and standard error. The composition baselines are counted features (nucleotide and codon frequencies, length) on each region set, with inner-validated ridge.

**What the numbers mean.**

- Counted features of the whole transcript already reach 0.542 on half-life and 0.702 on translation efficiency. Those are the floors an encoder has to clear, not zero. The UTRs add about +0.055 to half-life and +0.019 to translation efficiency over coding-sequence composition, before any encoder is involved.
- The 5' UTR alone carries real signal (0.178 and 0.433), and the full transcript beats the coding sequence alone on half-life (0.512 against 0.473), so there is information outside the coding sequence that the Phase A benchmark could never see.
- The protein encoder is the best single representation on both tasks (0.552 and 0.682), which is not what a transcript-level task would predict. A plausible reading: half-life and translation efficiency both track codon optimality and ORF length, and amino-acid composition and protein length are strongly tied to both; the protein encoder captures that side cleanly, while the RNA encoder has to read a transcript whose median length is about 3,000 nucleotides in windows of its 1,024-nucleotide context and average them, which loses long-range structure.
- The DNA encoder (Nucleotide Transformer v2, 6-mer tokens, a context of about 12,000 nucleotides, so it reads most transcripts whole) on the full transcript reaches 0.660 on translation efficiency, above the RNA encoder's 0.610, which supports the context explanation.
- Fusing the full transcript with the protein gains +0.053 on half-life (0.606) and +0.030 on translation efficiency (0.712) over the best single encoder. The three-encoder fusion is within error of the two-encoder one (0.602 and 0.713).

**Why this is not yet proof.** These are means and standard errors over nine splits, and the notebook does not keep per-split predictions, so there is no paired interval yet. The fused rows are within each other's error, and the gain over composition is +0.064 on half-life but only +0.010 on translation efficiency. Mouse is embedded and not yet scored.

**One sentence to say.** "Once the input is a full transcript with its protein, fusion leads the best single encoder on both tasks (+0.053 and +0.030, no paired interval yet), which fits the pattern: fusion adds something when the second input carries information the first one could not see."

---

## Slide 13. IsoFormer on GTEx: physically distinct molecules, no codon grid, fusion wins

**What the slide shows.** IsoFormer's six-row modality ablation (DNA, RNA, protein, and their combinations) on the 2,000-transcript GTEx pilot, ours with frozen encoders next to the paper's with fine-tuned encoders, and below it the manuscript's figure of the same rows (`r34_isoformer.pdf`), with the two resolved gains annotated on it.

**Why we did this.** Gantt 44, and the counterpart to BioLangFusion. Every benchmark where fusion failed gives all three encoders the same coding sequence. GTEx gives them three different molecules: 6,000 nucleotides of genomic DNA centered on the transcription start site (promoter and first exons), the transcript, and the protein. There is no codon grid that could align those, so IsoFormer fuses them with successive cross-attention between pooled modality representations. If fusion helps anywhere, it should help here, and Stage 1 found cross-modal alignment lowest exactly here (DNA shares almost no retrievable correspondence with RNA or protein on GTEx).

**How we did it.** Nucleotide Transformer v2 100M reads both the DNA window and the transcript, ESM-2 150M reads the protein, all frozen; IsoFormer's aggregator on top; predicting expression in 30 tissues; Spearman averaged over the tissues; split by gene (no gene shared between halves); one seed; paired bootstrap over the 1,000 test transcripts.

**What the numbers mean.** All three modalities beat RNA alone by +0.047 (interval +0.027 to +0.069), resolved; DNA with RNA beats RNA alone by +0.055 (+0.032 to +0.078), resolved. What carries the gain is RNA with DNA; protein adds nothing measurable (all three minus DNA with RNA is -0.007, a tie). Three of the paper's four orderings reproduce: RNA is the strongest single modality, all three beat the best single, DNA with RNA beats DNA with protein. The gap between our absolute values (0.50) and IsoFormer's (0.65) is what freezing the encoders costs: IsoFormer fine-tunes its encoders end to end and reports that RNA encoder specialization during fine-tuning is what makes its representation strong. Freezing is this project's premise, so that gap is a measurement, not a failure.

**Why this is the key slide for the alignment question.** Low alignment and high complementarity went together here: the DNA window and the transcript share almost nothing retrievable, and that is exactly why combining them adds information. Where alignment was high (three readings of one coding sequence), fusion added nothing. This is the empirical content of the Stage 2 definition: alignment measured after the composition control, reported separately from complementarity, and the two can move in opposite directions.

**One sentence to say.** "Where the three inputs are three different molecules, fusion wins by a margin that resolves, with frozen encoders, and the gain comes from DNA with RNA."

---

## Slide 14. What it adds up to, and what runs next

**What the slide shows.** Five conclusions on the left, the run order on the right, and in the footer the four things that need an answer from outside the project.

**The five conclusions, with the evidence behind each.**

1. *The build is right.* Single encoders within 0.01 of the paper, RNA-FM on fungal aside; cross-attention nowhere more than 0.01 below it (slide 4).
2. *The encoder matters more than the fusion head.* Better RNA encoders lift fungal and E. coli on their own; with mRNA-FM or CaLM in place fusion adds nothing resolved, and four single-seed gains remain (slides 9 and 10).
3. *Fusion helps where the second input is a different molecule or region.* GTEx +0.047 resolved (slide 13); full transcript with protein +0.053 and +0.030, no paired interval yet (slide 12).
4. *One jointly pretrained model beats three fused ones on stability.* LucaOne +0.092 resolved, on the leaky split (slide 11).
5. *With the paper's encoders the gated head mostly keeps one modality.* RNA gets almost nothing on CoV-Vac, E. coli and stability, and on CoV-Vac it drops the best encoder (slide 6). The paper reports no error bars, and its own fungal gain is the size of noise on that split (slide 5).

**What runs next, in order.**

1. *Seeds on the four surviving Phase B gains* (cross-attention with HyenaDNA and with RiNALMo on CoV-Vac, the gated head and concatenation with RiNALMo on E. coli). Each rests on one seed, and the intervals measure test-row sampling rather than training variation, so this is what turns them from candidates into results.
2. *Fusion built around mRNA-FM* (DNA with mRNA-FM, mRNA-FM with protein) and the other codon-level RNA encoders, then the plan's two-encoder replacements, whose candidates the single swaps have now identified (Gantt 24).
3. *The sequence-grouped stability re-split with seeds.* The published split repeats a training sequence in 68% of its test rows, so this is the only way to settle whether fusion helps there (slide 7).
4. *LucaOne on E. coli, CoV-Vac and fungal,* which fit its 1,280-token context where GTEx does not (Gantt 41); the gated head at the paper's 1,280 channels on the three benchmarks not yet run that way; the mouse half-life and translation-efficiency scores, whose embeddings already exist.
5. *Then the project's own designs, from January.* Coupled Mamba on the codon grid (Gantt 20 and 21), and IsoFormer-style cross-attention (Gantt 25) run on both redundant inputs (BioLangFusion's four benchmarks) and complementary ones (full transcripts, GTEx), which is the controlled version of the comparison GTEx alone cannot make.

**The footer: four things that need an answer from outside the project.** Whether Ab1 can be obtained through UCF, since it is from Yazdani-Jahromi et al. there and is the benchmark where the paper's three heads lead by the widest margins (+0.017 to +0.022; on fungal all three also lead, by +0.001 to +0.020). Whether seeds plus a paired bootstrap becomes the reporting standard for everything that follows. GPU access beyond one laptop and one desktop, which Coupled Mamba needs. Which two DNA encoders and which RNA encoder join the twelve Stage 1 teachers for distillation.

**One sentence to say.** "The reproduction is faithful, the paper's fusion gains mostly do not survive a stronger single encoder, and the project's question is now which encoder to put in each slot and when a second molecule is worth fusing."
