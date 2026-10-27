# Guide to the October 22 progress slides

A slide-by-slide explanation of `Mina_Progress_Slides_2026-10-22.pptx`, written after the October 22 meeting with Mina. Every slide gets the same five parts: what the slide shows, why we ran that experiment, how we ran it, what the numbers mean, and the answer to what Mina asked about it. The same text is in the speaker notes of each slide. Julian and Chase: use this to put each slide in your own words, and change anything that does not match how you would say it.

Every number here is read from the result files under `Code/stage4/results/` and `Code/stage3/results/`, and from the run records in `docs/run-records.md`. Nothing is quoted from memory.

## The whole deck in five sentences

1. We rebuilt BioLangFusion (the paper that fuses a DNA, an RNA and a protein language model on a shared codon grid) from its text, and our copy gives the same scores as the paper for every single encoder and for its cross-attention head, so the build is right.
2. The paper's best fusion head, the gated attention head, does not reproduce, and its own attention weights show why: it picks one modality instead of mixing three, and on one benchmark it picks the wrong one.
3. The paper's central claim, that fusing three encoders beats the best single encoder, holds on two of six settings we can measure, and the biggest of those (E. coli, +6.9 accuracy points) disappeared when we swapped the paper's weak RNA encoder for a better one: the better encoder alone gained 8 points and fusion then added nothing.
4. Fusion still clearly helps where the three inputs are physically different molecules (a genomic DNA window, a transcript and a protein on GTEx; a full transcript with its protein on half-life and translation efficiency), and a single jointly pretrained model (LucaOne) beats three separately pretrained encoders fused on mRNA stability.
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

**What the slide shows.** The one-paragraph summary of the whole deck, and three chips: what has results (plan priorities 1, 2, 5, and 3 and 4), and what is left to run (seeds on the three new Phase B gains, and the sequence-grouped stability re-split). All eight encoder swaps have now run (slide 10).

**Why this framing.** Mina's research plan of October 9 lists five priorities in order. The deck follows that order so that she can see each item as a line in her own plan: Priority 1 is the BioLangFusion reproduction (Phase A) and the encoder swaps (Phase B); Priority 2 is LucaOne; Priorities 3 and 4 are full-transcript half-life and translation efficiency; Priority 5 is the mRFP control.

**One sentence to say.** "Everything on your list has a first result except the rest of the encoder swaps; the reproduction holds, and the paper's fusion gains hold on fewer tasks than they appeared to, because a better single encoder closes the biggest of them."

---

## Slide 2. Timeline: what happened between October 7 and October 22

**What the slide shows.** Seven dated steps, in the order they were run.

**Why this order.** The plan's own rule is: reproduce first, then change one thing at a time. So the order was: adopt the plan and place it in the Gantt chart (October 9 to 15); build the pipeline (October 15 to 19); check that the pipeline reproduces the paper on the cleanest benchmark, fungal expression (October 17 to 18); run the two things that do not need the BioLangFusion grid at all, IsoFormer on GTEx and the LucaOne feasibility check (October 18 to 19); run the cheap control, mRFP (October 19); run Phase A on the other three benchmarks (October 19 to 20, 27 hours on Julian's laptop); then Phase B, LucaOne, and the half-life and translation-efficiency probes on Chase's GPU (October 21 to 22).

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
- *The transposed convolution.* The paper specifies the DNA upsampling twice: Equation 1 (kernel 2, stride 2) and the appendix lists (kernel 3, stride 2, padding 2). We ran an impulse test: feed a single token and see where its output lands. The appendix setting places the DNA stream two codons out of register with the protein frame and three codons short of it; Equation 1 lands exactly. Retraining under Equation 1 raised the two heads that combine streams position by position (concatenation +0.015, gated +0.008) and did not change cross-attention, which is exactly the signature of an alignment fault. Equation 1 is used everywhere.
- Also noted: the paper gives its "fixed" TextCNN two different widths (1,280 channels for the gated head, 100 for cross-attention). We use 100 for every head so that capacity is matched; the gated head at 1,280 is a diagnostic (slide 6).

**Error bars.** The paper reports one number per cell with no error bar. Every one of our runs saves its per-row test predictions, so any two models can be compared with a paired bootstrap. This is what lets us say "resolved" or "tie" on later slides.

**One sentence to say.** "The paper is the only spec, it contradicts itself twice, and we settled both contradictions by measurement rather than by guessing."

---

## Slide 4. Phase A reproduction: the single encoders and cross-attention reproduce; the gated head does not

**What the slide shows.** Our test score next to the paper's, for every model variant on the four benchmarks we have (Ab1, the fifth, is not public). Format: ours (paper). Spearman correlation everywhere except E. coli, which is accuracy in percent. Cells where the gated head misses the paper by more than 0.01 are shaded. Below the table, two things. On the left, the manuscript's version of the same numbers (`r34_reproduction.pdf`), one panel per benchmark, filled marker ours and open marker published, so the size of each agreement or gap is visible rather than read off. On the right, a second small table built from the paper's own Table 1: each of its three heads minus its own best single encoder.

| Paper's Table 1, head minus its best single | CoV-Vac | Fungal | E. coli | Stab. | Ab1 |
| --- | --- | --- | --- | --- | --- |
| Concatenation | -0.010 | +0.001 | -2.67 | -0.014 | +0.021 |
| MIL + entropy | +0.023 | +0.020 | -0.84 | +0.010 | +0.017 |
| Cross-attention | -0.013 | +0.008 | +0.98 | -0.003 | +0.022 |

**Why that second table is on the slide.** The paper's sentence is "fusion-based models consistently outperform the best single-modality baselines." Its own numbers give nine wins and six losses across the fifteen comparisons. Concatenation loses on three of five, cross-attention on two of five, and only MIL with entropy reaches four of five, losing E. coli. The claim holds only if you take the best head per task after seeing the test scores, and a different head wins on different tasks, so no method you could have committed to in advance wins everywhere. Add that there is no interval anywhere in the paper and that four of the five winning margins are 0.010 to 0.023, and the claim is weaker than the sentence. Two further details worth having ready: RNA-FM is the best single encoder on four of the five tasks, so "the best single baseline" is essentially RNA-FM throughout, and the head credited with beating it is MIL, the head that in our hands gives RNA a weight of 0.00 everywhere. And Ab1, the only task where all three heads win, is the one not in the public release.

**Why we did this.** This is the proof that the build is the paper's. If the single encoders alone give the paper's numbers, then the token caches, the splits, the codon grid, the TextCNN and the training loop are all right, because a single-encoder run uses all of them except the fusion head. If a fusion head then misses, the miss is in the head or in the paper's description of it, not in our plumbing.

**How to read the table.**

- Rows 1 to 4 are single encoders. "RNA-FM read on the codon grid" is RNA-FM with its nucleotide tokens averaged three to one before the TextCNN, which the paper does not report but which we need because that is how RNA-FM enters every fusion head.
- Rows 5 to 8 are the three fusion heads, with the gated head shown with and without its entropy term (the paper's Table 4 ablation).
- "Within 0.01" is what we count as reproduced: for a Spearman correlation on a few hundred test rows the 95% interval of the score itself is about plus or minus 0.03, so 0.01 is well inside noise. E. coli has 712 test rows and accuracy carries about plus or minus 3.7 points, so every E. coli difference in this table is within noise.

**What the numbers mean.**

- Every single encoder matches the paper within 0.01 on CoV-Vac and stability, and Nucleotide Transformer and ESM-2 match on fungal. The one single-encoder number we cannot place is RNA-FM on fungal. Read as the encoder writes it, one token per nucleotide, it is 0.722 against the paper's 0.767; pooled three to one onto the codon grid, the way every fusion head sees it, it is 0.794. The published value sits between our two readings, and both of ours are a single seed there. On the other three benchmarks the paper's RNA-FM numbers track the unpooled reading (CoV-Vac 0.841 against our 0.838, E. coli 52.9 against 50.7, stability 0.553 against 0.552), so the paper's baselines look unpooled and fungal is the outlier. Avoid saying the paper's baseline was codon-pooled: that would explain fungal and contradict CoV-Vac.

**What the RNA-FM-on-the-grid row is for, in one line of arithmetic.** Every fusion head does two things to RNA-FM: it averages the stream three to one onto codons, and it combines it with DNA and protein. When a fusion head beats RNA-FM alone, either step could be responsible. The grid row is RNA-FM with only the first step applied, so it splits the gap in two: fusion minus RNA-FM as written = (grid row minus RNA-FM as written) + (fusion minus grid row) = what the averaging did + what the fusion did. On the clean stability rows: 0.359 - 0.319 = 0.040 = (0.340 - 0.319) + (0.359 - 0.340) = 0.021 + 0.019. Half the apparent fusion gain is the averaging. Without the row, the averaging's effect gets credited to fusion.

**Why RNA-FM is the only encoder with two rows (worth having ready, since it looks arbitrary).** The codon grid exists to make three streams the same length so a head can combine them position by position. A single encoder has nothing to line up against, so an "alone" row is just that encoder's own tokens going into the TextCNN. ESM-2 writes one token per amino acid, which is already one per codon, so its native and gridded readings are the same object. Nucleotide Transformer writes one token per 6-mer, one per two codons, and putting it on the grid requires the learned transposed convolution that only exists inside a fusion model, so alone it is read natively. RNA-FM writes one token per nucleotide, three per codon, and pooling them three to one is a fixed average that needs no training, so both readings are available and they are genuinely different inputs: the TextCNN's kernels of 3, 4 and 5 span 3 to 5 nucleotides on the native stream and 3 to 5 codons, 9 to 15 nucleotides, on the gridded one. Same encoder, different receptive field.

- Pooling is not uniformly better, which is why it is reported as two rows and not as a correction. It gains 0.072 on fungal and costs a little everywhere else: CoV-Vac 0.823 against 0.838, E. coli 49.3 against 50.7, stability 0.547 against 0.552. The natural reading is that fungal expression is a codon-scale signal and CoV-Vac degradation is a nucleotide-scale one, but E. coli, which is also codon-scale, does not follow that, so treat it as an observation rather than a rule.
- Cross-attention reproduces everywhere: within 0.01 of its published value on all four benchmarks, and above it on three.
- The gated head, which the paper reports as its best head on three of five benchmarks, is the best head on none here. It is below the paper on fungal (0.801 against 0.824), far below on CoV-Vac (0.783 against 0.864), and 0.012 low on stability. On E. coli it is above the paper (54.5 against 52.1), but every E. coli difference in this table is inside the plus or minus 3.7 points of accuracy noise, so that cell decides nothing either way. So the honest statement is not "it misses everywhere" but "it misses on the two benchmarks where the comparison is tight," and CoV-Vac is the largest miss in the whole reproduction. Slide 6 explains why.
- The same holds without the entropy term, which is the paper's own ablation (row 7, not a different attention mechanism): fungal 0.802 against 0.826 and CoV-Vac 0.792 against 0.854 still miss, and stability matches (0.558 against 0.556). Turning the entropy term off does not rescue the head, which is the point of showing both rows.
- Concatenation is close to the paper on fungal and stability, above it on E. coli (58.7 against 50.3) and below it on CoV-Vac (0.799 against 0.831).

**Mina's note: "another table."** The point of this table is only the comparison with the paper, not the comparison between heads. Read it as: do the single-encoder rows match (yes), does cross-attention match (yes), does the gated head match (no). Slide 5 is the table for the fusion question.

**One sentence to say.** "Our build gives the paper's numbers for every encoder alone and for cross-attention; the paper's headline head, gated attention, is the one thing that does not come back, on the two benchmarks where the comparison is tight enough to tell."

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

**Reading one row out loud, concatenation.** Fungal -0.015 with an interval from -0.049 to +0.017: the gap is a sixth of the interval's width, so the benchmark cannot settle it, and roughly one resample in five came out the other way. CoV-Vac -0.038 with an interval entirely below zero: concatenation is really worse than RNA-FM alone there. E. coli +6.9 entirely above zero: a real win, and the one Phase B dissolved. Stability +0.007 straddling zero: a tie, on the split with 68% overlap. So concatenation is one for four, and the one does not survive a better encoder. The habit to teach: look at whether the interval crosses zero before reading the number. If it crosses, the number is not a result.

**Which rows are BioLangFusion and which are not.** The slide tests BioLangFusion's central claim on its four public benchmarks, and all four are there: fungal, CoV-Vac and stability in panel (a), E. coli in panel (b), separate only because it is scored in accuracy rather than Spearman. The table under the figure contains exactly those four and nothing else. Panel (a) carries two extra rows that are not BioLangFusion benchmarks, and they are the ones that confuse people. *Stability, unseen sequences* is not a fifth benchmark; it is the same stability models rescored on only the 1,135 test rows whose sequence never appears in training, because the published split puts copies of the same sequence on both sides (slide 7). *GTEx pilot* is not from BioLangFusion at all; it is IsoFormer's setup, where the three inputs are three different molecules, and it is on the slide as the contrast case, the one place fusion clearly wins. It is suggestive, not a controlled comparison: GTEx changes the inputs and the fusion method at the same time (BioLangFusion's heads need a codon grid, which a genomic window does not have), so the difference in outcome cannot be pinned on the inputs alone. Inside GTEx the comparison is fair, fusion against the best single modality with the same encoders, split and head. The controlled version of the cross-benchmark question is the project's IsoFormer-style cross-attention (Gantt 25, January) run on BioLangFusion's four benchmarks, which needs no grid, so it holds the method fixed and changes only the inputs. Read the nine rows above them as the test of the paper, and those two as context for it.

**How to read the figure in one pass.** Look at which markers are filled, and which side of the dashed zero line they sit on. Three are filled on the right: E. coli concatenation, E. coli gated, and GTEx. Two are filled on the left: CoV-Vac concatenation and CoV-Vac gated, where fusion is resolved *worse* than RNA-FM alone. The other nine sit on the line, which means this is a tie, not a small win. The E. coli pair is the win Phase B dissolved, which is why the figure annotates it.

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
- *E. coli: fusion won in Phase A (+6.9 accuracy points, interval +4.2 to +9.6).* This was the one clear win, and slide 9 shows it was fusion compensating for a weak RNA encoder rather than fusion adding information: swapping RNA-FM for mRNA-FM raised the single encoder by 8 points and the fusion gain fell to +0.3.
- *mRNA stability, published split: tie (+0.007).* But this split repeats a training sequence in 68% of its test rows, so every model's score here is mostly memory (slide 7).
- *mRNA stability, unseen sequences: leans fusion (+0.040, interval -0.003 to +0.083).* On the 1,135 test rows whose sequence never appears in training, the fusion heads sit 0.03 to 0.04 above RNA-FM as the encoder writes it, just short of resolving at one seed. Against RNA-FM on the codon grid (0.340 on these rows), the input every head actually receives, the gap is 0.007 to 0.019, so most of the lean is the pooling, not the fusion.
- *GTEx with IsoFormer's aggregator: fusion wins (+0.047, interval +0.027 to +0.069).* The only setting where the three inputs are physically different molecules (slide 13).

**The pattern, which is the interpretation Mina asked for.** Fusion of three readings of one coding sequence does not beat the best single encoder, because the three encoders are seeing the same information in three alphabets; the one that saw the most relevant data in pretraining wins alone. Fusion beats the best single encoder when the inputs carry information no single encoder could have seen (the genomic window around the transcription start site on GTEx; the untranslated regions on half-life and translation efficiency).

**Mina's note: "Interpret why these work better. For example: if someone wants to train a foundation model, we should train on task-specific datasets."** Her reading matches what the table shows, and the next slides give the evidence for it: on each benchmark the strongest single encoder is the one whose pretraining data is closest to the task's biology. RNA-FM (pretrained on non-coding RNA, structure-rich) wins CoV-Vac, a structure-driven degradation task on RNA fragments. mRNA-FM and CaLM (pretrained on coding sequences with codon tokens) win E. coli, a codon-usage-driven expression task (slide 9), and mRNA-FM carries by far the most synonymous-codon information in the mRFP control (slide 8). Nucleotide Transformer (pretrained on whole genomes of many species) wins fungal expression, a task on genes from many fungal genomes. So the practical rule is: match the encoder's pretraining domain to the task before asking whether to fuse, and expect fusion to add something only when the second input is a different molecule or region.

**Mina's note: "Did it report 1,000 rows or 3,000 rows in BioLangFusion?"** This is about the stability rows and is answered on slide 7: the paper's number is on the full 3,579-row published test split.

**One sentence to say.** "Fusion beats the best single encoder on two of six settings, and the largest of those two did not survive a better single encoder."

---

## Slide 6. The three fusion heads, and why only the gated one fails

**What the slide shows.** Two things. On top, one figure with three panels, one per fusion head, all starting from the same three streams on the codon grid and ending in the same fixed TextCNN. Below it, the measured gated weights for every benchmark, which are the evidence that the head is a selector rather than a mixer.

**How to read the figure, panel by panel.** Panel a, concatenation: at each codon the three vectors are glued end to end into 1,560 numbers (DNA is first projected from 512 to 600) and one linear layer maps that to 600. Nothing is discarded, and the TextCNN decides which features of which modality to use. Panel b, gated attention: each stream is averaged over its codons into one summary, a gate scores the three summaries, a softmax turns the scores into three weights that add to one, and those same three weights multiply the full streams codon by codon. The weights drawn (DNA 0.14, RNA 0.00, protein 0.86) are the measured E. coli weights for seed 0, and the greyed-out RNA row is what a weight of zero does: the whole stream is gone from every position. Panel c, cross-modal attention: the three streams are laid end to end into one context of 3T positions, every position of every modality queries that context, and the updated streams are merged back; arrow width stands for attention weight. The same figure is Figure 12 in the short manuscript and Figure 31 in the long one, and the file is `docs/notes/figures/figures_overleaf/fusion_heads_slide.pdf`.

**The weights behind the footer** (entropy weight 0.01, averaged over the test split; three values are seeds 0, 1 and 2).

| Benchmark | DNA | RNA | Protein |
| --- | --- | --- | --- |
| Fungal | 0.20 / 0.16 / 0.14 | 0.20 / 0.25 / 0.26 | 0.61 / 0.59 / 0.60 |
| CoV-Vac | 0.83 / 0.92 / 0.94 | 0.01 / 0.03 / 0.00 | 0.16 / 0.05 / 0.06 |
| E. coli | 0.11 / 0.18 / 0.10 | 0.00 | 0.89 / 0.82 / 0.90 |
| Stability | 0.28 | 0.00 | 0.71 |

At entropy weight 0.5 or 1.0, one modality takes 1.00 and the other two 0.00 on every benchmark. The two diagnostic numbers are the CoV-Vac miss (0.783 against 0.864 published) and the width test on fungal (0.801 at 100 channels, 0.812 at 1,280, against 0.824 published).

**What the gated head is, in detail (Mina's question "what is the gated head and how does it work").** After the three streams are projected to a common width, each stream is summarized by one vector, the mean of its codon vectors. Each summary goes through a gate: tanh(V h) multiplied elementwise by sigmoid(U h), then a linear score W. The three scores go through a softmax (we also learn a temperature, which the paper does not mention), giving three weights that add to one. The fused stream is weight_DNA times the DNA stream plus weight_RNA times the RNA stream plus weight_protein times the protein stream, codon by codon. The whole decision is three numbers per sequence. The paper calls these weights an interpretability output: "how much each modality is used."

**Yes, it is the MIL attention (Mina's question "is gated MIL attention related?").** The gate is the attention-based multiple instance learning pooling of Ilse, Tomczak and Welling (2018), where a bag of instances gets one attention weight per instance and the bag representation is the weighted sum. BioLangFusion uses the same equations with the three modalities as the three instances of one bag. That is why the paper calls it "MIL."

**The entropy term.** The loss is the task loss plus lambda times the negative entropy of the three weights. Lower entropy means sharper weights, so the term pushes the head toward putting most of the weight on one modality. The paper says this makes the weights "sharp and task-specific." Lambda is chosen on validation among 0.01, 0.5 and 1; validation chose 0.01 on every benchmark. At 0.5 or 1.0 one modality takes a weight of 1.00 and the others 0.00 on every benchmark.

**What the weights actually show here.** They are nearly one-hot even at lambda 0.01. RNA is at zero on all three Phase A benchmarks (CoV-Vac 0.01, E. coli 0.00, stability 0.00). On E. coli the head keeps protein (0.89). On CoV-Vac it keeps DNA (0.83 to 0.94 across seeds) and scores 0.783, almost exactly the DNA-only score of 0.779, even though RNA-FM alone is the best single encoder at 0.838. So on CoV-Vac the head selected against the best encoder, and that one choice is the largest miss in the whole reproduction. The entropy term never helps: on all four benchmarks the head without it scores at or above the head with it.

**Why RNA goes to zero (Mina's question "E. coli, why is RNA 0").** Three things combine. First, the softmax plus the entropy penalty is a winner-take-all mechanism: once one modality's weight pulls ahead early in training, the other two get almost no gradient and cannot recover, which is the same collapse seen in gating networks generally. Second, the RNA stream carries the least usable codon information of the three for an expression task: in the mRFP control (slide 8) RNA-FM has the smallest gain above the codon-frequency controls of any nucleotide encoder, and on E. coli it is also redundant with the DNA stream, which is the same letters read as 6-mers. Third, the gate decides from a mean over all codons, which erases exactly the positional detail that makes a stream useful. So the head finds nothing in the RNA summary that the protein summary does not already give it, and it drops RNA. The clearest demonstration is in Phase B: with mRNA-FM in the RNA slot on E. coli, mRNA-FM alone is the best single encoder at 58.7%, and the gated head still gives RNA a weight of 0.0007 and scores 53.1%, five points below the encoder it threw away.

**How this relates to fusion and concatenation (Mina's question "how does it help with fusion / concatenation").** The gated head does not add information; it reweights. Because its weights are nearly one-hot, it behaves like a single encoder plus a projection, and its score tracks the single-encoder score of whichever modality it kept. Concatenation keeps all three streams at every codon and lets the TextCNN pick features from any of them, so where the modalities are complementary (E. coli in Phase A) concatenation wins by seven points and the gated head by three. Cross-attention can move information across modalities and positions, so it is the only head that never loses to the best single encoder in Phase A. The ordering we measure (cross-attention above concatenation above gated) is the reverse of the paper's, and on fungal the cross-attention-minus-gated difference is resolved (+0.022, interval +0.009 to +0.036).

**The width question.** The paper's lists give the gated head a 1,280-channel TextCNN and the other heads 100. Running the gated head at 1,280 on fungal recovers about half of its gap to the paper (0.801 to 0.812 against 0.824). Running it at 1,280 on the other three benchmarks is the last check before the verdict on the head is final.

**One sentence to say.** "The gated head is a three-number switch, not a mixer; it drops RNA everywhere, and on CoV-Vac it drops the best encoder, which is the paper's largest unreproduced number."

---

## Slide 7. mRNA stability: the published split is mostly memorization

**What the slide shows.** Panel (b) of the manuscript's leakage figure (`r34_gated_leak.pdf`): the same eight trained models scored twice, once on the whole published test split (3,579 rows, filled marker) and once on the 1,135 of those rows whose sequence never appears in training (open marker). The number above each line is the drop. Every model loses about 0.2, and the shaded block at the bottom is the fusion heads, which lose as much as the single encoders do.

**Why we did this.** The Stage 3 dataset audit found that the published mRNA stability split repeats a training sequence in 2,444 of its 3,579 test rows (68%), usually with a different label (the same sequence measured again, median label spread 0.49). Validation leaks the same way (2,360 of 3,608 rows), so early stopping was also watching a leaky split. A score on this split is therefore mostly a test of whether the model remembers sequences it has seen, not whether it understands stability. We had to know how much of the published 0.55 is memory.

**How we did it.** No retraining. The models trained on the published split were scored on all 3,579 test rows (the published number) and then on only the 1,135 unseen-sequence rows. One seed.

**What the numbers mean.** Published-split scores cluster at 0.53 to 0.56 for every model, ours and the paper's, and the range is carried by the repeated rows. On unseen sequences the scores are 0.27 to 0.36. That is the honest generalization range for this benchmark, and it is a third lower than what the paper reports. On the unseen rows the three fusion heads (0.347 to 0.359) sit above RNA-FM as the encoder writes it (0.319) by 0.03 to 0.04, just short of resolving at one seed (concatenation +0.040, interval -0.003 to +0.083). But RNA-FM pooled onto the codon grid, the stream every fusion head actually receives, scores 0.340 on the same rows, so the part of that gap fusion itself adds is 0.007 to 0.019. Whether the leak hides a real fusion gain is open, and the sequence-grouped re-split with seeds is the run that settles it.

**Mina's question: "Did it report 1,000 rows or 3,000 rows in BioLangFusion?"** The 3,000. BioLangFusion reports its stability numbers on the full published test split, which is 3,579 rows after its own 999-nucleotide filter. The evidence is that our full-split numbers match the paper's within 0.01 for every single encoder (RNA-FM 0.552 against 0.553, Nucleotide Transformer 0.539 against 0.530, ESM-2 0.531 against 0.539). The 1,135 unseen rows are our construction; the paper never scores them, and no published number exists for them. This matters for how the paper's stability column should be read: it is the score on a split where two thirds of the test sequences were seen in training.

**One sentence to say.** "The paper's 0.55 on stability is on a split where 68% of test sequences are in training; on the sequences the models have never seen, everyone is at about 0.3, and fusion is slightly ahead."

---

## Slide 8. The mRFP synonymous control (plan item 5)

**What the slide shows.** The manuscript's mRFP figure (`r34_mrfp.pdf`), both panels. Panel (a) is what each encoder reads from the 1,000 synonymous recodings on its own, held-out R-squared, with the two control bars marked as dashed lines: the Track A controls alone reach 0.190 and all controls together 0.306. Panel (b) is the part that matters, the gain each encoder adds *above every control*, with the count of fold assignments in which that gain stayed above zero printed beside each bar (10/10 for almost all of them). mRNA-FM adds 0.18, the other nucleotide encoders 0.02 to 0.08, and the four protein encoders exactly 0.000, marked in the figure as the control holding.

**Why we did this.** Mina's plan item 5. The mRFP data set is one gene (a red fluorescent protein) recoded 1,000 ways with synonymous codons, so every sequence encodes the identical protein, and the measured expression differs only because of the codons chosen. This gives two clean tests. The protein encoders must be invariant: they see the same amino-acid string every time, so they must predict nothing (a negative control that checks the pipeline). And a nucleotide encoder is only interesting if it predicts expression beyond what simple codon statistics already predict; otherwise it has just memorized codon frequencies.

**How we did it.** Stage 1's pooled embeddings of the twelve encoders. Controls: GC content, GC3 (GC at the third codon position), the 64 codon frequencies, a Nussinov base-pairing proxy on the first 200 nucleotides, and ViennaRNA minimum free energy of the first 60 nucleotides and of the whole sequence. The controls together reach R-squared 0.306. For each encoder, a ridge probe is fit on the controls with inner cross-validation, the encoder is fit on the residual, and the gain in held-out R-squared is recorded over ten grouped fold assignments (identical sequences in one fold).

**What the numbers mean.** The protein control holds: 0.000 gain, in every fold assignment. Every nucleotide encoder beats the Track A controls (GC3, codon use, Nussinov) in 10 of 10 fold assignments, so all of them carry codon information beyond frequencies. mRNA-FM carries far more than the others (+0.18 above every control, +0.26 above the Track A set), and RNA-FM, BioLangFusion's own RNA encoder, carries the least (+0.02). Among the controls, the one structure feature that matters is the folding energy of the first 60 nucleotides after the start codon (R-squared 0.11 on its own; a weaker fold means higher expression), which is the classic Kudla et al. 2009 result for this kind of library. The Nussinov proxy carries nothing.

**Why mRNA-FM.** mRNA-FM and CaLM read coding sequences one codon per token and were pretrained on coding sequences, so their vocabulary is codon identity in context. RNA-FM reads single nucleotides and was pretrained on non-coding RNA, so synonymous codon choice is not something its pretraining ever rewarded. Nucleotide Transformer reads 6-mers across whole genomes, which contain codon structure but dilute it. The ranking is the ranking of how much codon context each encoder's pretraining exposed it to.

**Mina's note: "mRNA-FM performs best; maybe we want to do fusion against mRNA-FM, like DNA or RNA fusion with mRNA-FM; try different mRNA encoders as well."** Agreed, and this is already the direction Phase B took: mRNA-FM went into the RNA slot first because this control predicted it (slide 9). The follow-up she is asking for is to make mRNA-FM the anchor: DNA + mRNA-FM, mRNA-FM + protein, and mRNA-FM with other codon-level RNA encoders (CaLM, and the remaining RNA swap, RiNALMo), on the coding-sequence benchmarks. That goes into the plan as a Phase B extension.

**One sentence to say.** "Every nucleotide encoder knows something about codons beyond their frequencies, mRNA-FM knows by far the most, and the protein encoders know nothing, exactly as they should."

---

## Slide 9. Phase B: one encoder swapped at a time

**What the slide shows.** CoV-Vac and E. coli with three encoders in the RNA slot: RNA-FM (the paper's own), mRNA-FM and CaLM. For each: the encoder alone, and all three fusion heads with that encoder in place. Seed 0; everything else held at Phase A's settings. Under the table is the manuscript's Phase B figure (`r34_phaseB.pdf`): panels (a) and (b) are the swapped encoder alone against concatenation built on it, and panels (c) and (d) are the paired differences with their intervals, which is where the +8.0 and the unresolved +0.3 can be seen side by side.

**Why we did this.** Priority 1, Phase B: keep the splits, preprocessing, heads and training fixed and change encoders only. The plan's success criterion is that at least one swap improves test performance under the same architecture, and the plan's measure is Δfusion against the best single encoder of that triple. The RNA slot went first because the mRFP control predicted that mRNA-FM should carry the most, and CaLM is its control (another codon-level encoder with a different pretraining set).

**How we did it.** Token embeddings for mRNA-FM and CaLM were extracted on the benchmarks (both read one token per codon, so they drop onto the grid without the three-to-one average RNA-FM needs). The Phase A sweep was rerun with the swapped encoder: the encoder alone, concatenation, the gated head at the validation-chosen lambda, and cross-attention. Each swap is compared with a paired bootstrap on the same test rows both against its own best single encoder (Δfusion) and against the same head with the original encoders ("vs original"). Run on Chase's GPU, scored independently twice.

**What the numbers mean.**

- *E. coli.* mRNA-FM alone scores 58.7%, +8.0 points over RNA-FM alone (interval +3.9 to +11.9, resolved) and +8.6 over ESM-2 alone, the validation-chosen single encoder in Phase A. CaLM alone 57.6%, +6.8 (+2.5 to +10.8, resolved). With mRNA-FM in place, concatenation scores 59.0%, +0.3 over mRNA-FM alone (interval -2.9 to +3.5, not resolved); the gated head 53.1% (-5.6, resolved worse); cross-attention 54.8% (-3.9, interval -7.9 to 0.0, not resolved). With CaLM: concatenation 57.7% (+0.1), gated 56.7% (-0.8), cross-attention 53.9% (-3.7, resolved worse). So Phase A's +6.9 for fusion on E. coli was the fusion heads compensating for a weak RNA stream: give the model the right RNA encoder and the gain is gone. The 8-point single-encoder jump is the largest effect anywhere in Stage 4, and it is not a fusion effect.
- *CoV-Vac.* The swap does not help: mRNA-FM alone is 0.812 (-0.021 against RNA-FM alone, interval -0.056 to +0.013, a tie), CaLM alone 0.844 (+0.011, a tie). With either codon-level encoder, concatenation and the gated head are resolved worse than the encoder alone (mRNA-FM: concatenation -0.054, gated -0.050; CaLM: -0.027, -0.042) and cross-attention is a tie (+0.009 and -0.017). This is the same pattern as Phase A: on a structure-driven task on RNA fragments, a codon-level encoder has no advantage and fusing the three readings adds noise.

**Mina's notes on this slide.**

- *"We want to see if fusion replaces RNA-FM and see if it works."* That is exactly Phase B's comparison: each head with the swapped encoder against that encoder alone. On E. coli, replacing RNA-FM with mRNA-FM makes the single encoder as good as the fusion was, and fusion on top of it adds +0.3, which does not resolve. So fusion does not replace a good RNA encoder; a good RNA encoder replaces fusion.
- *"Check RNA-FM."* RNA-FM is the paper's RNA encoder and it is the weakest of the four RNA encoders on codon information in the mRFP control (+0.02 above the controls, against +0.18 for mRNA-FM). On CoV-Vac it is still the best choice, because that task is about RNA structure, which is what RNA-FM was pretrained on.
- *"BioLangFusion: explain fusion and what these mean."* The three heads are explained once at the top of this guide and on slide 6; in the table, "concat" keeps every feature of every modality at every codon, "gated" keeps a weighted average with one weight per modality, "cross" lets each modality attend to every position of every modality.
- *"Mark if we use cross-attention, will the result be better."* With the swapped encoders, no. Cross-attention is a tie with the encoder alone on CoV-Vac and on E. coli with mRNA-FM, and resolved worse on E. coli with CaLM. The project's own cross-attention design (January, Gantt 25) will be scored against these same baselines.

**What came next.** Chase ran the other swaps between October 25 and 27: the DNA swaps (NT 500M, HyenaDNA), the protein swaps (ESM-2 35M, ESM-2 150M, ProtBERT), RiNALMo in the RNA slot, and the fungal benchmark, plus three swaps on stability. Slide 10 has all eight. DNABERT-2 stayed out because its variable-length tokens need a grid adapter.

**One sentence to say.** "Swapping one frozen encoder moved E. coli by eight points; fusion on top of the better encoder adds nothing we can resolve, and cross-attention does not change that."

---

## Slide 10. Phase B complete: all eight swaps

**What the slide shows.** Two things. On top, a grid: each fusion head minus the best single encoder validation chose, for every swap on every benchmark, one seed, with the 95% paired interval deciding the shading. Blue means fusion is ahead and the interval excludes zero, orange means fusion is behind and the interval excludes zero, grey is a tie. The dashed top row is the original BioLangFusion encoders at seed 0, for reference. Underneath, a table: each swapped encoder read alone, with the original encoder it replaces in brackets. The table is the encoder question; the grid is the fusion question.

**What ran.** Eight single-encoder swaps. DNA slot: NT 500M human-ref and HyenaDNA large in place of NT v2 100M. RNA slot: RiNALMo 150M, mRNA-FM and CaLM in place of RNA-FM. Protein slot: ESM-2 35M, ESM-2 150M and ProtBERT in place of ESM-2 8M. Each on fungal, CoV-Vac and E. coli, and three of them (mRNA-FM, CaLM, ESM-2 150M) on mRNA stability: 27 swap-benchmark pairs, each with the swapped encoder alone and all three heads. Seed 0, everything else at Phase A's settings, including the gated head's entropy weight, which is inherited rather than retuned. Chase ran them on his GPU between October 25 and 27; the intervals come from his notebook (2,000 draws, identical sequences resampled together). The figure is `r34_phaseB_full.pdf` in the manuscripts and `r34_phaseB_grid_slide.pdf` on the slide, made by `docs/notes/figures/make_phaseB_full_figure.py`.

**The encoder question, from the table.**

- *The RNA swaps change the most.* On fungal, RiNALMo, mRNA-FM and CaLM alone score 0.844, 0.852 and 0.851, against 0.722 for RNA-FM and 0.813 for NT v2, the best original encoder. A single swapped encoder alone now beats the paper's best fusion on fungal (0.824). On E. coli the same three lift the single encoder from 50.7% to 57.3% to 58.7%.
- *The DNA swaps are worse* than NT v2 100M on fungal (0.740 and 0.737 against 0.813) and CoV-Vac.
- *The protein swaps move little,* except ESM-2 150M on E. coli (53.9% against 50.1%).

**The fusion question, from the grid.** 81 comparisons: 15 resolved ahead, 14 resolved behind, 52 ties. Read it in four parts.

| Pattern | Where | What it means |
| --- | --- | --- |
| Fusion ahead, but only because the reference is weak | Fungal with ProtBERT (the three cells marked †) | Validation chose ProtBERT alone (0.761) although NT v2 alone scores 0.813; against NT v2 these are ties |
| Fusion ahead because the RNA slot is still weak | E. coli with the DNA and protein swaps (8 cells) | RNA-FM still sits in the RNA slot, so fusion compensates for it, as in Phase A |
| No fusion gain once the RNA encoder is good | E. coli and fungal with mRNA-FM or CaLM | Fusion ties, and on E. coli two heads are resolved worse (gated -5.6 with mRNA-FM, cross -3.7 with CaLM) |
| New gains no weak encoder explains | CoV-Vac cross-attention with HyenaDNA (+0.039) and RiNALMo (+0.031); E. coli gated with RiNALMo (+6.5) | Candidates, one seed each |

Two more patterns: on CoV-Vac, concatenation and the gated head are resolved worse than the best single encoder in 12 of 16 configurations, while cross-attention is never resolved worse there; and on stability all nine comparisons tie.

**What is genuinely new.** Cross-attention with HyenaDNA reaches 0.872 on CoV-Vac, above the paper's best CoV-Vac number (0.864), and with RiNALMo 0.854. Both are nucleotide-resolution encoders and CoV-Vac is the folding of 81-nucleotide fragments, so this is the first sign that letting nucleotide-level streams attend to each other helps a nucleotide-scale task. The gated head with RiNALMo on E. coli reaches 62.5%, the highest E. coli accuracy in the project, and there its weights actually mix (RNA 0.42, protein 0.50) instead of collapsing.

**The gated head, revisited.** With a strong RNA encoder the gate now gives RNA real weight: fungal 0.48 to 0.60, stability with mRNA-FM 0.79, CoV-Vac with RiNALMo 0.94. So slide 6's "RNA is zero everywhere" holds for the paper's RNA-FM, not for every RNA encoder; the gate tends to pick the strongest stream. But on E. coli with mRNA-FM, the best single encoder there, it still gives RNA 0.00 and scores 53.1%: it selects, and sometimes selects wrong.

**A side result: gridded DNA.** The sweep also ran the swapped DNA encoders stretched onto the codon grid. For NT 500M the grid helps (fungal 0.740 to 0.766); for HyenaDNA it hurts (fungal 0.737 to 0.676). So whether placing an encoder on the grid helps depends on the encoder, which is why the original NT v2 100M grid run is still worth doing.

**Caveats to say before Mina asks.** One seed per configuration, and the intervals measure which test rows were drawn, not training variation, so the three new gains are candidates. The entropy weight is inherited. The single reference is chosen on validation, which picked a weak one once (fungal, ProtBERT).

**What comes next.** Seeds on the three new gains; the mRNA-FM-anchored fusions Mina asked for; the plan's two-encoder replacements, chosen from these single swaps; DNABERT-2 once it has a grid adapter.

**One sentence to say.** "Across all eight swaps a better single encoder does most of the work, fusion gains on E. coli only appear while the weak RNA encoder is still in place, and the three new gains, two of them cross-attention on CoV-Vac, need seeds before we claim them."

---

## Slide 11. LucaOne, one jointly pretrained model (Priority 2)

**What the slide shows.** The manuscript's LucaOne figure (`r34_lucaone.pdf`): ridge-probe Spearman on the published mRNA stability test split for a composition baseline, each BioLangFusion encoder's pooled vector, the three concatenated, and LucaOne's gene stream, protein stream and both together. The two bracketed annotations under the bars are the paired comparisons with their intervals, so the claim and its uncertainty are both on the figure.

**Why we did this.** Priority 2 asks whether one model pretrained jointly on nucleotide and protein sequences (LucaOne, 1.8 billion parameters, trained on both alphabets with one vocabulary) gives a stronger representation than three separately pretrained encoders and their fusion. If joint pretraining already captures the cross-modal structure, then fusing separately pretrained models is solving a problem that pretraining can solve better.

**How we did it.** LucaOne cannot be dropped into the BioLangFusion grid (its tokenizer and context are different), so the comparison is made at the level of pooled vectors with an identical ridge probe for every row: the same rows, the same split, the same probe. LucaOne reads the coding sequence as its gene stream and the translated protein as its protein stream; "both streams" is the two pooled vectors concatenated. The bootstrap groups identical test sequences together.

**What "BLF three fused" means (Mina's question).** It is the pooled vector of each of BioLangFusion's three encoders (Nucleotide Transformer v2 100M, 512 wide; RNA-FM, 640; ESM-2 8M, 320) concatenated into one 1,472-wide vector, with the same ridge probe fit on it. It is the probe-level equivalent of concatenation fusion, included so that LucaOne is compared against the three encoders combined and not only against each alone. It is not the trained TextCNN concatenation head from slides 4 and 5; those are trained on token streams, these are linear probes on pooled vectors, and the two are not interchangeable.

**What the numbers mean.** LucaOne's two streams reach 0.483, against 0.390 for the three BioLangFusion encoders concatenated: +0.092, interval +0.066 to +0.120, resolved. Against the best single BioLangFusion encoder (Nucleotide Transformer, 0.312) it is +0.170. Even LucaOne's gene stream alone (0.443) beats the three fused. So one jointly pretrained model carries more linearly accessible stability signal than three separately pretrained encoders combined.

**Two caveats that go with the number.** First, this is the published split with its 68% train/test sequence overlap; grouping repeated sequences in the bootstrap does not remove the overlap, so read the number beside slide 7. Second, LucaOne is 1.8B parameters against 108M for the three encoders combined, so this compares representations, not matched budgets.

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

**One sentence to say.** "Once the input is a full transcript with its protein, fusion helps on both tasks, which fits the pattern: fusion adds something when the second input carries information the first one could not see."

---

## Slide 13. IsoFormer on GTEx: physically distinct molecules, no codon grid, fusion wins

**What the slide shows.** IsoFormer's six-row modality ablation (DNA, RNA, protein, and their combinations) on the 2,000-transcript GTEx pilot, ours with frozen encoders next to the paper's with fine-tuned encoders, and below it the manuscript's figure of the same rows (`r34_isoformer.pdf`), with the two resolved gains annotated on it.

**Why we did this.** Gantt 44, and the counterpart to BioLangFusion. Every benchmark where fusion failed gives all three encoders the same coding sequence. GTEx gives them three different molecules: 6,000 nucleotides of genomic DNA centered on the transcription start site (promoter and first exons), the transcript, and the protein. There is no codon grid that could align those, so IsoFormer fuses them with successive cross-attention between pooled modality representations. If fusion helps anywhere, it should help here, and Stage 1 found cross-modal alignment lowest exactly here (DNA shares almost no retrievable correspondence with RNA or protein on GTEx).

**How we did it.** Nucleotide Transformer v2 100M reads both the DNA window and the transcript, ESM-2 150M reads the protein, all frozen; IsoFormer's aggregator on top; predicting expression in 30 tissues; Spearman averaged over the tissues; split by gene (no gene shared between halves); one seed; paired bootstrap over the 1,000 test transcripts.

**What the numbers mean.** All three modalities beat RNA alone by +0.047 (interval +0.027 to +0.069), resolved; DNA with RNA beats RNA alone by +0.055 (+0.032 to +0.078), resolved. What carries the gain is RNA with DNA; protein adds nothing measurable (all three minus DNA with RNA is -0.007, a tie). Three of the paper's four orderings reproduce: RNA is the strongest single modality, all three beat the best single, DNA with RNA beats DNA with protein. The gap between our absolute values (0.50) and IsoFormer's (0.65) is what freezing the encoders costs: IsoFormer fine-tunes its encoders end to end and reports that RNA encoder specialization during fine-tuning is what makes its representation strong. Freezing is this project's premise, so that gap is a measurement, not a failure.

**Why this is the key slide for the alignment question.** Low alignment and high complementarity went together here: the DNA window and the transcript share almost nothing retrievable, and that is exactly why combining them adds information. Where alignment was high (three readings of one coding sequence), fusion added nothing. This is the empirical content of the Stage 2 definition: alignment measured after the composition control, reported separately from complementarity, and the two can move in opposite directions.

**One sentence to say.** "Where the three inputs are three different molecules, fusion wins by a margin that resolves, with frozen encoders, and the gain comes from DNA with RNA."

---

## Slide 14. What it adds up to, and what Mina asked for

**What the slide shows.** Five conclusions, and the questions and follow-ups from the October 22 meeting.

**The five conclusions, with the evidence behind each.**

1. *The build is right.* Single encoders within 0.01 of the paper; cross-attention within 0.01 on all four benchmarks (slide 4).
2. *The encoder matters more than the fusion head.* One encoder swap moved E. coli by 8 points and dissolved the fusion gain that had been compensating for the weak encoder (slide 9).
3. *Fusion still wins where the modalities are different molecules.* GTEx +0.047 resolved (slide 13); full transcript with protein +0.053 and +0.030 (slide 12).
4. *One jointly pretrained model beats three fused ones on stability.* LucaOne +0.092 resolved, on the leaky split (slide 11).
5. *The gated head is a selector, not a mixer.* Its weights are nearly one-hot, RNA is dropped everywhere, and on CoV-Vac it drops the best encoder (slide 6). The paper reports no error bars, and its own fungal gain is the size of noise on that split (slide 5).

**What Mina asked for on October 22, and where each lands.**

- Interpret why things work better, not only that they do: this guide, the speaker notes and the Overleaf now carry the interpretation on every slide (pretraining domain predicts the best single encoder; fusion helps when the second input is a different molecule or region).
- Build fusion around mRNA-FM (DNA + mRNA-FM, mRNA-FM + protein) and try other mRNA encoders: a Phase B extension on the coding-sequence benchmarks (Gantt 24).
- Check whether cross-attention does better with the swapped encoders: on CoV-Vac, yes with two nucleotide-level swaps (HyenaDNA +0.039, RiNALMo +0.031, one seed), and it is never resolved worse there; on E. coli and fungal, no (slide 10). The project's own cross-attention design is still scored in January (Gantt 25).
- Run LucaOne on the other datasets: E. coli, CoV-Vac and fungal fit its context; GTEx does not (Gantt 41).
- Which rows the paper reports on stability: the full 3,579-row published split (slide 7).

**Still open from the deck's own list.** Ab1 (not public; can it be obtained through UCF); seeds plus a paired bootstrap as the reporting standard for everything; GPU access for Coupled Mamba; the distillation teachers (which two DNA encoders and which RNA encoder join the twelve).

**Open runs, in order.** Seeds on the three new Phase B gains; the gated head at 1,280 channels on the other three benchmarks; the sequence-grouped stability re-split with seeds; seeds on the swaps that matter; the mouse half-life and translation-efficiency scores; LucaOne on the three coding-sequence benchmarks; the mRNA-FM-anchored fusions.

**One sentence to say.** "The reproduction is faithful, the paper's fusion gains mostly do not survive a stronger single encoder, and the project's question is now which encoder to put in each slot and when a second molecule is worth fusing."
