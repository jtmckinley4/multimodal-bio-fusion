# Meeting notes with Mina

- Check esm encoder where it can be part of the similarity in the embedding space where it can be in direction of low and high variance, pca could be applicable here
- hyena couple model; good for very long sequence
- start to use same model in bio lang and use same model they use in alignment since they argue about tokenization.
- sometimes i need to put these together to see which correlation is being shown for a more comprehensive analysis using some of the simpler fusion methods from bioLang with many pre-trained models
- think of some biological use cases, but keep in general
- if there are 3 RNA based models can they be used together since they are trained on different datasets
- DNA and mRNA has huge overlap information wise, so an experiment is surrounding mRNA and proteins
- one thing really want: overleaf project, create GitHub, share it with Mina
- next meeting is october 5th or 7th

## October 22, 2026: Stage 4 round 1 review

Mina's questions on the progress deck (`Slides/Mina_Progress_Slides_2026-10-22.pptx`), as noted in the meeting, and where each is answered. The full answers are in `Slides/Mina_Progress_Slides_2026-10-22_Guide.md` and in the deck's speaker notes; the Overleaf and `Mina_Project_Notes.docx` (Section 13) carry the same.

- Slide 4 (reproduction table): "another table." The table is only the comparison with the paper; the fusion question is slide 5.
- Slide 5 (central claim): interpret why these work better; for example, if someone wants to train a foundation model, train on task-specific datasets. Answer: on each coding-sequence benchmark the strongest single encoder is the one whose pretraining is closest to the task's biology (RNA-FM on CoV-Vac, mRNA-FM and CaLM on E. coli, Nucleotide Transformer on fungal); fusion helps only when the second input is a different molecule or region.
- Slide 6 (gated head): E. coli, why is RNA 0? What is the gated head and how does it work? Is gated MIL attention related? How does it help with fusion and concatenation? Answer: it is the attention-MIL pooling of Ilse et al. 2018 with the three modalities as the bag; softmax plus the entropy penalty is winner-take-all; RNA-FM carries the least codon information and repeats the DNA letters, so it is dropped; the head reweights and does not add information, which is why concatenation and cross-attention beat it.
- Slides 7 and 5 (stability): did BioLangFusion report 1,000 rows or 3,000 rows? Answer: the full 3,579-row published test split (ours match within 0.01); the 1,135 unseen-sequence rows are ours.
- Slide 8 (mRFP): mRNA-FM performs best; maybe fusion against mRNA-FM (DNA or RNA fusion with mRNA-FM); try different mRNA encoders as well. Adopted as a Phase B extension (Gantt 24).
- Slide 9 (Phase B): see whether fusion replaces RNA-FM and whether it works; check RNA-FM; explain the BioLangFusion fusion heads and what they mean; what does "BLF three fused" mean (slide 10); mark whether cross-attention does better; run LucaOne on the other datasets. Answers: a good RNA encoder replaces the fusion, not the other way around; cross-attention is not better on the four swaps run; "BLF three fused" is the three pooled vectors concatenated under the same ridge probe; LucaOne on E. coli, CoV-Vac and fungal adopted (Gantt 41).

Follow-ups adopted, as additions to the plan: an interpretation layer on every result (deck notes, slide guide, both Overleaf versions, project notes Section 14); fusion built around mRNA-FM and the other codon-level RNA encoders; LucaOne on the three coding-sequence benchmarks; cross-attention reported beside every swap.

Still with Mina: Ab1 access; seeds plus a paired bootstrap as the reporting standard; GPU access for Coupled Mamba; the distillation teachers.

## October 29, 2026: follow-up on the Stage 4 round 1 deck

Julian's and Chase's notes, combined. Every item is an addition to the plan, not a replacement.

Resources and contacts
- GPU: ask UCF for access to GPUs on the Stokes cluster (the note says "100", typically a few hundred; confirm the GPU type and allocation). The lab's H100s are not open to students, though access may be possible close to a paper deadline.
- Sean Moore (UCF Burnett School of Biomedical Sciences, lab page med.ucf.edu/semoore; the lab lists an applied industrial microbiology project): ask about E. coli datasets; read his current publications first.

Framing: task and data first
- A well-designed fusion method can still fail on the wrong task, so the first step is the data or the task. Look for task-specific datasets, ideally truly multimodal biology datasets (inputs that are different molecules), to see where fusion works.
- The BioLangFusion benchmarks can still be used for tasks where they make sense. Other fusion papers mostly use their own datasets; we can show this and argue why.
- BioLangFusion is a workshop paper, not a main-track one, which helps explain the leaky stability split.
- Focus now: literature (arXiv, NeurIPS) and Hugging Face. For these datasets, find similar tasks where fusion works and where it does not, spot the differences, and argue why.

Slide 6 (gated head) and new fusion designs
- Training is "lazy": the model takes the easiest signal first, often the protein encoder, which may be why the gated head collapses onto one modality. Cross-attention fuses better because it has no gate, but it weights the modalities in a fixed way; an adaptive gate is sometimes better.
- Flamingo (vision and language) fuses frozen models with gated cross-attention. Gated cross-attention at the codon level could be a real novelty for this project. Coupled Mamba stays in the plan; BioLangFusion applied three methods (concatenation, gated, cross-attention).

Slide 8 (mRFP; mRNA-FM against CaLM)
- mRNA-FM and CaLM both read codons and were trained only on coding sequences, yet mRNA-FM does best. Compare their training data and look for the specific features mRNA-FM has that CaLM lacks. From the model cards: mRNA-FM, 45 million RefSeq coding sequences, about 0.24B parameters; CaLM, about 9.9 million ENA coding sequences clustered at 40% identity, about 86M parameters. Stage 1 CKA between the two is only 0.163.
- Hyperparameter tuning on mRNA-FM: vary how many of its layers are used (more or fewer) and look at a heat map of layer importance.
- Consider fusion together with fine-tuning.

Slide 12 (half-life and translation efficiency)
- Test whether RNA secondary structure helps half-life prediction. Coding and non-coding regions both matter, and structure is heavily involved in regulation, so the proteins involved (RNA-binding proteins) are worth looking into as well. Look for datasets with RNA sequence plus secondary structure, and possibly secondary structure prediction.

Slide 13 (GTEx)
- Rerun the pilot with BioLangFusion's encoders (NT v2 100M, RNA-FM, ESM-2 8M) so the comparison with the BioLangFusion benchmarks is controlled.
- Fine-tune IsoFormer the way the paper does, ourselves if GPU access allows.
