# Stage 4: run the experiments

Each notebook follows one Stage 4 tier using models built and checked in [stage3/](../stage3/README.md), importing them from [shared_code/fusion/](../shared_code/fusion/). A notebook may inspect saved results while a separate runner performs training. Tiers and Gantt IDs follow the Overleaf (Section 12.4) and `docs/Research_Project_Plan_Gantt.xlsx`.

**Who runs which of these, and in what order, is in [RUNBOOK.md](RUNBOOK.md).** Read it before starting anything: the results folders are kept disjoint so that two machines can work the same night without colliding.

The table's "research plan item" column is the numbered priority list in Mina's October 9 plan (`docs/Multimodal_Research Plan .docx`), which is the order the project follows.

| Notebook | Tier | Research plan item | Gantt | Status |
| --- | --- | --- | --- | --- |
| [tier1_mrfp_control.ipynb](tier1_mrfp_control.ipynb) | 1 | 5, the mRFP synonymous control | 18 | **run 2026-10-19.** Protein encoders invariant; every nucleotide encoder predicts expression above GC3, codon use and the structure controls, mRNA-FM by far the most ([run record](../../docs/run-records.md#2026-10-19-stage4-mrfp-control-the-mrfp-synonymous-control-research-plan-item-5)) |
| [tier2_blf_phaseA.ipynb](tier2_blf_phaseA.ipynb) | 2 | 1, Phase A: BioLangFusion on E. coli, CoV-Vac and mRNA stability (fungal is in Stage 3 step 3; Ab1 is not public) | 17 | **run 2026-10-20, 26.9 hours.** Single encoders and cross-attention reproduce; the gated head does not, and its weights show it selecting one modality; fusion beats the best single encoder on E. coli, loses on CoV-Vac, ties on stability's published split, where every model drops 0.2 on unseen sequences ([run record](../../docs/run-records.md#2026-10-20-stage4-blf-phasea-biolangfusion-phase-a-on-cov-vac-e-coli-and-mrna-stability-research-plan-item-1)) |
| [tier2_encoder_swaps.ipynb](tier2_encoder_swaps.ipynb) | 2 | 1, Phase B: one encoder swapped at a time; eight swaps on CoV-Vac, fungal and E. coli, three on mRNA stability | 24 | **built, not run.** Chase's; needs a GPU night and about 53 GB of token cache, which the notebook projects before extracting |
| [tier2_lucaone_isoformer.ipynb](tier2_lucaone_isoformer.ipynb) | 2 | 2, LucaOne frozen on mRNA stability; IsoFormer's GTEx table at three seeds | 41, 44 | **built 2026-10-20, not run.** Chase's. GTEx is ruled out for LucaOne by its 1,280-token context, measured in Part 1 rather than assumed |
| [tier3_halflife_te.ipynb](tier3_halflife_te.ipynb) | 3 | 3 and 4: full-transcript half-life and translation efficiency, behind the plan's gate | 42 | **built 2026-10-20, not run.** Julian's. Parts 1 and 2 (the gate and the composition floor) are half an hour on a laptop; Part 3 is a GPU night and waits for the gate |
| `tier4_designs.ipynb` | 4 | Coupled Mamba and cross-attention against every baseline | 20, 21, 25 | planned, January |
| `tier5_ablations.ipynb` | 5 | Alignment loss, codon grid, modality dropping | 23 | planned |

Results are small files under `results/`, one folder per notebook (`tier1_mrfp/`, `phaseA/<benchmark>/`, `phaseB/<benchmark>/<encoder>/`, `lucaone/`, `gtex_isoformer/`, `halflife_te/`): JSON per training run with its per-row test predictions, CSV for the probe tables. They are tracked; the checkpoints beside them are not. Token caches for the swapped encoders are written next to Stage 3's under `../stage3/embeddings/` and stay out of Git. Every reported result gets a [run record](../../docs/run-records.md).

## What the three new notebooks add to `shared_code/`

| Module | What it is for | Checked by |
| --- | --- | --- |
| [`lucaone.py`](../shared_code/lucaone.py) | LucaOne's checkpoints, its context arithmetic, and pooled embeddings for its gene and protein streams. It is deliberately not an entry in `encoders.ENCODERS`: its tokenizer takes a stream-type argument no other tokenizer takes, and loading 1.8B parameters should never be something a sweep does by accident | [`test_lucaone.py`](../../tests/test_lucaone.py) |
| [`mrnabench.py`](../shared_code/mrnabench.py) | The mRNABench catalog, the transcript regions recovered from its codon-start track, the plan's feasibility gate, gene-grouped splits, the benchmark's own ridge probe, and chunked pooling for transcripts longer than an encoder's context | [`test_mrnabench.py`](../../tests/test_mrnabench.py) |
| [`matrix_cache.py`](../shared_code/matrix_cache.py) | One matrix per file, reused only when every setting that produced it matches. Shared by the two above so that neither widens Stage 1's embedding cache metadata, which would invalidate caches that took days to fill | exercised by both suites |
| `fusion/token_embeddings.pooled_matrix` | Mean-pools a benchmark's token cache into one vector per row, so LucaOne can be compared against BioLangFusion's own encoders on the same rows without extracting anything twice | [`test_fusion_model.py`](../../tests/test_fusion_model.py) |

## The appendix-convolution cohort on E. coli (Chase, October 19 to 20)

[tier2_blf_phaseA_appendix.ipynb](tier2_blf_phaseA_appendix.ipynb) and [run_blf_phaseA.py](run_blf_phaseA.py) are Chase's Phase A reader and runner, kept under their own names. They ran E. coli at three seeds under the appendix convolution (kernel 3, stride 2, padding 2), all nine candidates per seed, on an RTX 2060 SUPER; the 27 results and their provenance records are in `results/blf_phaseA/ecoli_proteins/` and the [run record](../../docs/run-records.md#2026-10-19t220959z-phasea-ecoli-three-seed-classification-comparison). Beside the Equation 1 results in `results/phaseA/ecoli_proteins/`, they are the convolution comparison on a second benchmark. Both files were written against a trainer refactor (run identities, checkpointed random state) that is not in `shared_code/fusion/train.py` on this branch; that refactor is reviewed separately, so these two files do not run as they stand against the shared code here.

