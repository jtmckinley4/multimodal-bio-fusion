# Stage 3: build the fusion models

Each notebook here builds one model, or the data it needs, and checks that it works. The model code itself lives in [shared_code/fusion/](../shared_code/fusion/) so that the Stage 4 experiments import exactly what was checked here. Experiments with results belong in [stage4/](../stage4/README.md).

Two rounds, following the research plan (Overleaf Section 11):

**Round 1, now: the pipeline and the research plan's baselines**

| Notebook | What it builds and how it is checked |
| --- | --- |
| [01_blf_datasets.ipynb](01_blf_datasets.ipynb) **done** | The BioLangFusion benchmarks with their published splits. Four of five are in hand (Ab1 is not in the public release). Recovers the paper's preprocessing: a 999-nucleotide limit reproduces all four published "used" counts. Records four deviations for Phase A, including 68% train/test sequence overlap in the published mRNA stability split |
| [02_token_embeddings.ipynb](02_token_embeddings.ipynb) **done** | Per-token embeddings for BioLangFusion's encoders (NT v2 100M, RNA-FM, ESM-2 8M) into a resumable shard cache under `embeddings/`, about 34 GB in float16 across the four benchmarks. Ran on the Mac 2026-10-16 in 53 minutes: 34.2 GB, 405 shards, every row verified ([run record](../../docs/run-records.md#2026-10-16-stage3-token-embeddings-token-level-caches-for-the-four-biolangfusion-benchmarks)). Settles deviation 4: the protein stream uses `grid_protein`, one token per codon |
| [03_blf_build_check.ipynb](03_blf_build_check.ipynb) **run 2026-10-17, half reproduced** | Codon grid, the three fusion heads, and the shared TextCNN head, assembled from [shared_code/fusion/](../shared_code/fusion/) and checked against fungal expression: published 0.804 for the best single encoder and 0.824 for the gated head, with the full Table 1 and the Table 4 ablations as the target. Nine runs including the paper's lambda sweep over {0.01, 0.5, 1}; resumable, and it checkpoints every ten epochs. Not checked against mRNA stability, whose published scores sit on a test split with 68% sequence overlap. Result: the three single encoders reproduce in the published order, cross-attention exceeds its published value, and concatenation and the gated head fall short; the appendix's transposed convolution misaligns the DNA stream by two codons, and the rerun under Equation 1 is the open test ([run record](../../docs/run-records.md#2026-10-17-stage3-blf-build-check-biolangfusion-reproduced-on-fungal-expression)) |
| [04_lucaone_isoformer_check.ipynb](04_lucaone_isoformer_check.ipynb) **run 2026-10-19** | LucaOne's input limit, measured: its 1,280-token context admits none of the GTEx pilot's 6,000-nt DNA window and only 40% of transcripts, so a LucaOne GTEx baseline is not possible; joint checkpoints do exist on the hub, so context rather than availability is the blocker. Then IsoFormer's aggregator (successive cross-attention, no codon grid) on the GTEx pilot, run as the six-way modality ablation of its Table 2. The pilot is a 2,000-row sample of IsoFormer's own released dataset, split by gene |

**Round 2, January: the project's designs**

| Notebook (planned) | What it builds |
| --- | --- |
| `05_coupled_mamba_check.ipynb` | Coupled Mamba as a fourth head in the same pipeline |
| `06_cross_attention_check.ipynb` | IsoFormer-style cross-attention with the encoders Stage 1 points to |

Token-level embeddings are large; keep them out of Git. Training results are small JSON
files and are tracked; the checkpoints beside them are not.
`results/fungal_expression/reproducible/` holds Chase's fifteen seed-0 fungal fits from his Windows
desktop (nine appendix, six Equation 1), which record a run identity but not per-row predictions; the
Stage 3 comparisons read `results/fungal_expression/` directly.

**Why fungal expression is the build check.** Of the four benchmarks in hand it is the only
one with clean published splits, published values for all six variants, and no decision of
ours in the way. mRNA stability's published scores are measured on a test split where 68% of
rows reuse a training sequence, usually with a different label, so matching them would be
evidence about leakage rather than about the build. CoV-Vac depends on how its 81-nucleotide
fragments are translated, which was our choice, not the paper's. E. coli is classification and
is a third check that the loss switch works.
