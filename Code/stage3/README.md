# Stage 3: build the fusion models

Each notebook here builds one model, or the data it needs, and checks that it works. The model code itself lives in [shared_code/fusion/](../shared_code/fusion/) so that the Stage 4 experiments import exactly what was checked here. Experiments with results belong in [stage4/](../stage4/README.md).

Two rounds, following the research plan (Overleaf Section 11):

**Round 1, now: the pipeline and the research plan's baselines**

| Notebook (planned) | What it builds and how it is checked |
| --- | --- |
| `01_blf_datasets.ipynb` | The five BioLangFusion datasets (Fungal, E. coli, mRNA stability, Ab1, CoV-Vac) with their published splits |
| `02_token_embeddings.ipynb` | Per-token embeddings for BioLangFusion's encoders (NT v2 100M, RNA-FM, ESM-2 8M); run on the cluster |
| `03_blf_build_check.ipynb` | Codon grid, the three fusion heads, and the TextCNN head; checked by reproducing the published mRNA stability scores (0.553 single, 0.563 fused) |
| `04_lucaone_isoformer_check.ipynb` | LucaOne's frozen embeddings (input limit checked) and IsoFormer's aggregator on the GTEx pilot |

**Round 2, January: the project's designs**

| Notebook (planned) | What it builds |
| --- | --- |
| `05_coupled_mamba_check.ipynb` | Coupled Mamba as a fourth head in the same pipeline |
| `06_cross_attention_check.ipynb` | IsoFormer-style cross-attention with the encoders Stage 1 points to |

Token-level embeddings are large; keep them on cluster scratch storage, not in Git.
