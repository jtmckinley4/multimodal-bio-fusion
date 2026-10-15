# Stage 4: run the experiments

Each notebook runs one Stage 4 tier on models built and checked in [stage3/](../stage3/README.md), importing them from [shared_code/fusion/](../shared_code/fusion/). Tiers and Gantt IDs follow the Overleaf (Section 12.4) and `docs/Research_Project_Plan_Gantt.xlsx`.

| Notebook (planned) | Tier | Research plan item | Gantt |
| --- | --- | --- | --- |
| `tier1_mrfp_control.ipynb` | 1 | 5, the mRFP synonymous control | 18 |
| `tier2_blf_phaseA.ipynb` | 2 | 1, Phase A: BioLangFusion on its five datasets | 17 |
| `tier2_encoder_swaps.ipynb` | 2 | 1, Phase B: one encoder swapped at a time | 24 |
| `tier2_lucaone_isoformer.ipynb` | 2 | 2, LucaOne; and IsoFormer on GTEx | 41, 44 |
| `tier3_halflife_te.ipynb` | 3 | 3 and 4: half-life and translation efficiency | 42 |
| `tier4_designs.ipynb` | 4 | Coupled Mamba and cross-attention against every baseline | 20, 21, 25 |
| `tier5_ablations.ipynb` | 5 | Alignment loss, codon grid, modality dropping | 23 |

Save small result tables and figures in `results/`, with a [run record](../../docs/run-records.md) for anything reported.
