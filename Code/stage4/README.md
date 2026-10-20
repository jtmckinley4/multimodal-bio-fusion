# Stage 4: run the experiments

Each notebook follows one Stage 4 tier using models built and checked in [stage3/](../stage3/README.md), importing them from [shared_code/fusion/](../shared_code/fusion/). A notebook may inspect saved results while a separate runner performs training. Tiers and Gantt IDs follow the Overleaf (Section 12.4) and `docs/Research_Project_Plan_Gantt.xlsx`.

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

## BioLangFusion Phase A

[`tier2_blf_phaseA.ipynb`](tier2_blf_phaseA.ipynb) reads one explicitly selected benchmark and result directory. It explains the fixed protocol, checks saved evidence, selects models on validation, and compares their test scores. Its coverage and missing-fit preview come from the result files; opening the notebook does not launch training. Choose `fungal_expression` or `ecoli_proteins` in setup. Existing fungal results remain in their Stage 3 directories.

[`run_blf_phaseA.py`](run_blf_phaseA.py) executes the same declared candidate grid separately. From the repository root, preview an E. coli run with:

```shell
python -B Code/stage4/run_blf_phaseA.py --benchmark ecoli_proteins --cache-root PATH_TO_EXISTING_STAGE3_EMBEDDINGS --results-dir Code/stage4/results/blf_phaseA/ecoli_proteins --device cuda
```

The preview reads the retained data and existing caches, hashes the cache contents, and checks every planned result before any fit. It does not extract embeddings. Add `--execute` only to start the reviewed grid. `--device mps` selects Apple acceleration; `--device cpu` requests CPU execution; `--device auto` uses the available backend. A requested unavailable backend is an error rather than a silent fallback.

Compatible completed fits with validated per-row predictions are reused. Interrupted fits resume through the shared trainer's checkpoints; work after the latest checkpoint may repeat. The trainer removes a checkpoint after completing its result: saved predictions support repeated evaluation, but the result JSON does not retain fitted weights for new inference. Incompatible files and older score-only results are preserved and block execution in that directory. Use a separately reviewed output directory for a new cohort instead of deleting or replacing those records.

The notebook's analysis manifest identifies all candidate result files and their hashes, the selected files, selection rules, reader sources, and bootstrap settings. Save that manifest with reported tables or figures so the analysis can be reconstructed. Keep execution details, numerical interpretations, and unresolved producer evidence in [run records](../../docs/run-records.md); notebook prose explains the reusable method and its limits. A repeated analysis reads saved predictions and does not need another fit. Preserve earlier exports and write a new analysis directory when the reader, selection rule, or inputs change.
