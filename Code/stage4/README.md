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

[`tier2_blf_phaseA.ipynb`](tier2_blf_phaseA.ipynb) reads one explicitly selected benchmark and result directory. Its retained settings describe the completed appendix-convolution comparison: three native-resolution single encoders and the declared fusion candidates, with validation selection and paired uncertainty calculated separately within each seed. Its coverage preview describes the selected local records; a record absent here does not establish that the team needs another fit. Full-grid readiness applies to this historical comparison, not to every Phase A study. Opening the notebook does not launch training. Choose `fungal_expression` or `ecoli_proteins` in setup. Existing fungal results remain in their Stage 3 directories.

The intended team comparison uses Equation 1 convolution and adds an RNA-FM control that averages each nonoverlapping nucleotide triplet before the existing TextCNN. That control helps distinguish the effect of RNA pooling from the effect of fusion. Julian has reported implementing this work, including fewer repeated candidates and a seed-averaged paired bootstrap. The exact repeat, validation-selection, and uncertainty rules still need review against his implementation. Keep the completed appendix results distinct; adding the control must not silently change the earlier best-baseline comparison.

[`run_blf_phaseA.py`](run_blf_phaseA.py) is the retained producer for the completed appendix workflow. It remains available to check and reuse that cohort; it is not yet the implementation of the intended team comparison. From the repository root, check the retained E. coli results with:

```shell
python -B Code/stage4/run_blf_phaseA.py --benchmark ecoli_proteins --cache-root PATH_TO_EXISTING_STAGE3_EMBEDDINGS --results-dir Code/stage4/results/blf_phaseA/ecoli_proteins --device cuda
```

The preview reads the retained data and existing caches, hashes the cache contents, and checks every result in that producer's grid before any fit. It does not extract embeddings. Its `--execute` option starts that retained grid, so it is not a shortcut to the intended team protocol. `--device mps` selects Apple acceleration; `--device cpu` requests CPU execution; `--device auto` uses the available backend. A requested unavailable backend is an error rather than a silent fallback.

Compatible completed fits with validated per-row predictions are reused. Interrupted fits resume through the shared trainer's checkpoints; work after the latest checkpoint may repeat. The trainer removes a checkpoint after completing its result: saved predictions support repeated evaluation, but the result JSON does not retain fitted weights for new inference. Incompatible files and older score-only results are preserved and block execution in that directory. Use a separately reviewed output directory for a new cohort instead of deleting or replacing those records.

The notebook's analysis manifest identifies all candidate result files and their hashes, the selected files, selection rules, reader sources, and bootstrap settings. Save that manifest with reported tables or figures so the analysis can be reconstructed. Keep execution details, numerical interpretations, and unresolved producer evidence in [run records](../../docs/run-records.md); notebook prose explains the reusable method and its limits. A repeated analysis reads saved predictions and does not need another fit. Preserve earlier exports and write a new analysis directory when the reader, selection rule, or inputs change.

The next step is to review Julian's implementation and existing results together with these retained artifacts, identify compatible work, and agree on any remaining gaps. Local file availability alone does not establish whether he has completed a comparison. Do that integration review before scheduling another benchmark or fit; CoV-Vac remains a later candidate, not an automatic continuation.

### Equation 1 and RNA-control integration

Equation 1 already has a shared implementation in [`CodonGrid`](../shared_code/fusion/codon_grid.py): `tconv="equation"` selects kernel 2, stride 2, and padding 0. For complete 6-mer tokens, input position `t` contributes to output positions `2t` and `2t + 1`, before length fitting and masking. This is a learned upsampling operation. The two-position interpretation does not resolve the individual nucleotide tokens that NT emits for a remainder; making the final shape match the codon grid does not prove their biological correspondence.

The additional RNA baseline should reuse the grid's RNA operation: average each consecutive group of three nucleotide embeddings, retain only complete triplets from the chosen origin, and pass those vectors and their codon mask to the existing TextCNN. For scalar token values `[1, 2, 3, 4, 5, 6, 100, 100]`, the two valid pooled values are `[2, 5]`; the last two nucleotides do not form a codon. Positions beyond a row's complete codons must stay masked when that row shares a padded batch. Pooling also changes the positions covered by a TextCNN window: a width-3 kernel reads three nucleotide-token positions in the native RNA baseline and three pooled triplets in the control. This is a resolution/preprocessing comparison, not an isolated test of a biological mechanism.

Use these checks when integrating Julian's reported baseline:

- The RNA vectors and mask entering TextCNN match the verified grid operation. Changing only the DNA or protein values must not change the control's prediction; the control needs no DNA upsampling or fusion layers.
- The native RNA baseline remains unchanged. The control uses the same RNA embedding width and TextCNN settings, with the same trainable parameter count, and produces one output for regression or three logits for E. coli classification. Give the control its own variant and result identity; never relabel a native RNA result.
- Report native RNA and codon-pooled RNA separately. Before selecting a comparator, state whether the control participates in the new analysis's “best single encoder” selection or is reported separately. The historical reader's three native baselines retain their existing eligibility.
- Inspect which models are repeated, whether the baseline and MIL weight are selected once or within each seed, and how paired rows and seeds enter the bootstrap. Keep these decisions explicit before comparing or combining records; repeated predictions on the same test rows are not independent observations.

The [forward-only checks](../../tests/README.md#forward-only-codon-grid-checks) verify the existing grid with synthetic values. They do not implement or validate the standalone RNA baseline, certify a biological reading frame, or complete the reproduction. Review the contributing model/factory connection and apply the acceptance checks above when its source is available; no second RNA baseline implementation is needed in preparation.

Verified token caches can serve Equation 1 and the RNA control without new encoder inference. Native single-encoder models bypass the DNA convolution, so earlier native results may be scientifically comparable after reviewing their inputs and remaining protocol. Exact execution-resume identity still includes configuration and source hashes; do not rewrite those identities or weaken the reuse checks. Appendix fusion predictions cannot become Equation 1 results. Inspect saved Equation 1 results for matching configuration, provenance, ordered test rows, and prediction fields before deciding whether further fits are needed. These integration checks support [Gantt/Details tasks 15–17 and 28](../../docs/Research_Project_Plan_Gantt.xlsx); they schedule no training.
