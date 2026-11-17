# Correction input inspection

The executed copies use the maintained notebook's opt-in fold and bounded-cache inspections. They support keeping the five-fold design for all three regression benchmarks; they do not establish that the future anchors or correction models will train well. Grouped stability initially stopped at its missing canonical split record. A separate inspection then supplied a temporary candidate built with the already-agreed protocol, preserving the strict production loader and leaving the canonical split unpublished.

## Read the evidence

| Dataset | Fold audit | Cache/control audit | Executed copies |
| --- | --- | --- | --- |
| CoV-Vac | [Fold report](../../../anchor_correction/cov_vac_degradation/analyses/decision-support-folds-dec1224df8f458e3.json) | [Control report](../../../anchor_correction/cov_vac_degradation/analyses/decision-support-controls-c078ea33b71691ff.json) | [Folds](local_kernel_retry/cov_vac_degradation-folds.ipynb), [controls](local_kernel_retry/cov_vac_degradation-controls.ipynb) |
| Fungal expression | [Fold report](../../../anchor_correction/fungal_expression/analyses/decision-support-folds-2446530b0b409a5f.json) | [Control report](../../../anchor_correction/fungal_expression/analyses/decision-support-controls-793e6199de20b58d.json) | [Folds](local_kernel_retry/fungal_expression-folds.ipynb), [controls](local_kernel_retry/fungal_expression-controls.ipynb) |
| Grouped stability, candidate split | [Fold report](../../../anchor_correction/mrna_stability/analyses/decision-support-folds-a98018b61aaa682c.json) | [Control report](../../../anchor_correction/mrna_stability/analyses/decision-support-controls-40495cadc1850b33.json) | [Folds](stability_candidate/mrna_stability-folds.ipynb), [controls](stability_candidate/mrna_stability-controls.ipynb) |

CoV-Vac's five folds contain 976–1,063 fitting groups, 224–270 stopping groups and 298–365 held-out groups. Fungal expression contains 1,366–1,398 fitting groups, 325–361 stopping groups and 416–446 held-out groups. Grouped stability contains 5,010–5,135 fitting groups, 1,242–1,285 stopping groups and 1,534–1,679 held-out groups. Every inspected fold role has varied targets. These counts support mechanical feasibility, not an optimal fold count or equivalent residual distributions between the smaller fold models and the eventual final anchor.

For each dataset, the feature check samples 32 rows per partition, giving 96 rows per modality and encoder combination. None of the sampled real/donor feature arrays is identical in either the original or mRNA-FM combination. This verifies that this recorded shuffle changes those inputs; it is not an exhaustive check of all arrays or training seeds.

CoV-Vac retains its fixed 81-nucleotide length under shuffling. Fungal donors change length for 2,144/2,153 training rows, 554/555 validation rows and 430/430 test rows. The fungal control therefore disrupts length as well as sequence content. It supports a whole-input mismatch comparison, not a content-only claim holding length fixed. Retaining that interpretation is reasonable; a length-stratified control would be a separately agreed comparison.

Grouped stability changes length for 16,895/16,940 training rows, 3,468/3,493 validation rows and 3,473/3,496 test rows. More consequentially, each receiver group uses its donor group's inputs with the receiver's observation count. Unequal group sizes therefore change donor exposure: total-variation distances between original and shuffled group frequencies are 0.361, 0.363 and 0.372. This statistic is half the sum of absolute probability differences; it measures distribution shift, not prediction error. CoV-Vac is zero and fungi are at most 0.0028. The stability control tests pairing plus length and input-frequency changes. Decide whether that broad disruption is the intended control or whether a frequency-preserving alternative is needed; no donor, weighting or exclusion rule was changed here.

The [candidate split](stability_candidate/candidate-split.json) uses seed 42 and the approved exact-sequence grouping, with 16,940/3,493/3,496 train/validation/test rows. Its fingerprint is `793991aa9bec95e23d6a2c055c61bb5e09f718bf57418f628d6c48a7edce5d40`. The [candidate runner](run_stability_candidate.py) passes a temporary file through the existing row/order/target validation and uses canonical sequence strings for OOF grouping. Later canonical preparation should reproduce this fingerprint from unchanged inputs; this inspection does not establish that any grouped fit exists.

Pooled feature norms differ across encoders. They describe untrained representation scale, not predictive value or modality importance. No standardization, donor policy, fold allocation or model was selected from these observations.

## Execution and preservation

[Execution records](local_kernel_retry/cov_vac_degradation-folds.json) beside each executed copy identify its source and runner hashes, CPU environment, timestamps and outcome. [The original source snapshot](source_snapshot.zip) preserves the notebook and shared code; [the retry runner](run_inspections_local_kernel.py) records the explicit CPU settings and blocks training, extraction and bootstrap calls. The maintained notebook was not saved or changed.

The first Windows-sandbox attempt showed no observed cell progress and was interrupted. Its [attempt record](cov_vac_degradation-folds.json) is preserved. The separate local-kernel retry completed CoV-Vac and fungi. Stability's [initial missing-input failure](local_kernel_retry/mrna_stability-folds.json) is retained alongside the later successful candidate inspection; neither represents a failed model or replacement training.
