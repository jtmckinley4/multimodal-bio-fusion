# A6 input inspection before Newton

These executions inspect the fixed candidate inputs for human RNA half-life and translation efficiency before allocating encoder extraction or training. They support Gantt/Details task 42. The source [region-aware notebook](../../../../region_aware_fusion.ipynb) was unchanged; only executed copies enabled `RUN_INPUT_AUDIT`. Preparation publication, extraction, training, learned diagnostics and performance analysis remained disabled. Passing these checks establishes input consistency under the declared policy, not scientific acceptance of that policy.

## Read the evidence

| Dataset | Tables and embedded plots | Saved input measurements | Execution and source snapshots |
| --- | --- | --- | --- |
| Human RNA half-life | [Executed notebook](rnahl-human-retry1/executed-input-audit.ipynb) | [Audit JSON](../../../region_aware_v1/rnahl-human/input-audit-v1.json) | [Execution record](rnahl-human-retry1/execution.json), [sources](rnahl-human-retry1/sources/) |
| Human translation efficiency | [Executed notebook](translation-efficiency-human/executed-input-audit.ipynb) | [Audit JSON](../../../region_aware_v1/translation-efficiency-human/input-audit-v1.json) | [Execution record](translation-efficiency-human/execution.json), [sources](translation-efficiency-human/sources/) |

The first [sandboxed startup attempt](rnahl-human/execution.json) was interrupted before verified cell progress; it produced no audit. The successful executions used the same source cells outside that kernel-communication restriction. `run_input_inspections.py` records the bounded execution route. Source hashes, environment and timing are in the execution records; canonical `preparation.json` files were not created.

## Which rows and groups remain?

| Measurement | RNA half-life | Translation efficiency |
| --- | ---: | ---: |
| Supplied rows | 12,939 | 10,281 |
| Retained rows | 12,888 | 10,255 |
| Excluded rows | 51 | 26 |
| Mapped retained rows | 6,719 / 12,888 | 5,180 / 10,255 |
| Unmapped retained rows, grouped by gene | 6,169 / 12,888 | 5,075 / 10,255 |
| Train / validation / test rows | 9,023 / 1,932 / 1,933 | 7,179 / 1,538 / 1,538 |
| Distinct evaluation groups | 8,908 | 7,473 |

Half-life excludes 34 rows whose final marked codon extends beyond the supplied transcript and 17 with an internal stop. Translation efficiency excludes 20 internal-stop rows, four without marked coding regions and two whose final marked codon extends beyond the transcript. No alternative exclusions, split seed or grouping policy was selected after inspecting these results.

Both candidate splits pass the implemented group and exact-sequence overlap checks. Approximately half of retained rows lack a homology-map assignment and use the already-declared gene fallback. These are consequently mixed homology/gene-group evaluations; they do not establish that every test gene lacks a related training gene. The largest single group is below 1.7% of its partition in each dataset.

## What context and storage would the encoders use?

| Measurement | RNA half-life | Translation efficiency |
| --- | ---: | ---: |
| Median supplied transcript length, nt | 3,100.5 | 3,396 |
| Maximum supplied transcript length, nt | 12,288 | 17,497 |
| Median pooled positions | 1,034 | 1,133 |
| Median protein length, residues | 475 | 467 |
| Rows needing multiple 1,022-nt RNA windows | 12,234 / 12,888 | 9,749 / 10,255 |
| Rows with an RNA window boundary inside CDS | 10,309 / 12,888 | 7,862 / 10,255 |
| Rows needing multiple 1,022-residue protein windows | 1,712 / 12,888 | 1,301 / 10,255 |
| RNA-FM / RiNALMo / ESM-2 token payload, decimal GB | 61.613 / 61.613 / 5.000 | 51.232 / 51.232 / 3.834 |
| Combined payload, decimal GB | 128.226 | 106.297 |

The combined estimate is **234.523 GB** for float16 token features across both tasks. It excludes file-format overhead, compression effects, model weights and working memory. Native and pooled arms reuse these same token caches; pooling changes model-grid positions rather than requiring a second extraction. Native versus pooled position totals are 48,135,045 versus 16,053,512 for half-life and 40,024,669 versus 13,348,405 for translation efficiency.

About 95% of rows use multiple independent RNA windows under the chosen 1,022-position policy. Keeping every supplied base therefore does not give an encoder unrestricted transcript-wide context. A bounded Newton resource trial is still needed before scheduling the full panel; the payload calculation cannot establish runtime or peak GPU memory.

Half-life retains 15 rows with no 3′ UTR; translation efficiency retains two with no 5′ UTR and three with no 3′ UTR. These rows cannot supply a diagnostic for the absent region. Half-life also has 168 retained and 34 excluded inputs exactly 12,288 nt long. This inspection establishes the supplied length distribution, not the upstream mechanism that produced that maximum or the content of any omitted sequence.

## How much does the mask control change?

All denominators below are retained transcripts. Overlap is the intersection-over-union of true and permuted CDS masks. Retained update share is first calculated within each transcript, then averaged across transcripts.

| Dataset and grid | Mean mask overlap | Mean retained protein-update share | Unchanged effective support | No retained protein-update support |
| --- | ---: | ---: | ---: | ---: |
| Half-life, native | 0.5034 | 68.94% | 2,656 / 12,888 | 433 / 12,888 |
| Half-life, pooled | 0.5032 | 68.94% | 2,666 / 12,888 | 435 / 12,888 |
| Translation efficiency, native | 0.4861 | 67.88% | 2,075 / 10,255 | 391 / 10,255 |
| Translation efficiency, pooled | 0.4859 | 67.88% | 2,087 / 10,255 | 392 / 10,255 |

The fixed permutation alters effective protein-update support for roughly four fifths of transcripts. For roughly one fifth, the permuted interval still covers every protein-supported position. “Unchanged support” does not mean every model receives identical inputs: the explicit region-mask feature used by C/D can still change outside that support. The control neither creates protein residues in UTRs nor moves their representations.

These measurements support retaining the agreed permutation with an explicit coverage limitation. They do not justify silently reseeding it, forcing larger perturbations or choosing a stronger control after seeing the distribution. Whether this coverage is sufficient for the intended biological claim remains a review decision. No such decision or training run was made here.
