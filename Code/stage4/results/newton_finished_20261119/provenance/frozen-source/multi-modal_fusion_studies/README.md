# Prepare Mina's controlled fusion studies

This recipe prepares auxiliary correction (A.2), gates (A.3–A.4), anchor attention (A.5) and transcript regions (A.6) for a separate Newton workspace. The notebooks own scientific settings; the [batch plan](../../../docs/research-execution-plan.md) connects them to Mina's proposal and Gantt/Details tasks **45, 46, 25 and 42**. The [run checklist](RUN_CHECKLIST.md) gives upload, verification, dependency, execution and return steps. Preparation is not evidence that a research job has run or that its resource budget is sufficient.

## Keep source, reused inputs and new results separate

`prepare.py --source-root <repository>` inventories the execution source. Adding `--destination <new sibling directory>` creates a new package with four notebooks, shared Python, recipes, input-verification specifications and source hashes. Existing destinations are refused. Datasets, embeddings, research results and weights are not copied. The current preparation uses the sibling **`ARCC Fusion Studies 2026-11-17 Controls`**, uploaded later to **`$HOME/multi-modal_fusion_studies`**. The separate grouped-stability package goes to **`$HOME/multi-modal_fusion_grouped_stability`**.

Reuse the existing **`$HOME/multi-modal_fusion`** inputs and completed swaps in place. These packages and Newton workspaces do not synchronize automatically with the maintained repository. Relative research-document links in this README navigate the checkout; the prepared copy is an execution snapshot, with this README and launchers at its root.

| Launcher route | Contents and role |
| --- | --- |
| `--data-root` | Existing BLF CSVs in `multi-modal_fusion/datasets/`; unchanged row order and targets. |
| `--cache-root` | Existing `multi-modal_fusion/Code/stage3/embeddings/`; selected NT-v2, RNA-FM, mRNA-FM and ESM-2 caches. No repeat transfer or extraction. |
| `--reference-root` | Completed swap workspace, including `outputs/phaseA/`, `outputs/phaseB/` and required best-head weights. |
| `--grouped-split` | New `multi-modal_fusion_grouped_stability/outputs/study/split.json`; compatible grouped single-input fits and weights must remain alongside it. Historical supplied-split stability cannot substitute. |
| `--region-data-root` | Staged half-life/TE parquet files and `homology_maps/human_homology_map.csv`. |
| `--region-cache-root` | New A.6 token-cache destination, separate from BLF caches and result directories. Pooled half-life/TE arrays cannot supply token positions. |

Only routes used by the selected study are required. Paths identify where to look; recorded hashes, row identities and settings determine compatibility. Missing inputs never authorize replacement training or automatic downloads.

## Verify weights for their actual purpose

**Trained prediction-head weights** support correction-anchor validation inference and gate diagnostics. The reference verifier checks 30 candidate-single files for CoV-Vac/fungi and 12 MIL files for CoV-Vac/E. coli, across seeds 0/1/2. This 42-file superset is approximately 240 MB; the files can remain on Newton. Each descriptor's `model_artifact.file`, SHA-256 and byte length must agree with the actual file. Later correction repeats need only the selected winners; grouped-stability weights are separate outputs of its new 36-fit experiment. A.5 uses the old baseline JSONs but does not require their weights.

**Pretrained encoder weights** support A.6 extraction. Stage only the following three model directories in the Newton Hugging Face hub cache, preserving snapshot files and recorded references:

| Alias | Model directory | Recorded snapshot | Weight bytes |
| --- | --- | --- | ---: |
| `rnafm` | `models--multimolecule--rnafm` | [`7d6e73ad3b48e042b378f9a788a56ccb4d573a27`](https://huggingface.co/multimolecule/rnafm/tree/7d6e73ad3b48e042b378f9a788a56ccb4d573a27) | 398,117,588 |
| `rinalmo` | `models--multimolecule--rinalmo-mega` | [`5e3a682eb856148fe81f12a25301962c9ffaf0ed`](https://huggingface.co/multimolecule/rinalmo-mega/tree/5e3a682eb856148fe81f12a25301962c9ffaf0ed) | 592,312,640 |
| `esm2_8m` | `models--facebook--esm2_t6_8M_UR50D` | [`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`](https://huggingface.co/facebook/esm2_t6_8M_UR50D/tree/c731040fcd8d73dceaa04b0a8e6329b345b0f5df) | 31,384,292 |

The [input manifest](input-manifest.json) names the required configuration, tokenizer, weight and reference files. RNA-FM and ESM resolve through the recorded `refs/main`; RiNALMo's revision is explicit. Do not replace these references with newer versions. The approximately 1.02 GB already exists locally; no mRNA-FM, RiNALMo giga, Torch Hub, TensorFlow or duplicate `.bin` weights are required for this panel. Preserve snapshot/reference files and any linked blobs in these three model directories; do not upload the entire model cache.

The required human map comes from mRNABench's [released archive](https://zenodo.org/records/13910050/files/homology_maps_homologene.zip) and [pinned splitter](https://github.com/morrislab/mRNABench/blob/778d691ab91c51c03addeb9219229f2ec81d195c/mrna_bench/data_splitter/homology_split.py). The 485,941-byte CSV has SHA-256 `b3318df10d702fe7898a6ed1296c6b0ca81c36a8392f77c24b8a04ae16b9772a`. Its map is incomplete: the approved policy groups mapped families and groups unmapped rows by recorded gene identity. It does not establish complete homology separation. The [input preparation record](../../../docs/run-records.md#2026-11-17-controlled-fusion-input-preparation) and [grouping decision](../../../docs/run-records.md#2026-11-17-a6-grouping-for-unmapped-genes) retain source details and limitations.

## Review the workload before scheduling

The launcher runs one study, benchmark, seed and action. It does not submit a grid. Training seeds **0/1/2** repeat model fitting; split seed **42** fixes group allocation. The [grouped-stability recipe](../grouped_stability/README.md) is a prerequisite only for the stability correction arm, not for every study.

| Study | Prepared comparison and new workload |
| --- | --- |
| Grouped stability, separate package | 36 neural fits: 12 per seed, with the new exact-sequence-grouped split. |
| `correction` — [A.2 notebook](../../../Code/stage4/anchor_correction.ipynb) | CoV-Vac, fungi and grouped stability; original/mRNA-FM bundles; separate/joint auxiliaries with real, shuffled and random inputs. **209–258 neural fits**, depending on shared anchor selection and NT-grid selection, plus **162 ridge records selecting among five penalties each**. Five OOF anchor fits are additional to final anchors and correction fits. |
| `gates` — [A.3–A.4 notebook](../../../Code/stage4/gate_diagnostics.ipynb) | CoV-Vac/E. coli, two bundles; **84 new neural fits** and 12 reused MIL fits. Includes the joint-input sequence gate matched to the position gate's scorers. |
| `attention` — [A.5 notebook](../../../Code/stage4/anchor_cross_attention.ipynb) | CoV-Vac/E. coli, two bundles; **72 new neural fits**, including an anchor-skip nonlinear concatenation control with active parameters matched to gated attention. |
| `regions` — [A.6 notebook](../../../Code/stage4/region_aware_fusion.ipynb) | Two RNA encoders × two grids × twelve core arms × three seeds × two datasets, plus six native protein-only fits = **294 new neural fits**. Run the 147 half-life fits before reviewing whether to expand to the 147 TE fits. |

The full prepared core and grouped prerequisite therefore represent **695–744 neural fits plus 810 ridge penalty fits**, excluding extraction, diagnostic inference and statistical reporting. These are workload counts, not GPU-hour estimates or a direction to submit everything. Previously fitted full-training anchors cannot supply honest out-of-fold training predictions.

[batches.tsv](batches.tsv) lists 49 possible study operations: 27 fitting jobs, 18 diagnostic jobs, two CPU preparation jobs and two extraction jobs. One fitting job contains several fits. The table records prerequisites and resource type; it is neither a scheduler nor completion evidence. The separate grouped package owns its three additional fitting jobs.

The correction control remains the frequency-preserving `row_shuffle_v1`; historical `shuffle` fits do not fill it. A.4 retains `position_residual_control`, a full-width half-sum plus a rank-2 affine update, with 5,402 parameters versus 5,403 for its gates. The added A.5 control has 2,889,003 mixing parameters versus 2,888,402 for gated attention at width 600. Similar counts do not establish equal expressivity or runtime. A.6 adds sequence/nongated controls within D and E and selects its strongest single on seed-0 validation from all four RNA-only configurations plus native protein-only; that configuration stays fixed across training seeds. Its cohort, masks, chunking and training budgets remain unchanged. The [control-repair plan](../../../docs/research-execution-plan.md#control-repairs-before-newton) owns the precise comparisons and optional exclusions.

## Execute and interpret separate operations

| Operation | What it does |
| --- | --- |
| `--verify-only` | Source hashes only; no notebook kernel, data, cache or model. |
| `--preflight --kernel … --expected-gpu …` | Allocated-kernel imports and small CUDA check; no research notebook, encoder load or fitting. |
| `--action prepare` | A.6 seed 0 only: CPU cohort, group split and mask-control preparation. |
| `--action extract` | A.6 seed 0 only: read prepared metadata and staged encoders; produce reusable token caches. |
| `--action train` | One benchmark and training seed, using compatible existing inputs. No implicit extraction or statistical report. |
| `--action diagnostics` | Gates, attention or A.6: inference from saved best weights for one benchmark/seed. This is additional GPU work, not fitting. |

Correction's saved-prediction diagnostics and all three-seed statistical reports remain explicit notebook operations after fitting. Default notebook reading does not create missing reports. For A.6, repeated `--fit-key` arguments select exact core keys for an agreed bounded trial; incomplete subsets cannot produce a full-panel report.

Use the existing Newton environment; `requirements.txt` is a compatibility reference, not permission to upgrade packages. Extraction is offline and needs the staged weights. Verify imports and actual GPU access, then review a representative resource trial before choosing CPU/RAM/time and concurrency. Full-transcript attention can be substantially heavier than the shorter BLF tasks; its runtime and peak memory remain unmeasured.

Source and input verification, software tests, resource readiness and scientific acceptance remain distinct. The launcher refuses GPU fallback and incompatible-result replacement, relays epoch progress and retains launch records/executed notebooks on failure. Follow the [checklist](RUN_CHECKLIST.md) to return compact evidence without transferring embedding arrays.
