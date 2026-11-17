# Prepare Mina's controlled fusion studies

This source-only recipe connects the four Stage 4 notebooks for auxiliary correction (A.2), gates (A.3–A.4), anchor attention (A.5) and transcript regions (A.6). The [batch plan](../../../docs/research-execution-plan.md) records agreed scientific choices and human review gates. Preparation and synthetic tests do not establish research results or Newton readiness.

## Prepare and transfer deliberately

Run `prepare.py --source-root <repository>` to inspect the source inventory. Adding `--destination <new sibling directory>` copies only the four notebooks, Python helpers and execution recipes and publishes their hashes. An existing destination is refused. No dataset, cache, result, weight or checkpoint is copied. Keep the completed encoder-swap and grouped-stability execution snapshots unchanged.

Upload that new directory's contents to a separate Newton workspace when transfer is authorized. Keep its `Code/` layout. Point to existing Newton inputs instead of copying them again:

| Explicit route | Required contents |
| --- | --- |
| `--data-root` | Existing BLF CSVs in `datasets/`; original row order and targets remain unchanged. |
| `--cache-root` | Existing `Code/stage3/embeddings/`; original NT-v2/RNA-FM/ESM-2 caches plus mRNA-FM for each selected benchmark. |
| `--reference-root` | Completed Newton workspace with `outputs/phaseA/` and `outputs/phaseB/`. Compatible controls stay in place. Retain their best weights for correction-anchor validation inference and gate diagnostics. Missing weights hold those operations without retraining. |
| `--grouped-split` | The new grouped-stability `split.json` and its completed single-input results/weights in the same study root, produced only by an authorized run of that separate protocol. Supplied-split stability evidence cannot substitute. |
| `--region-data-root` | Local mRNABench half-life/TE parquet files and the required `homology_maps/human_homology_map.csv`. Mapped families retain their groups; unmapped rows use recorded gene identities. No automatic download occurs. |
| `--region-cache-root` | A new A.6 token-cache directory, separate from BLF caches and results. Existing pooled half-life/TE arrays cannot supply token positions. |

Only routes used by the selected study are required. Returned fit records retain the input/source identities; paths alone do not establish compatibility. Missing baseline weights can block diagnostics without invalidating saved score comparisons. They never trigger replacement training automatically.

The current correction recipe uses the frequency-preserving `row_shuffle_v1` control; historical `shuffle` fits cannot fill that arm. A.4 uses `position_residual_control`, a full-width additive path plus a small affine update, instead of the earlier primary rank-150 bottleneck. These replace controls without expanding the fit panel. Their result/report identities keep historical evidence separate. Prepare a fresh source snapshot for this revision rather than changing a previously launched workspace in place. A.6's scientific policies and model panel are unchanged; its repeated cohort validation and completed-result reuse have been streamlined. The [revision record](../../../docs/run-records.md#2026-11-17-frequency-preserving-shuffle-and-full-width-control) describes the implementation checks, not authorization to transfer or run it.

### Verify trained reference heads separately from encoder weights

Correction and gate diagnostics need some **trained prediction-head weights** from the completed Newton batch. A.6 extraction instead needs **pretrained encoder weights**. These are different artifacts; neither can replace the other.

The reference set to check on Newton is 30 single-predictor files for CoV-Vac/fungal expression and 12 MIL files for CoV-Vac/E. coli, across seeds 0/1/2 and the original/mRNA-FM combinations. Use each selected result JSON's `model_artifact.file`, resolved beside that JSON, and compare both its SHA-256 and byte length with `model_artifact.sha256` and `size_bytes`. Do not rename weights, use an optimizer checkpoint as a substitute, or infer integrity from a matching filename. Missing or mismatched files require investigation, not automatic replacement training.

For each correction benchmark, the five candidate singles are Phase A's `unimodal:dna`, `unimodal:rna`, `unimodal-grid:rna`, `unimodal:prot`, and Phase B/mRNA-FM's `unimodal:rna`. Seed-0 selection needs ten files across the two benchmarks; later repeats need only the selected winners. MIL diagnostics use `mil` with entropy weight 0.01 from both combinations. Provisioning all 42 files is a small superset, approximately 240 MB, while the ten initial candidates plus twelve MIL files total approximately 166 MB. A.5's new controlled fits use the existing baseline JSONs but do not require their old weights.

The [input verification record](../../../docs/run-records.md#2026-11-17-controlled-fusion-input-preparation) distinguishes verified local result descriptors from absent local weight bytes. It does not establish whether those files are still present on Newton. They may remain there for execution; downloading them is not a prerequisite to an on-cluster hash check. Grouped-stability reference weights are separate outputs of the new 36-fit experiment.

### Stage the A.6 inputs at their recorded versions

The required human map is the exact CSV inside mRNABench's [released homology archive](https://zenodo.org/records/13910050/files/homology_maps_homologene.zip), selected by its [pinned splitter](https://github.com/morrislab/mRNABench/blob/778d691ab91c51c03addeb9219229f2ec81d195c/mrna_bench/data_splitter/homology_split.py). Verify the 708,362-byte archive against MD5 `c6fe06cc0a4ae71358c6cdfefbcce3c6`, then extract only `homology_maps_homologene/human_homology_map.csv` to `homology_maps/human_homology_map.csv` beneath the selected data root. The unmodified CSV is 485,941 bytes with SHA-256 `b3318df10d702fe7898a6ed1296c6b0ca81c36a8392f77c24b8a04ae16b9772a`. Local downloaded data remain Git-ignored.

**Coverage is not complete.** The downloaded map passes the A.6 map reader, but leaves 6,199 of 12,939 raw half-life rows and 5,089 of 10,281 raw TE rows without a name match. These are input-coverage counts before cohort filtering, not a new cohort or split. The [approved A.6 policy](../../../docs/run-records.md#2026-11-17-a6-grouping-for-unmapped-genes) retains mapped homology groups and groups unmapped rows by recorded gene identity. The source splitter assigns unmapped rows individually, so this is an explicit project adaptation. The map remains required; aliases are not replaced, and missing names are not grounds for exclusion. An absent mapping does not establish absent homologs: the split cannot promise complete homology separation. The older pooled notebook can also discover this shared map during a future computation, so historical gene-only results must remain identified as such.

Stage only these three Hugging Face model directories into the hub cache used by the Newton job, preserving their `snapshots/` and `refs/` layout:

| A.6 alias | Model directory | Verified snapshot | Weight bytes |
| --- | --- | --- | ---: |
| `rnafm` | `models--multimolecule--rnafm` | [`7d6e73ad3b48e042b378f9a788a56ccb4d573a27`](https://huggingface.co/multimolecule/rnafm/tree/7d6e73ad3b48e042b378f9a788a56ccb4d573a27) | 398,117,588 |
| `rinalmo` | `models--multimolecule--rinalmo-mega` | [`5e3a682eb856148fe81f12a25301962c9ffaf0ed`](https://huggingface.co/multimolecule/rinalmo-mega/tree/5e3a682eb856148fe81f12a25301962c9ffaf0ed) | 592,312,640 |
| `esm2_8m` | `models--facebook--esm2_t6_8M_UR50D` | [`c731040fcd8d73dceaa04b0a8e6329b345b0f5df`](https://huggingface.co/facebook/esm2_t6_8M_UR50D/tree/c731040fcd8d73dceaa04b0a8e6329b345b0f5df) | 31,384,292 |

Each RNA snapshot needs `config.json`, `tokenizer_config.json`, `vocab.txt` and `model.safetensors`; ESM additionally needs `special_tokens_map.json`. RNA-FM and ESM currently resolve through cached `refs/main`, whereas RiNALMo's revision is explicit in the registry. Preserve the observed references and verify the snapshots above rather than substituting a newer `main`. These identities are recorded for staging; the registry and extraction code have not been changed. The three local model directories total approximately 1.02 GB and need no new weight downloads. No mRNA-FM, RiNALMo giga, Torch Hub, TensorFlow or duplicate `.bin` weights are needed for this A.6 panel.

File integrity and recorded package availability do not establish a successful offline load on Newton. A separately authorized runtime check remains necessary before extraction. The [verification record](../../../docs/run-records.md#2026-11-17-controlled-fusion-input-preparation) records checkpoint hashes, local availability and this limit.

## Review the workload before submission

The launcher selects exactly one study, benchmark, seed and action. It does not submit a grid. Scientific panels remain visible in each notebook; the wrapper changes execution paths, seed, device, flags and observed producer evidence only. Training seeds are 0/1/2; split seed 42 has a different role.

| Study | Prepared panel and dependency |
| --- | --- |
| `correction` | CoV-Vac, fungal expression and grouped stability; original/mRNA-FM combinations; linear and TextCNN corrections, separate and joint auxiliaries, real/shuffled/random controls. Seed 0 establishes fixed anchor selection before seeds 1/2. Five OOF anchor fits per selected configuration are additional to final anchors and correction fits. Ridge penalty trials also cost fits; one saved ridge record is not one estimator fit. |
| `gates` | CoV-Vac and E. coli; original/mRNA-FM combinations. Reuse compatible Newton singles/concatenation/MIL controls. New sequence and positional variants are separately identified. No automatic extension to all encoder swaps. |
| `attention` | CoV-Vac and E. coli; original/mRNA-FM combinations. Fixed seed-0 anchor selection precedes repeats. Projected anchor-only, ordinary, skip, sequence-gated and position-gated attention test distinct changes. |
| `regions` | Human half-life first, then TE. Two RNA encoders × two resolutions × eight core arms × three training seeds × two datasets = **192 prepared head fits**. Optional capacity/attention controls are listed separately in the notebook and disabled by default. This count excludes token extraction and diagnostic inference. |

Complete preparation does not mean run every prepared arm immediately. Review scientific settings, missing evidence, memory/storage and the authorized resource budget. Begin with a representative, separately recorded resource trial before deciding concurrency or expanding to TE. Full-transcript attention may be much heavier than the ≤999-base BLF comparisons; no runtime estimate has been validated.

With all compatible reference fits available, the prepared new workload is:

| Work | New fits across the declared panel |
| --- | --- |
| Separate grouped-stability prerequisite | 36 neural fits. |
| A.2 correction | 209–258 neural fits, depending on whether the two bundles select the same anchor and whether NT-grid wins; plus 162 ridge result records, each selecting from five fitted penalties. |
| A.3–A.4 gates | 72 neural fits; 12 existing MIL fits are reused. |
| A.5 anchor attention | 60 neural fits. |
| A.6 core | 192 neural fits; optional arms remain disabled. |

That is **569–618 new neural fits plus 810 ridge penalty fits** if the entire prepared core and grouped prerequisite are eventually approved. Counts exclude embedding extraction, diagnostic inference and statistical reporting. Baseline inference/reuse is not a new training fit. The OOF anchors genuinely need new training; previously fitted full-training anchors cannot supply honest OOF predictions. These are workload counts, not GPU-hour estimates or an instruction to launch everything.

## Execute only the selected operation

`python run_fusion_study.py --verify-only` checks source hashes without starting a kernel or opening research data. Use the existing Newton environment and compare its versions with `requirements.txt`; do not upgrade it automatically. A.6 additionally needs a parquet reader and separately staged model weights. The launcher sets Hugging Face/Transformers offline mode, including extraction: acquiring missing weights is a separate preparation action.

The Slurm example requires the established `ARCC_PROJECT_ROOT`, `ARCC_ANACONDA_MODULE`, `ARCC_CONDA_ENV` and `ARCC_KERNEL`. GPU work also requires `ARCC_EXPECTED_GPU`. Supply account, partition, GPU, CPU, RAM, time and concurrency limits after resource review. Create `outputs/logs/` before submission, because Slurm opens logs before the script starts.

Pass script arguments after `newton_fusion_study.sbatch.example`, for example:

```shell
--study correction --benchmark fungal_expression --seed 0 --action train \
  --data-root "$HOME/multi-modal_fusion/datasets" \
  --cache-root "$HOME/multi-modal_fusion/Code/stage3/embeddings" \
  --reference-root "$HOME/multi-modal_fusion"
```

This is an argument example, not a submission command. Gates, attention and supplied-split correction require `--reference-root`. Grouped stability instead requires `--grouped-split` and new compatible anchors. A.6 uses the two region routes and has separate `prepare`, `extract`, `train` and `diagnostics` actions. Preparation is CPU metadata work; extraction is performed once per dataset before its head fits. No training action automatically prepares a cohort, extracts embeddings or downloads weights.

For an agreed A.6 subset, repeat `--fit-key` with exact keys from the notebook's core inventory. This selects fits without changing their recipe; incomplete subsets cannot produce the full-panel summary. Diagnostic jobs also handle one selected benchmark and seed, rather than silently repeating all seeds.

The kernel observes its actual GPU before training; a failed CUDA/GPU check refuses CPU fallback. Epoch streams are relayed to Slurm logs. Duplicate writers for the same study/benchmark/seed are refused. An executed notebook and launch record are retained on success or failure. Completed incompatible results are preserved, not overwritten.

## Return compact evidence

Return new fit/prediction JSONs, selected diagnostics/reports, launch/environment logs, executed notebooks, `source-snapshot.json` and its listed source files. Keep this cohort separate from the completed swaps. Retain final weights and recovery checkpoints where subsequent inference/resumption needs them; do not commit large weights. A.6 cohort, split, exclusion, permutation and chunk/cache manifests are part of its evidence. Unchanged BLF caches and newly produced token arrays need not return merely to read the results.

New statistical comparisons remain explicit local analysis operations. Three-seed spread and paired test-group uncertainty answer different questions. Neither passing synthetic checks nor producing a fit establishes biological interpretation or authorizes the next experiment.
