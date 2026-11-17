# Prepare an encoder-swap execution workspace

This recipe maintains preparation and launch code for the existing encoder-swap experiment: four benchmarks, training seeds 0/1/2 and fixed historical seed-0 validation choices. It does not implement Mina's proposed correction, gate or attention experiments. In the repository, read the [common ARCC guide](../README.md) for environment initialization, GPU preflight, transfer selection and evidence retention. Repository-relative links in this README are navigation for the checkout; a prepared flat workspace does not contain those linked repository guides.

The repository notebook owns scientific settings and calculations. Preparation copies it byte-for-byte, including historical saved outputs. The launcher creates a disposable execution copy, clears that copy's outputs, and inserts one cell containing paths, mode, benchmark/seed, device and producer checks. It leaves the source notebook unchanged. Completed run copies and logs go under the selected output root.

The completed historical Newton batch used a different, version-1 preparation. Its original sibling `ARCC Execution/` workspace remains preserved evidence. The maintained version-2 recipe has synthetic checks; those checks are not a new cluster execution. See the repository's [November 16 verification record](../../../docs/run-records.md#2026-11-16-arcc-recipe-layout-and-returned-input-deduplication) for the historical batch and returned-input mapping. A new snapshot is not a reason to repeat completed fits.

## Inspect before preparing

From the maintained repository, this command prints an inventory and hashes without creating a workspace:

```powershell
python -B execution/arcc/encoder_swaps/prepare.py --source-root . --selection-workspace '../ARCC Execution'
```

`--selection-workspace` still requires the full separate original workspace, including its four CSVs, historical selection records and original snapshot. The script verifies those inputs against that snapshot. The [compact returned root](../../../Code/stage4/results/newton_h100_seeds012/README.md) is for reading evidence, not preparation. Its duplicate inputs and historical selection records resolve through original snapshot membership and byte hashes to canonical repository files. The archived `provenance/` layout does not change this preparation contract.

The inventory includes the maintained notebook and shared Python source, four source CSVs, preserved historical selection JSONs, the original selection manifest, the saved Stage 1 alignment capture, recipe files and focused runtime tests. The alignment JSON is historical context used by the notebook after fitting; packaging neither recomputes it nor treats it as a Newton result. The manifest identifies the maintained source commit and uncommitted changes separately from the historical selection source. A hash identifies bytes, not execution or scientific acceptance.

To materialize a reviewed inventory, add `--destination '../ARCC Execution next'`. The destination must be a new sibling directory. An existing directory, overlap with preserved sources, or escaping symlink/junction is refused. Copying uses exclusive writes and publishes `source-snapshot.json` last. Failed preparation leaves an incomplete directory for inspection; it does not delete it or mark it complete.

The seven runtime files belong to this recipe: `prepare.py`, `run_encoder_swaps.py`, the three `newton_*.sbatch.example` wrappers, `requirements-notebooks.txt` and `batches.tsv`. Preparation places them and this README at the generated workspace root, preserving the existing flat runtime layout. The preflight is recipe-specific because it verifies the swap launcher; it is not a generic standalone ARCC utility.

No embedding arrays, model weights or recovery checkpoints are copied. Preparation's `--cache-root` declares an external cache location and checks separation; it does not inventory or verify its arrays. The execution command's `--cache-root` selects the actual cache location. Preparation performs no network transfer, environment installation or job submission.

## Verify the prepared workspace

Inside the new workspace, with the intended Python environment active, use the flat runtime paths:

```shell
python -B run_encoder_swaps.py --verify-only
python -B -m unittest discover -s tests -p 'test_arcc_*.py'
```

The first command verifies every prepared file, including code and historical inputs. It does not start a notebook kernel or open cache arrays. Tests use invented inputs and mocked kernel/GPU interfaces; they do not fit research models or establish GPU readiness. This version-2 launcher deliberately refuses the original version-1 workspace.

## Choose the operation and destination

| Operation | Execution choice | What it may do |
| --- | --- | --- |
| Inspect package | `--verify-only` | Check source/input bytes; no kernel or research artifact writes. |
| Newton fitting | `--mode train --purpose newton` | One benchmark/seed; requires a Slurm allocation, CUDA and an observed GPU matching `--expected-gpu`. Reuses compatible completions and recovery state. |
| Saved-prediction analysis | `--mode report --purpose newton` | CPU analysis of completed seeds 0/1/2; no extraction or replacement fits. This produces a report and requires research-analysis authorization. |
| Explicit local pilot | `--mode train --purpose local_pilot --device cpu` or `cuda` | Same declared scientific settings, separate `pilot_outputs` destination and pilot producer label. It is a real fit, not a synthetic check. |

Newton defaults to `outputs/`; local pilots default to `pilot_outputs/`. An explicit `--output-root` must remain within the prepared workspace and separate from code, historical inputs, caches and the other purpose's default destination. Records carry their purpose and actual output root, so changing a path does not promote a pilot into a Newton result. Imported completed results are not training destinations.

Fitting requires complete compatible token caches. Missing shards, incompatible evidence or metadata requiring repair stop before fitting; they never trigger extraction or downloads. Transfer required caches and provenance only through a separately authorized operation, retaining `recovery-records.json` at the cache root when legacy verification needs it. Fully completed fits do not require reopening their cache arrays.

## Preflight and launch in the selected allocation

Configure `ARCC_PROJECT_ROOT`, `ARCC_ANACONDA_MODULE`, `ARCC_CONDA_ENV`, `ARCC_KERNEL`, `ARCC_EXPECTED_GPU` and, if needed, `ARCC_CACHE_ROOT` / `ARCC_OUTPUT_ROOT`. Follow the common guide's environment, allocation and log-directory procedure before submission. The wrappers retain inherited-Conda initialization and verify the package before proceeding.

- `newton_preflight.sbatch.example` verifies this package and observes the allocated GPU and installed environment without running a research notebook.
- `newton_swaps.sbatch.example` runs one explicit `ARCC_BENCHMARK` / `ARCC_SEED`, or the selected `batches.tsv` row for an array. The table describes the intended four-benchmark, three-seed grid; it is not a submission instruction or a completion record.
- `newton_report.sbatch.example` explicitly analyzes saved predictions on CPU after the intended results are available.

For example, the Python fit command inside an authorized allocation is:

```shell
python -u run_encoder_swaps.py --mode train --purpose newton \
  --benchmark cov_vac_degradation --seed 0 --kernel biofusion-arcc \
  --expected-gpu 'NVIDIA H100 PCIe' --cache-root /absolute/path/to/embeddings
```

Confirm the exact benchmark/seed matrix, source snapshot, allocation, resource limits and output destination before a new compute batch. Do not submit a job merely to validate package construction. A prior H100 observation does not establish the next allocation's GPU.

## Inspect progress and preserve failures

Epoch text is forwarded immediately to stdout/stderr while rich displays remain in the executed notebook. Per-launch records retain prepared source hashes, fixed-selection evidence and the environment observed inside the kernel. Results and checkpoints retain the launch link; cohort checks precede the shared trainer's identity and recovery checks.

Cooperative locks reject concurrent writers for the same benchmark/seed. Interrupted launches retain their notebook/record when Python can finalize them, and completed fits remain reusable. Forced termination can prevent final evidence saving; inspect scheduler status, logs and actual artifacts before deciding whether to resume. An incompatible completed result is a blocker, never permission to replace it.

Use the common guide's return whitelist and retain source/input mappings alongside the returned evidence. In the repository, the [result-set contract](../../../Code/stage4/RUNBOOK.md#select-a-returned-result-set) owns import and reading. Its named Newton reader expects the default `outputs/` layout; a custom output root needs an explicit matching import/reader decision. Returning evidence and calculating a saved-prediction report remain separate operations.
