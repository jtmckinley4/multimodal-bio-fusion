# Prepare an encoder-swap execution workspace

This recipe maintains preparation and launch code for the existing encoder-swap experiment: four benchmarks, training seeds 0/1/2 and fixed historical seed-0 validation choices. The focused `ecoli_ernie` extension uses the same notebook for two additional RNA encoders and only the existing MIL gated and concatenation heads. It does not implement the separate correction, gate-variant or attention studies. In the repository, read the [common ARCC guide](../README.md) for environment initialization, GPU preflight, transfer selection and evidence retention. Repository-relative links in this README are navigation for the checkout; a prepared flat workspace does not contain those linked repository guides.

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

The preparation script, launcher, Slurm wrappers, requirements and batch inventories belong to this recipe. Preparation places them and this README at the generated workspace root, preserving the existing flat runtime layout. The preflight is recipe-specific because it verifies the swap launcher; it is not a generic standalone ARCC utility.

No embedding arrays, model weights or recovery checkpoints are copied. Preparation's `--cache-root` declares an external cache location and checks separation; it does not inventory or verify its arrays. The execution command's `--cache-root` selects the actual cache location. Preparation performs no network transfer, environment installation or job submission.

## Focused E. coli RNA extension

Prepare a fresh snapshot with `--experiment ecoli_ernie`. This retains the original E. coli rows, NT-v2 100M and ESM-2 8M caches, codon grid and training recipe. It schedules four seed-0 fits: the two new encoders' MIL heads first, followed by their concatenation heads. The inherited MIL penalty is selected from the preserved original seed-0 tuning records; the expected value for these inputs is 0.01. No original-reference, single-encoder or cross-attention fits are scheduled. New single controls would be needed to estimate gains over a validation-selected single; this panel does not claim that endpoint.

```powershell
python -B execution/arcc/encoder_swaps/prepare.py --source-root . `
  --selection-workspace '../ARCC Execution' --experiment ecoli_ernie `
  --destination '../ARCC Ecoli RNA Extension'
```

This profile copies only the E. coli CSV and original E. coli Phase A selection JSONs, alongside source, tests and the original selection manifest. It excludes the other three CSVs and historical replacement results. Keep the old Newton workspace and running study workspaces intact. Upload the new package into its own Newton directory, for example `$HOME/multi-modal_fusion_ecoli_ernie`; merge neither its source nor its outputs into the completed swap workspace.

The approved RNAErnie checkpoint is the authors' newer [LLM-EDA/RNAErnie](https://huggingface.co/LLM-EDA/RNAErnie/tree/df26681ed44bb671b61536c5e2a196d6058a44f7), not the shorter `multimolecule/rnaernie` conversion. Its 2,048-token context accommodates the retained 999-nucleotide inputs with framing tokens. ERNIE-RNA uses the pinned base [multimolecule/ernierna](https://huggingface.co/multimolecule/ernierna/tree/9f6099b85d61f7567d4e1ed1e92e1de85fdfd292). The shared loader implements the author RNAErnie tokenizer and reads hidden representations. Actual pinned-weight loading and GPU extraction remain runtime checks; synthetic tests do not establish them.

Before the separately authorized extraction, stage only the required files in the existing Hugging Face cache. These commands download weights; they are not verification-only commands. Do not upgrade the environment used by running jobs.

```shell
hf download LLM-EDA/RNAErnie --revision df26681ed44bb671b61536c5e2a196d6058a44f7 \
  model.safetensors config.json tokenizer_config.json special_tokens_map.json vocab.txt
hf download multimolecule/ernierna --revision 9f6099b85d61f7567d4e1ed1e92e1de85fdfd292 \
  model.safetensors config.json tokenizer_config.json vocab.txt
```

After package verification and environment preflight, the execution sequence is explicit: extract the two new RNA caches once, then train the four requested heads. Both commands below are for an authorized GPU allocation; neither submits a job.

```shell
python -u run_encoder_swaps.py --experiment ecoli_ernie --mode extract \
  --kernel biofusion-arcc --expected-gpu 'NVIDIA H100 PCIe' \
  --reference-root "$HOME/multi-modal_fusion" \
  --cache-root "$HOME/multi-modal_fusion/Code/stage3/embeddings"

python -u run_encoder_swaps.py --experiment ecoli_ernie --mode train --seed 0 \
  --kernel biofusion-arcc --expected-gpu 'NVIDIA H100 PCIe' \
  --reference-root "$HOME/multi-modal_fusion" \
  --cache-root "$HOME/multi-modal_fusion/Code/stage3/embeddings"
```

The reference workspace and existing cache root are read-only inputs. New RNA caches default to `extension_embeddings/`; new fits, launch records and executed notebook copies go under `outputs/ecoli_ernie/`. Those directories must remain separate from the completed cohort. Extraction is offline, requires the staged weights and cannot train. Training requires complete compatible caches and cannot extract. An incomplete original DNA/protein cache blocks both operations instead of being regenerated. Completed extension fits are reused only after their identities and producer evidence match.

For Slurm, use `newton_ernie.sbatch.example` with `ARCC_OPERATION=extract` or `train`, `ARCC_REFERENCE_ROOT` pointing to the completed workspace and `ARCC_CACHE_ROOT` pointing to its embeddings. It defaults to seed 0 for training; `ARCC_SEED=1` or `2` selects a separately approved repeat. `ecoli_ernie_batches.tsv` records four fits per seed, not authorization to schedule all three. Follow the existing resource/queue procedure and use an `afterok` dependency when scheduling fitting after extraction. The old `batches.tsv` and `newton_swaps.sbatch.example` still describe the original full panel.

Return `outputs/ecoli_ernie/`, the new source snapshot and its execution sources into `Code/stage4/results/newton_ecoli_ernie/`, retaining the fits under `outputs/ecoli_ernie/phaseB/`. The new RNA caches can stay on Newton. The [encoder-swaps notebook](../../../Code/stage4/encoder_swaps.ipynb#Replacing-RiNALMo-with-RNAErnie-and-ERNIE-RNA) shows both heads and all three seeds beside the RiNALMo reference. Its default saved-result mode reads both Newton returns at their standard locations. To read custom locations, select `EXPERIMENT = "ecoli_ernie"` and set `RETURNED_WORKSPACE` for the extension and `REFERENCE_WORKSPACE` for the original Newton cohort.

The reader checks the fit recipe and encoder combination and displays a mean only for complete rows. The [return record](../../../docs/run-records.md#2026-11-19-verified-newton-return-for-e-coli-gates-and-attention) documents the producer and matched-row checks. This panel supports descriptive head comparisons. Its launcher disables `--mode report`, which requires the larger panel's baselines.

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
