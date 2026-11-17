# ARCC execution and returned evidence

Use this guide for the shared operating procedure: keep execution copies separate, establish the environment inside an allocation, transfer an explicit file selection, and preserve evidence when results return. The [encoder-swap recipe](encoder_swaps/README.md) owns the supplied-split repeat workflow; the [grouped-stability recipe](grouped_stability/README.md) prepares the separate 36-fit evaluation. Scientific settings remain in the maintained notebooks and shared code. Preparing either recipe does not submit jobs or make its results interchangeable with the other.

The completed historical Newton batch and the maintained version-2 recipe are different execution sources. The [November 16 verification record](../../docs/run-records.md#2026-11-16-arcc-recipe-layout-and-returned-input-deduplication) owns the completed batch's counts, GPU observations, source/input verification and returned-input mapping. Synthetic checks of the maintained recipe do not establish that it ran on Newton.

## Keep workspaces separate

The [controlled-fusion recipe](fusion_studies/README.md) prepares A.2–A.6 in a new source-only workspace. It references existing Newton CSVs, BLF caches and completed controls in place. Full-transcript A.6 inputs and token caches have separate routes. The recipe does not automatically submit the full prepared panel or acquire missing weights.

| Location | Purpose |
| --- | --- |
| Maintained repository | Scientific source, reusable execution recipes, canonical `datasets/` and selected shareable evidence. |
| Separate local execution workspace | A prepared source/input snapshot, its runtime files and local transfer or recovery material. The original sibling `ARCC Execution/` remains the full historical workspace. |
| Uploaded Newton workspace | The explicitly transferred copy used by the cluster. Changes in the repository or local workspace do not update it. |
| Returned result root | Immutable results, reports, logs and executed notebooks, with producer sources under `provenance/`. Verified duplicate inputs/selection records can resolve to canonical files. This reading package is not a full execution workspace. |

Create a distinct preparation when source changes are authorized; do not replace an active or completed execution's scripts or initial snapshot. Capture-time flags in an old manifest describe its preparation, not later job completion. Preserve the actual source versions identified by launch records when they differ from the initial snapshot.

## Establish the environment and allocation

Use the selected recipe's Slurm examples and the intended installed environment. The dependency list is a starting specification, not a pinned environment lock. Compare the recorded `pip freeze` and kernel environment before deciding whether a dependency change is needed; packaging alone does not require an upgrade or reinstall.

Slurm can inherit an active Conda environment without inheriting Conda's shell function. The wrappers initialize that function before loading a module that may try to deactivate the inherited environment, then initialize and activate the selected environment:

```shell
if (( ${CONDA_SHLVL:-0} > 0 )); then
    ARCC_BOOTSTRAP_CONDA_BASE=$(conda info --base)
    source "$ARCC_BOOTSTRAP_CONDA_BASE/etc/profile.d/conda.sh"
fi
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARCC_CONDA_ENV"
```

Configure the absolute project path, installed module/environment, registered notebook kernel and expected GPU through the recipe's `ARCC_*` settings. Select account, partition, CPU/RAM/time and GPU requests from the available allocation and measured workload. The examples do not estimate training duration or install packages.

Run GPU preflight in an authorized Slurm allocation. It must verify the selected package, observe CUDA availability and the actual GPU name, and record package versions. A requested GPU, an `expected_gpu` field or a startup banner is not an observed device. The encoder-swap preflight remains with its recipe because it invokes that recipe's launcher. A future allocation needs its own check even when a previous run used an H100.

Slurm opens stdout/stderr files before the script starts. Create the selected log directory before submission and align `--output` / `--error` with the runtime destination. Preparation supplies empty default log/notebook directories; changing the output root does not change Slurm's log paths automatically.

## Transfer an explicit selection

Prepare and review the file list before transferring. Do not upload or return the whole repository `Code/` tree: it can contain unrelated results and large embedding caches. Transfer caches separately only when the selected run requires them and the transfer is authorized.

| Direction | Include deliberately | Handle separately |
| --- | --- | --- |
| Upload | The prepared snapshot's listed scientific source, runtime files, dependency specification, benchmark CSVs, fixed-selection evidence and preparation manifests. Preserve their relative layout. | Required frozen cache shards and provenance metadata, including any required `recovery-records.json`; verify their actual destination. Model downloads, unrelated experiments and old results are not implied by package preparation. |
| Return | Fresh fit/prediction/report JSONs, launch/environment and scheduler logs, executed notebooks, preparation manifests, actual execution source copies and historical selection records. | Retain or transfer final weights and unfinished recovery checkpoints according to the intended reuse/recovery need. Unchanged verified embedding caches need not return. |

Use binary transfer mode so text-like files, including CSV, JSON, Python and notebooks, retain their exact bytes. Compare hashes after transfer; a transfer listing or matching filename is insufficient. Preserve manifests and producer records as written. Resolve historical paths to retained evidence during verification rather than rewriting recorded identities to fit a new location.

## Retain shared evidence and the local execution archive

The repository's [artifact policy](../../README.md#results-caches-and-intermediate-files) distinguishes shareable results and provenance from local cache, weight and recovery files. A compact returned result set keeps the evidence needed to read and assess that cohort; the full local execution archive retains operational material needed to reconstruct or resume its execution. Keeping a compact shared copy does not authorize deleting the archive.

The [returned encoder-swap guide](../../Code/stage4/results/newton_h100_seeds012/README.md) maps canonical benchmark CSVs and historical selection JSONs through the original snapshot's recorded paths and byte hashes. The snapshot now resides under `provenance/`; actual producer source is retained under `provenance/execution-source/`. Fresh results, reports, logs and executed notebooks stay separate. This deduplication does not change the full original workspace required by the recipe's `--selection-workspace`.

The [Stage 4 result-set contract](../../Code/stage4/RUNBOOK.md#select-a-returned-result-set) owns reading and analysis destinations. Keep returned Newton results separate from desktop/Mac cohorts, and keep new local analyses separate from producer reports. Import and hash verification do not themselves perform statistical analysis or establish scientific acceptance. Record execution, interruption and reviewed reuse through the [provenance workflow](../../docs/run-records.md#apply-the-provenance-workflow).
