# Run the grouped-stability comparison on Newton

This recipe prepares the [grouped-stability notebook](../../../Code/stage4/grouped_stability.ipynb) for the agreed 36-fit comparison: one exact-sequence-grouped split, two encoder combinations, twelve unique fits per training seed and seeds 0, 1 and 2. It changes the evaluation partition, not the encoder weights or the existing research results. The notebook owns the scientific settings and the distinction between the advisor's requirements and the team's implementation choices.

The implementation and synthetic checks are complete; the grouped evaluation has not run. Split preparation and fitting are explicit notebook operations. Use the [common ARCC guide](../README.md) for allocation, environment, transfer and evidence-retention practices. These relative documentation links navigate the checkout; the prepared workspace places this README and the runtime entrypoints at its root, with the notebook under `Code/stage4/`.

The [run and transfer checklist](RUN_CHECKLIST.md) gives the current cache location, environment checks, seed-0 resource review, repeat-job dependencies and evidence-return steps. It keeps submission separate from preparation.

## Prepare a separate workspace

From the maintained repository, inspect the source inventory without copying files:

```powershell
python -B execution/arcc/grouped_stability/prepare.py --source-root .
```

The current prepared workspace is the sibling `ARCC Grouped Stability 2026-11-17`. For a future source revision, add `--destination <new sibling directory>` after reviewing the inventory. Existing destinations, symlinks/junctions and overlap with a declared cache root are refused. The original `ARCC Execution/` and its completed swap results stay unchanged. Failed preparation leaves an incomplete directory for inspection; the completion manifest is written only after every copied source is verified.

The package contains this notebook, shared Python source, the stability CSV, execution files and synthetic checks. It excludes old fit results, historical selection inputs, embedding arrays, model weights and recovery checkpoints. `arcc_package_support.py` and `arcc_execution_support.py` are byte-identical, hash-bound copies of the existing encoder-swap preparation/launcher modules; only their generic file verification, path protection, locking, record publication and live-text helpers are reused. Their swap-specific selection and execution methods are not called.

Inside the prepared workspace:

```shell
python -B run_grouped_stability.py --verify-only
python -B -m unittest discover -s tests -p 'test_arcc_grouped_stability.py'
python -B -m unittest discover -s tests -p 'test_grouped_stability.py'
```

Verification checks source/input bytes without starting a notebook kernel or reading cache arrays. Adding `--kernel biofusion-arcc` also checks that the registered kernel uses the activated Python interpreter; both Slurm wrappers require this check. Tests use invented inputs and mocked execution interfaces. The copied requirements list is the existing provisional dependency specification, not an environment lock: use the installed Newton environment and compare its recorded versions instead of upgrading automatically.

## Reuse the caches already on Newton

Upload only the prepared package's contents into a new Newton directory. Point `ARCC_CACHE_ROOT` to the existing directory containing the benchmark folders, for example `/home/ch806988/multi-modal_fusion/Code/stage3/embeddings`. This panel uses `mrna_stability/ntv2_100m`, `rnafm`, `mrnafm` and `esm2_8m` beneath that root. Each cache must already record its ordered-input identity and complete shard hashes; the recipe does not repair legacy metadata. No duplicate transfer or extraction is implied.

The training route requires complete compatible caches and refuses extraction or download fallback. Its launcher additionally sets Hugging Face and Transformers offline flags. Changing the partition does not change the cached per-sequence representations, provided their input and row identities remain compatible.

## Select and launch the approved workload

Set `ARCC_PROJECT_ROOT` to the new prepared workspace, together with `ARCC_ANACONDA_MODULE`, `ARCC_CONDA_ENV`, `ARCC_KERNEL`, `ARCC_EXPECTED_GPU` and `ARCC_CACHE_ROOT`. The wrappers initialize inherited Conda correctly before module loading. GPU availability and the observed device name are checked inside the executing notebook kernel; production fits refuse CPU fallback.

| File | Operation |
| --- | --- |
| `newton_preflight.sbatch.example` | Verify the package and observe the allocated environment without running the research notebook. |
| `newton_grouped_stability.sbatch.example` | Execute twelve fits for one seed, supplied as `ARCC_SEED=0`, `1` or `2`, or array index 0–2. |
| `batches.tsv` | The declared three-seed, 36-fit workload; neither submission nor completion evidence. |

Supply account, partition, GPU type, CPU/RAM/time limits and allowed concurrency through reviewed `sbatch` options. No training-duration estimate or allocation budget is implied by the templates. One job per seed runs its twelve fits sequentially. Review seed 0's resource use before expanding to seeds 1 and 2; their fits are independent, but later baseline selection requires seed-0 validation results and the paired report requires all 36 fits. No out-of-fold correction fits belong to this batch.

The Python command inside an approved allocation is:

```shell
python -u run_grouped_stability.py --seed 0 --kernel biofusion-arcc \
  --expected-gpu 'NVIDIA H100 PCIe' \
  --cache-root /home/ch806988/multi-modal_fusion/Code/stage3/embeddings
```

Do not submit from the maintained repository or overwrite the historical Newton workspace. A prepared package is required. Slurm opens stdout/stderr before the script runs, so the selected log directory must exist before submission. The default paths are prepared as empty directories; custom output roots also need matching Slurm `--output` and `--error` options.

## Read progress and return evidence

Text progress is forwarded to Slurm logs as the notebook executes. The source notebook remains unchanged; a disposable execution copy is saved under `outputs/notebooks/`. Each launch records the exact package hashes, seed and actual kernel environment. Cooperative per-seed locks prevent duplicate writers. Compatible completed fits and recovery checkpoints can be reused, while incompatible evidence is preserved and rejected.

The default destinations are:

- `outputs/study/split.json`: saved grouped allocation and input identity.
- `outputs/study/fits/`: twelve separately identified fit records per seed and their linked weights/recovery state.
- `outputs/logs/` and `outputs/notebooks/`: launch, environment, scheduler and executed-notebook evidence.
- `outputs/study/reports/`: separately authorized saved-prediction analyses; training keeps comparison execution disabled.

Return fit/prediction JSONs, the split record, launch/environment logs, executed notebooks and source-snapshot evidence. Retain linked final weights and incomplete checkpoints in the execution archive when recovery or later use needs them. Do not return embedding caches. Preserve the package's source files so returned hashes remain checkable, and keep this experiment separate from the completed supplied-split Newton results. A returned record or passed synthetic check does not establish a scientific conclusion.
