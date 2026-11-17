# Grouped-stability transfer and run checklist

Use this checklist for the separately prepared 36-fit package. It implements the grouped evaluation in Gantt/Details task 28; preparation is not a completed experiment or permission to submit jobs. Scientific settings remain in `Code/stage4/grouped_stability.ipynb`.

## Stage only the new package

Upload the prepared package's contents into a new Newton directory such as `$HOME/multi-modal_fusion_grouped_stability`. Keep the completed encoder-swap workspace and the older local package unchanged. This package contains code, the stability CSV and synthetic tests; it contains no embedding arrays, trained weights or historical results.

The existing Newton caches supply `mrna_stability/ntv2_100m`, `rnafm`, `mrnafm` and `esm2_8m`. All four require complete manifests, shard hashes and ordered-input identities. No encoder weights need staging for these frozen-cache fits, and no cache transfer or extraction is required.

In the Newton shell, after transfer is authorized and complete:

```shell
export ARCC_PROJECT_ROOT="$HOME/multi-modal_fusion_grouped_stability"
export ARCC_CACHE_ROOT="$HOME/multi-modal_fusion/Code/stage3/embeddings"
export ARCC_ANACONDA_MODULE="anaconda/anaconda-2024.10"
export ARCC_CONDA_ENV="$HOME/.conda/envs/biofusion-arcc"
export ARCC_KERNEL="biofusion-arcc"
export ARCC_EXPECTED_GPU="NVIDIA H100 PCIe"
export ARCC_OUTPUT_ROOT="$ARCC_PROJECT_ROOT/outputs"
cd "$ARCC_PROJECT_ROOT"
mkdir -p outputs/logs outputs/notebooks
"$ARCC_CONDA_ENV/bin/python" -B run_grouped_stability.py --verify-only --kernel "$ARCC_KERNEL"
```

Verification checks package bytes and the registered kernel's interpreter path without starting a kernel. It does not check GPU access or cache-array contents. Do not upgrade the working Newton environment automatically; retain its observed package versions.

## Observe the allocated environment

After approval for this environment-only job:

```shell
PREFLIGHT_JOB=$(sbatch --parsable --account=igaribay --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  newton_preflight.sbatch.example)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
sacct -j "$PREFLIGHT_JOB" --format=JobID,State,Elapsed,ExitCode
```

Once it finishes, inspect `outputs/logs/preflight-JOBID.out` and `.err`, replacing `JOBID` with that actual number. Require `COMPLETED`, exit `0:0`, the expected environment and an H100 observation. The preflight checks the activated interpreter's GPU access and the registered kernel's path; training additionally verifies the GPU inside the executing notebook kernel. Neither check establishes cache compatibility until the training preflight verifies the manifests and shard hashes.

## Review resources with seed 0 before expanding

Agree CPU count, RAM and a bounded first-job wall time, then set `GROUPED_CPUS`, `GROUPED_MEM` and `GROUPED_TRIAL_TIME` to those Slurm values. The following command refuses unset limits. It launches the normal seed-0 panel, up to twelve fits, within that allocation; it does not reduce the scientific epoch budget or silently choose a sample. Initial cache verification can leave the GPU idle before training starts.

```shell
: "${GROUPED_CPUS:?Set the approved CPU count}"
: "${GROUPED_MEM:?Set the approved RAM limit}"
: "${GROUPED_TRIAL_TIME:?Set the approved first-job wall time}"
SEED0_JOB=$(sbatch --parsable --account=igaribay --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task="${GROUPED_CPUS:?}" \
  --mem="${GROUPED_MEM:?}" --time="${GROUPED_TRIAL_TIME:?}" \
  --dependency="afterok:$PREFLIGHT_JOB" --export=ALL,ARCC_SEED=0 \
  newton_grouped_stability.sbatch.example)
SEED0_JOB=${SEED0_JOB%%;*}
squeue -j "$SEED0_JOB"
sacct -j "$SEED0_JOB" --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

Inspect the matching `outputs/logs/grouped-*` logs for fit names and advancing epochs. Review elapsed time, peak memory and fit progress before agreeing the full run time and concurrency. If the trial times out, retain its outputs and resubmit the same seed/package after review: compatible completed fits are reused and valid checkpoints resume. Use the final successful seed-0 job ID for the dependency below, not a timed-out job ID.

Seeds 1 and 2 do not statistically depend on seed 0: each fits the same twelve predictors independently. Waiting for seed 0 is a resource and startup gate. **Baseline selection does depend on complete seed-0 validation results**, and the later three-seed report requires all 36 fits; training jobs do neither selection nor bootstrap analysis.

After reviewing seed 0, set `GROUPED_RUN_TIME` and `GROUPED_CONCURRENCY` (1 or 2) to the approved limits:

```shell
: "${GROUPED_RUN_TIME:?Set the approved repeat-job wall time}"
if [[ "${GROUPED_CONCURRENCY:-}" =~ ^[12]$ ]]; then
  sbatch --account=igaribay --partition=normal --qos=normal \
    --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task="${GROUPED_CPUS:?}" \
    --mem="${GROUPED_MEM:?}" --time="${GROUPED_RUN_TIME:?}" \
    --dependency="afterok:${SEED0_JOB:?}" --array="1-2%$GROUPED_CONCURRENCY" \
    newton_grouped_stability.sbatch.example
else
  echo 'Set approved concurrency to 1 or 2 before submitting.'
fi
```

All jobs share the new output root and external caches. Per-seed locks reject duplicate writers, and all jobs independently derive and verify the same grouped split. A successful seed job reports twelve compatible fits. Check scheduler exit status and logs as well as file counts.

## Return evidence without caches

Download the new package's `outputs/` into a separate grouped-stability result destination, along with `source-snapshot.json` and the exact execution sources needed to verify it. Preserve `outputs/study/split.json`, fit JSONs and linked final weights, launch/environment logs and executed notebooks; retain recovery files for any incomplete fits. The weights are needed for downstream anchor-correction preparation, even though a score-only summary needs just saved predictions.

Do not download the external embedding directory or merge these results into `newton_h100_seeds012`. Keep the package archive and its source/dataset hashes; an already retained byte-identical CSV can satisfy the source evidence without another copy. Baseline selection, paired bootstrap and interpretation remain a separately authorized local analysis after all 36 fits are verified.
