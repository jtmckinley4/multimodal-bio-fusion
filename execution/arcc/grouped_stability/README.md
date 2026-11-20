# Grouped stability on Newton

This comparison measures how well predictors generalize to mRNA sequences they did not train on. All rows with the same nucleotide sequence stay in one partition, even if their measured values differ. The three training seeds share that split, with 12 fits per seed. [Grouped-stability notebook](../../../Code/stage4/grouped_stability.ipynb)

[Review the 36 completed fits](#review-saved-results) · [Run or resume on Newton](#run-or-resume-on-newton) · [Update an existing workspace](#update-an-existing-workspace)

## Review saved results

Open the notebook locally and point `RESULTS_ROOT` at the returned folder. Use `READ_SAVED_RESULTS = True`, `RUN_TRAINING = False`, and `RUN_COMPARISONS = False` to read compatible saved fits and the paired report. Set `RUN_COMPARISONS = True` only when you want to calculate a report from the 36 saved fits. [Returned results](../../../Code/stage4/results/grouped_stability_v1/)

## Run or resume on Newton

### Prepare and upload

From the repository root on your computer, omit `--destination` to see the upload and cache requirements. Include it to create a folder beside the repository with the stability CSV, notebook, shared Python code, requirements and launch files.

```powershell
python -B execution/arcc/prepare.py grouped_stability --destination "../ARCC Grouped Stability"
```

Upload the contents of `ARCC Grouped Stability` to `~/multi-modal_fusion_grouped_stability/`. On Newton, set the workspace path and point `ARCC_CACHE_ROOT` to a readable directory containing `mrna_stability/`. Adjust the paths if your directories differ. [Upload to Newton](../README.md#upload-to-newton)

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal_fusion_grouped_stability"
export ARCC_CACHE_ROOT="$HOME/multi-modal_fusion/Code/stage3/embeddings"
export ARCC_OUTPUT_ROOT="$ARCC_PROJECT_ROOT/outputs"
```

### Verify the upload and cache

With the shared environment active, check that the uploaded files match the snapshot and that the requested kernel uses the active Python. [Set up your Newton session](../README.md#set-up-your-newton-session)

```bash
cd "$ARCC_PROJECT_ROOT" && python -B run_grouped_stability.py --verify-only --kernel "$ARCC_KERNEL"
```

Run the separate cache command from the preparation preview to check sample files under `mrna_stability/{ntv2_100m,rnafm,mrnafm,esm2_8m}`. Training verifies the embedding hashes and their correspondence to the input rows.

### Submit a seed

Set `ARCC_ACCOUNT` for this session. For a new snapshot alongside earlier results, set `ARCC_OUTPUT_ROOT` to a fresh directory under the workspace’s `outputs/`. To resume an interrupted seed, use the original workspace, source snapshot, and output directory so compatible fits and checkpoints can be reused. Submit the GPU check from the Newton workspace. [Choose the Slurm allocation](../README.md#choose-the-slurm-allocation) · [Check and submit a job](../README.md#check-and-submit-a-job)

```bash
mkdir -p outputs/logs "$ARCC_OUTPUT_ROOT/logs" "$ARCC_OUTPUT_ROOT/notebooks"
PREFLIGHT_JOB=$(sbatch --parsable --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  grouped_stability_preflight.sbatch)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
echo "$PREFLIGHT_JOB"
```

Replace `CPUS`, `MEMORY` and `HH:MM:SS` with the requested CPUs per job, memory including units (such as `32G`), and time limit. Start a new comparison at seed 0. For a retry, set `seed` to the interrupted seed number. The dependency waits for the GPU check to succeed.

```bash
export ARCC_CPUS=CPUS ARCC_MEM=MEMORY ARCC_TIME=HH:MM:SS
seed=0
TRAIN_JOB=$(sbatch --parsable --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --job-name="grouped-stability-seed${seed}" \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task="${ARCC_CPUS:?}" \
  --mem="${ARCC_MEM:?}" --time="${ARCC_TIME:?}" \
  --dependency="afterok:${PREFLIGHT_JOB:?}" --export="ALL,ARCC_SEED=$seed" \
  newton_grouped_stability.sbatch.example)
TRAIN_JOB=${TRAIN_JOB%%;*}
echo "$TRAIN_JOB"
```

After seed 0 succeeds, review its resource use before submitting seeds 1 and 2. To queue both as an array, replace the `--export` option above with `--array=1-2%1` for one job at a time or `--array=1-2%2` for up to two GPUs. Use `--job-name=grouped-stability-repeats` and set the dependency to `afterok:` followed by the seed-0 job ID. If Slurm no longer knows that ID, verify the successful seed-0 run in job history before omitting the dependency. [Batch inventory](batches.tsv) · [Monitor or retry](../README.md#monitor-or-retry)

## Return results

From `$ARCC_OUTPUT_ROOT/study/` on Newton, copy `split.json` and the 12 `fits/<configuration>/seed<N>.json` files per seed. The existing local cohort lives at `Code/stage4/results/grouped_stability_v1/`. For a new cohort, choose a separate local folder and point the notebook’s `RESULTS_ROOT` at it. Keep `split.json` at that folder’s root and preserve the `fits/` paths. Save the workspace’s `source-snapshot.json` and the matching launch records from `$ARCC_OUTPUT_ROOT/logs/` with the returned evidence. [Download results](../README.md#download-results)

Keep the original `study/` directory on Newton, including the split, fit records and best weights. Grouped-stability correction reads this evidence for the matching seed. [Fusion mechanism experiments](../fusion_studies/README.md)

## Update an existing workspace

From the local repository root, preview what would change in the existing upload folder. Add `--apply` when ready to update it locally, then follow the shared instructions to sync and verify the Newton copy. [Update an existing workspace](../README.md#update-an-existing-workspace)

```powershell
python -B execution/arcc/prepare.py grouped_stability --update "../ARCC Grouped Stability"
```
