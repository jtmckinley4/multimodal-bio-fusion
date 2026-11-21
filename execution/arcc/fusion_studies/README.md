# Fusion mechanism experiments on Newton

The correction, gate and attention experiments use frozen benchmark embeddings to test how auxiliary encoders help a predictor. The region-aware study asks whether matching protein features to the coding part of a full RNA transcript helps. The same upload folder carries the notebooks and launch scripts for all four studies. [Research plan](../../../docs/research-execution-plan.md)

[Prepare a workspace](#prepare-the-workspace) · [Update an existing workspace](#update-an-existing-workspace) · [Run correction, gates or attention](#run-correction-gates-or-attention) · [Run region-aware fusion](#run-region-aware-fusion) · [Review results locally](#return-and-review-results)

## Prepare the workspace

On your computer, open a terminal at the repository root. Run the command without `--destination` to see which files are included and which inputs you need on Newton. Include it to create an upload folder beside the repository.

```powershell
python -B execution/arcc/prepare.py fusion_studies --destination "../ARCC Fusion Studies"
```

Upload the contents to `~/multi-modal-fusion/fusion-mechanisms/`. In your Newton terminal, set the path to the workspace you uploaded. [Upload to Newton](../README.md#upload-to-newton)

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal-fusion/fusion-mechanisms"
```

Enter the workspace and activate the shared environment. From there, check the upload against its source snapshot. The study sections cover their input checks separately. [Set up your Newton session](../README.md#set-up-your-newton-session)

```bash
python -B run_fusion_study.py --verify-only
```

### Update an existing workspace

From the repository root on your computer, point `--update` at the upload folder you previously prepared. Add `--apply` to write the previewed changes there, then follow the shared guide to sync those changes to Newton and verify the new upload. [Sync an update](../README.md#update-an-existing-workspace)

```powershell
python -B execution/arcc/prepare.py fusion_studies --update "../ARCC Fusion Studies"
```

## Check the GPU and set job resources

In the Newton workspace, submit the GPU check with your Slurm allocation set. Save the returned job ID. If Slurm can no longer resolve a prerequisite job ID, confirm its successful exit in accounting before leaving that ID out of a dependency. [Choose the Slurm allocation](../README.md#choose-the-slurm-allocation) · [Check and submit a job](../README.md#check-and-submit-a-job)

```bash
mkdir -p outputs/logs
PREFLIGHT_JOB=$(sbatch --parsable --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  fusion_studies_preflight.sbatch)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
echo "$PREFLIGHT_JOB"
```

Replace `CPUS`, `MEMORY` and `WALLTIME` with the approved limits for the next job. Include units for memory. Set these options again when the resource limits change.

```bash
job_options=(--account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal
  --cpus-per-task=CPUS --mem=MEMORY --time=WALLTIME)
```

The training and diagnostics logs are `outputs/logs/study-<job-id>.out` and `.err`. Follow the output log for epochs and check the study’s result directory for completed fits. [Monitor or retry](../README.md#monitor-or-retry)

The commands write results under `outputs/`. To keep a new run alongside earlier results, add `--output-root "$ARCC_PROJECT_ROOT/outputs/RUN_NAME"` to `study_inputs` or `region_inputs` below. Use the same output root for every action in that run, including diagnostics and continuations.

## Run correction, gates or attention

Set `swap_root` to the Newton workspace with the encoder-swap datasets and frozen embeddings.

```bash
swap_root="$HOME/multi-modal_fusion"
```

### Verify trained reference heads separately from encoder weights

For correction on CoV-Vac or fungal expression, verify the saved reference heads in that workspace. Gate diagnostics also use saved MIL head weights. Attention uses the matching saved reference predictions for comparison and its own trained weights for diagnostics. The check below covers the shared inventory of 42 result JSONs and 42 best-weight files. For grouped stability, use the grouped results described below.

```bash
python -B verify_inputs.py --scope references --reference-root "$swap_root"
```

### Choose the study and inputs

Set `study` to `gates` or `attention` for `cov_vac_degradation` or `ecoli_proteins`. For `correction`, choose `cov_vac_degradation` or `fungal_expression`. Use the appropriate pair in this block.

```bash
study=gates
benchmark=ecoli_proteins
study_inputs=(--data-root "$swap_root/datasets"
  --cache-root "$swap_root/Code/stage3/embeddings" --reference-root "$swap_root")
```

For grouped-stability correction, use this block instead. The notebook finds the grouped baseline fits and weights beside `split.json`. Keep that directory with the matching-seed anchor weights and the seed-0 evidence used to select the anchor. Use the grouped split from the run you intend to compare. [Grouped stability guide](../grouped_stability/README.md#return-results)

```bash
study=correction
benchmark=mrna_stability
grouped_results="$HOME/multi-modal_fusion_grouped_stability/outputs/study"
study_inputs=(--data-root "$swap_root/datasets"
  --cache-root "$swap_root/Code/stage3/embeddings" --grouped-split "$grouped_results/split.json")
```

### Train and run diagnostics

Submit seed 0 first and save its ID separately before you reuse `TRAIN_JOB`. For seeds 1 and 2, change `seed` and set `train_after` to that same seed-0 ID. Correction fixes the anchor choice using seed-0 validation. [Batch inventory](batches.tsv)

```bash
seed=0
train_after="${PREFLIGHT_JOB:?}"
TRAIN_JOB=$(sbatch --parsable "${job_options[@]}" --gres=gpu:nvidia_h100_pcie:1 \
  --job-name="$study-$benchmark-seed$seed-train" --dependency="afterok:$train_after" \
  newton_fusion_study.sbatch.example --study "$study" --benchmark "$benchmark" \
  --seed "$seed" --action train "${study_inputs[@]}")
TRAIN_JOB=${TRAIN_JOB%%;*}
echo "$TRAIN_JOB"
```

For gates or attention, submit diagnostics using the same study, benchmark and seed, with that seed’s training ID. For correction, proceed to local review after training.

```bash
sbatch "${job_options[@]}" --gres=gpu:nvidia_h100_pcie:1 \
  --job-name="$study-$benchmark-seed$seed-diagnostics" --dependency="afterok:${TRAIN_JOB:?}" \
  newton_fusion_study.sbatch.example --study "$study" --benchmark "$benchmark" \
  --seed "$seed" --action diagnostics "${study_inputs[@]}"
```

## Run region-aware fusion

### Stage the A.6 inputs at their recorded versions

Under `inputs/mrnabench/`, stage `rnahl-human.parquet` and `translation-efficiency-human.parquet`. Put `homology_maps/human_homology_map.csv` from the Orthrus archive under the same directory. The byte check covers both human tables even when you plan to run half-life only. The split keeps known related genes together and groups unmapped rows by recorded gene identity. [Half-life table](https://huggingface.co/datasets/morrislab/rnahl-saluki-human) · [Translation-efficiency table](https://huggingface.co/datasets/morrislab/translation-efficiency-human) · [Orthrus archive](https://zenodo.org/records/13910050)

Set `HF_HUB_CACHE` to a readable cache on Newton, including its linked weight files. If weights are missing, use `hf download MODEL --revision HASH FILES` with the filenames and revisions in the manifest. In a new cache, RNA-FM and ESM-2 also need `refs/main` files with the recorded revisions and no trailing newline. Check your quota before extraction creates token caches. [Input manifest](input-manifest.json) · [Input preparation record](../../../docs/run-records.md#2026-11-17-controlled-fusion-input-preparation) · [Storage quota](../README.md#prepare-on-newton)

```bash
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
```

After the inputs are staged, check their bytes and set the paths for the region jobs.

```bash
region_inputs=(--region-data-root "$ARCC_PROJECT_ROOT/inputs/mrnabench"
  --region-cache-root "$ARCC_PROJECT_ROOT/token_caches")
python -B verify_inputs.py --scope regions \
  --region-data-root "$ARCC_PROJECT_ROOT/inputs/mrnabench" --hf-hub-root "$HF_HUB_CACHE"
```

### Prepare the cohort and extract tokens

Start with human half-life. Review its results before deciding to run `translation-efficiency-human`. CPU preparation records the cohort and split. After that job and the GPU check succeed, extract the token embeddings once for reuse by all seeds. [Batch inventory](batches.tsv)

```bash
dataset=rnahl-human
PREPARE_JOB=$(sbatch --parsable "${job_options[@]}" --job-name="regions-$dataset-prepare" \
  newton_fusion_study.sbatch.example --study regions --benchmark "$dataset" \
  --seed 0 --action prepare "${region_inputs[@]}")
PREPARE_JOB=${PREPARE_JOB%%;*}
echo "$PREPARE_JOB"
```

```bash
EXTRACT_JOB=$(sbatch --parsable "${job_options[@]}" --gres=gpu:nvidia_h100_pcie:1 \
  --job-name="regions-$dataset-extract" --dependency="afterok:${PREPARE_JOB:?}:${PREFLIGHT_JOB:?}" \
  newton_fusion_study.sbatch.example --study regions --benchmark "$dataset" \
  --seed 0 --action extract "${region_inputs[@]}")
EXTRACT_JOB=${EXTRACT_JOB%%;*}
echo "$EXTRACT_JOB"
```

### Train the panel and run diagnostics

An empty `fit_args` selects the full 49-fit panel for one seed. For a bounded trial, replace the empty list with one `--fit-key` per fit. For example, `fit_args=(--fit-key rnafm__native__rna_only__true --fit-key rnafm__native__attention_cds__true)` selects two native RNA-FM fits. After the trial succeeds, clear the list and use the same output directory so the full panel can reuse compatible fits. Save the full seed-0 job ID separately and use it as `train_after` for both seed 1 and seed 2.

```bash
seed=0
fit_args=()
train_after="${EXTRACT_JOB:?}"
TRAIN_JOB=$(sbatch --parsable "${job_options[@]}" --gres=gpu:nvidia_h100_pcie:1 \
  --job-name="regions-$dataset-seed$seed-train" --dependency="afterok:$train_after" \
  newton_fusion_study.sbatch.example --study regions --benchmark "$dataset" \
  --seed "$seed" --action train "${region_inputs[@]}" "${fit_args[@]}")
TRAIN_JOB=${TRAIN_JOB%%;*}
echo "$TRAIN_JOB"
```

Queue diagnostics after each successful full training seed, using that seed’s job ID and the same paths.

```bash
sbatch "${job_options[@]}" --gres=gpu:nvidia_h100_pcie:1 \
  --job-name="regions-$dataset-seed$seed-diagnostics" --dependency="afterok:${TRAIN_JOB:?}" \
  newton_fusion_study.sbatch.example --study regions --benchmark "$dataset" \
  --seed "$seed" --action diagnostics "${region_inputs[@]}"
```

### Retry the native half-life attention fit

The size-four stress probe finished two Adam updates and a validation batch on one H100. Use this option for the RNA-FM native true-CDS attention fit at seed 0. It keeps one optimizer update per 32 examples and uses four rows at a time on the GPU for training and scoring. The fit records the slice size in its execution identity. [Half-life checkpoint](../../../docs/agent-context.md#current-execution-and-retained-evidence)

After the existing jobs finish, follow [Update an existing workspace](#update-an-existing-workspace) and verify the transferred files. From Newton, use the original preparation and token caches. The retry gets a separate output directory so its checkpoints and result can be read alongside the earlier fits.

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal-fusion/fusion-mechanisms"
export ARCC_ANACONDA_MODULE="anaconda/anaconda-2024.10"
export ARCC_CONDA_ENV="$HOME/.conda/envs/biofusion-arcc"
export ARCC_KERNEL="biofusion-arcc"
export ARCC_EXPECTED_GPU="NVIDIA H100 PCIe"
retry_root="$ARCC_PROJECT_ROOT/outputs/halflife-native-sliced"

(
  set -e
  cd "$ARCC_PROJECT_ROOT"
  "$ARCC_CONDA_ENV/bin/python" -B run_fusion_study.py --verify-only
  original="outputs/regions/rnahl-human/preparation.json"
  retry_prep="$retry_root/regions/rnahl-human/preparation.json"
  mkdir -p outputs/logs "$(dirname "$retry_prep")"
  cp -n -- "$original" "$retry_prep"
  cmp -- "$original" "$retry_prep"
)
```

Once the checks pass, submit one fit with the resource limits you intend to use. The 48-hour walltime below is a cap. If the job times out after a checkpoint has been saved, resume with the same package, options, and output directory.

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

sbatch --chdir="$ARCC_PROJECT_ROOT" --account="${ARCC_ACCOUNT:?Set your Newton account}" \
  --partition=normal --qos=normal --gres=gpu:nvidia_h100_pcie:1 \
  --cpus-per-task=4 --mem=64G --time=2-00:00:00 \
  --job-name=halflife-native-attention-seed0 \
  "$ARCC_PROJECT_ROOT/newton_fusion_study.sbatch.example" \
  --study regions --benchmark rnahl-human --seed 0 --action train \
  --fit-key rnafm__native__attention_cds__true --slice-size 4 \
  --region-data-root "$ARCC_PROJECT_ROOT/inputs/mrnabench" \
  --region-cache-root "$ARCC_PROJECT_ROOT/token_caches" \
  --output-root "$retry_root"
```

Follow the ordinary Slurm output under `outputs/logs/`. The returned preparation, seed-0 predictions, best weights and launch record form a pilot cohort. The full-panel report still requires matching implementation hashes and all declared seeds.


## Return and review results

Download the fit and prediction JSONs from the study’s output directory into its notebook’s `RESULTS_ROOT`. The paths below use the default Newton output directory and paths from the repository root on your computer. Keep the benchmark and configuration subfolders.

- `outputs/correction/` → `Code/stage4/results/anchor_correction/`. Include the selection and out-of-fold prediction records. The notebook checks the fit JSONs and best-weight files named in each prediction record. Bring back those files or reuse verified local copies. For reference anchors, point `REFERENCE_ROOT` or `GROUPED_SPLIT_PATH` at the matching returned baseline so the notebook can find its original fits and weights. [Anchor correction notebook](../../../Code/stage4/anchor_correction.ipynb)
- `outputs/gates/` → `Code/stage4/results/gate_diagnostics_v1/`. Keep each diagnostic JSON with its companion `.npz` array. [Gate diagnostics notebook](../../../Code/stage4/gate_diagnostics.ipynb)
- `outputs/attention/` → `Code/stage4/results/anchor_cross_attention_v1/`. Keep the selected-anchor JSON and each diagnostic JSON with its companion `.npz` array. [Anchor cross-attention notebook](../../../Code/stage4/anchor_cross_attention.ipynb)
- `outputs/regions/` → `Code/stage4/results/region_aware_v1/`. Keep the dataset subfolder, including `preparation.json`, `fits/<configuration>/seed<N>.json` and the fit’s `seed<N>.diagnostics-v2.json` if present. A completed trial supplies individual fit records. The three-seed report needs the full panel. [Region-aware fusion notebook](../../../Code/stage4/region_aware_fusion.ipynb)

Bring back the launch records and matching source evidence separately. Keep token caches and training checkpoints on Newton for reuse. If you are reviewing a separate run, download to a new folder and point the notebook’s `RESULTS_ROOT` at it. [Download results](../README.md#download-results)

In the notebook, keep `READ_SAVED_RESULTS = True` and the training and diagnostics switches off. Leave `RUN_ANALYSIS = False` to view an existing summary. For a new local summary, enable `RUN_ANALYSIS` for that pass. In correction, also set `BENCHMARK` to the returned benchmark you want to review. Gates and attention use the matching local encoder-swap results at `REFERENCE_ROOT`. [Saved-result settings](../../../Code/stage4/RUNBOOK.md#read-saved-results-without-starting-work)

For a new region-aware summary, select the dataset in `DATASET_KEY` and record the local analysis environment in the notebook’s setup cell. These settings calculate a report from the full set of saved fits.

```python
RUN_ANALYSIS = True
PRODUCER = run_identity.producer_record(device="cpu", repository=REPO_DIR)
```
