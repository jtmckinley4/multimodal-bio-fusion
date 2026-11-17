# Newton upload, run and return checklist

Use this after reviewing the [prepared workload](README.md#review-the-workload-before-scheduling). Follow source preparation, transfer, input verification, a resource trial and production runs in order. Agree the selected workload and resource limits before submission.

## 1. Upload the new source packages

In WinSCP, upload the **contents** of each prepared sibling folder to its new remote directory. Use binary transfer and preserve the relative paths. Do not add another nested folder named `ARCC Fusion Studies 2026-11-17 Controls` on Newton.

| Local prepared folder | Newton directory |
| --- | --- |
| `ARCC Fusion Studies 2026-11-17 Controls` | `/home/ch806988/multi-modal_fusion_studies/` |
| `ARCC Grouped Stability 2026-11-17` | `/home/ch806988/multi-modal_fusion_grouped_stability/` |

Use the Controls package for the repaired panel; the earlier fusion package is retained as a superseded preparation snapshot. The grouped package remains valid and unchanged. The fusion package contains source, launchers, documentation and verification manifests, not the scientific inputs. The grouped package additionally contains its stability CSV. Leave `/home/ch806988/multi-modal_fusion/` in place: its datasets, BLF caches, completed swap records and best-head weights are reused. Do not re-upload the approximately 137 GB of embeddings or merge the new packages into that older workspace.

For A.6, upload these additional inputs directly from their original local locations. `INPUT_TRANSFER.tsv` in the prepared local folder lists exact source/destination paths; it is a transfer aid, not another data copy.

| Local source | Newton destination |
| --- | --- |
| Repository `.data-cache/mrnabench/rnahl-human.parquet` | `multi-modal_fusion_studies/inputs/mrnabench/rnahl-human.parquet` |
| Repository `.data-cache/mrnabench/translation-efficiency-human.parquet` | `multi-modal_fusion_studies/inputs/mrnabench/translation-efficiency-human.parquet` |
| Repository `.data-cache/mrnabench/homology_maps/human_homology_map.csv` | `multi-modal_fusion_studies/inputs/mrnabench/homology_maps/human_homology_map.csv` |
| The three named model directories in `D:/model-cache/huggingface/hub/` | The same directory names beneath `/home/ch806988/.cache/huggingface/hub/` |

The [model table](README.md#verify-weights-for-their-actual-purpose) names those three directories. Preserve their `snapshots/`, `refs/` and any linked `blobs/` files. These are approximately 1.02 GB, not the entire Hugging Face cache. If matching directories already exist remotely, verify them before replacing anything. An incompatible existing reference needs its own separate cache destination and `HF_HUB_CACHE` setting.

Before A.6 extraction, confirm quota and filesystem space for **new** token arrays. The retained CPU estimate was approximately 234.5 GB of float16 payload across both datasets, excluding file overhead, weights, buffers and results; it is not measured disk use or a runtime estimate. Existing pooled caches cannot replace these arrays.

## 2. Verify the environment, source and inputs

On the Newton login node, activate the established environment:

```bash
module load anaconda/anaconda-2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/.conda/envs/biofusion-arcc"
cd "$HOME/multi-modal_fusion_studies"
python -m pip check
python -B run_fusion_study.py --verify-only
```

Use this environment rather than automatically installing or upgrading `requirements.txt`. Its recorded swap baseline includes PyTorch `2.10.0+cu126` and Transformers `5.15.1`; A.6 also needs the parquet reader `pyarrow`. A missing dependency should lead to a specific, reviewed environment change, followed by a new preflight. The kernel and shell interpreter must refer to the intended environment.

The package also carries the synthetic model/fit checks. They use invented tensors and temporary records, without opening research datasets or training research models:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m unittest discover -s tests -p "test_controlled_*.py"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m unittest discover -s tests -p "test_region_training.py"
```

Verify existing trained reference weights and the staged A.6 inputs:

```bash
python -B verify_inputs.py --scope references \
  --reference-root "$HOME/multi-modal_fusion"

python -B verify_inputs.py --scope regions \
  --region-data-root "$HOME/multi-modal_fusion_studies/inputs/mrnabench" \
  --hf-hub-root "$HOME/.cache/huggingface/hub"
```

The second command can check the data root or hub root separately while the other transfer finishes. Verification checks file identities; it does not prepare a cohort, open encoder weights as models, extract tokens or fit. Missing old weights are investigated in the existing Newton workspace; they do not trigger automatic retraining. Grouped-stability weights do not exist until its new experiment runs.

The [manifest](input-manifest.json) covers 42 reference JSONs plus 42 linked head weights, three region-data files (about 43.7 MB) and 15 pinned model/reference files (about 1.02 GB). It does not rehash all old embedding shards; the study readers separately enforce their cache identity and completeness contracts.

Set the execution environment for the new workspace:

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal_fusion_studies"
export ARCC_ANACONDA_MODULE="anaconda/anaconda-2024.10"
export ARCC_CONDA_ENV="$HOME/.conda/envs/biofusion-arcc"
export ARCC_KERNEL="biofusion-arcc"
export ARCC_EXPECTED_GPU="NVIDIA H100 PCIe"
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
cd "$ARCC_PROJECT_ROOT"
mkdir -p outputs/logs outputs/notebooks
```

Slurm opens log files before the script starts, so those directories must already exist. The launcher sets model-hub offline mode; staging missing weights is a separate preparation step.

## 3. Run preflight, then review resource limits

The established allocation example is account `igaribay`, partition/QOS `normal`, and one `gpu:nvidia_h100_pcie:1`. Availability and permission still come from Newton. CPU count, RAM, walltime and maximum concurrent jobs must be explicitly reviewed for the selected operation; this checklist does not invent a budget.

After setting `ARCC_CPUS`, `ARCC_MEM` and `ARCC_TIME` to the reviewed **preflight** values, submit:

```bash
PREFLIGHT_JOB=$(sbatch --parsable \
  --account=igaribay --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 \
  --cpus-per-task="${ARCC_CPUS:?Set reviewed CPU count}" \
  --mem="${ARCC_MEM:?Set reviewed RAM}" \
  --time="${ARCC_TIME:?Set reviewed walltime}" \
  newton_fusion_preflight.sbatch.example)
echo "$PREFLIGHT_JOB"
```

The preflight checks source integrity, imports and a small CUDA operation in the actual notebook kernel. It does not load pretrained encoders, run a research notebook or measure training/extraction memory. Inspect `sacct -j "$PREFLIGHT_JOB" --format=JobID,State,Elapsed,ExitCode` and `outputs/logs/preflight-${PREFLIGHT_JOB}.out`/`.err` before a resource trial. A verified H100 does not establish that a full-transcript batch fits in GPU memory.

Record the selected trial's exact study, dataset, seed/fit keys and resource limits. Do not silently reduce batch size, precision, sequence length or stopping criteria after an OOM; those changes affect the comparison. Review trial measurements before selecting the full panel's time, RAM and concurrency. For A.6, extraction and head fitting require separate resource checks.

## 4. Follow the dependency order

| Work | What must finish first | Seeds/actions |
| --- | --- | --- |
| Grouped stability | Its separate package verification/preflight and compatible existing stability caches | Three jobs: 12 fits each for seeds 0, 1 and 2. |
| A.2 CoV-Vac/fungi | Verified old reference records/weights and existing BLF caches | Seed 0 before seeds 1/2; one `train` job per benchmark/seed. |
| A.2 stability | New grouped split and compatible grouped single-input results/weights | Same seed order; pass `--grouped-split`. Do not use old supplied-split stability baselines. |
| A.3–A.4 gates | Compatible CoV-Vac/E. coli reference records and BLF caches | One `train` job per benchmark/seed; separate `diagnostics` jobs afterwards. |
| A.5 attention | Compatible CoV-Vac/E. coli references and fixed seed-0 validation selection | Complete seed 0 first, then seeds 1/2; separate `diagnostics`. |
| A.6 half-life | Staged parquet/map and encoder weights; reviewed storage/resource plan | `prepare` on CPU once with seed 0 → `extract` on GPU once with seed 0 → `train` for 0/1/2 → separate diagnostics. |
| A.6 TE | Review half-life evidence/resource use before expanding | Repeat the A.6 sequence for `translation-efficiency-human`; do not share its cohort/split with half-life. |

The two packages are independent except that stability correction consumes grouped-stability evidence. CoV-Vac/fungal correction, gates and attention do not need to wait for all grouped fits if their own inputs and resource allowance are ready. [batches.tsv](batches.tsv) lists the 49 possible study operations and prerequisites; select the authorized rows rather than submitting the whole table. Limit total concurrency across packages; a separate array cap does not cap all other jobs.

Use the [grouped-stability instructions](../grouped_stability/README.md) in the checkout, or that package's own root README. Set `ARCC_PROJECT_ROOT` to `$HOME/multi-modal_fusion_grouped_stability` and `ARCC_CACHE_ROOT` to `$HOME/multi-modal_fusion/Code/stage3/embeddings` for those jobs. Its wrapper accepts `ARCC_SEED` or a Slurm array index 0–2. Afterwards restore the fusion workspace environment from step 2 before using the following commands.

## 5. Submit one reviewed study job

Reset `ARCC_CPUS`, `ARCC_MEM` and `ARCC_TIME` for the selected **research operation**, rather than carrying the preflight budget into training. The following is an A.2 CoV-Vac seed-0 example; it includes the successful-preflight dependency:

```bash
cd "$HOME/multi-modal_fusion_studies"
TRAIN_JOB=$(sbatch --parsable \
  --account=igaribay --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 \
  --cpus-per-task="${ARCC_CPUS:?Set reviewed CPU count}" \
  --mem="${ARCC_MEM:?Set reviewed RAM}" \
  --time="${ARCC_TIME:?Set reviewed walltime}" \
  --dependency="afterok:${PREFLIGHT_JOB:?Set successful preflight job ID}" \
  newton_fusion_study.sbatch.example \
  --study correction --benchmark cov_vac_degradation --seed 0 --action train \
  --data-root "$HOME/multi-modal_fusion/datasets" \
  --cache-root "$HOME/multi-modal_fusion/Code/stage3/embeddings" \
  --reference-root "$HOME/multi-modal_fusion")
echo "$TRAIN_JOB"
```

Select the next job by changing the explicit study arguments, not the maintained notebook. For seed 1/2 jobs that depend on seed 0, use `--dependency="afterok:$TRAIN_JOB"` with that benchmark's seed-0 job ID. A failed prerequisite blocks its dependent job; inspect and resolve it instead of removing the dependency blindly. Do not submit duplicate writers for the same study/benchmark/seed, even for different A.6 fit subsets.

| Study | Valid benchmark argument | Additional route/action |
| --- | --- | --- |
| `correction` | `cov_vac_degradation`, `fungal_expression`, `mrna_stability` | `train` only; stability needs `--grouped-split "$HOME/multi-modal_fusion_grouped_stability/outputs/study/split.json"`. |
| `gates`, `attention` | `cov_vac_degradation`, `ecoli_proteins` | Keep data/cache/reference routes; `train` or `diagnostics`. |
| `regions` | `rnahl-human`, `translation-efficiency-human` | Use region routes below; `prepare`, `extract`, `train` or `diagnostics`. |

For A.6, replace the study arguments after the wrapper with:

```bash
--study regions --benchmark rnahl-human --seed 0 --action prepare \
  --region-data-root "$HOME/multi-modal_fusion_studies/inputs/mrnabench" \
  --region-cache-root "$HOME/multi-modal_fusion_studies/token_caches"
```

`prepare` is a CPU Slurm job: omit `--gres`, use its reviewed CPU/RAM/time, and keep `--seed 0`. Use the same routes for GPU `extract`, dependent on preparation, then GPU `train`, dependent on extraction. Each dataset's frozen caches are extracted once and reused across training seeds. For an agreed smaller fitting trial, append repeated `--fit-key` arguments copied exactly from the notebook inventory. Without those arguments, training selects its entire declared core panel for the chosen dataset and seed.

Diagnostic jobs keep the same benchmark/seed/input routes and use `--action diagnostics` after successful fitting. They reload best weights and consume additional GPU time. Neither fitting nor diagnostic jobs generate the three-seed bootstrap report. Correction's saved OOF/validation diagnostics are local notebook analyses, not a launcher action.

## 6. Monitor and preserve failures

Use the numeric job ID printed by `sbatch` directly:

```bash
squeue -u "$USER" -o "%.18i %.24j %.12T %.12M %R"
sacct -j "$TRAIN_JOB" --format=JobID,JobName%24,State,Elapsed,ExitCode
tail -n 60 -f "outputs/logs/study-${TRAIN_JOB}.out" \
  "outputs/logs/study-${TRAIN_JOB}.err"
```

`Ctrl+C` stops `tail`, not the research job. Pending jobs may have no log yet. Training logs must show the observed GPU and advancing fit/epoch messages; job state alone does not establish progress. For a running GPU job, inspect the allocation without opening another GPU job:

```bash
srun --jobid="$TRAIN_JOB" --overlap --nodes=1 --ntasks=1 \
  --cpus-per-task=1 nvidia-smi
```

Keep failed launch records, executed notebooks and compatible recovery state. Investigate the error, then rerun the same compatible batch only after its prior writer has ended. Completed fits are reused; incompatible files are refused and preserved. Changing the source package or scientific settings requires a new identified execution rather than rewriting a launched snapshot.

## 7. Return compact evidence, then report across seeds

Download each new workspace into its own new local results folder after checking completion. Keep grouped stability and the four studies identifiable. Return:

- Fit/prediction JSONs, selected diagnostic summaries/arrays and any already-produced reports from `outputs/`.
- A.6 preparation/cohort/split/exclusion/permutation records and cache/chunk manifests; the grouped-stability `outputs/study/split.json`.
- `outputs/logs/`, `outputs/notebooks/`, `source-snapshot.json`, input manifests and the source files listed by that snapshot. A launch record alone cannot reconstruct its executed source.

Keep best weights and incomplete recovery checkpoints on Newton while inference or resumption still needs them. They need not be downloaded to calculate a report from saved predictions. Do not return `token_caches/`, the old BLF `embeddings/`, pretrained model weights or duplicated source datasets merely to read results. If downloading `outputs/`, select compact files/directories and exclude linked large weights/checkpoints deliberately; do not download the workspace root indiscriminately.

Point each maintained notebook's explicit result/reference paths at its returned cohort. Leave training, extraction and preparation off. Once all required seed-0/1/2 fits are present, separately authorize `RUN_ANALYSIS=True` to publish the saved-prediction comparisons; default reading only displays existing reports. Three-seed variation and group-bootstrap uncertainty remain separate quantities. Review the results before authorizing the next workload or updating scientific conclusions.
