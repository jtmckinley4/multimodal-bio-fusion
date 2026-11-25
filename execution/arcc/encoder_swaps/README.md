# Encoder swaps on Newton

The encoder-swap notebook compares what changes when we replace one DNA, RNA or protein encoder at a time. It already contains the saved [four-benchmark findings](../../../Code/stage4/encoder_swaps.ipynb#Newton-three-seed-findings) and [E. coli RNA comparison](../../../Code/stage4/encoder_swaps.ipynb#Replacing-RiNALMo-with-RNAErnie-and-ERNIE-RNA). For account access, file transfers, Conda and job monitoring, follow the [shared Newton guide](../README.md).

[Prepare the original panel](#prepare-the-original-panel) · [Prepare the E. coli RNA extension](#prepare-the-e-coli-rna-extension) · [Update an existing workspace](#update-an-existing-workspace)

## Prepare the original panel

From the repository root on your computer, create the local upload folder. The package includes the four benchmark CSVs and the historical seed-0 validation records used to keep selection fixed. Omit `--destination` to preview the upload and input checks.

```powershell
python -B execution/arcc/prepare.py encoder_swaps --destination "../ARCC Encoder Swaps"
```

[Upload the folder's contents](../README.md#upload-to-newton) to your Newton workspace. These paths follow the original layout. If you are keeping the completed workspace as it is, choose a new `ARCC_PROJECT_ROOT` and reuse its embeddings through `ARCC_CACHE_ROOT` if you can read them there.

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal_fusion"
export ARCC_CACHE_ROOT="$HOME/multi-modal_fusion/Code/stage3/embeddings"
export ARCC_OUTPUT_ROOT="$ARCC_PROJECT_ROOT/outputs"
```

If the required embeddings are missing, transfer the benchmark and encoder folders that the run needs. For an older cache that relies on `recovery-records.json`, bring its copy of that file to the cache root so the notebook can verify the original input order and embedding file hashes. [Cache recovery history](../../../docs/run-records.md#2026-10-25-encoder-swap-cache-and-checkpoint-recovery)

[Set up the Newton session](../README.md#set-up-your-newton-session) and continue to [check the upload and GPU](#check-the-upload-and-gpu).

## Prepare the E. coli RNA extension

The extension runs the existing MIL gated and concatenation heads with RNAErnie and ERNIE-RNA on the same E. coli split. It keeps the DNA and protein encoders, codon grid and historical MIL penalty (0.01). The RiNALMo comparison comes from the original Newton batch.

From the repository root on your computer, prepare the extension as a separate upload folder. Omit `--destination` to preview this package and its input checks.

```powershell
python -B execution/arcc/prepare.py encoder_swaps --experiment ecoli_ernie --destination "../ARCC E. coli RNA Extension"
```

[Upload the folder's contents](../README.md#upload-to-newton) to the extension workspace. Point `ARCC_REFERENCE_ROOT` to the original workspace containing `outputs/phaseB/ecoli_proteins/rinalmo/` and `ARCC_CACHE_ROOT` to its readable E. coli DNA and protein embedding caches.

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal_fusion_ecoli_ernie"
export ARCC_REFERENCE_ROOT="$HOME/multi-modal_fusion"
export ARCC_CACHE_ROOT="$ARCC_REFERENCE_ROOT/Code/stage3/embeddings"
export ARCC_OUTPUT_ROOT="$ARCC_PROJECT_ROOT/outputs/ecoli_ernie"
```

[Set up the Newton session](../README.md#set-up-your-newton-session) in the extension workspace. If the checkpoints are missing from your Hugging Face cache, download these pinned files before extraction. The RNAErnie checkpoint has room for the 999-nucleotide inputs and their framing tokens. [RNAErnie weights](https://huggingface.co/LLM-EDA/RNAErnie/tree/df26681ed44bb671b61536c5e2a196d6058a44f7) · [ERNIE-RNA weights](https://huggingface.co/multimolecule/ernierna/tree/9f6099b85d61f7567d4e1ed1e92e1de85fdfd292)

```bash
hf download LLM-EDA/RNAErnie --revision df26681ed44bb671b61536c5e2a196d6058a44f7 \
  model.safetensors config.json tokenizer_config.json special_tokens_map.json vocab.txt
hf download multimolecule/ernierna --revision 9f6099b85d61f7567d4e1ed1e92e1de85fdfd292 \
  model.safetensors config.json tokenizer_config.json vocab.txt
```

## Check the upload and GPU

With Conda active in the Newton workspace, check the upload against its source snapshot.

```bash
python -B run_encoder_swaps.py --verify-only
```

Run the external-input checks printed by preparation to check that the embeddings are readable. For the extension, also check the reference directory and staged weights. The launcher and notebook verify the inputs they actually use at runtime.

For a resumed run, keep its original output directory. If you are running a changed snapshot alongside earlier results, set `ARCC_OUTPUT_ROOT` to a fresh subfolder under the workspace's `outputs/`. With [your allocation set](../README.md#choose-the-slurm-allocation), submit the GPU check from the workspace and keep its job ID.

```bash
mkdir -p outputs/logs "${ARCC_OUTPUT_ROOT:?}/logs" "$ARCC_OUTPUT_ROOT/notebooks"
PREFLIGHT_JOB=$(sbatch --parsable --job-name=encoder-swaps-preflight \
  --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  encoder_swaps_preflight.sbatch)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
echo "$PREFLIGHT_JOB"
```

## Run the original panel

Set `ARCC_CPUS`, `ARCC_MEM` and `ARCC_TIME` to the agreed limits before submitting. This example submits E. coli seed 0 after the GPU check succeeds. For another run, set the benchmark and seed from the [batch list](batches.tsv). The launcher also supports arrays using that list's indexes 0–11.

```bash
export ARCC_BENCHMARK=ecoli_proteins ARCC_SEED=0
sbatch --job-name="encoder-swaps-${ARCC_BENCHMARK}-seed${ARCC_SEED}" \
  --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task="${ARCC_CPUS:?}" \
  --mem="${ARCC_MEM:?}" --time="${ARCC_TIME:?}" --dependency="afterok:${PREFLIGHT_JOB:?}" \
  newton_swaps.sbatch.example
```

Expect 42 fit records per benchmark and seed, 504 across the full panel. From `ARCC_OUTPUT_ROOT`, download the fit and prediction JSONs in `phaseA/` and `phaseB/` and the matching `logs/` records. Keep those paths under `outputs/` in a distinct local result folder. [Download results](../README.md#download-results)

Keep the current workspace's `source-snapshot.json` with the launch evidence. The notebook checks a separate snapshot of the original validation selection. Copy the upload's `evidence/selection-source-snapshot.json` to `provenance/source-snapshot.json` under the local result folder. Keep `RESULT_SET = "newton_h100_seeds012"` and point `FOUR_BENCHMARK_WORKSPACE` to that folder containing `outputs/`, or leave it at `None` for the repository default. [Saved result layout and settings](../../../Code/stage4/RUNBOOK.md#select-a-returned-result-set)

To calculate a report from previously saved full-panel predictions on Newton, use the optional [CPU report wrapper](newton_report.sbatch.example). For inspection, keep the notebook's default [saved-results settings](../../../Code/stage4/RUNBOOK.md#read-saved-results-without-starting-work).

## Focused E. coli RNA extension

[Prepare the extension workspace](#prepare-the-e-coli-rna-extension) and use its [GPU check job ID](#check-the-upload-and-gpu). Set `ARCC_CPUS`, `ARCC_MEM` and `ARCC_TIME` to the agreed limits for extraction. Extract both RNA embedding caches once. The job uses the staged weights offline and writes the caches under `extension_embeddings/`.

```bash
export ARCC_OPERATION=extract
EXTRACT_JOB=$(sbatch --parsable --job-name=ecoli-ernie-extract \
  --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task="${ARCC_CPUS:?}" \
  --mem="${ARCC_MEM:?}" --time="${ARCC_TIME:?}" --dependency="afterok:${PREFLIGHT_JOB:?}" \
  newton_ernie.sbatch.example)
EXTRACT_JOB=${EXTRACT_JOB%%;*}
echo "$EXTRACT_JOB"
```

Set the training limits before submitting. This submits seed 0 after extraction succeeds. Change `ARCC_SEED` to 1 or 2 if submitting another seed. If you are reusing completed caches from an earlier session, use `PREFLIGHT_JOB` in place of `EXTRACT_JOB` in the dependency below. Training checks cache compatibility before starting. Expect four fits per seed. [Batch list](ecoli_ernie_batches.tsv)

```bash
export ARCC_OPERATION=train ARCC_SEED=0
sbatch --job-name="ecoli-ernie-train-seed${ARCC_SEED}" \
  --account="${ARCC_ACCOUNT:?}" --partition=normal --qos=normal \
  --gres=gpu:nvidia_h100_pcie:1 --cpus-per-task="${ARCC_CPUS:?}" \
  --mem="${ARCC_MEM:?}" --time="${ARCC_TIME:?}" --dependency="afterok:${EXTRACT_JOB:?}" \
  newton_ernie.sbatch.example
```

From `ARCC_OUTPUT_ROOT`, download the fit and prediction JSONs in `phaseB/` and the matching `logs/` records. Put them under `outputs/ecoli_ernie/` in a separate local result folder and keep the source snapshot for those launches. Keep `extension_embeddings/` on Newton. [Download results](../README.md#download-results)

To read a new return, keep `READ_SAVED_RESULTS = True` and set `ECOLI_FOLLOWUP_WORKSPACE` to the extension's local folder containing `outputs/`. `FOUR_BENCHMARK_WORKSPACE` selects the original cohort used as its reference. Either setting can remain `None` to use that study's repository default. The default four-benchmark Newton view can read both returns together; select `EXPERIMENT = "ecoli_ernie"` for a focused view. [Verified extension return](../../../docs/run-records.md#2026-11-19-verified-newton-return-for-e-coli-gates-and-attention)

## Update an existing workspace

From the repository root on your computer, preview the update for the folder you already have. Use the same experiment option you used to create it.

```powershell
python -B execution/arcc/prepare.py encoder_swaps --update "../ARCC Encoder Swaps"
```

For the E. coli RNA extension, use this command.

```powershell
python -B execution/arcc/prepare.py encoder_swaps --experiment ecoli_ernie --update "../ARCC E. coli RNA Extension"
```

After reviewing the preview, add `--apply` to update the local folder. Then sync that folder to its Newton workspace and verify the upload. [Sync an update](../README.md#update-an-existing-workspace)
