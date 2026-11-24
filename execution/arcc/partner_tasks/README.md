# RPI2241 partner-task fits on Newton

This recipe runs a fresh RPI2241 comparison on Newton: six arms for each of three held-out-side splits and three training seeds, giving **54 fits in nine training jobs**. Each job fits the six arms for one split and seed. Julian's existing results remain separate; they are not copied into this workspace or counted as Newton completions. The recipe uses the existing pair-task methods without changing the table, grouping, splits, architecture or training settings.

Follow the [shared Newton guide](../README.md) for account access, binary file transfer, Conda, Slurm allocation and monitoring. This page gives the experiment-specific inputs and commands. The [partner-task notebook](../../../Code/stage4/partner_tasks.ipynb) explains the experiment; use the dedicated launcher below for this batch, rather than running its local continuation and analysis cells. Repository-relative links in a copied README refer to the maintained checkout.

## Get the prepared inputs from Julian

Ask for the complete built RPI2241 folder and both complete per-entity cache folders below. These are existing inputs to copy, not a request to rebuild the table or extract new embeddings. Keep original filenames and bytes, including metadata and any accompanying manifests.

| Folder in Julian's checkout | What to obtain | Where it goes in the new upload folder |
| --- | --- | --- |
| `Code/stage4/data/pairs/rpi2241/` | The whole folder: `pairs.csv`, `rna.fasta`, `protein.fasta`, `dataset.json`, `groups.json` when present in the original built version, and any accompanying provenance files. | `inputs/rpi2241/` |
| `Code/stage4/cache/pair_entities/rpi2241/left__rna__rinalmo/` | `entities.json` and every `shard-*.npz`, plus any accompanying manifest. | `inputs/pair_entities/rpi2241/left__rna__rinalmo/` |
| `Code/stage4/cache/pair_entities/rpi2241/right__protein__esm2_150m/` | `entities.json` and every `shard-*.npz`, plus any accompanying manifest. | `inputs/pair_entities/rpi2241/right__protein__esm2_150m/` |

The expected table has 4,479 pairs: 2,238 measured positives and 2,241 supplied sampled negatives, with 968 RNA and 3,582 protein identifiers. It supplies pair labels, RNA/protein strings, entity names, group assignments and construction metadata. The caches supply one token array per entity; repeated appearances of an entity in different pairs reuse that array. They are not the row-indexed BioLangFusion caches. The reader restores the table through [pair_sources.py](../../../Code/shared_code/fusion/pair_sources.py); [entity_embeddings.py](../../../Code/shared_code/fusion/entity_embeddings.py) defines the cache names, metadata and shard layout. The packaged [input identity](input-identity.json) records the expected fingerprints.

If Julian already has transfer checksums, obtain those too. A `SHA256SUMS` list covering the table, metadata and shards gives an additional byte-for-byte transfer check; creating a special checksum package is not a prerequisite for sending the full folders. Keep any checksum list's relative paths intact when checking it. The launcher checks declared input identity and cache completeness separately.

No pretrained RiNALMo/ESM-2 weights are needed when these complete caches are staged. Do not copy Julian's `results/partner_tasks/`, old fit JSONs, trained heads, notebook outputs or checkpoints into the new result directory. The small `input-identity.json` supplied by the recipe records expected input fingerprints and fixed settings; it is not a set of fitted results. If the inputs fail verification, resolve the missing or different files before fitting instead of substituting another table, encoder or extraction.

## Prepare and stage the local upload folder

On your computer, open a terminal at the repository root. Preview the source package by omitting `--destination`, or create a new upload folder beside the repository:

```powershell
python -B execution/arcc/prepare.py partner_tasks --destination "../ARCC RPI2241 Newton"
```

Preparation copies the launcher, Slurm wrapper, declared settings and maintained shared Python source. It does not bundle Julian's inputs or any old fit outputs. After the upload folder exists, copy the three input folders into it using the destinations above. The resulting layout is:

```text
ARCC RPI2241 Newton/
  run_partner_tasks.py
  newton_partner_tasks.sbatch
  batches.tsv
  input-identity.json
  requirements.txt
  source-snapshot.json
  Code/shared_code/
  inputs/
    rpi2241/
      pairs.csv
      rna.fasta
      protein.fasta
      dataset.json
      groups.json                 if supplied by the original table version
    pair_entities/rpi2241/
      left__rna__rinalmo/
        entities.json
        shard-*.npz
      right__protein__esm2_150m/
        entities.json
        shard-*.npz
  outputs/
    logs/
    rpi2241-newton/
```

Preserve the full source folders, including files not expanded in this tree. Do not rename cache folders or rebuild group assignments. The frozen comparison expects the same table, entity identities and split membership as the declared RPI2241 inputs.

## Upload and verify on Newton

Upload the folder's contents in binary mode to a new workspace, keeping the layout above. The example destination is separate from the completed swap and mechanism-study workspaces:

```bash
export ARCC_PROJECT_ROOT="$HOME/multi-modal-fusion/partner-tasks-rpi2241-newton"
```

Follow [Set up your Newton session](../README.md#set-up-your-newton-session), using the shared `ARCC_ANACONDA_MODULE`, `ARCC_CONDA_ENV` and `ARCC_EXPECTED_GPU` settings, and enter this workspace. Check the uploaded source against its snapshot:

```bash
cd "$ARCC_PROJECT_ROOT"
python -B run_partner_tasks.py --verify-only
```

Source verification does not establish that the separately transferred inputs are complete. Run input verification through the Slurm wrapper on a compute node before training. Replace `PARTITION`, `QOS`, `CPUS`, `MEMORY` and `WALLTIME` below with the approved values for that job; memory needs units. Set the account through the shared guide. No walltime, memory requirement or permitted concurrency is inferred from a different experiment's resource use.

```bash
mkdir -p outputs/logs
job_options=(--account="${ARCC_ACCOUNT:?}" --partition=PARTITION --qos=QOS
  --cpus-per-task=CPUS --mem=MEMORY --time=WALLTIME)
PREFLIGHT_JOB=$(sbatch --parsable "${job_options[@]}" \
  newton_partner_tasks.sbatch --verify-inputs)
PREFLIGHT_JOB=${PREFLIGHT_JOB%%;*}
echo "$PREFLIGHT_JOB"
```

The equivalent Python input check inside that allocation is `python -B run_partner_tasks.py --verify-inputs`. It validates the table, recorded splits, entity metadata and required shards, and hashes the staged inputs. It requires no CUDA GPU, loads no pretrained encoders and starts no extraction or fitting. Keep its logs and confirm successful completion. This verification job is separate from the nine training jobs; the GPU check and creation of the cohort manifest happen when training starts.

## Submit the nine training jobs

All jobs use split seed 42 and the existing 70/15/15 grouping rule. Training seeds 0, 1 and 2 change the fit's random seed while leaving each split's membership fixed.

| Split argument | Held-out entities | Training seeds | Fits |
| --- | --- | --- | --- |
| `left` | RNA groups | `0`, `1`, `2` | 18 |
| `right` | Protein groups | `0`, `1`, `2` | 18 |
| `both` | RNA and protein groups | `0`, `1`, `2` | 18 |

Each split/seed job runs `left_only`, `right_only`, `concat`, `bilinear`, `cross_attention` and `interleaved_mamba`. Optional gate variants and reversed Mamba order are excluded. The existing cross-attention arm uses its declared default without an added gate; the Mamba arm retains its declared order and architecture. These are the six keys defined in [pair_study.py](../../../Code/shared_code/fusion/pair_study.py), using [pair_models.py](../../../Code/shared_code/fusion/pair_models.py).

The uploaded `batches.tsv` lists the nine combinations. Set approved training resources in `job_options`, then submit each split/seed combination. This example submits only `left`, seed `0`; change those two arguments for the other eight jobs. Each training job requests one H100, and the launcher checks the actual allocated GPU. Keep simultaneous jobs within the agreed allocation limit.

```bash
sbatch "${job_options[@]}" --gres=gpu:nvidia_h100_pcie:1 \
  --dependency="afterok:${PREFLIGHT_JOB:?}" \
  newton_partner_tasks.sbatch --split left --seed 0
```

The first training invocation creates `outputs/rpi2241-newton/run-inputs.json`, binding the source snapshot, staged-file hashes, all three splits and observed environment for the nine jobs. Later jobs must match it. A retry can reuse completed arms from this Newton cohort; historical fits without that cohort link are refused.

The fixed fit settings are AdamW with learning rate `0.001` and weight decay `0.00001`, batch size 32 accumulated from microbatches of 8, at most 50 epochs, patience 5, training-only balanced positive-class weighting and validation AUPRC selection. Length bucketing stays off. The requested token cap is 1,024; the supplied RiNALMo and ESM-2 caches retain at most 1,022 content tokens after encoder framing tokens. Keep the supplied cache metadata's truncation counts with the evidence. [PairConfig and training loop](../../../Code/shared_code/fusion/pair_training.py)

The split's groups currently represent exact sequence identity. Three training seeds do not add similarity clustering or change the generalization claim. The dual-disjoint split retains only pairs whose two entity groups fall in the same partition, so its smaller test set remains a limitation. Statistical comparison and interpretation follow separately; this training recipe does not run the notebook's paired bootstrap.

## Retain and return the new evidence

Use the shared guide to [monitor the jobs](../README.md#monitor-or-retry). Each fit is saved at `outputs/rpi2241-newton/rpi2241/held_out_<split>/<arm>__seed<seed>.json`. Scheduler logs are `outputs/logs/partner-<job-id>.out` and `.err`; the package environment is saved as `outputs/logs/environment-<job-id>.txt`. Expect six completed fit records per split/seed and 54 across the full grid; a submission, job count or an old local fit is not evidence that the new grid completed.

Return the whole `outputs/rpi2241-newton/` directory, including `run-inputs.json`, together with the relevant launch/environment and scheduler logs, `input-identity.json`, `source-snapshot.json` and the matching source package. Preserve the original JSON bytes and directory layout. Keep the staged table and cache manifests available with their source/transfer evidence; unchanged cache shards need not be downloaded again when a verified local copy exists. Place this cohort in a separate local destination rather than overlaying Julian's historical results. Use a separately scoped saved-prediction analysis to compare or aggregate the returned fits. This recipe supports the RNA–protein comparison in `Gantt`/`Details` task 47; preparation does not mark that research task complete.
