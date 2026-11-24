# Running experiments on Newton

We keep the Stage 4 setup here so that anyone on the team can run an experiment on Newton under comparable conditions. We use the same Python package versions and check that Newton assigns each training job the same GPU model (NVIDIA H100 PCIe).

You prepare an upload folder on your computer and copy it to a workspace on Newton. The jobs fill the workspace’s output directories. You bring the files the notebook needs back to your computer for review.

An easy way to follow the commands is to keep the guide for the experiment you’re running open in VS Code or another Markdown reader alongside your Newton terminal.

- [Grouped stability](grouped_stability/README.md)
- [Encoder swaps](encoder_swaps/README.md)
- [Fusion mechanism experiments](fusion_studies/README.md)
- [RPI2241 partner-task fits](partner_tasks/README.md)

Already uploaded the workspace? [Start a new Newton session](#enter-the-uploaded-workspace) to use it. If you’ve changed the files, go to [Update an existing workspace](#update-an-existing-workspace).

## Prepare on your computer

On your computer, open a terminal at the top level of the repository, such as the PowerShell terminal in VS Code on Windows. Copy the preparation command from the README in the corresponding ARCC experiment folder linked above and run it there. The shared script [prepare.py](prepare.py) reads the experiment’s *recipe*, which lists the files to copy and where each belongs in the upload folder. These can include notebooks, shared Python modules, Slurm scripts, batch lists and data files bundled with the experiment.

Include `--destination` with a new path beside the repository to create the **local upload folder**. Omit it if you want a preview first. The preview shows the file count, total size and a summary of what’s included. Add `--json` to either mode when you need the full file list.

Preparing or previewing the upload prints the external inputs you’ll need on Newton and commands to check them. `prepare.py` gets that checklist from a local helper. [Python source for the checklist (input_guidance.py)](input_guidance.py)

The local upload folder includes an empty `outputs/` structure for logs and executed notebooks. Keep the path handy for the [upload step](#upload-to-newton). Later, use the experiment’s return instructions and the notebook’s result-path setting to place the finished files.

## Prepare on Newton

If you’re new to Newton, request ARCC access with your UCF email and complete the orientation. Ask your lab which Slurm allocation to use for the project. [ARCC account setup](https://arcc.ist.ucf.edu/docs/access/)

Off the UCF network, connect to the UCF VPN using Cisco Secure Client before using SSH or WinSCP. Sign in to `newton.ist.ucf.edu` with your UCF NID. The experiment’s README gives the **Newton workspace** path where you’ll upload the prepared files and where jobs write by default. [UCF login guide](https://arcc.ist.ucf.edu/docs/access/windows/)

Check your storage quota before transferring files.

```bash
lfs quota -h -u "$USER" /lustre/fs1
```

Use the experiment’s README and the preparation checklist to identify which embedding caches the run needs. Reuse a copy you can read on Newton or transfer the required benchmark and encoder directories with their metadata. If you need to regenerate a BioLangFusion cache, follow its extraction notebook. [BioLangFusion token embedding extraction](../../Code/stage3/blf_token_embeddings.ipynb)

The RPI2241 partner-task recipe uses a built pair table and two per-entity caches supplied separately; they have a different layout from BioLangFusion caches. Follow its [exact input list](partner_tasks/README.md#get-the-prepared-inputs-from-julian) and retain the original folder names and metadata.

Set the input paths from the experiment’s README and run the preview’s file-access checks from the Newton account that will submit the jobs. Run checks that call an uploaded script after the workspace and environment are ready. If the run generates new inputs on Newton, follow the job order in its guide. [Verify the upload and required inputs](#verify-the-upload-and-required-inputs)

## Upload to Newton

1. In WinSCP, open **New Site**. Choose **SFTP**, enter `newton.ist.ucf.edu` as the host, use port **22** and enter your UCF NID as the username. Complete the sign-in prompts. You can save the site for later transfers. [WinSCP login settings](https://winscp.net/eng/docs/ui_login)
2. Open the [local upload folder](#prepare-on-your-computer) in the left pane and the [Newton workspace](#prepare-on-newton) in the right pane. Create the remote folder for a first upload. Select the **contents** of the local folder and click **Upload** so the launcher, `requirements.txt` and `source-snapshot.json` land at the top of the Newton workspace. If you’ve updated the local folder, leave its `.arcc-history/` on your computer.
3. In the transfer dialog, choose **Transfer settings → Transfer mode → Binary** to preserve the file bytes. Use the same setting when you download or synchronize. For large cache transfers, **Transfer in background** lets you queue the remaining folders. Under **Preferences → Transfer → Endurance**, transfer resume with temporary filenames lets WinSCP continue an interrupted SFTP copy. Resume against the same source and destination. [Transfer settings](https://winscp.net/eng/docs/ui_transfer_custom) · [Transfer queue](https://winscp.net/eng/docs/transfer_queue) · [Resume settings](https://winscp.net/eng/docs/ui_pref_resume)

If a required cache is missing, upload it separately to the input location identified during Newton preparation. When refreshing a workspace you’ve already uploaded, use the [one-way synchronization steps](#update-an-existing-workspace).

## Set up your Newton session

Start here once the [experiment folder is on Newton](#upload-to-newton). In each new SSH session, enter that workspace, activate Conda and select the Slurm allocation. If you’re creating the environment for the first time, install the packages from the workspace’s `requirements.txt`.

### Enter the uploaded workspace

In your Newton terminal, copy the `ARCC_PROJECT_ROOT` assignment from the experiment’s guide. Then enter the workspace and check that it contains the uploaded package list. The path printed by `pwd` should be the Newton workspace for the experiment you intend to run.

```bash
cd "${ARCC_PROJECT_ROOT:?Set the Newton workspace path from the experiment guide}" &&
  pwd && ls -l requirements.txt
```

In the same shell, set the module, environment, kernel and GPU names used by the launch scripts.

```bash
export ARCC_ANACONDA_MODULE=anaconda/anaconda-2024.10
export ARCC_CONDA_ENV="$HOME/.conda/envs/biofusion-arcc"
export ARCC_KERNEL=biofusion-arcc
export ARCC_EXPECTED_GPU="NVIDIA H100 PCIe"
```

If `biofusion-arcc` already exists, go to [Activate an existing environment](#activate-an-existing-environment). Otherwise, create it here.

### Create the Conda environment on first use

Load Conda, create the environment with Python 3.11.18, then activate it.

```bash
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create --prefix "$ARCC_CONDA_ENV" python=3.11.18 pip
conda activate "$ARCC_CONDA_ENV"
```

Once activation succeeds, install the pinned packages from the `requirements.txt` you checked in the workspace. Those versions include the Linux x86-64 PyTorch build with CUDA 12.6. [Package versions](requirements.txt) · [Recorded Newton environment](../../Code/stage4/results/newton_finished_20261119/provenance/studies/outputs/logs/environment-1220728.txt)

```bash
python -m pip install -r requirements.txt
```

Register the new environment as a notebook kernel so the launcher uses the same Python installation.

```bash
python -m ipykernel install --user --name "$ARCC_KERNEL"
```

Continue to [Check Python and package versions](#check-python-and-package-versions).

### Activate an existing environment

After entering the workspace and setting the variables in your new SSH session, load the Anaconda module, initialize Conda, and activate your existing environment.

```bash
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARCC_CONDA_ENV"
```

### Check Python and package versions

With the Conda environment active in the Newton workspace, check which Python the shell is using. The first command should point inside `ARCC_CONDA_ENV`. The second should print Python 3.11.18. If you see Newton’s system Python, return to the activation step before running the comparison.

```bash
command -v python
python --version
```

On first use, after changing the pinned versions or when troubleshooting, compare the active environment’s packages with the workspace’s `requirements.txt`. An empty diff means the versions match.

```bash
diff -u <(grep -E '^[[:alnum:]].*==' "$ARCC_PROJECT_ROOT/requirements.txt" | LC_ALL=C sort) \
        <(python -m pip freeze | LC_ALL=C sort)
```

After installing or changing packages, check their dependencies.

```bash
python -m pip check
```

### Choose the Slurm allocation

On first use or when switching allocations, check the **Account** column for the lab allocation linked to your Newton login.

```bash
sacctmgr -P show assoc where cluster=newton user="$USER" format=Account,Partition,QOS
```

If the lab’s account is missing, ask your lab or ARCC to confirm your access. In each session you’ll submit from, replace `LAB_ACCOUNT` with the account you’ll use.

```bash
export ARCC_ACCOUNT=LAB_ACCOUNT
```

The submission command passes this value to Slurm with `--account` to select which allocation the job uses. Slurm uses your configured default when the account is omitted. [Slurm submission options](https://slurm.schedmd.com/sbatch.html) · [Slurm default accounts](https://slurm.schedmd.com/sacctmgr.html)

To check usage, run this command and confirm the account named in the output.

```bash
myusage
```

## Check and submit a job

### Verify the upload and required inputs

With your environment active, run the experiment’s `--verify-only` command to check the uploaded files against `source-snapshot.json`. Run the external-input checks for your run. A sample cache check shows whether you can read those files. The training run checks the identity of the embeddings it uses.

### Run GPU preflight and submit

Submit the GPU preflight from the experiment’s README. You can queue training while the preflight is pending or running. The recipe’s `afterok` dependency keeps training queued until the preflight succeeds. If Slurm has already dropped the preflight job from its controller, check the preflight log for the expected GPU and verify `COMPLETED` with exit code `0:0` in job history before submitting without the dependency option. [Slurm job dependencies](https://slurm.schedmd.com/sbatch.html)

Save the job ID for [monitoring](#monitor-or-retry) and the output path for [downloading results](#download-results).

## Monitor or retry

Use the command that answers the question you have. For commands that reference a single job, replace `JOB_ID` with its actual Slurm ID and set the variable once in your terminal.

```bash
job_id=JOB_ID
```

### What is running or waiting?

Check current jobs, including why a job is pending. [Slurm squeue](https://slurm.schedmd.com/squeue.html)

```bash
squeue -u "$USER" -o "%.10i %.25j %.12T %.12M %R"
```

### When might a queued job start?

Slurm may show an estimated start time for a pending job. The estimate can change. [Slurm start estimates](https://slurm.schedmd.com/squeue.html)

```bash
squeue --start -j "$job_id"
```

### How did jobs finish?

Check recent jobs for completed, cancelled, failed or timed-out runs. For an earlier batch, replace `today` with its submission date in YYYY-MM-DD format. [Slurm sacct](https://slurm.schedmd.com/sacct.html)

```bash
sacct -X -u "$USER" -S today --format=JobID,JobName%25,State%24,Elapsed,ExitCode
```

### Where are the logs?

Find the paths Slurm set for standard output and standard error.

```bash
scontrol show job "$job_id" | grep -E 'StdOut=|StdErr='
```

For older jobs, look for the job ID in the workspace’s `outputs/logs/` if Slurm no longer shows the paths. Copy the output or error path in place of `LOG_PATH` to follow the log. Press Ctrl+C when you’re done watching it.

```bash
tail -n 40 -f -- "LOG_PATH"
```

### Is a running job using its resources?

Check CPU time, peak memory and disk activity for the batch step. [Slurm sstat](https://slurm.schedmd.com/sstat.html)

```bash
sstat -j "${job_id}.batch" --format=JobID,AveCPU,MaxRSS,MaxDiskRead,MaxDiskWrite
```

Use this command to inspect the GPU allocated to a running GPU job.

```bash
srun --jobid="$job_id" --overlap -N1 -n1 -c1 nvidia-smi
```

Compare the completed fit count with the expected count for the seed in the experiment’s guide. If a prerequisite fails, start with its error log. Keep completed fits and checkpoints. Use the new job ID when submitting jobs that depend on a retry.

## Download results

1. Use the [same WinSCP connection](#upload-to-newton). In the right pane, open the run’s output directory, usually under `outputs/` in the Newton workspace. In the left pane, open the local result path set in the notebook. If you’re keeping an earlier run, use a new local path and point the notebook to it.
2. In the right pane, select the files and subfolders listed in the experiment’s return instructions and click **Download** in binary mode. Copy the results directly, keeping the paths the notebook expects. Include the fit records, any split or diagnostic files the notebook reads, and the launch records for those jobs. Leave the embedding caches and model weights on Newton unless the experiment’s guide calls for them.
3. Keep the matching source snapshot and verify the recorded source and input hashes. Use the local snapshot and source files when they match the launch records. In the notebook, choose the downloaded result folder and use its saved-result settings to view the scores or recalculate a comparison table. [Shared result records](../../Code/stage4/RUNBOOK.md#select-a-returned-result-set)

For repeated downloads from the same run, you can synchronize the required result subfolders. In **Commands → Synchronize**, choose **Local/Left** as the destination and **Synchronize files** as the mode. Use binary transfer, turn **Preview changes** on and leave **Delete files** off. Review the list before copying. If you keep a local copy under the upload folder’s `outputs/`, follow the experiment’s return layout and point the notebook there if it supports an alternate result path. [WinSCP synchronization](https://winscp.net/eng/docs/ui_synchronize)

## Update an existing workspace

After editing the repository, preview the changes with the experiment’s `--update` command. Review the files it lists and save the **Local snapshot SHA-256**. Add `--apply` to update the local upload folder. If the updater reports an unsupported snapshot format or a different experiment, prepare a fresh upload folder and Newton workspace. [Update script](package_update.py)

1. Let jobs using that Newton workspace finish, including queued jobs. Activate the environment and run the workspace’s existing package check (`--verify-only`). Run this command from the Newton workspace and compare its output with the local snapshot hash you saved before applying the update.

   ```bash
   sha256sum source-snapshot.json
   ```

2. In WinSCP, put the updated local **upload folder** on the left and that experiment’s existing Newton workspace on the right. Choose **Commands → Synchronize**, direction **Remote/Right**, and mode **Synchronize files**. Turn **Preview changes** on. Leave **Delete files** and **Existing files only** off so the sync can add new files and preserve unrelated ones. Under **Transfer settings**, set the transfer mode to **Binary** and paste the exclusion mask below into **File mask**. [WinSCP synchronization](https://winscp.net/eng/docs/ui_synchronize) · [File masks](https://winscp.net/eng/docs/file_mask)

   ```text
   | .arcc-history/; outputs/; pilot_outputs/; embeddings/; extension_embeddings/; token_caches/; inputs/; checkpoints/; __pycache__/; *.npz; *.pt; *.safetensors
   ```

3. In the WinSCP preview, confirm the added and changed files match the local update preview, including `source-snapshot.json`. If the comparison skips a changed file because of its timestamp, upload that file in binary mode. Check any new path that already exists on Newton before overwriting it. Move only explicitly retired, verified source files out of the remote workspace. [WinSCP comparison criteria](https://winscp.net/eng/docs/ui_synchronize#comparison_criteria)
4. After syncing, repeat the hash command and compare it with the new local snapshot hash. Run the updated package check and verify any new inputs before submitting. If the transfer stops partway, finish syncing that version first. The local `.arcc-history/` holds earlier versions.
