#!/usr/bin/env bash
# Supply reviewed account, partition, CPU/RAM/time and GPU options to sbatch.
# This checks the actual notebook kernel; it does not execute a research notebook.
#SBATCH --job-name=fusion-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=outputs/logs/preflight-%j.out
#SBATCH --error=outputs/logs/preflight-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${ARCC_PROJECT_ROOT:?Absolute path to the new source-only study workspace}"
: "${ARCC_ANACONDA_MODULE:?Installed Newton module}"
: "${ARCC_CONDA_ENV:?Prepared environment}"
: "${ARCC_KERNEL:?Registered notebook kernel}"
: "${ARCC_EXPECTED_GPU:?Reviewed GPU-name substring}"
cd "$ARCC_PROJECT_ROOT"
[[ -d outputs/logs ]] || {
    echo "Create outputs/logs before submitting; Slurm and environment records use it." >&2
    exit 2
}
# Initialize inherited Conda before module loading tries to deactivate it.
if (( ${CONDA_SHLVL:-0} > 0 )); then
    ARCC_BOOTSTRAP_CONDA_BASE=$(conda info --base)
    source "$ARCC_BOOTSTRAP_CONDA_BASE/etc/profile.d/conda.sh"
fi
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARCC_CONDA_ENV"
printf 'Using Python: %s on %s\n' "$(command -v python)" "$(hostname)"
python -u run_fusion_study.py --verify-only
python -m pip freeze > "outputs/logs/environment-preflight-${SLURM_JOB_ID}.txt"
python -u run_fusion_study.py --preflight --kernel "$ARCC_KERNEL" \
    --expected-gpu "$ARCC_EXPECTED_GPU"
