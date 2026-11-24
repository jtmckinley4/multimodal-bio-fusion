#!/usr/bin/env bash
# Supply the account, partition, walltime, CPUs, RAM and (for training) one H100.
# Submit from the prepared partner workspace after creating outputs/logs.
#SBATCH --job-name=rpi2241
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=outputs/logs/partner-%j.out
#SBATCH --error=outputs/logs/partner-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Allocated Slurm job required}"
: "${ARCC_PROJECT_ROOT:?Absolute path to the prepared RPI2241 workspace}"
: "${ARCC_ANACONDA_MODULE:?Installed Newton Anaconda module}"
: "${ARCC_CONDA_ENV:?Path to the prepared Conda environment}"

cd "$ARCC_PROJECT_ROOT"
if (( ${CONDA_SHLVL:-0} > 0 )); then
    ARCC_BOOTSTRAP_CONDA_BASE=$(conda info --base)
    source "$ARCC_BOOTSTRAP_CONDA_BASE/etc/profile.d/conda.sh"
fi
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARCC_CONDA_ENV"

export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OPENBLAS_NUM_THREADS"
printf 'Using Python: %s on %s\n' "$(command -v python)" "$(hostname)"
python -m pip freeze > "outputs/logs/environment-${SLURM_JOB_ID}.txt"
python -B -u run_partner_tasks.py \
    --expected-gpu "${ARCC_EXPECTED_GPU:-NVIDIA H100 PCIe}" "$@"
