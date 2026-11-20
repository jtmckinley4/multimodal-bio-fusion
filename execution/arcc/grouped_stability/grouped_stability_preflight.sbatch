#!/usr/bin/env bash
# Package verification and allocated environment only; no notebook, split or fits.
# Supply approved account, partition, GPU, CPU, RAM and time through sbatch options.
#SBATCH --job-name=grouped-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=outputs/logs/preflight-%j.out
#SBATCH --error=outputs/logs/preflight-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Use an allocated Newton Slurm job}"
: "${ARCC_PROJECT_ROOT:?Absolute path to the NEW grouped-stability workspace}"
: "${ARCC_ANACONDA_MODULE:?Module confirmed on Newton}"
: "${ARCC_CONDA_ENV:?Prepared environment name or absolute path}"
: "${ARCC_KERNEL:?Kernel registered in that environment}"
: "${ARCC_EXPECTED_GPU:?Expected GPU-name substring}"
cd "$ARCC_PROJECT_ROOT"
printf 'Preparing Conda environment on %s\n' "$(hostname)"
if (( ${CONDA_SHLVL:-0} > 0 )); then
    ARCC_BOOTSTRAP_CONDA_BASE=$(conda info --base)
    source "$ARCC_BOOTSTRAP_CONDA_BASE/etc/profile.d/conda.sh"
fi
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARCC_CONDA_ENV"
printf 'Using Python: %s\n' "$(command -v python)"
python -u run_grouped_stability.py --verify-only --kernel "$ARCC_KERNEL"
ARCC_OUTPUT_ROOT="${ARCC_OUTPUT_ROOT:-$ARCC_PROJECT_ROOT/outputs}"
[[ -d "$ARCC_OUTPUT_ROOT/logs" ]] || {
    echo "Prepare the selected logs directory before submission." >&2; exit 2;
}
python - "$ARCC_EXPECTED_GPU" <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import sys
import numpy as np  # Import before PyTorch on supported environments.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable; refusing CPU fallback.")
gpu = torch.cuda.get_device_name(0)
expected = sys.argv[1].strip()
if not expected or expected.casefold() not in gpu.casefold():
    raise RuntimeError(f"Observed GPU {gpu!r} differs from requested {expected!r}.")
names = ("numpy", "pandas", "scipy", "scikit-learn", "torch", "nbconvert", "nbclient", "ipykernel")
print(json.dumps({"python": sys.version, "platform": platform.platform(),
    "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": gpu, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
    "packages": {name: metadata.version(name) for name in names}}, indent=2))
print("Environment observed only; no notebook, split generation, cache arrays or training ran.")
PY
python -m pip freeze > "$ARCC_OUTPUT_ROOT/logs/environment-preflight-${SLURM_JOB_ID}.txt"
