#!/usr/bin/env bash
# Source/input verification and actual allocated environment only; no research notebook.
# Supply account, partition, GPU class, CPU, RAM and time through approved sbatch options.
# Resource requirements must come from the selected workload, not this example.
#SBATCH --job-name=biofusion-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=outputs/logs/preflight-%j.out
#SBATCH --error=outputs/logs/preflight-%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Run this GPU preflight in an allocated Newton Slurm job}"
: "${ARCC_PROJECT_ROOT:?Set the absolute Newton path to the prepared workspace}"
: "${ARCC_ANACONDA_MODULE:?Set a module available on Newton}"
: "${ARCC_CONDA_ENV:?Set the prepared environment name or absolute path}"
: "${ARCC_EXPECTED_GPU:?Set the approved GPU-name substring for this preflight}"
cd "$ARCC_PROJECT_ROOT"
printf 'Preparing Conda environment on %s\n' "$(hostname)"
# Slurm can inherit an active environment without Conda's shell function.
# Initialize it before module loading can attempt to deactivate that environment.
if (( ${CONDA_SHLVL:-0} > 0 )); then
    ARCC_BOOTSTRAP_CONDA_BASE=$(conda info --base)
    source "$ARCC_BOOTSTRAP_CONDA_BASE/etc/profile.d/conda.sh"
fi
module load "$ARCC_ANACONDA_MODULE"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ARCC_CONDA_ENV"
printf 'Using Python: %s\n' "$(command -v python)"

# Verify the declared snapshot and inputs before inspecting the allocated GPU.
python -u run_encoder_swaps.py --verify-only
ARCC_OUTPUT_ROOT="${ARCC_OUTPUT_ROOT:-$ARCC_PROJECT_ROOT/outputs}"
[[ -d "$ARCC_OUTPUT_ROOT/logs" ]] || {
    echo "Prepare the selected output root's logs directory before submission." >&2; exit 2;
}
python - "$ARCC_EXPECTED_GPU" <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import sys
import numpy as np  # Preserve NumPy-before-PyTorch import order.
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; refusing CPU fallback for a GPU job.")
gpu = torch.cuda.get_device_name(0)
expected_gpu = sys.argv[1].strip()
if not expected_gpu or expected_gpu.casefold() not in gpu.casefold():
    raise RuntimeError(f"Allocated GPU {gpu!r} does not match {expected_gpu!r}.")
names = ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch",
         "transformers", "multimolecule", "nbconvert", "nbformat", "nbclient", "ipykernel"]
print(json.dumps({"python": sys.version, "platform": platform.platform(),
    "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "gpu": gpu, "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "packages": {name: metadata.version(name) for name in names}}, indent=2))
print("Environment observed only; no notebook, weights, embeddings or training were run.")
PY
python -m pip freeze > "$ARCC_OUTPUT_ROOT/logs/environment-${SLURM_JOB_ID}.txt"
