#!/bin/bash
#SBATCH --job-name=hw6-cifar
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --constraint=work
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Supply your GPU allocation at submission: sbatch --account=YOUR_PROJECT-gpu ...
# Run from the transferred project directory. Create logs/ BEFORE sbatch.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the project directory}"
module reset
module load pytorch-conda/2.8
module list
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Optional overlay environment, created in the guide if the site module lacks packages.
if [[ -x .venv/bin/python ]]; then
    PYTHON="$PWD/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi
RUN_DIR="${1:-runs/full}"
nvidia-smi
srun "$PYTHON" check_environment.py --gpu
# prepare is performed beforehand, without holding an idle GPU during download.
srun "$PYTHON" -u experiment.py train --run "$RUN_DIR" --device cuda
# Deliberately no finalize here. Freeze the complete run before opening test scores.
