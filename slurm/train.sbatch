#!/bin/bash
#SBATCH --job-name=personality_train
#SBATCH --partition=low-gn
#SBATCH --exclude=gn05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

#SBATCH --output=outputs/logs/slurm/%x_%j.out
#SBATCH --error=outputs/logs/slurm/%x_%j.err

set -euo pipefail

PROJECT_DIR="/home/c.demartino/projects/personality-trait-prediction"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Avoid CUDA stub libraries if present in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/share/libs/nvidia/cuda-12.8.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | awk '!/stubs/ && NF' | paste -sd: -)"

echo "========== SLURM INFO =========="
echo "Job ID      : ${SLURM_JOB_ID:-N/A}"
echo "Job name    : ${SLURM_JOB_NAME:-N/A}"
echo "Node        : $(hostname)"
echo "Partition   : ${SLURM_JOB_PARTITION:-N/A}"
echo "CPUs        : ${SLURM_CPUS_PER_TASK:-N/A}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Started     : $(date)"
echo

echo "========== NVIDIA-SMI =========="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not found or not available"
echo

echo "========== CHECK PYTORCH / CUDA =========="
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: CUDA is not available. Stop training to avoid CPU run.")
PY

echo
echo "========== START TRAINING =========="
python scripts/train.py

echo
echo "========== FINISHED =========="
echo "Finished: $(date)"