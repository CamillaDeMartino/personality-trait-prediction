#!/bin/bash
#SBATCH --job-name=personality_eval
#SBATCH --partition=low-gn
#SBATCH --exclude=gn05
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/c.demartino/projects/personality-trait-prediction/outputs/logs/slurm/personality_eval-%j.out
#SBATCH --error=/home/c.demartino/projects/personality-trait-prediction/outputs/logs/slurm/personality_eval-%j.err

set -euo pipefail

PROJECT_DIR="/home/c.demartino/projects/personality-trait-prediction"
VENV_DIR="$PROJECT_DIR/.venv"

cd "$PROJECT_DIR"
source "$VENV_DIR/bin/activate"

mkdir -p "$PROJECT_DIR/outputs/logs/slurm"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

export LD_LIBRARY_PATH=/opt/share/libs/nvidia/cuda-12.8.0/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | awk '!/stubs/ && NF' | paste -sd: -)"

echo "========== CHECK PYTORCH / CUDA =========="
python - <<'PY'
import torch
print("torch version:", torch.__version__)
print("torch cuda version:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: CUDA is not available.")
PY

echo "========== START EVALUATION =========="
python scripts/evaluate.py