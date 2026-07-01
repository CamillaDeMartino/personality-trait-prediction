#!/bin/bash
#SBATCH --job-name=personality_smoke
#SBATCH --partition=low-gn
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

#SBATCH --output=outputs/logs/slurm/%x_%j.out
#SBATCH --error=outputs/logs/slurm/%x_%j.err

echo "========== Personality Smoke Test =========="

cd ~/projects/personality-trait-prediction

source .venv/bin/activate

python scripts/train_smoke_test.py

echo
echo "Finished: $(date)"