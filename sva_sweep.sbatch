#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --output=logs/sva_%x_%j.out
#SBATCH --error=logs/sva_%x_%j.err
# Thin runner: everything after the script name is forwarded to eval_sva.py.
# usage: sbatch -J <name> sva_sweep.sbatch --task nounpp --nodes mlp --method ig --loss ce ...
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python scripts/eval_sva.py "$@"
