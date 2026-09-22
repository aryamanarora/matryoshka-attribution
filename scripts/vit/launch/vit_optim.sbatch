#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/vit_%x_%j.out
#SBATCH --error=logs/vit_%x_%j.err
# Thin runner for the optimizer / LR / eps grid on the ViT teaser. Everything after the script
# name is forwarded to scripts/vit/vit_teaser_optim.py.
#   for s in 42 43 44; do sbatch -J vito$s scripts/vit/launch/vit_optim.sbatch --seed $s --out results/vit_optim/seed$s; done
# Same env caveat as scripts/vit/launch/vit_teaser.sbatch: the `vit` uv group in .venv-vit, not the default env.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UV_PROJECT_ENVIRONMENT=.venv-vit uv run --no-default-groups --group vit python scripts/vit/vit_teaser_optim.py "$@"
