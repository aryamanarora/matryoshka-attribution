#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/vit_%x_%j.out
#SBATCH --error=logs/vit_%x_%j.err
# Thin runner for the optimizer / LR / eps grid on the ViT teaser. Everything after the script
# name is forwarded to scripts/vit_teaser_optim.py.
#   for s in 42 43 44; do sbatch -J vito$s vit_optim.sbatch --seed $s --out results/vit_optim/seed$s; done
# Same venv caveat as vit_teaser.sbatch: .venv-vit (torchvision), NOT the project .venv.
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-vit/bin/python scripts/vit_teaser_optim.py "$@"
