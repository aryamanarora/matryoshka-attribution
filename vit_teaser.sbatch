#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/vit_%x_%j.out
#SBATCH --error=logs/vit_%x_%j.err
# Thin runner for the AttnLRP-Fig-1-style ViT teaser. Everything after the script name is
# forwarded to scripts/vit_teaser_attr.py.
#   sbatch -J vit_ld vit_teaser.sbatch --target logit_diff --out results/vit_teaser_logitdiff
# NOTE: uses .venv-vit (torchvision + lxt + zennit), NOT the project .venv -- lxt pins
# transformers<5, which would break the rest of the repo.
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv-vit/bin/python scripts/vit_teaser_attr.py "$@"
