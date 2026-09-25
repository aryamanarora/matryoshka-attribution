#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/vit_%x_%j.out
#SBATCH --error=logs/vit_%x_%j.err
# Thin runner for the AttnLRP-Fig-1-style ViT teaser. Everything after the script name is
# forwarded to scripts/vit/vit_teaser_attr.py.
#   sbatch -J vit_ld scripts/vit/launch/vit_teaser.sbatch --target logit_diff --out results/vit_teaser_logitdiff
# NOTE: runs in the `vit` uv dependency group (torchvision + lxt + zennit, transformers<5),
# synced into .venv-vit from uv.lock -- it conflicts with the default `lm` group, see pyproject.toml.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
UV_PROJECT_ENVIRONMENT=.venv-vit uv run --no-default-groups --group vit python scripts/vit/vit_teaser_attr.py "$@"
