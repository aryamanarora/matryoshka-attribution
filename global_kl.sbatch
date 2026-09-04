#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/gkl_%x_%j.out
#SBATCH --error=logs/gkl_%x_%j.err
# Thin runner: everything after the script name is forwarded to eval_global_kl.py.
#   sbatch -J gkl_sgd global_kl.sbatch --method mattr --optimizer sgd
# .venv (TL-independent path: pure HF transformers) is fine here -- the CLAUDE.md Gemma-2
# forward bug is TransformerLens-only and this script never touches gemma2.
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python scripts/eval_global_kl.py "$@"
