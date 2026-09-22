#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs_arith/%x_%j.out
#SBATCH --error=logs_arith/%x_%j.err
# usage: sbatch -J arith-probes scripts/arith_formats/launch/probes.sbatch --model llama3.1-8b
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
uv run python scripts/arith_formats/probes.py "$@"
