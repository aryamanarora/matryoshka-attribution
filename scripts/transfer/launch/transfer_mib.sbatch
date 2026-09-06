#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/xfer_%x_%j.out
#SBATCH --error=logs/xfer_%x_%j.err
# Thin runner: everything after the script name is forwarded to eval_transfer_mib.py.
# usage: sbatch -J xfer_mib_ioi scripts/transfer/launch/transfer_mib.sbatch --model llama3 --task ioi --sources ...
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python scripts/transfer/eval_transfer_mib.py "$@"
