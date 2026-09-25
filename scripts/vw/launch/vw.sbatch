#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs_vw/%x_%j.out
#SBATCH --error=logs_vw/%x_%j.err
# Thin runner for the interference-weights replication. Everything after the sbatch script
# is forwarded verbatim:   sbatch -J vw_lr1e3 scripts/vw/launch/vw.sbatch scripts/vw/vw_train.py --lr 1e-3 ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Dispatch on the first argument so this one runner covers both the python entry points
# and the multi-stage shell drivers (scripts/vw_stage*.sh).
if [[ "$1" == *.sh ]]; then bash "$@"; else uv run python "$@"; fi
