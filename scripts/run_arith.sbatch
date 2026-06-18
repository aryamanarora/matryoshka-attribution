#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
# Usage: sbatch -J <name> -o logs_arith/<name>.out scripts/run_arith.sbatch <args...>
set -u
cd /home/guests/aryaman/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "ARGS: $@"
.venv/bin/python scripts/attribute_arith.py "$@"
