#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mib_%x_%j.out
#SBATCH --error=logs/mib_%x_%j.err
# usage: sbatch -J <name> mib_node_seed.sbatch <model> <task> <masking> <kschedule> <outdir> <opt> <seed> <bs>
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$1; TASK=$2; MASK=$3; KS=$4; OUT=$5; OPT=${6:-adam}; SEED=${7:-42}; BS=${8:-20}
uv run python scripts/eval_mib.py --model "$MODEL" --task "$TASK" --steps 500 \
  --k-schedule "$KS" --masking "$MASK" --mode sufficient --optimizer "$OPT" \
  --seed "$SEED" --split validation --train-split train --include-input --batch-size "$BS" \
  --output "results/$OUT"
