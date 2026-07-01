#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mibedge_%x_%j.out
#SBATCH --error=logs/mibedge_%x_%j.err
# usage: sbatch -J <name> mib_edge.sbatch <model> <task> <masking> <kschedule> <outdir> <opt> <seed> <bs>
set -euo pipefail
cd ~/learning-to-attribute
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL=$1; TASK=$2; MASK=$3; KS=$4; OUT=$5; OPT=${6:-adam}; SEED=${7:-42}; BS=${8:-5}; EVAL=${9:-0}
uv run python scripts/eval_mib_edge.py --model "$MODEL" --task "$TASK" --steps 5000 \
  --k-schedule "$KS" --masking "$MASK" --mode sufficient --optimizer "$OPT" \
  --seed "$SEED" --split validation --train-split train --batch-size "$BS" \
  --eval-examples "$EVAL" --output "results/$OUT"
