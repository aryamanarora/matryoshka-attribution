#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs_arith/%x_%j.out
#SBATCH --error=logs_arith/%x_%j.err
# Runs one model per GPU, in parallel, inside a single allocation (the per-user QOS caps
# NODES, not GPUs, so packing several single-GPU runs into one job is what keeps the queue
# moving).  Usage:
#   sbatch -J arith --gres=gpu:3 --mem=192G scripts/arith_formats/launch/arith.sbatch qwen3-0.6b llama3.2-1b gemma2-2b
#   ARGS="--stage behav" sbatch ... (extra args forwarded to run_model.py)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"   # submit from the repo root
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
ARGS=${ARGS:-}
i=0
pids=()
for m in "$@"; do
  echo "[$(date)] launching $m on GPU $i"
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/arith_formats/run_model.py --model "$m" $ARGS \
    > "logs_arith/${m}_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!)
  i=$((i + 1))
done
rc=0
for p in "${pids[@]}"; do
  wait "$p" || rc=1
done
echo "[$(date)] done rc=$rc"
exit $rc
