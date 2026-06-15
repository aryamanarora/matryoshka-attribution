#!/bin/bash
# Node-level uniform-k + Gumbel-perturbed hard top-k sweep (train->val + learned input).
# Forward selection ranks scores + Gumbel(0,1); straight-through grad. Run on sc in tmux.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  case $model in
    gpt2|qwen2.5) res="-c 2 -r 32G"; bs="";;
    gemma2)       res="-c 3 -r 64G"; bs="--batch-size 4";;
    llama3)       res="-c 4 -r 96G"; bs="--batch-size 2";;
  esac
  name="reg-${task}-${model}"
  nlprun -g 1 -q jag -d a6000 $res -n "$name" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform --masking hard_topk_gumbel --mode sufficient --split validation --train-split train --include-input $bs --output results/mib_node_hard_topk_gumbel"
  sleep 1
done
echo "ALL GUMBEL NODE JOBS SUBMITTED"
