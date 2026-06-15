#!/bin/bash
# Edge-level uniform-k hard-fwd sweep (train->val): mirror of mib_edge_hard_topk but
# --k-schedule uniform. Output -> results/mib_edge_hard_topk_uniform. Run on sc in tmux.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy"
)

for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  case $model in
    gpt2|qwen2.5) res="-c 3 -r 64G";;
    gemma2)       res="-c 4 -r 96G";;
    llama3)       res="-c 5 -r 128G";;
  esac
  name="reu-eht-${task}-${model}"
  nlprun -g 1 -q jag -d a6000 $res -n "$name" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split validation --train-split train --batch-size 5 --eval-examples 0 --output results/mib_edge_hard_topk_uniform"
  sleep 1
done
echo "ALL UNIFORM-K EDGE JOBS SUBMITTED"
