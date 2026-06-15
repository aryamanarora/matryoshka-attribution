#!/bin/bash
# Node-level uniform-k hard-fwd, TRAIN on train, EVAL on TEST. Run on sc in tmux.
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
  nlprun -g 1 -q jag -d a6000 $res -n "tnu-${task}-${model}" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform --masking hard_topk --mode sufficient --split test --train-split train --include-input $bs --output results/test_node_hard_topk_uniform"
  sleep 1
done
echo "ALL TEST-NODE-UNIFORM JOBS SUBMITTED"
