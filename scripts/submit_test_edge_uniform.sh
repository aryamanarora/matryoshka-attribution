#!/bin/bash
# Edge-level uniform-k hard-fwd, TRAIN on train, EVAL on TEST. Run on sc in tmux.
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
  nlprun -g 1 -q jag -d a6000 $res -n "teu-${task}-${model}" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split test --train-split train --batch-size 5 --eval-examples 0 --output results/test_edge_hard_topk_uniform"
  sleep 1
done
echo "ALL TEST-EDGE-UNIFORM JOBS SUBMITTED"
