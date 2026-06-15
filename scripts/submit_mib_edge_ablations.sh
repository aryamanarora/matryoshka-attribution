#!/bin/bash
# Rerun edge-level L2A ablations train->val (honest). Run from repo root on sc in tmux.
# Edge is memory-heavy + slow (5000 steps, full eval). 3 ablations x 9 pairs = 27 jobs.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy"
)

# name|masking|output-dir|eval-batch-size
ABL=(
  "eht|hard_topk|mib_edge_hard_topk|5"
  "et|topk|final_edge|5"
  "edt|topk_detached|mib_edge_detached_tau|10"
)

for a in "${ABL[@]}"; do
  IFS='|' read -r an mask out bs <<< "$a"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) res="-c 3 -r 64G";;
      gemma2)       res="-c 4 -r 96G";;
      llama3)       res="-c 5 -r 128G";;
    esac
    name="ree-${an}-${task}-${model}"
    nlprun -g 1 -q jag -d a6000 $res -n "$name" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule log --masking $mask --mode sufficient --split validation --train-split train --batch-size $bs --eval-examples 0 --output results/$out"
    sleep 1
  done
done
echo "ALL EDGE ABLATION JOBS SUBMITTED"
