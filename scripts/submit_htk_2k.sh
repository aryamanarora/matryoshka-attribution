#!/bin/bash
# hard_topk at the default lr (0.01) but 2000 steps, across all 11 tasks -> htk_lr_0.01_2k.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PAIRS=( "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction" \
        "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge" )
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  case $model in gpt2|qwen2.5) res="-c 2 -r 32G"; bs="";; gemma2) res="-c 3 -r 64G"; bs="--batch-size 4";; llama3) res="-c 4 -r 96G"; bs="--batch-size 2";; esac
  nlprun -g 1 -q jag -d a6000 $res -n "h2k-${task}-${model}" \
    "$EXP uv run python scripts/eval_mib.py --model $model --task $task --steps 2000 --k-schedule uniform --masking hard_topk --mode sufficient --lr 0.01 --split validation --train-split train --include-input $bs --output results/htk_lr_0.01_2k"
  sleep 1
done
echo "HTK 2k SUBMITTED"
