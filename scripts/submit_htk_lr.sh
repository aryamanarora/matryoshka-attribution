#!/bin/bash
# lr sweep for hard_topk (MAttr main) on ioi/gpt2 to check if lr=0.01 is optimal.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for lr in 0.005 0.05 0.1 0.3; do
  nlprun -g 1 -q jag -d a6000 -c 2 -r 32G -n "htk-lr$lr" \
    "$EXP uv run python scripts/eval_mib.py --model gpt2 --task ioi --steps 500 --k-schedule uniform --masking hard_topk --mode sufficient --lr $lr --split validation --train-split train --include-input --output results/htk_lr_$lr"
  sleep 1
done
echo "HTK LR SWEEP SUBMITTED"
