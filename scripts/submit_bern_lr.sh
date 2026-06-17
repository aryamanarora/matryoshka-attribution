#!/bin/bash
# Quick lr sweep for bernoulli_reinforce (+hard) on ioi/gpt2 to test under-training.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for lr in 0.05 0.1 0.3; do
  nlprun -g 1 -q jag -d a6000 -c 2 -r 32G -n "bern-lr$lr" \
    "$EXP uv run python scripts/eval_mib.py --model gpt2 --task ioi --steps 500 --k-schedule uniform --masking bernoulli_reinforce --mode sufficient --lr $lr --split validation --train-split train --include-input --output results/bern_lr_$lr"
  sleep 1
done
# also one with more steps at lr 0.1
nlprun -g 1 -q jag -d a6000 -c 2 -r 32G -n "bern-lr0.1-2k" \
  "$EXP uv run python scripts/eval_mib.py --model gpt2 --task ioi --steps 2000 --k-schedule uniform --masking bernoulli_reinforce --mode sufficient --lr 0.1 --split validation --train-split train --include-input --output results/bern_lr_0.1_2k"
echo "BERN LR SWEEP SUBMITTED"
