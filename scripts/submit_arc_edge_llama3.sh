#!/bin/bash
# Fill the missing arc edge llama3 cells (val+test), 200-example subset like the
# other edge llama3 cells (-> daggered). sphinx h100.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for t in arc_easy arc_challenge; do
  nlprun -g 1 -q sphinx -d h100 -r 128G -n se-arc-${t}-val \
    "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split validation --train-split train --batch-size 2 --eval-examples 200 --output results/mib_edge_hard_topk_uniform"
  sleep 1
  nlprun -g 1 -q sphinx -d h100 -r 128G -n se-arc-${t}-tst \
    "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split test --train-split train --batch-size 2 --eval-examples 200 --output results/test_edge_hard_topk_uniform"
  sleep 1
done
echo "ARC EDGE LLAMA3 SUBMITTED"
