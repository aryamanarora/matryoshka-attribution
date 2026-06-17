#!/bin/bash
# Fill ALL remaining LR-sweep slots for the 10 non-gpt2 tasks (node, uniform-k, full eval).
# hard_topk: lr 0.005/0.05/0.1/0.3 ; bernoulli: lr 0.01/0.05/0.3 + lr0.1@2k.
# (bern lr0.1@500 is the main mib_node_bernoulli_reinforce rerun, not repeated here.)
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PAIRS=( "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction" \
        "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge" )
res_for() { case $1 in qwen2.5) echo "-c 2 -r 32G|";; gemma2) echo "-c 3 -r 64G|--batch-size 4";; llama3) echo "-c 4 -r 96G|--batch-size 2";; esac; }
launch() { # mask lr steps outdir tag
  local mask=$1 lr=$2 steps=$3 out=$4 tag=$5
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"; IFS='|' read -r res bs <<< "$(res_for $model)"
    nlprun -g 1 -q jag -d a6000 $res -n "${tag}-${task}-${model}" \
      "$EXP uv run python scripts/eval_mib.py --model $model --task $task --steps $steps --k-schedule uniform --masking $mask --mode sufficient --lr $lr --split validation --train-split train --include-input $bs --output results/$out"
    sleep 1
  done
}
for lr in 0.005 0.05 0.1 0.3; do launch hard_topk $lr 500 "htk_lr_$lr" "h$lr"; done
for lr in 0.01 0.05 0.3;       do launch bernoulli_reinforce $lr 500 "bern_lr_$lr" "b$lr"; done
launch bernoulli_reinforce 0.1 2000 "bern_lr_0.1_2k" "b2k"
echo "LR SWEEP (all tasks) SUBMITTED"
