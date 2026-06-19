#!/bin/bash
# Re-run all +hard bwd (bernoulli_reinforce) node cells at lr 0.1 (was undertrained at 0.01).
# Node llama3 fits on jag a6000 (full eval), unlike edge. Overwrites the existing dirs.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PAIRS=( "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction" \
        "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge" )
# k-schedule -> output dir
for ks_out in "uniform mib_node_bernoulli_reinforce" "log mib_node_bernoulli_reinforce_log"; do
  read -r ks out <<< "$ks_out"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) res="-c 2 -r 32G"; bs="";;
      gemma2)       res="-c 3 -r 64G"; bs="--batch-size 4";;
      llama3)       res="-c 4 -r 96G"; bs="--batch-size 2";;
    esac
    nlprun -g 1 -q jag -d a6000 $res -n "b01-${ks:0:1}-${task}-${model}" \
      "$EXP uv run python scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule $ks --masking bernoulli_reinforce --mode sufficient --lr 0.1 --split validation --train-split train --include-input $bs --output results/$out"
    sleep 1
  done
done
echo "BERN lr0.1 RERUN SUBMITTED"
