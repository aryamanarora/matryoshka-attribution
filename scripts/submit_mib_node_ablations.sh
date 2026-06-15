#!/bin/bash
# Rerun all node-level L2A ablations train->val (honest: train on train, eval on validation).
# Run from repo root on sc inside tmux.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

# name|masking|k-schedule|output-dir
ABL=(
  "htlog|hard_topk|log|mib_node_hard_topk_log"
  "tlog|topk|log|mib_node_topk_log"
  "dtlog|topk_detached|log|mib_node_detached_tau_log"
  "brlog|bernoulli_reinforce|log|mib_node_bernoulli_reinforce_log"
  "htuni|hard_topk|uniform|mib_node_hard_topk"
  "tuni|topk|uniform|final_node"
  "dtuni|topk_detached|uniform|mib_node_detached_tau"
  "bruni|bernoulli_reinforce|uniform|mib_node_bernoulli_reinforce"
)

for a in "${ABL[@]}"; do
  IFS='|' read -r an mask ks out <<< "$a"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) res="-c 2 -r 32G"; bs="";;
      gemma2)       res="-c 3 -r 64G"; bs="--batch-size 4";;
      llama3)       res="-c 4 -r 96G"; bs="--batch-size 2";;
    esac
    name="re-${an}-${task}-${model}"
    nlprun -g 1 -q jag -d a6000 $res -n "$name" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule $ks --masking $mask --mode sufficient --split validation --train-split train --include-input $bs --output results/$out"
    sleep 1
  done
done
echo "ALL NODE ABLATION JOBS SUBMITTED"
