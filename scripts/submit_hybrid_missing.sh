#!/bin/bash
# Fill in the 5 missing hybrid MLP-swap evals (to reach all 11 task/model columns).
# Uses uniform MAttr (mib_node_hard_topk) x NAP-IG; val split. Prereqs verified present.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
# "model task batch"
PAIRS=(
  "llama3 ioi 2"
  "llama3 mcqa 2"
  "gemma2 arc_easy 4"
  "llama3 arc_easy 2"
  "llama3 arc_challenge 2"
)
for p in "${PAIRS[@]}"; do
  read -r model task bs <<< "$p"
  case $model in
    gemma2) res="-c 3 -r 64G";;
    llama3) res="-c 4 -r 96G";;
  esac
  nlprun -g 1 -q jag -d a6000 $res -n "hyb2-${task}-${model}" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_hybrid_scores.py --model $model --task $task --split validation --batch-size $bs --output results/hybrid_eval"
  sleep 1
done
echo "ALL MISSING HYBRID JOBS SUBMITTED"
