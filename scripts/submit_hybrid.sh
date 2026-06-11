#!/bin/bash
# Re-eval NAP-IG x L2A hybrids on the HONEST L2A scores (mib_node_hard_topk_log), val split.
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa"
)
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  case $model in
    gpt2|qwen2.5) res="-c 2 -r 32G";;
    gemma2)       res="-c 3 -r 64G";;
    llama3)       res="-c 4 -r 96G";;
  esac
  nlprun -g 1 -q jag -d a6000 $res -n "hyb2-${task}-${model}" \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/eval_hybrid_scores.py --model $model --task $task --split validation --output results/hybrid_eval"
  sleep 1
done
echo "ALL HYBRID JOBS SUBMITTED"
