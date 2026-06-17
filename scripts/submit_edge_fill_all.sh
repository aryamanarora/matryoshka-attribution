#!/bin/bash
# Fill ALL missing edge ablation cells (validation, k-schedule=log, mode=sufficient).
#  - small models (gpt2/qwen/gemma): only bernoulli_reinforce row was never run -> jag, full eval.
#  - llama3 cells: a6000 OOMs at full eval -> sphinx h100, 200-example subset (daggered).
set -u
cd /juice2/scr2/aryaman/learning-to-attribute
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# ---- small-model bernoulli_reinforce cells (jag a6000, full eval) ----
for p in "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "qwen2.5 mcqa" "gemma2 mcqa" "gemma2 arc_easy"; do
  read -r model task <<< "$p"
  case $model in gpt2|qwen2.5) res="-c 3 -r 64G";; gemma2) res="-c 4 -r 96G";; esac
  nlprun -g 1 -q jag -d a6000 $res -n "eb-${task}-${model}" \
    "$EXP uv run python scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule log --masking bernoulli_reinforce --mode sufficient --split validation --train-split train --batch-size 5 --eval-examples 0 --output results/mib_edge_bernoulli_reinforce"
  sleep 1
done

# ---- llama3 cells (sphinx h100, subset 200) ----
# spec: <output-dir> <masking> <tasks...>
run_llama() {
  out=$1; mask=$2; shift 2
  for t in "$@"; do
    nlprun -g 1 -q sphinx -d h100 -r 128G -n "el-${mask}-${t}" \
      "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule log --masking $mask --mode sufficient --split validation --train-split train --batch-size 2 --eval-examples 200 --output results/$out"
    sleep 1
  done
}
ALL5="ioi arithmetic_subtraction mcqa arc_easy arc_challenge"
run_llama final_edge                   topk               $ALL5
run_llama mib_edge_detached_tau        topk_detached      $ALL5
run_llama mib_edge_bernoulli_reinforce bernoulli_reinforce $ALL5
run_llama mib_edge_hard_topk           hard_topk          arc_easy arc_challenge   # only arc missing here
echo "ALL MISSING EDGE CELLS SUBMITTED"
