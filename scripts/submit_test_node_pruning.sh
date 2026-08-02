#!/bin/bash
# TEST-set evals for the Node Pruning baseline at the headline budget (s=0.9, the unsuffixed
# results/eprun_node dir).
#
# EVAL_ONLY=1 is the point: the mask is trained on the TRAIN split
# (eval_mib_edge_pruning.py --train-split train, independent of --split), so the graph already
# on disk from the validation pass is the same circuit we owe the test set. Retraining would
# not just waste ~11 GPU-hours, it would produce a *different* circuit and quietly break the
# "same circuit, two splits" claim. Nothing here touches results/eprun_node/graph_*.json.
#
# 11 cells, one job each -> results/eprun_eval/EdgePruning_patching_node/
#   <task-dash>_<model>_test_abs-False.pkl   (the _test_ suffix keeps it clear of validation)
# DRYRUN=1 to preview.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

n=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  graph="$L2A/results/eprun_node/graph_${task}_${model}.json"
  if [ ! -f "$graph" ]; then
    echo "SKIP $task/$model: no trained graph at $graph"
    continue
  fi
  name="np-test-${task}-${model}"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY $name"
  else
    EVAL_ONLY=1 sbatch --job-name="$name" \
      scripts/run_edge_pruning.sbatch "$model" "$task" node 3000 test >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n Node Pruning test-set eval jobs =="
