#!/bin/bash
# Fill the two llama3 ARC cells (arc_easy, arc_challenge) that every edge-level MAttr dir is
# missing -- the "---" holes in the edge block of paper/tabs/mib_results.tex and the reason the
# edge rows average 9 cells where the node rows average 11.
#
# This is a TRAINING gap, not an eval gap: no <task>_llama3_scores.pt exists for these cells in
# any dir except mib_edge_bernoulli_reinforce (which did get them), so there is nothing to
# re-evaluate. Edge-level masks over an 8B model on ARC-length contexts are the single most
# expensive cell in the grid, which is presumably why submit_edge_lr05.sh shipped with
# "9 cells (no llama arc)".
#
# Hyperparameters are NOT re-chosen here: each row below is copied verbatim from the `args` dict
# inside that dir's existing mcqa_llama3_scores.pt, so the new ARC cells are trained exactly like
# their nine siblings and the row average stays internally comparable. llama3 keeps the
# eval-examples 200 cap (the dagger), as every other llama3 edge cell does.
#
# BATCH 1, not the siblings' 2: at batch 2 this OOMs on the FIRST forward (79 GiB, before step 1
# -- ARC contexts are long and an edge mask over 8B keeps every edge's activation live). Batch is
# the one hyperparameter that could not be copied. It changes gradient noise, not the objective
# or the 5000-step budget, so the cells stay comparable in the way that matters for the row.
#
#   bash scripts/submit_edge_arc_llama3.sh            # submit
#   DRYRUN=1 bash scripts/submit_edge_arc_llama3.sh   # preview
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}

# outdir | masking | k-schedule | lr | optimizer | split
CONFIGS=(
  "mib_edge_topk_log_lr05|topk|log|0.05|adam|validation"
  "mib_edge_hard_topk_log_lr05|hard_topk|log|0.05|adam|validation"
  "mib_edge_hard_topk_uniform_lr05|hard_topk|uniform|0.05|adam|validation"
  "mib_edge_detached_tau|topk_detached|log|0.01|adam|validation"
  "mib_edge_identity_sgd_log|hard_topk_identity|log|0.01|sgd|validation"
  "mib_edge_identity_sgd_uniform|hard_topk_identity|uniform|0.01|sgd|validation"
  "test_edge_topk_log_lr05|topk|log|0.05|adam|test"
  "test_edge_hard_topk_log_lr05|hard_topk|log|0.05|adam|test"
  "test_edge_hard_topk_uniform_lr05|hard_topk|uniform|0.05|adam|test"
  "test_edge_hard_topk_uniform|hard_topk|uniform|0.01|adam|test"
)
n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r out mask sched lr opt split <<< "$c"
  for task in arc_easy arc_challenge; do
    # skip anything already trained, so this script is safe to re-run after a partial wave
    if [ -f "$ABS/results/$out/${task}_llama3_scores.pt" ]; then
      echo "SKIP $out/$task: already trained"
      continue
    fi
    # split goes in the name: the val and test dirs share a config suffix, so without it the two
    # jobs would write to the same logs/<name>.out and the first one's log would be lost.
    name="ea-${split:0:3}-${out#*_edge_}-${task}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib_edge.py --model llama3 --task $task --steps 5000 --k-schedule $sched \
--masking $mask --mode sufficient --lr $lr --optimizer $opt --split $split --train-split train \
--batch-size 1 --eval-examples 200 --output results/$out"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name -> $out"
    else
      # 24h is the association's MaxWall (sacctmgr show assoc); 36h is rejected outright with
      # AssocMaxWallDurationPerJobLimit, so a longer request buys nothing but a failed submit.
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=5 --mem=128G --time=24:00:00 \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n llama3 ARC edge jobs =="
