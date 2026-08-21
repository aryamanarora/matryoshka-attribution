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
# BATCH 2, same as the nine siblings -- so nothing about these cells is re-chosen at all.
#
# Getting here took three OOM waves and two wrong diagnoses. Batch was blamed first (it OOM'd at
# batch 1 too), then a constant factor in eval_mib_edge.py's memory (halving it did not help). The
# actual driver is the LENGTH TAIL: stacks cost ~286 MB per token position on llama3, ARC medians
# are 52/61 tokens but the maxima are 178/186, and loss_fn draws a random example per step. So a
# median step fit and a tail step did not, which is why a run could pass step 1 and die later, and
# why mcqa (1.19x tail) and ioi (1.53x) never showed it. eval_mib_edge.py now checkpoints the
# stacks, making peak memory independent of destination count; a probe pinned to the longest
# example in each split trains at batch 2 with room, so the deviation is no longer needed.
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
  # soft-fwd SGD, LRs imported from the node optima (submit_mib_edge_soft_sgd.sh). Unlike every
  # other line here the LR is NOT this dir's swept optimum -- but that is exactly why these two
  # cells must exist: without them the row averages 9 cells against its siblings' 11, and the two
  # missing cells are ARC/llama3, the highest-scoring columns in the edge section (id-STE reads
  # 8.96/8.55 there vs a 6.57 row average), so omitting them biases the row DOWNWARD and would
  # make the already-negative LR-transfer result look worse than it is.
  "mib_edge_softlog_sgd_lr_1.0|topk|log|1.0|sgd|validation"
  "mib_edge_softuni_sgd_lr_3.0|topk|uniform|3.0|sgd|validation"
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
--batch-size 2 --eval-examples 200 --output results/$out"
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
