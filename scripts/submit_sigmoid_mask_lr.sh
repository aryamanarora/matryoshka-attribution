#!/bin/bash
# Learning-rate sweep for the pyvene sigmoid-mask baseline (node, validation, logit_diff).
#
# Why: at pyvene's published lr=1e-3 the baseline lands at CPR AUC ~0.77 (avg over the 6
# cells that had finished on 2026-08-03), well under Node Pruning's ~1.44 and MAttr's ~1.84.
# The training logs say it is not a divergence -- final logits sit at 0.9--1.4 with tau
# annealed to 0.1, so the gate DOES saturate -- but Adam at lr=1e-3 can displace a logit by
# at most ~3.0 in 3000 steps, and it used a third of that. That is the signature of a run
# that is still in transient, not one that has converged to a bad optimum.
#
# pyvene picked 1e-3 for a handful of rotation parameters on a single intervention site; we
# are optimizing 156--1056 independent gate logits against a task loss. There is no reason
# their lr transfers, and pinning it there would be the un-fair-chance version of this
# baseline. So lr is the one hyperparameter we tune, and the paper should report the tuned
# number (noting the sweep) rather than the 1e-3 one.
#
# Protocol: sweep on the three CHEAP cells first (gpt2/qwen2.5, ~10-20 min each including
# eval), pick the best lr by mean CPR AUC, then run that lr over all 11. Sweeping 11 cells x
# 4 lrs up front would burn ~40 GPU-h on gemma2/llama3 evals to answer a question the small
# cells answer for ~2.
#
#   bash scripts/submit_sigmoid_mask_lr.sh            # 3 cells x 3 lrs = 9 jobs
#   LRS="0.5" bash scripts/submit_sigmoid_mask_lr.sh  # add a point
#   DRYRUN=1 bash scripts/submit_sigmoid_mask_lr.sh
#
# lr=1e-3 is NOT in the default list -- results/eprun_eval_ld_sig/ already is that point.
# Dirs: results/eprun_node_ld_sig_lr<LR>/ + results/eprun_eval_ld_sig_lr<LR>/.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
STEPS=${STEPS:-3000}
LRS=${LRS:-"0.01 0.1 1.0"}
PAIRS=${PAIRS:-"gpt2 ioi|qwen2.5 ioi|qwen2.5 mcqa"}

n=0
IFS='|' read -ra CELLS <<< "$PAIRS"
for lr in $LRS; do
  for p in "${CELLS[@]}"; do
    read -r model task <<< "$p"
    graph="$L2A/results/eprun_node_ld_sig_lr${lr}/graph_${task}_${model}.json"
    if [ -f "$graph" ]; then
      echo "SKIP $task/$model lr=$lr: already trained"
      continue
    fi
    name="siglr${lr}-${task}-${model}"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name -> results/eprun_node_ld_sig_lr${lr}"
    else
      GATE=sigmoid LOSS=logit_diff LR="$lr" sbatch --job-name="$name" \
        scripts/run_edge_pruning.sbatch "$model" "$task" node "$STEPS" validation >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n sigmoid-mask lr-sweep jobs =="
