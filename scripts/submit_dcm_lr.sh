#!/bin/bash
# Learning-rate sweep for the DCM baseline (node, validation, logit_diff), at three PINNED
# densities.
#
# What DCM is here: Prakash et al. (ICLR 2024) `experiment_2/DCM.py`, as re-implemented in
# `roonbug/belief_dynamics` (`causal-experiments/exps/patching/dcm/run.py`) -- a raw
# coefficient clamped to [0,1], no sigmoid, no temperature anneal, no lr schedule, circuit =
# round(mask) at 0.5, sparsity term `mult * task_loss.detach() * mask.mean()` with a PID
# controller on log(mult). See learning_to_attribute/edge_pruning.py:learn_scores_dcm.
#
# WHY THE DENSITY IS PINNED, AND WHY THERE ARE THREE OF THEM.
# DCM produces a SET, not a ranking: clamp_(0,1) piles the final scores onto exactly 0.0 and
# 1.0, so handing them to MIB's rank-then-sweep-top-k harness would let tie order decide the
# interior of the CPR curve. At a pinned density that problem disappears -- round(mask) has
# exactly k units and top-k selects exactly them -- so DCM's number is only apples-to-apples
# with the other mask rows AT the pin. Three pins (1%/5%/20%) because there is no reason the
# best lr is density-independent, and finding out costs three cheap cells rather than a
# claim.
#
# Protocol is submit_sigmoid_mask_lr.sh's, for the same reason: sweep the three CHEAP cells
# (gpt2/qwen2.5, ~10-20 min each including eval), pick the best lr per density by mean CPR,
# then run those over all 11. 11 cells x 5 lrs x 3 densities up front would burn ~150 GPU-h
# on gemma2/llama3 evals to answer a question the small cells answer for ~8.
#
# lr grid is centred on DCM's published 1e-1, which is identical in Prakash et al. and in
# belief_dynamics -- so unlike pyvene's 1e-3 there is a real prior here, and the sweep is
# checking whether it transfers off attention-heads-at-one-position, not replacing it.
#
#   bash scripts/submit_dcm_lr.sh                    # 3 cells x 5 lrs x 3 densities = 45 jobs
#   LRS="3.0" bash scripts/submit_dcm_lr.sh          # add a point
#   DENSITIES="0.05" bash scripts/submit_dcm_lr.sh   # one pin only
#   DRYRUN=1 bash scripts/submit_dcm_lr.sh
#
# Dirs: results/eprun_node_ld_dcm_d<D>_lr<LR>/ + results/eprun_eval_ld_dcm_d<D>_lr<LR>/.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
STEPS=${STEPS:-3000}
LRS=${LRS:-"0.01 0.03 0.1 0.3 1.0"}
DENSITIES=${DENSITIES:-"0.01 0.05 0.2"}
PAIRS=${PAIRS:-"gpt2 ioi|qwen2.5 ioi|qwen2.5 mcqa"}

n=0
IFS='|' read -ra CELLS <<< "$PAIRS"
for d in $DENSITIES; do
  for lr in $LRS; do
    for p in "${CELLS[@]}"; do
      read -r model task <<< "$p"
      graph="$L2A/results/eprun_node_ld_dcm_d${d}_lr${lr}/graph_${task}_${model}.json"
      if [ -f "$graph" ]; then
        echo "SKIP $task/$model d=$d lr=$lr: already trained"
        continue
      fi
      name="dcm${d}lr${lr}-${task}-${model}"
      if [ "$DRYRUN" = "1" ]; then
        echo "DRY $name -> results/eprun_node_ld_dcm_d${d}_lr${lr}"
      else
        GATE=dcm LOSS=logit_diff DENSITY="$d" LR="$lr" sbatch --job-name="$name" \
          scripts/run_edge_pruning.sbatch "$model" "$task" node "$STEPS" validation >/dev/null \
          && echo "submitted $name"
      fi
      n=$((n+1))
    done
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n DCM lr-sweep jobs =="
