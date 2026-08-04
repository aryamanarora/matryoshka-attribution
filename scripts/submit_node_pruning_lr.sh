#!/bin/bash
# LR sweep for Node Pruning (logit-diff), at the two budgets that actually matter.
#
# Every other method in paper/tabs/lr_sweep.tex gets five LR points; Node Pruning got one. Its
# rows are a SPARSITY sweep at a single learning rate, so "Node Pruning underperforms MAttr"
# currently rests on the baseline's default LR being a good one. This closes that gap -- same
# argument that put the DBM block in the table.
#
# WHICH TWO BUDGETS: s=0.5 and s=0.8, the best and second-best logit-diff rows by Avg CPR in
# mib_results.tex (1.67 and 1.46; the rest run 1.36 / 1.28 / 1.24 / 0.84 / 0.34). They also
# bracket the peak -- the curve rises to s=0.5 and falls after it -- so if a different LR moves
# the optimum, sweeping on both sides is what will show it. Sweeping all seven budgets would be
# 4x the jobs to sharpen rows that are already well off the pace.
#
# WHICH LRs: 0.1 / 0.3 / 1.5 / 3.0, bracketing the hard-concrete default of 0.8 on both sides
# (roughly log-spaced). With the existing default-LR runs as the 0.8 row that is a 5-point
# sweep, matching \ourmethod{} and DBM. Note the default differs by gate -- 0.8 for
# hard_concrete, 1e-3 for sigmoid -- so this grid is NOT comparable to the DBM block's, and the
# two are swept around their own defaults rather than on a shared grid.
#
# The runner already does the things that are easy to get wrong: it trains and evaluates in the
# MIB venv (TL 2.15.4, so the gemma2 cells avoid the Gemma-2 forward bug), and it applies
# --head 200 to llama3 validation only. Nothing here overrides either.
#
# Cost: 4 LRs x 2 budgets x 11 cells = 88 jobs, ~64 GPU-hours. They are submitted at a lower
# priority than the llama3 ARC edge wave, which is filling actual "---" holes in the paper --
# this sharpens a baseline row that already has numbers, so it should not be in front.
#
#   bash scripts/submit_node_pruning_lr.sh            # submit all 88
#   bash scripts/submit_node_pruning_lr.sh 0.5        # just one budget
#   DRYRUN=1 bash scripts/submit_node_pruning_lr.sh   # preview
#
# Register new dirs in scripts/make_lr_table.py:METHODS to get their table rows.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
NICE=${NICE:-3000}

SPARSITIES=("${@:-}")
[ -z "${1:-}" ] && SPARSITIES=(0.5 0.8)
LRS=(0.1 0.3 1.5 3.0)

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

n=0; skipped=0
for s in "${SPARSITIES[@]}"; do
  for lr in "${LRS[@]}"; do
    for p in "${PAIRS[@]}"; do
      read -r model task <<< "$p"
      # Suffix must mirror run_edge_pruning.sbatch exactly (_s<S> then _ld then _lr<LR>), or
      # the skip check reads a dir that never gets written and every job reruns.
      dir="eprun_node_s${s}_ld_lr${lr}"
      if [ -f "$L2A/results/$dir/graph_${task}_${model}.json" ]; then
        echo "SKIP $task/$model s=$s lr=$lr: already trained"
        skipped=$((skipped+1))
        continue
      fi
      name="nplr-s${s}-lr${lr}-${task}-${model}"
      if [ "$DRYRUN" = "1" ]; then
        echo "DRY $name -> results/$dir"
      else
        LOSS=logit_diff LR="$lr" sbatch --nice="$NICE" --job-name="$name" \
          scripts/run_edge_pruning.sbatch "$model" "$task" node 3000 validation "$s" >/dev/null \
          && echo "submitted $name"
      fi
      n=$((n+1))
    done
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n Node Pruning LR jobs (${skipped} skipped) across budgets ${SPARSITIES[*]}, LRs ${LRS[*]} =="
