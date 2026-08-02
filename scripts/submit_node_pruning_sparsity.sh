#!/bin/bash
# Node Pruning validation runs at additional L0 budgets (default 0.8 and 0.5).
#
# Why these two: the three budgets already on disk (0.9 / 0.95 / 0.99) all sit on the SPARSE
# side of the mean CPR curve's peak, which is at 50% of nodes kept. CPR AUC orders them
# 0.9 > 0.95 > 0.99 -- monotone toward *less* sparsity, with the best value at the edge of
# what we ran -- so "s=0.9 is the best budget" currently means "the least sparse one we tried".
# s=0.8 and s=0.5 (20% and 50% kept, both MIB grid points) bracket the peak from below and
# settle it. They also stay well inside the range where the Lagrangian actually binds: at
# s=0.99 the constraint is not met on 10 of 11 cells (gemma2/arc_easy asks for 2.3 nodes and
# converges at 11.0, i.e. an achieved s of 0.953), so that row is not the budget it claims.
#
# Each sparsity gets its OWN dirs -- results/eprun_node_s<S>/ and results/eprun_eval_s<S>/ --
# so nothing already computed is touched or overwritten. Trains from scratch (a different
# budget is a different circuit; EVAL_ONLY would be wrong here).
#
#   bash scripts/submit_node_pruning_sparsity.sh            # 0.8 and 0.5, 22 jobs
#   bash scripts/submit_node_pruning_sparsity.sh 0.8         # just one budget
#   DRYRUN=1 bash scripts/submit_node_pruning_sparsity.sh    # preview
#   LOSS=logit_diff bash scripts/submit_node_pruning_sparsity.sh 0.9   # MAttr's objective, _ld dirs
#
# Register a new budget in scripts/make_mib_table.py:EPRUN_SPARSITIES to get its table rows.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
SPARSITIES=("${@:-}")
[ -z "${1:-}" ] && SPARSITIES=(0.8 0.5)

# LOSS=logit_diff trains on MAttr's objective instead of Edge Pruning's KL (dirs get _ld).
# Suffix logic must mirror run_edge_pruning.sbatch or the skip check reads the wrong dir.
LOSS=${LOSS:-kl}
LD=""; [ "$LOSS" = "kl" ] || LD="_ld"
# Trap: the s=0.9 KL runs live in the UNSUFFIXED results/eprun_node, because that is the
# sbatch's own default and it only adds _s<S> when a sparsity is passed explicitly. Asking
# for 0.9 + KL here would build a second, redundant results/eprun_node_s0.9.
if [ "$LOSS" = "kl" ]; then
  for s in "${SPARSITIES[@]}"; do
    [ "$s" = "0.9" ] && { echo "refusing s=0.9 with LOSS=kl: those runs are results/eprun_node (unsuffixed)"; exit 1; }
  done
fi

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

n=0
for s in "${SPARSITIES[@]}"; do
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    # Skip a cell that already converged at this budget, so the script is re-runnable after
    # a partial failure without redoing ~40 min of llama3 training.
    graph="$L2A/results/eprun_node_s${s}${LD}/graph_${task}_${model}.json"
    if [ -f "$graph" ]; then
      echo "SKIP $task/$model s=$s $LOSS: already trained"
      continue
    fi
    name="np-s${s}${LD}-${task}-${model}"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name -> results/eprun_node_s${s}${LD}"
    else
      LOSS="$LOSS" sbatch --job-name="$name" \
        scripts/run_edge_pruning.sbatch "$model" "$task" node 3000 validation "$s" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n Node Pruning jobs across budgets: ${SPARSITIES[*]} =="
