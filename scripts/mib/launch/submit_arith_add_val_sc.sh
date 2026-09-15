#!/bin/bash
# VALIDATION-split arithmetic_addition / llama3 for every MAttr row of the two validation tables
# (make_mib_table.py -> paper/tabs/mib_results.tex; make_mib_accauc_table.py -> mib_accauc_results.tex,
# whose rows are make_mib_table.OUR_METHODS). The test-split counterpart is
# submit_arith_add_fill_sc.sh. sc / nlprun, run INSIDE tmux.
#
# WHY. On 2026-09-15 arithmetic_addition became the 12th COLUMN of every MIB table. Both
# generators suppress a row's Avg and drop it from the section best/avg when it has fewer cells
# than COLUMNS, so until these land every MAttr row of the validation tables shows no average.
#
# PROTOCOL IS READ OFF EACH DIR'S OWN arithmetic_subtraction_llama3_scores.pt `args` dict
# (dumped 2026-09-15), not re-chosen: 500 node steps / 5000 edge steps, batch 2 (batch 4 for the
# two hard_topk_identity node dirs), --include-input at node level only, node = full validation
# split, edge = --eval-examples 200 (the $\dagger$ cap). Three node dirs (mib_node_detached_tau_log,
# mib_node_detached_tau, mib_node_hard_topk_gumbel) stored mode="necessary": they were produced
# 2026-06-10/11, BEFORE the 2026-06-15 label flip (CLAUDE.md), when that label meant denoising --
# so today's flag is --mode sufficient for every row, same as the post-flip edge twin
# mib_edge_detached_tau already stores. Dirs whose args carried no optimizer key ran on the
# default (adam); it is passed explicitly here so the record is unambiguous.
#
# 15 node (jag a6000) + 10 edge (sphinx h100) = 25 jobs. A cell with a validation pkl already on
# disk is skipped. LEVELS="node" / DRYRUN=1 / ONLY=<substring> to restrict.
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"
DRYRUN=${DRYRUN:-0}; LEVELS=${LEVELS:-"node edge"}; ONLY=${ONLY:-}
MODEL=llama3; TASK=arithmetic_addition
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
# dir | masking | k-schedule | optimizer | lr | batch
NODE=(
  "topklog_lr_0.05|topk|log|adam|0.05|2"
  "softlog_sgd_lr_1.0|topk|log|sgd|1.0|2"
  "htklog_lr_0.05|hard_topk|log|adam|0.05|2"
  "mib_node_detached_tau_log|topk_detached|log|adam|0.01|2"
  "mib_node_bernoulli_reinforce_log|bernoulli_reinforce|log|adam|0.1|2"
  "mib_node_identity_sgd_log|hard_topk_identity|log|sgd|0.01|4"
  "mib_node_identity_gumbel_sgd_log|hard_topk_identity_gumbel|log|sgd|0.01|2"
  "mib_node_topk_uniform_lr05|topk|uniform|adam|0.05|2"
  "softuni_sgd_lr_3.0|topk|uniform|sgd|3.0|2"
  "htk_lr_0.05|hard_topk|uniform|adam|0.05|2"
  "mib_node_hard_topk_gumbel|hard_topk_gumbel|uniform|adam|0.01|2"
  "mib_node_detached_tau|topk_detached|uniform|adam|0.01|2"
  "mib_node_bernoulli_reinforce|bernoulli_reinforce|uniform|adam|0.1|2"
  "mib_node_identity_sgd|hard_topk_identity|uniform|sgd|0.01|4"
  "mib_node_identity_gumbel_sgd_uniform|hard_topk_identity_gumbel|uniform|sgd|0.01|2"
)
EDGE=(
  "mib_edge_topk_log_lr05|topk|log|adam|0.05|2"
  "mib_edge_softlog_sgd_lr_3.0|topk|log|sgd|3.0|2"
  "mib_edge_hard_topk_log_lr05|hard_topk|log|adam|0.05|2"
  "mib_edge_detached_tau|topk_detached|log|adam|0.01|2"
  "mib_edge_bernoulli_reinforce|bernoulli_reinforce|log|adam|0.01|2"
  "mib_edge_identity_sgd_log|hard_topk_identity|log|sgd|0.01|2"
  "mib_edge_topk_uniform_lr05|topk|uniform|adam|0.05|2"
  "mib_edge_softuni_sgd_lr_3.0|topk|uniform|sgd|3.0|2"
  "mib_edge_hard_topk_uniform_lr05|hard_topk|uniform|adam|0.05|2"
  "mib_edge_identity_sgd_uniform|hard_topk_identity|uniform|sgd|0.01|2"
)
n=0; skip=0
go() {  # level spec
  local level=$1; IFS='|' read -r out mask sched opt lr bs <<< "$2"
  [ -n "$ONLY" ] && [[ "$out" != *"$ONLY"* ]] && return
  if [ -f "results/$out/${TASK}_${MODEL}_validation.pkl" ]; then echo "SKIP $out: already on disk"; skip=$((skip+1)); return; fi
  local name="aav-${level}-${out}" res cmd
  if [ "$level" = node ]; then
    res="-q jag -d a6000 -c 4 -r 96G"
    cmd="$EXP uv run python scripts/mib/eval_mib.py --model $MODEL --task $TASK --steps 500 --k-schedule $sched \
--masking $mask --optimizer $opt --mode sufficient --lr $lr --split validation --train-split train \
--include-input --batch-size $bs --output results/$out"
  else
    res="-q sphinx -d h100 -r 128G"
    cmd="$EXP uv run python scripts/mib/eval_mib_edge.py --model $MODEL --task $TASK --steps 5000 --k-schedule $sched \
--masking $mask --optimizer $opt --mode sufficient --lr $lr --split validation --train-split train \
--batch-size $bs --eval-examples 200 --output results/$out"
  fi
  n=$((n+1))
  if [ "$DRYRUN" = "1" ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
    nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
}
case " $LEVELS " in *" node "*) for s in "${NODE[@]}"; do go node "$s"; done ;; esac
case " $LEVELS " in *" edge "*) for s in "${EDGE[@]}"; do go edge "$s"; done ;; esac
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n validation arithmetic_addition jobs, $skip skipped =="
