#!/bin/bash
# pyvene sigmoid-mask baseline on MIB (node level, validation).
#
# Same environment, same task loss and the SAME 3000 steps as the Node Pruning rows -- the
# only thing that changes is how the mask is parameterized:
#
#   Node Pruning : stochastic hard-concrete gate, fixed temperature 2/3, Lagrangian L0 that
#                  anneals a *sparsity target*; AdamW lr 0.8.
#   this script  : pyvene's SigmoidMaskIntervention -- deterministic sigmoid(mask/temp), mask
#                  init 0, temperature annealed 50 -> 0.1 (i.e. it anneals how *binary* the
#                  gate is, not how sparse it is), no sparsity term at all; Adam lr 1e-3.
#
# Because there is no L0 term this method has no notion of a budget; it is usable in MIB only
# because the eval ranks nodes by score and sweeps sparsity itself. So there is one run per
# cell, not one per budget, and no --target-sparsity anywhere.
#
# Loss is logit_diff, matching MAttr and the _ld Node Pruning rows -- comparing against the KL
# rows instead would confound the parameterization with the objective.
#
#   bash scripts/submit_sigmoid_mask.sh            # 11 cells at pyvene's lr=1e-3
#   LR=0.3 bash scripts/submit_sigmoid_mask.sh     # 11 cells at the swept lr
#   DRYRUN=1 bash scripts/submit_sigmoid_mask.sh   # preview
#
# LR is the tuned knob: pyvene's published 1e-3 averages 0.61 CPR AUC on the three cheap
# cells and 0.3 averages 1.13 (see scripts/submit_sigmoid_mask_lr.sh for why we sweep it and
# paper/tabs/lr_sweep.tex for the curve). 0.1/0.3/1.0 are a plateau within noise of each
# other; 0.3 is the argmax, taken as-is rather than adjudicated.
#
# Dirs: results/eprun_node_ld_sig[_lr<LR>]/ (circuits) + results/eprun_eval_ld_sig[_lr<LR>]/.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
STEPS=${STEPS:-3000}
LR=${LR:-}
[ -z "$LR" ] && LRSUF="" || LRSUF="_lr${LR}"

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

n=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  graph="$L2A/results/eprun_node_ld_sig${LRSUF}/graph_${task}_${model}.json"
  if [ -f "$graph" ]; then
    echo "SKIP $task/$model: already trained"
    continue
  fi
  name="sig${LRSUF}-${task}-${model}"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY $name -> results/eprun_node_ld_sig${LRSUF}"
  else
    GATE=sigmoid LOSS=logit_diff LR="$LR" sbatch --job-name="$name" \
      scripts/run_edge_pruning.sbatch "$model" "$task" node "$STEPS" validation >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n pyvene sigmoid-mask jobs =="
