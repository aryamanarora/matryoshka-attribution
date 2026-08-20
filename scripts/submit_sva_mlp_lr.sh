#!/bin/bash
# LR sweep on ONE neuron-substrate MAttr cell: addition / llama3 / --nodes mlp, headline config
# (soft top-k fwd `topk`, Adam, log k-schedule, logit_diff loss).
#
# WHY THIS CELL. It is the cheapest clear failure. At --nodes mlp the headline MAttr scores
# acc-AUC 0.361 where IG scores 0.500, and the arithmetic tasks are also where MAttr misses
# Feucht et al.'s published layer-18 neurons (1/12 cells vs IG's 12/12). Within the four arith
# tasks, `addition` is 5 tokens, so a run is ~4 minutes; `hours` is 38 and scores worse (0.257)
# but costs ~8x. Reference points on this exact cell, all already on disk in results/sva_sweep:
#
#     IG                          0.500      <- the number to close on
#     id-STE / SGD                0.437
#     +hard (hard_topk / Adam)    0.393
#     MAttr headline (this cell)  0.361      <- lr=0.05, the point this sweep brackets
#     I x G                       0.244
#
# WHY IT IS THE RIGHT SUBSTRATE. The node-level MIB sweep (submit_softlog_sgd_lr.sh) was a
# control: Adam already wins at node level (~5.3k mask logits), so a flat result there says
# nothing about the ~2.3M-logit neuron substrates where MAttr actually collapses. This is the
# sweep that tests the hypothesis rather than a proxy for it.
#
# WHY THE GRID GOES TO 10. The gate slope at init is ~k/n, so the useful LR should scale with
# n/k -- 458k MLP neurons here against ~157 units at MIB node level. If the collapse is an LR
# artifact, the fix is at the TOP of the grid, and a grid that stops at the node-level optimum
# (0.3) would return "no effect" for the same reason a thermometer that stops at 40C does.
# lr=0.05 is deliberately absent: it is the existing run in results/sva_sweep, reused as the
# sweep's centre rather than recomputed.
#
# OUTPUT DIR PER LR IS LOAD-BEARING. eval_sva.run_tag() does NOT encode the learning rate --
# every LR here produces the SAME filename, addition_llama3_mlp_sufficient_topk_adam_bs1.json.
# Writing them all to one --output would leave the last job to finish silently overwriting the
# rest, and the sweep would look like it ran while holding one run. One subdir per LR is what
# keeps them apart; do not "tidy" them into a shared dir.
#
# DRY=1 to preview.  LRS="1.0 3.0" to override the grid.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

TASK=${TASK:-addition}
MODEL=${MODEL:-llama3}
NODES=${NODES:-mlp}
DATASET=${DATASET:-arith}
# Held fixed at the headline so the only thing varying across jobs is --lr. Same 2000 steps and
# bs=1 as submit_sva_sweep.sh's MATTR_COMMON, so every result here is directly comparable to
# the sweep dir this cell's centre point comes from.
VARIANT=${VARIANT:-topk}
OPT=${OPT:-adam}
LRS=${LRS:-"0.001 0.005 0.01 0.1 0.3 1.0 3.0 10.0"}
OUTBASE=${OUTBASE:-results/sva_mlp_lr}

n=0
for lr in $LRS; do
  out="$OUTBASE/${VARIANT}_${OPT}/lr_$lr"
  name="svalr-${TASK}-${NODES}-${VARIANT}-${OPT}-lr${lr}"
  args=(--model "$MODEL" --task "$TASK" --dataset "$DATASET" --nodes "$NODES"
        --method mattr --variant "$VARIANT" --optimizer "$OPT" --k-schedule log
        --loss logit_diff --mode sufficient --train-batch-size 1 --steps 2000
        --eval-examples 100 --lr "$lr" --output "$out")
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out"
  else mkdir -p "$out"; sbatch -J "$name" sva_sweep.sbatch "${args[@]}" >/dev/null && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== ${DRY:+DRY }total $n jobs (~4 min each) -> $OUTBASE/${VARIANT}_${OPT} =="
