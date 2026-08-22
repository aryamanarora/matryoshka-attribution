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
# THE THREE ARMS (2026-08-20). VARIANT/OPT/LRS are env overrides, one output subdir per arm,
# so all three live side by side under $OUTBASE. They complete the optimizer x gate cross that
# submit_sva_sweep.sh's MATTR_CONFIGS never runs -- it bundles variant WITH optimizer
# (hard_topk:adam, hard_topk_identity:sgd, topk:adam) and never crosses them, which is why no
# neuron-substrate cell had a controlled optimizer contrast before this.
#
#   topk / adam                 lr 0.001..10   the headline; DONE, flat, argmax at 0.05 = 0.361
#   hard_topk_identity / adam   lr 0.001..10   id-STE with the optimizer swapped
#   topk / sgd                  lr 0.05..300   soft gate with the optimizer swapped
#
# WHY id-STE+ADAM IS THE DECISIVE ONE. At node level the same contrast (mib_node_identity_sgd
# vs ident_adam_lr_0.01, uniform-k, 500 steps) gives SGD 5/5 with mean +0.404 CPR AUC. Under
# identity STE dL/ds = dL/dm exactly -- the raw effect size -- and SGD keeps that magnitude
# while Adam divides by sqrt(v) per node, so only sign-consistency survives. If that is what
# id-STE's 0.437 is buying at 2.29M units, this arm drops to ~0.36.
#
# AND IT DOUBLES AS A FALSIFICATION TEST. hard fwd + identity bwd should be EXACTLY lr-invariant
# (the mask reads only the score RANKING, the backward reads no magnitude, so scores = lr x a
# fixed vector and the selection never moves). Verified at node level: Spearman +1.0000 across
# lr, std exactly x2/x10, byte-identical validation.pkl. If this arm returns 9 identical
# acc_auc values, that is the prediction confirmed at neuron scale, NOT a broken sweep -- check
# scores.pt md5s (they should DIFFER while the ranking matches) before calling it a bug.
#
# WHY topk/SGD NEEDS A GRID 1000x HIGHER. The sigmoid-slope backward carries dm/ds ~ k/n
# (~6e-3 here). Adam normalises that factor away, which is exactly why the n/k lr prediction
# failed for the topk/adam arm; SGD does NOT normalise it, so the same prediction should hold
# here and the useful lr should sit ~n/k ~ 160x above Adam's optimum. A grid stopping at 0.3
# would return a guaranteed-uninformative null -- the same trap as the first sweep, in reverse.
#
# THE STEPS ARM (2026-08-22). Every Adam run above is STILL RISING at step 2000 (+0.02 to +0.09
# acc-AUC over its last 800 probe steps) while the three best topk/sgd runs are flat (±0.009), so
# the 0.135 best-vs-best gap is confounded with training budget. STEPS=20000 settles whether
# Adam's deficit is structural (its scores are flat at EVERY lr -- mass@top-200 0.0010-0.0037
# across four decades, vs SGD 0.12-0.19) or just undertrained.
#
#   SUBMIT THE SGD CONTROL TOO. A 20k Adam run that gains 0.1 proves nothing on its own if SGD
#   gains as much over the same span; the claim is about the GAP, so both arms have to move to
#   the same step count. That is why the 2026-08-22 launch is three jobs, not two.
#
#   AND RAISE THE PROBE. At the default --train-eval-examples 20 the probe SATURATES (see
#   eval_sva.py:773 -- flat at 2000 and 6250 on nounpp/mlp while the real test acc-AUC rose
#   +0.04). Reading "has Adam converged" off a 20-example probe is precisely the mistake that
#   comment documents, so this arm runs it at 64. --train-eval-every rises with it to keep the
#   probe cost per run roughly where it was at 2000 steps.
#
# DRY=1 to preview.  LRS="1.0 3.0" to override the grid.
# VARIANT=hard_topk_identity OPT=adam LRS="0.001 0.005 0.01 0.05 0.1 0.3 1.0 3.0 10.0" bash $0
# VARIANT=topk OPT=sgd LRS="0.05 0.3 1.0 3.0 10.0 30.0 100.0 300.0" bash $0
# STEPS=20000 OUTBASE=results/sva_mlp_steps20k LRS="0.005 0.05" bash $0            # Adam, long
# STEPS=20000 OUTBASE=results/sva_mlp_steps20k VARIANT=topk OPT=sgd LRS="1.0" bash $0  # control
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
# 2000 keeps every default invocation byte-identical to the runs already in results/sva_mlp_lr.
# Override ONLY together with OUTBASE: run_tag() encodes neither lr nor steps, so a 20k run
# written into the 2000-step tree would silently overwrite its own control.
STEPS=${STEPS:-2000}
PROBE_EVERY=${PROBE_EVERY:-200}
PROBE_EX=${PROBE_EX:-20}

n=0
for lr in $LRS; do
  out="$OUTBASE/${VARIANT}_${OPT}/lr_$lr"
  name="svalr-${TASK}-${NODES}-${VARIANT}-${OPT}-lr${lr}"
  args=(--model "$MODEL" --task "$TASK" --dataset "$DATASET" --nodes "$NODES"
        --method mattr --variant "$VARIANT" --optimizer "$OPT" --k-schedule log
        --loss logit_diff --mode sufficient --train-batch-size 1 --steps "$STEPS"
        --train-eval-every "$PROBE_EVERY" --train-eval-examples "$PROBE_EX"
        --eval-examples 100 --lr "$lr" --output "$out")
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out"
  else mkdir -p "$out"; sbatch -J "$name" sva_sweep.sbatch "${args[@]}" >/dev/null && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== ${DRY:+DRY }total $n jobs (~4 min each) -> $OUTBASE/${VARIANT}_${OPT} =="
