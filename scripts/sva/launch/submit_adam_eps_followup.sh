#!/bin/bash
# FOLLOW-UP to submit_adam_vs_sgd_mlp.sh arm A, which found the answer: Adam's EPS.
#
# Arm A result (addition/llama3/mlp, 2000 steps, logit_diff, log-k, 2,293,760 neurons):
#
#   eps      best acc-AUC over lr   median|s| / (lr*sqrt(T))   p99/p50 of |s|   top-2082 overlap w/ IG
#   1e-8         0.388                    0.51 - 2.18              2.8 - 4.9          0.04 - 0.08
#   1e-6         0.380                    0.28 - 1.97              2.4 - 4.6          0.02 - 0.08
#   1e-4         0.400                    0.05 - 0.23              7.1 - 11.8         0.09 - 0.23
#   1e-2         0.490                    0.001 - 0.004           35.7 - 72.6         0.61 - 0.73
#   (MAttr+SGD 0.496, IG 0.500)
#
# MECHANISM. At 2.29M mask logits nearly every per-step |grad| exceeds the default eps=1e-8, so
# Adam's update is m_hat/(sqrt(v_hat)+eps) ~ sign(g)*lr: every coordinate takes the SAME size
# step regardless of its effect size, and the learned score is a signed COUNT of steps. That
# predicts median|s| ~ lr*sqrt(steps) (an unweighted +-lr random walk) and a near-flat score
# distribution -- both measured, at every lr. Raising eps above the typical |g| restores
# magnitude weighting: the median score collapses 3 orders of magnitude, the tail ratio grows
# 20x, the ranking converges on the gradient-path ranking (IG overlap 0.08 -> 0.73) and the
# exam score matches SGD. Monotone in eps at all four learning rates.
#
# THIS SCRIPT CLOSES THE THREE REMAINING HOLES.
#
# 1. EPS TOP END. 1e-2 is the largest value swept and it is the argmax, so the bracket is
#    OPEN on the right -- as it stands the result reads "bigger is better, we stopped looking".
#    1e-1 and 1e0 either turn it over (interior optimum, a real tuning result) or do not
#    (Adam is simply being asked to become SGD, which is a different and more honest claim).
#
# 2. SECOND CELL. nounpp/llama3/mlp: the same 2.29M-unit substrate, a different task, and a
#    cell where the on-disk Adam/SGD/IG gap has the same signature (0.656 / 0.704 / 0.695).
#    One cell is an anecdote; the fix has to move a second one.
#
# 3. THE NODE-LEVEL CONTROL -- why this is neuron-SPECIFIC. Same task, same script, same loss,
#    --nodes node instead of --nodes mlp: ~1k units instead of 2.29M. The mechanism is about
#    |grad| relative to eps, not about neurons per se, and per-unit gradients at node
#    granularity are far larger. Prediction: eps does NOT matter at node level, which is why
#    the MIB node-level results have never shown this and why the deficit only ever appears on
#    the neuron substrates. If eps DOES move the node cell, the story is wrong and the MIB
#    node-level LR/optimizer conclusions need re-reading.
#
# DRY=1 to preview.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

OUT=${OUT:-results/adamsgd_mlp}
n=0
sub () {  # sub <name> <outdir> <args...>
  local name="$1" out="$2"; shift 2
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out :: $*"
  else mkdir -p "$out"; sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "$@" --output "$out" >/dev/null
       echo "submitted $name"; fi
  n=$((n+1))
}
COMMON=(--model llama3 --dataset arith --method mattr --variant topk --k-schedule log
        --mode sufficient --loss logit_diff --train-batch-size 1 --steps 2000
        --eval-examples 100 --train-eval-every 250 --train-eval-examples 64)

# ---- 1. eps top end, on the winning lr band
for eps in 1e-1 1e0; do
  for lr in 0.05 0.5 5.0; do
    sub "aeps2-${eps}-lr${lr}" "$OUT/A_eps/eps_${eps}_lr_${lr}" \
        "${COMMON[@]}" --task addition --nodes mlp --optimizer adam --lr "$lr" --adam-eps "$eps"
  done
done

# ---- 2. second cell: nounpp (SVA), same 2.29M-unit mlp substrate.
#         --dataset sva, and eps=1e-8 is the control it has to beat.
SVA=(--model llama3 --dataset sva --method mattr --variant topk --k-schedule log
     --mode sufficient --loss logit_diff --train-batch-size 1 --steps 2000
     --eval-examples 100 --train-eval-every 250 --train-eval-examples 64)
for eps in 1e-8 1e-2; do
  for lr in 0.05 0.5; do
    sub "nouneps-${eps}-lr${lr}" "$OUT/G_nounpp/eps_${eps}_lr_${lr}" \
        "${SVA[@]}" --task nounpp --nodes mlp --optimizer adam --lr "$lr" --adam-eps "$eps"
  done
done
sub "nounsgd-lr1.0" "$OUT/G_nounpp/sgd_lr_1.0" "${SVA[@]}" --task nounpp --nodes mlp \
    --optimizer sgd --lr 1.0

# ---- 3. node-level control on the SAME task (small substrate, same harness)
for eps in 1e-8 1e-2; do
  for lr in 0.05 0.5; do
    sub "nodeeps-${eps}-lr${lr}" "$OUT/H_node/eps_${eps}_lr_${lr}" \
        "${COMMON[@]}" --task addition --nodes node --optimizer adam --lr "$lr" --adam-eps "$eps"
  done
done
sub "nodesgd-lr1.0" "$OUT/H_node/sgd_lr_1.0" "${COMMON[@]}" --task addition --nodes node \
    --optimizer sgd --lr 1.0

# ---- 4. re-run the Expected Gradients reference that arm E lost to a kwarg bug (now fixed)
sub "steplessig" "$OUT/E_steplessig" --model llama3 --task addition --dataset arith \
    --nodes mlp --mode sufficient --method mc_ig --ig-steps 1 --seed 42 --loss logit_diff \
    --eval-examples 100

echo "== ${DRY:+DRY }total $n jobs -> $OUT =="
