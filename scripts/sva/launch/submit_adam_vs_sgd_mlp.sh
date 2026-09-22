#!/bin/bash
# WHY ADAM LOSES TO SGD (AND TO IG) AT MLP-NEURON SCALE -- the diagnostic matrix.
# One cell: addition / llama3 / --nodes mlp (2,293,760 neurons), sufficient/denoising, bs=1,
# log-k, 2000 steps, 100 eval pairs. Same cell as submit_sva_mlp_lr.sh, so every job here is
# directly comparable to what is already on disk in results/sva_mlp_lr and results/sva_sweep.
# ~4 min/job on an a6000.
#
# WHAT THE ON-DISK DATA ALREADY SAYS (read this before adding arms).
# Decomposing acc-AUC by decade of k, Adam's ENTIRE deficit sits in k in [1e3, 1e5]; at
# k >= 1e5 Adam and SGD are identical. And in exactly that window Adam's MEAN MARGIN is 3-5x
# the clean model's while its ACCURACY is 10 points below SGD's:
#
#     k=14066   MAttr+Adam lr.05   mean margin 30.84 (clean = 6.68)   acc 0.87
#               MAttr+SGD  lr1.0   mean margin  5.82                  acc 0.97
#
# i.e. the training objective (mean base-source margin, -d.mean()) is ANTI-correlated with the
# exam metric (accuracy) precisely where the exam discriminates. Across the 51 runs of
# results/sva_sweep on this cell, spearman(acc_auc, faith_auc) = +0.65 but
# spearman(acc_auc, CLIPPED faith_auc) = +0.94 -- every bit of the disagreement between the
# two exam metrics is over-recovery (faithfulness > 1). And the lr sweep shows this is not an
# Adam property: SGD at lr >= 3 also starts over-recovering (faith_max 1.37 -> 4.57) and its
# acc-AUC falls 0.496 -> 0.436. SGD wins at lr=1.0 because at lr=1.0 the gate has NOT engaged
# (faith_max exactly 1.00, std(scores) << T) -- it is IG in disguise, not a better optimizer.
#
# So the standing hypothesis is: the loss AVERAGES A RAW, UNBOUNDED PER-SAMPLE MARGIN, so the
# cheapest way to lower it is to pad an already-decided example rather than flip an undecided
# one. Whichever optimizer descends the objective hardest wins the objective and loses the exam.
# The three arms below try to break that hypothesis.
#
# ARM A -- adam-eps x lr. THE UNTESTED HPARAM. With 2.29M mask logits, nearly every per-step
#   gradient is far below Adam's default eps=1e-8, so m_hat/(sqrt(v_hat)+eps) ~ sign(g): the
#   learned score is a signed COUNT of steps and all effect MAGNITUDE is divided out (this is
#   the measured mechanism behind id-STE+Adam's collapse, see scripts/sva/report_adamsgd_mlp.py). Raising
#   eps above the typical |g| restores magnitude sensitivity and continuously interpolates
#   Adam -> SGD+momentum at effective lr = lr/eps. If the deficit is an OPTIMIZER artifact,
#   acc-AUC should climb toward 0.49 along the eps axis. If it is the OBJECTIVE, large-eps Adam
#   should just reproduce the SGD lr sweep -- good at the lr where the gate stays disengaged,
#   gap-padding and worse above it. lr must be co-swept because eps rescales the step.
#
# ARM B -- loss shape x optimizer. The direct test. `ld_tanh` (added with this script) is
#   logit_diff with each example's contribution squashed through tanh(d/scale): identical to
#   logit_diff for small |d|, bounded by 1 per example, so gap-padding buys nothing. `hinge`
#   is the hard-cap version (exactly zero gradient past the margin) and `prob` the bounded
#   base-token probability. If the objective is the problem, Adam on ld_tanh should recover.
#   Each loss gets its OWN lr bracket: their gradient scales differ by 1/ld_scale or more, so
#   a shared bracket would confound "the loss did nothing" with "the lr was wrong".
#
# ARM C -- train only where the exam discriminates. --fixed-k-frac pins k instead of sampling
#   it log-uniformly over [1, N]. The train k-distribution and the eval grid ARE already
#   matched (both log-uniform on [1, N] -- checked), but the train INTEGRAND (margin) has all
#   its headroom at k > 1e5 while the exam INTEGRAND (accuracy) has all its variance at
#   k in [1e3, 1e5]. Pinning k into that window removes the dense-end headroom entirely.
#   k/N = 9.1e-4 is the grid point where SGD reads 0.86 acc and Adam 0.27.
#
# ARM D -- per-example margin dump. Eval-only (--scores-from), no training: re-scores the
#   EXISTING Adam and SGD score vectors with --dump-per-example so the margin HISTOGRAM at
#   each k is on disk. "mean 30.8 / acc 0.87" is consistent with gap-padding but does not
#   prove it; the histogram does.
#
# DRY=1 to preview. ARMS="A C" to run a subset.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

OUTBASE=${OUTBASE:-results/adamsgd_mlp}
STEPS=${STEPS:-2000}
ARMS=${ARMS:-"A B C D"}
COMMON=(--model llama3 --task addition --dataset arith --nodes mlp
        --method mattr --variant topk --k-schedule log --mode sufficient
        --train-batch-size 1 --steps "$STEPS" --eval-examples 100
        --train-eval-every 250 --train-eval-examples 64)

n=0
sub () {  # sub <name> <outdir> <extra args...>
  local name="$1" out="$2"; shift 2
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out :: $*"
  else mkdir -p "$out"
       sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "${COMMON[@]}" "$@" --output "$out" >/dev/null
       echo "submitted $name"
  fi
  n=$((n+1))
}

if [[ " $ARMS " == *" A "* ]]; then
  # eps spans the plausible |grad| range; lr spans 4 decades so each eps gets its own optimum.
  for eps in 1e-8 1e-6 1e-4 1e-2; do
    for lr in 0.005 0.05 0.5 5.0; do
      sub "aeps-${eps}-lr${lr}" "$OUTBASE/A_eps/eps_${eps}_lr_${lr}" \
          --optimizer adam --loss logit_diff --lr "$lr" --adam-eps "$eps"
    done
  done
fi

if [[ " $ARMS " == *" B "* ]]; then
  # ld_tanh at two scales: 2.0 (~ the clean margin 6.68 / 3, so most examples are in the
  # linear part) and 8.0 (~ the clean margin, so the squash only bites on padded examples).
  for lr in 0.005 0.05 0.5; do
    sub "bl-tanh2-adam-lr${lr}"  "$OUTBASE/B_loss/ld_tanh2_adam_lr_${lr}"  --optimizer adam --loss ld_tanh --ld-scale 2.0 --lr "$lr"
    sub "bl-tanh8-adam-lr${lr}"  "$OUTBASE/B_loss/ld_tanh8_adam_lr_${lr}"  --optimizer adam --loss ld_tanh --ld-scale 8.0 --lr "$lr"
    sub "bl-hinge-adam-lr${lr}"  "$OUTBASE/B_loss/hinge_adam_lr_${lr}"     --optimizer adam --loss hinge --hinge-margin 2.0 --lr "$lr"
    sub "bl-prob-adam-lr${lr}"   "$OUTBASE/B_loss/prob_adam_lr_${lr}"      --optimizer adam --loss prob --lr "$lr"
  done
  # SGD control on the SAME losses. Its bracket sits ~100x higher (no per-coordinate
  # normalisation, and the sigmoid_topk gate slope carries a factor ~k/n).
  for lr in 1.0 10.0 100.0; do
    sub "bl-tanh2-sgd-lr${lr}"   "$OUTBASE/B_loss/ld_tanh2_sgd_lr_${lr}"   --optimizer sgd --loss ld_tanh --ld-scale 2.0 --lr "$lr"
    sub "bl-hinge-sgd-lr${lr}"   "$OUTBASE/B_loss/hinge_sgd_lr_${lr}"      --optimizer sgd --loss hinge --hinge-margin 2.0 --lr "$lr"
  done
fi

if [[ " $ARMS " == *" C "* ]]; then
  # 9.1e-4 = the k=2082 grid point (SGD 0.86 acc, Adam 0.27); the others bracket it by ~3x.
  for kf in 0.0003 0.0009 0.003 0.01; do
    for lr in 0.005 0.05; do
      sub "ck-${kf}-adam-lr${lr}" "$OUTBASE/C_fixedk/kf_${kf}_adam_lr_${lr}" \
          --optimizer adam --loss logit_diff --lr "$lr" --fixed-k-frac "$kf"
    done
  done
  for kf in 0.0009 0.01; do
    sub "ck-${kf}-sgd-lr1.0" "$OUTBASE/C_fixedk/kf_${kf}_sgd_lr_1.0" \
        --optimizer sgd --loss logit_diff --lr 1.0 --fixed-k-frac "$kf"
  done
fi

if [[ " $ARMS " == *" D "* ]]; then
  # Eval-only: no --steps/--optimizer semantics, just re-score two existing vectors with the
  # per-example margin dumped. One job, because the model load + eval-set build amortise.
  A=results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.scores.pt
  S=results/sva_mlp_lr/topk_sgd/lr_1.0/addition_llama3_mlp_sufficient_topk_sgd_bs1.scores.pt
  G=results/sva_sweep/addition_llama3_mlp_ig.scores.pt
  for f in "$A" "$S" "$G"; do [ -e "$f" ] || { echo "MISSING $f"; exit 1; }; done
  if [ "${DRY:-0}" = "1" ]; then echo "DRY perex -> $OUTBASE/D_perexample"
  else mkdir -p "$OUTBASE/D_perexample"
       sbatch -J "perex" scripts/sva/launch/sva_sweep.sbatch --model llama3 --task addition --dataset arith \
         --nodes mlp --mode sufficient --eval-examples 100 --dump-per-example \
         --scores-from "adam:$A" "sgd:$S" "ig:$G" \
         --output "$OUTBASE/D_perexample" >/dev/null
       echo "submitted perex"
  fi
  n=$((n+1))
fi

echo "== ${DRY:+DRY }total $n jobs -> $OUTBASE =="
