#!/bin/bash
# Is MAttr's failure on the residual-SAE basis just a mistuned LR? Sweep it on `addition`.
#
# THE DIAGNOSIS THIS TESTS. On resid_sae_span at the imported settings (SGD lr=1.0, Adam
# lr=0.05/eps=1e-2 -- both carried over from the MLP-neuron/MIB sweeps and never tuned here),
# MAttr's top-10,000 units are 100% DEAD: latents that never fire on this task, so f_b = f_c = 0
# and masking them is a no-op. IG's top-10,000 is 0% dead. The two methods are not ranking the
# same universe, which is why MAttr's IIA curve sits far to the right of IG's.
#
# The mechanism is score saturation. sigmoid_topk's backward is
#     grad_s_j = (sp_j / T) * (g_j - gsp/sp_sum),   sp_j = m_j (1 - m_j)
# so once |s_j - tau| >> T the unit's gradient dies and it freezes wherever it landed. At lr=1.0
# the scores run away -- range 2.6e7 on addition, 3.3e17 on hours -- and by the end 0.0000% of
# units are still inside the active band at k=1000. The probe plateauing by step ~500 is gradient
# death, not convergence. The prediction: a much smaller lr keeps the mask in the band, live
# latents accumulate real signal, and dead@k collapses.
#
# *** lr AND T ARE ONE KNOB, NOT TWO, so this sweeps lr alone at the default T=0.5. *** From the
# first step (zero init => every score equal => m = k/n and sp uniform):
#     Delta s = -lr * (sp/T) * (g_j - gbar)
# and the mask only ever sees (s - tau)/T, so the dynamics depend on lr/T^2. Sweeping both would
# re-measure the same axis twice.
#
# WHY `addition` (user's choice): it is mid-pack on every symptom -- score range 2.6e7 (vs 1e6 on
# nounpp, 3.3e17 on hours), 0.88%-4% live support, and it is one of the four arithmetic tasks that
# also drove the pre-fix divergence, so a fix that shows up here is not a quirk of the easiest cell.
#
# SEPARATE OUTPUT DIR PER LR IS MANDATORY, not tidiness: run_tag() does NOT encode --lr for
# method=mattr (eval_sva.py:635 says so outright), so every point of this sweep writes the SAME
# filename. Without the per-lr dir the sweep silently collapses to whichever job finished last.
#
# Reading the result: the headline is not acc_auc alone but acc_auc TOGETHER WITH dead@10k from
# scripts/sva/compare_sae_ranks.py. An lr that improves acc_auc while still selecting 100% dead
# latents has found a different bug, not fixed this one.
#
#   bash scripts/sva/launch/submit_sae_lr_sweep.sh        # 13 jobs
#   DRY=1 bash scripts/sva/launch/submit_sae_lr_sweep.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

TASK=${TASK:-addition}
DS=${DS:-arith}
NODES=${NODES:-resid_sae_span}
ROOT=${ROOT:-results/sae_lr_sweep}
STEPS=${STEPS:-2000}
KSCHED=${KSCHED:-log}
# Adam eps is an IDENTITY knob at this scale, not a numerical one: at eps=1e-8 the update is
# effectively sign(g), so the score is a VOTE COUNT over steps and no single large-gradient step
# can set the ranking -- the exact pathology the SAE trace shows. eps=1e-2 (the MLP-basis fix)
# preserves magnitude, which is what is toxic here. run_tag omits the fragment at the 1e-8
# default, so those runs land as `..._adam_bs1` and cannot collide with the `_eps1e-2` ones.
EPS=${EPS:-1e-2}
# Per-step gradient normalisation (0 = off). See trainer.learn_scores for why this is the
# middle term between full-magnitude SGD and sign(g) Adam.
GN=${GN:-0}
# SGD: bracket DOWNWARD from the shipped lr=1.0. The measured overshoot on addition is ~1e7x, so
# the grid has to reach 1e-8 to be sure the minimum is bracketed rather than clipped at the edge.
# `${VAR-default}`, NOT `${VAR:-default}`: the colon form treats SGD_LRS="" as unset and falls
# back to the full grid, so `SGD_LRS="" bash ...` (submit only the Adam half) silently
# re-submits all 9 SGD points on top of the ones already queued.
SGD_LRS=${SGD_LRS-"1.0 0.1 0.01 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8"}
# Adam: the OPPOSITE regime. Its scores never leave the band (range 3.76) and 83% of units stay
# exactly tied, and over 2000 steps at lr=0.05 the realised drift is ~0.0019/step -- 4% of the
# nominal step size, i.e. the updates are almost entirely cancelling. That is consistent with
# lr being too LOW (not enough signal accumulates) as well as with sign-flipping noise from
# --train-batch-size 1, so the grid must bracket UPWARD, not just downward. It initially did not,
# which would have made "Adam lr too low" untestable by construction.
ADAM_LRS=${ADAM_LRS-"2.0 1.0 0.5 0.1 0.05 0.01 1e-3"}

VARIANT=${VARIANT:-topk}
COMMON=(--model llama3 --task "$TASK" --dataset "$DS" --nodes "$NODES" --loss logit_diff
        --method mattr --variant "$VARIANT" --k-schedule "$KSCHED" --mode sufficient
        --train-batch-size 1 --eval-examples 100 --steps "$STEPS" --grad-norm "$GN")

n=0
sub() { # $1=jobname $2=outdir ; rest = extra args
  local name=$1 out=$2; shift 2
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $out :: $*"
  else mkdir -p "$out"; sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "${COMMON[@]}" "$@" \
       --output "$out" >/dev/null; fi
  n=$((n+1))
}

for lr in $SGD_LRS; do
  sub "saelr_gc${GN}_sgd_${lr}"  "$ROOT/${VARIANT}_${KSCHED}_gc${GN}_sgd_lr${lr}"  --optimizer sgd  --lr "$lr"
done
for lr in $ADAM_LRS; do
  sub "saelr_gc${GN}_ada${EPS}_${lr}" "$ROOT/${VARIANT}_${KSCHED}_gc${GN}_adam_eps${EPS}_lr${lr}" --optimizer adam --lr "$lr" --adam-eps "$EPS"
done
echo "== ${DRY:+DRY }submitted $n jobs on $TASK/$NODES -> $ROOT =="
