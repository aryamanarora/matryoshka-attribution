#!/bin/bash
# COMPUTE-MATCHED comparison on the SAE residual basis: every method gets the same budget of
# EXAMPLE-BACKWARDS (one backward pass over one example), instead of the arbitrary settings the
# per-substrate sweeps inherit from the MLP basis.
#
# WHY THE UNIT IS EXAMPLE-BACKWARDS. A gradient method does n_draws backward passes over a batch
# of B examples; MAttr does `steps` backward passes over batches of 1. Counting *passes* would
# call a B=100 pass equal to a B=1 pass, which is wrong by 100x in FLOPs. Counting
# draws x examples is the honest unit. At the sweep's default settings the three methods were
# nowhere near matched:  IxG 1x100 = 100,  IG 10x100 = 1,000,  MAttr 2000x1 = 2,000.
#
# THE TARGET IS 4,000, and only two of the three can actually reach it:
#   MAttr   --steps 4000                          -> 4,000   (exact)
#   IG      400 examples x 10 draws               -> 4,000   (exact; every task has >=433)
#   IxG     min(4000, usable) examples x 1 draw   -> 433..4,000, TASK-DEPENDENT
#
# *** IxG CANNOT SPEND THE BUDGET ON MOST TASKS, AND THAT IS A PROPERTY OF THE METHOD. *** It is
# one closed-form pass over the data, so its compute is bounded by the dataset, not by a knob.
# Usable pairs at the modal prompt length (the filter gradient_scores applies):
#   simple 433, addition 1267, nounpp 1717, hours 2057, months 2705, weekdays 2759,
#   within_rc 7777, rc 81668.
# So on simple/addition/nounpp/hours/months/weekdays IxG is budget-STARVED by construction and
# its number must be reported with the budget it actually got -- eval_sva logs
# "SAE gradient attribution: N examples x D draws = N*D example-backwards".
#
# --grad-batch 25 is a MEMORY bound, not a statistical one: scores sum over examples, so
# chunking is exact (verified against the single-batch run: relative max score difference 5e-3,
# Spearman 0.996 on the top-20k, i.e. fp nondeterminism from differing batch shapes). Without it
# the single-batch path stores 32 layers x [B,P,d_model] three times and OOMs well before B=4000.
#
# SEPARATE OUTPUT DIR because --grad-examples and --grad-batch are NOT in run_tag: an IG run at
# 400 examples writes the same filename as the sweep's 100-example one and would overwrite it.
# --steps IS in the tag (`_s4000`), but the dir keeps the whole experiment together anyway.
#
# logit-diff only -- this is a compute question, not a loss-robustness one.
#
#   bash scripts/submit_sae_budget.sh        # 32 jobs
#   DRY=1 bash scripts/submit_sae_budget.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

OUT=${OUT:-results/sae_budget4k}
NODES=${NODES:-resid_sae_span}
BUDGET=${BUDGET:-4000}
CHUNK=${CHUNK:-25}
IG_STEPS=${IG_STEPS:-10}
MATTR=(--method mattr --variant topk --k-schedule log --mode sufficient
       --train-batch-size 1 --eval-examples 100)
declare -A DS=( [nounpp]=sva [rc]=sva [simple]=sva [within_rc]=sva
                [addition]=arith [months]=arith [weekdays]=arith [hours]=arith )
TASKS=(nounpp rc simple within_rc addition months weekdays hours)
mkdir -p "$OUT"
n=0
sub() { # $1=name ; rest = args
  local name=$1; shift
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name :: $*"
  else sbatch -J "$name" sva_sweep.sbatch "$@" --output "$OUT" >/dev/null; fi
  n=$((n+1))
}
for t in "${TASKS[@]}"; do
  d=${DS[$t]}
  common=(--model llama3 --task "$t" --dataset "$d" --nodes "$NODES" --loss logit_diff)
  # IxG: one draw, as many examples as the budget (capped by the dataset).
  sub "b4k_ixg_$t"  "${common[@]}" --method ixg --eval-examples 100 \
      --grad-examples "$BUDGET" --grad-batch "$CHUNK"
  # IG: budget / ig_steps examples, ig_steps draws.
  sub "b4k_ig_$t"   "${common[@]}" --method ig  --eval-examples 100 --ig-steps "$IG_STEPS" \
      --grad-examples $((BUDGET / IG_STEPS)) --grad-batch "$CHUNK"
  # MAttr: one example per step, so steps == budget.
  sub "b4k_msgd_$t" "${common[@]}" "${MATTR[@]}" --optimizer sgd  --lr 1.0  --steps "$BUDGET"
  sub "b4k_mada_$t" "${common[@]}" "${MATTR[@]}" --optimizer adam --lr 0.05 --adam-eps 1e-2 \
      --steps "$BUDGET"
done
echo "== ${DRY:+DRY }submitted $n at budget $BUDGET example-backwards -> $OUT =="
