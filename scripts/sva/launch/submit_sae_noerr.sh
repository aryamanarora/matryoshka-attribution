#!/bin/bash
# SVA+ sweep on the MLP-output SAE basis WITHOUT the reconstruction-error node (--sae-no-error).
#
# WHY. On mlp_sae_span the error node takes 90-100% of MAttr's top 10 on every one of the 8
# tasks, and 30-98% of the top 100 -- against 1 unit in 32,769 by chance. IG does not do this
# (0-10% of its top 10). So the circuit MAttr reports on that basis is largely "the part the
# dictionary could not reconstruct", which is not an interpretable object. This sweep removes
# the error node from the substrate entirely and asks what MAttr selects when it cannot.
#
# *** THESE RUNS ARE NOT COMPARABLE TO THE WITH-ERROR ONES BY AUC. *** With the error node gone
# the term is frozen at its clean value, so the k=0 endpoint (F_patch) is a weaker intervention
# and the faithfulness denominator F_clean - F_patch is a smaller interval. eval_sva computes
# both endpoints through the same hook, so each run is internally consistent (faithfulness still
# runs 0 -> 1) -- but a no-error AUC against a with-error AUC compares two different
# normalisations. Compare RANKINGS across the two settings, and AUCs only within this one.
# See llama.py:_sae_interchange's sae_no_error branch for the exact form and why "keep the clean
# error" cannot be implemented as ke=1 (it is a no-op).
#
# SEPARATE OUTPUT DIR, not results/sva_sweep. The filenames still say mlp_sae_span but `total`
# is 32*seq*32768 rather than *32769, so a no-error json landing in the main sweep dir would be
# picked up by anything that globs there and reshapes by the with-error width. The `_noerr` tag
# keeps names unique, but the dir keeps the accident impossible rather than merely unlikely.
#
# METHODS mirror what the SAE columns of figs/sva_curves_iia.pdf already carry (IG, IxG, both
# MAttr arms, Random), so the no-error facet can be dropped in beside the with-error one without
# a second decision about which series to show. Baselines are mandatory, not optional: dead@k is
# defined as "outside IG's nonzero support" and every rho is against IG, so both are undefined
# without ig/ixg computed ON THIS SUBSTRATE -- the with-error support cannot be reused, the
# vectors are a different length and the indices mean different things.
#
#   SMOKE=1 bash scripts/sva/launch/submit_sae_noerr.sh   # 1 cheap job, run this FIRST
#   DRY=1   bash scripts/sva/launch/submit_sae_noerr.sh   # print only
#   bash scripts/sva/launch/submit_sae_noerr.sh           # 40 jobs (8 tasks x 5 methods)
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

NODES=${NODES:-mlp_sae_span}
ROOT=${ROOT:-results/sva_sae_noerr}
STEPS=${STEPS:-2000}
CHUNK=${CHUNK:-25}
EVAL=${EVAL:-100}
# SVA tasks take --dataset sva; the four arithmetic-wild ones take arith. Same split
# submit_sva_sweep.sh uses; keep them in step.
SVA_TASKS=${SVA_TASKS-nounpp rc simple within_rc}
ARITH_TASKS=${ARITH_TASKS-addition months weekdays hours}

n=0
sub() { local name=$1 out=$2; shift 2
  # A job name that does not encode every varying parameter makes the QUEUED guard below skip
  # whole waves silently -- that has happened three times in this repo (submit_sva_eps.sh,
  # submit_sae_grad.sh, submit_sva_mlp_lr.sh). Task AND method are both in $name.
  if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -qx "$name"; then
    echo "  SKIP $name (already queued)"; return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $out :: $*"
  else mkdir -p "$out"; sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch --model llama3 --nodes "$NODES" \
       --loss logit_diff --sae-no-error "$@" --output "$out" >/dev/null; fi
  n=$((n+1)); }

if [[ "${SMOKE:-0}" == 1 ]]; then
  # Cheapest cell that exercises the new intervention end to end: 8 examples, no training.
  # What to check in the json before launching the rest --
  #   total     == 32 * seq_len * 32768   (32769 would mean the error node is still in)
  #   F_clean   != F_patch                (equal = the intervention is a no-op)
  #   all finite faithfulness             (the new form has no >1 coefficient on the live
  #                                        activation, but verify rather than assume)
  sub noerr_smoke "$ROOT/smoke" --task addition --dataset arith --method ixg \
      --eval-examples 8 --grad-batch 8
  echo "== ${DRY:+DRY }submitted $n SMOKE job -> $ROOT/smoke =="
  exit 0
fi

for spec in "sva $SVA_TASKS" "arith $ARITH_TASKS"; do
  read -ra parts <<< "$spec"; ds=${parts[0]}
  for t in "${parts[@]:1}"; do
    C=(--task "$t" --dataset "$ds")
    sub "noerr_${t}_ig"   "$ROOT" "${C[@]}" --method ig  --ig-steps 10 --eval-examples "$EVAL" --grad-batch "$CHUNK"
    sub "noerr_${t}_ixg"  "$ROOT" "${C[@]}" --method ixg --eval-examples "$EVAL" --grad-batch "$CHUNK"
    sub "noerr_${t}_rand" "$ROOT" "${C[@]}" --method random --seed 42 --eval-examples "$EVAL"
    # Both optimizers at the LRs the with-error mlp_sae_span runs use, so the only difference
    # between the two settings is the substrate. eps=1e-2 is the shipped Adam.
    sub "noerr_${t}_sgd"  "$ROOT" "${C[@]}" --method mattr --variant topk --mode sufficient \
        --k-schedule log --optimizer sgd --lr 1.0 --train-batch-size 1 --steps "$STEPS" --eval-examples "$EVAL"
    sub "noerr_${t}_adam" "$ROOT" "${C[@]}" --method mattr --variant topk --mode sufficient \
        --k-schedule log --optimizer adam --lr 1.0 --adam-eps 1e-2 --train-batch-size 1 \
        --steps "$STEPS" --eval-examples "$EVAL"
  done
done
echo "== ${DRY:+DRY }submitted $n jobs on $NODES (--sae-no-error) -> $ROOT =="
