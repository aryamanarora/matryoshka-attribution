#!/bin/bash
# Adam eps x lr grid for MAttr on the MLP-output SAE basis, under a chosen --sae-error mode.
#
# WHY THIS AND NOT THE REST OF THE SWEEP. The first 4 SVA cells of the no-error sweep put
# MAttr+Adam BELOW IG (acc-AUC 0.477 vs 0.595; faith-AUC 0.611 vs 0.665), having led it
# comfortably with the error node in (0.808 vs 0.663). Before that is written down as "MAttr's
# advantage on this basis was the error nodes", it has to survive the hyperparameter check the
# MLP-neuron result already went through: at 2.29M mask logits torch's default eps=1e-8 makes
# Adam's update ~sign(g)*lr, so the learned score is a signed COUNT of steps and the run is
# measuring a degeneracy rather than the method. The no-error substrate is 5.2M logits, larger
# still, and lr=1.0 was imported from the WITH-error runs -- neither number has been checked here.
#
# THE GRID IS THE MLP-NEURON ONE, deliberately: same six eps and four lr values as
# submit_adam_vs_sgd_mlp.sh's ARM A / plot_adamsgd_mlp_diag.EPS_GRID x LR_GRID, so the resulting
# heatmap can be read against figs/adamsgd_epsgrid*.pdf without a scale change. eps spans the
# plausible |grad| range; lr spans four decades so each eps gets its own optimum.
#
# EVERYTHING ELSE IS HELD AT THE NO-ERROR SWEEP'S SETTINGS (topk / log-k / logit_diff /
# sufficient / bs 1 / 2000 steps / 100 eval examples), so the only things varying are eps and lr.
#
# ERRMODE PICKS THE INTERVENTION, and it decides what the grid can be compared against:
#   frozen (default) each error term is measured against its OWN reconstruction, so the error
#          node is an ordinary scored unit rather than a master switch. SAME endpoints as
#          `absorb` (F_clean 6.672 / F_patch -6.651 on addition), so these AUCs ARE directly
#          comparable to the runs already in results/sva_sweep.
#   none   the error node is dropped; F_patch moves to -4.143 and every AUC inflates (Random
#          alone goes 0.023 -> 0.142 acc-AUC on nothing but the easier k=0 state). Comparable
#          only within itself. This is what the first pass of this grid ran.
#   absorb the legacy shortcut; kept reachable but there is no reason to sweep it.
#
#   DRY=1 bash scripts/submit_saenoerr_epslr.sh              # print only
#   bash scripts/submit_saenoerr_epslr.sh                    # 24 jobs, frozen
#   ERRMODE=none bash scripts/submit_saenoerr_epslr.sh       # the original none grid
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

ERRMODE=${ERRMODE:-frozen}
# Which dictionary. eval_sva picks the Llama-Scope repo from this (LXM for mlp_sae_span, LXR for
# resid_sae_span), so nothing else needs to change. NODETAG keeps the ORIGINAL mlp paths intact
# -- results/saefrozen_epslr is already on disk and referenced by plots/plot_epsgrid_facets.py.
NODES=${NODES:-mlp_sae_span}
# NODETAG keeps the ORIGINAL mlp_sae_span paths intact -- results/saefrozen_epslr is on disk and
# plots/plot_epsgrid_facets.py points at it. NSHORT keeps job names unique per substrate, which the
# QUEUED guard below depends on.
#
# ERRARG IS EMPTY FOR NON-SAE SUBSTRATES. `node` and `mlp` have no reconstruction error to
# intervene on, so passing --sae-error there would be a flag eval_sva accepts and silently
# ignores -- and OUTBASE would then claim an intervention the run does not have.
case "$NODES" in
  mlp_sae_span)   NODETAG=""       ; NSHORT=""  ;;
  resid_sae_span) NODETAG="_resid" ; NSHORT="r" ;;
  node)           NODETAG="_node"  ; NSHORT="n" ;;
  mlp)            NODETAG="_mlpn"  ; NSHORT="m" ;;
  *) echo "unexpected NODES=$NODES"; exit 1 ;;
esac
case "$NODES" in
  *_sae_span) ERRARG=(--sae-error "$ERRMODE"); MODETAG="sae${ERRMODE}" ;;
  *)          ERRARG=()                      ; MODETAG=""              ;;
esac
# One dir and one job-name prefix per mode: run_tag encodes the mode (`_ferr` / `_noerr`) but
# NOT the lr, so two modes sharing a tree would collide cell for cell.
# MODEL is a first-class knob so the same grid can be run cross-model. Verified for gemma2:
# eval_sva loads through AutoModelForCausalLM and never imports transformer_lens, so CLAUDE.md's
# "never evaluate a gemma2 cell in the L2A venv" rule -- which is about eval_mib.py's
# HookedTransformer path -- does not apply here.
MODEL=${MODEL:-llama3}
MTAG=$([ "$MODEL" = llama3 ] && echo "" || echo "_$MODEL")
OUTBASE=${OUTBASE:-results/${MODETAG:+${MODETAG}_}epslr${NODETAG}${MTAG}}
PREFIX=${PREFIX:-$([ "$ERRMODE" = none ] && echo nesl || echo fesl)$NSHORT${MODEL:0:1}}
STEPS=${STEPS:-2000}
TASK=${TASK:-addition}
DS=${DS:-arith}
# Same six / four as plot_adamsgd_mlp_diag.EPS_GRID and LR_GRID. Override to extend a bracket,
# but keep the defaults if the figure is meant to sit beside the MLP-neuron one.
EPSES=${EPSES:-"1e-8 1e-6 1e-4 1e-2 1e-1 1e0"}
LRS=${LRS:-"0.005 0.05 0.5 5.0"}
COMMON=(--model "$MODEL" --task "$TASK" --dataset "$DS" --nodes "$NODES" "${ERRARG[@]}"
        --method mattr --variant topk --k-schedule log --mode sufficient --loss logit_diff
        --optimizer adam --train-batch-size 1 --steps "$STEPS" --eval-examples 100
        --train-eval-every 250 --train-eval-examples 64)

n=0
sub () {  # sub <name> <outdir> <extra args...>
  local name="$1" out="$2"; shift 2
  # The job name carries BOTH varying parameters. A name that does not has silently skipped
  # whole waves behind this guard three times in this repo (submit_sva_eps.sh,
  # submit_sae_grad.sh, submit_sva_mlp_lr.sh) -- run_tag encodes eps but not lr, so name
  # collisions here would also overwrite on disk.
  if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -qx "$name"; then
    echo "  SKIP $name (already queued)"; return; fi
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out :: $*"
  else mkdir -p "$out"
       sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "${COMMON[@]}" "$@" --output "$out" >/dev/null
       echo "submitted $name"
  fi
  n=$((n+1))
}

subraw () {  # like sub, but WITHOUT $COMMON -- COMMON is the MAttr+Adam grid config, and a
             # reference run that inherited its --method/--optimizer would be a duplicate grid
             # cell wearing a baseline's name.
  local name="$1" out="$2"; shift 2
  if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -qx "$name"; then
    echo "  SKIP $name (already queued)"; return; fi
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out :: $*"
  else mkdir -p "$out"
       sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "$@" --output "$out" >/dev/null
       echo "submitted $name"
  fi
  n=$((n+1))
}

# REFERENCE RUNS for the row's two rho panels and for the baseline columns. They are NOT part of
# the grid and land in $OUTBASE/refs: plot_epsgrid_facets correlates each grid cell against IG and
# against MAttr+SGD, and both references must be the SAME intervention as the grid or the
# correlation folds an intervention change into a hyperparameter figure.
if [[ "${REFS:-0}" == 1 ]]; then
  R=(--task "$TASK" --dataset "$DS" --model "$MODEL" --nodes "$NODES" "${ERRARG[@]}"
     --loss logit_diff)
  subraw "${PREFIX}ref-ig"   "$OUTBASE/refs" "${R[@]}" --method ig --ig-steps 10 --eval-examples 100 --grad-batch 25
  subraw "${PREFIX}ref-ixg"  "$OUTBASE/refs" "${R[@]}" --method ixg --eval-examples 100 --grad-batch 25
  subraw "${PREFIX}ref-rand" "$OUTBASE/refs" "${R[@]}" --method random --seed 42 --eval-examples 100
  subraw "${PREFIX}ref-sgd"  "$OUTBASE/refs" "${R[@]}" --method mattr --variant topk --mode sufficient \
      --k-schedule log --optimizer sgd --lr 1.0 --train-batch-size 1 --steps "$STEPS" --eval-examples 100
fi

for eps in $EPSES; do
  for lr in $LRS; do
    sub "${PREFIX}-${eps}-lr${lr}" "$OUTBASE/eps_${eps}_lr_${lr}" --lr "$lr" --adam-eps "$eps"
  done
done
echo "== ${DRY:+DRY }total $n jobs on $MODEL/$NODES${ERRARG:+ (--sae-error $ERRMODE)} -> $OUTBASE =="
