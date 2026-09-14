#!/bin/bash
# Replicate the residual-SAE diagnosis on the MLP-OUTPUT SAE basis (mlp_sae_span, Llama-Scope LXM).
#
# WHAT THIS IS FOR. Everything established on resid_sae_span was measured on ONE substrate, so it
# is not yet known whether the findings are about SAE bases in general or about the residual
# stream specifically. This is the same cell (addition/llama3/logit_diff) on the other dictionary.
# Three claims are up for replication, and they are independent -- any of them can fail alone:
#
#   1. THE SPIKE. On resid, |grad| swings ~10 orders of magnitude with k and one large-k draw
#      freezes the score vector by step 3 (|s|max 1609, never moves again) -- vs the MLP-NEURON
#      basis where |grad| peaks at 0.29 and training proceeds normally. Is the MLP-output SAE
#      like the residual SAE (dictionary is what matters) or like MLP neurons (site is what
#      matters)? MATTR_LOG_EVERY=1 on the two 60-step probes answers this directly.
#
#   2. THE PATH. MAttr's ranking is orthogonal to IG's (rho ~0.02 vs IG-IxG's 0.51) because IG
#      integrates over the INPUT EMBEDDING while MAttr interpolates latents toward CACHED patch
#      latents. That is a property of the two algorithms, not of the dictionary, so it SHOULD
#      replicate exactly. If it does not, the path story is wrong. Expected Gradients (mc_ig) is the
#      discriminating cell: MAttr's alpha draws on IG's path -- it scored rho=0.80 with IG on
#      resid, and must again here.
#
#   3. THE FIX. uniform k-schedule + grad clip=1.0 was the best MAttr on resid (0.198 vs 0.152
#      for the shipped log-k default). Both arms are included, and so is log-k+clip, because
#      uniform-k ALONE was WORSE (0.042): it sits at alpha~0.5 where the gate slope a(1-a) peaks,
#      which enlarges the spikes, so the clip is what makes it pay. Without the log-k+clip cell
#      the two effects cannot be separated.
#
# BASELINES ARE MANDATORY HERE, not optional. dead@k is defined as "outside IG's nonzero support"
# and every rho is against IG, so both are meaningless without ig/ixg ON THIS BASIS -- the resid
# support cannot be reused (different dictionary, different indices, different length). random is
# the floor that makes dead@k readable (it lands at the dead fraction of the basis, ~0.98 on
# resid) and is what showed MAttr was WORSE than chance there.
#
# --grad-batch 25 bounds memory only; scores sum over examples so chunking is exact.
# Smoke on nounpp/mlp_sae_span passed earlier (IG 0.664, IxG 0.591, 0/24 non-finite), so the
# substrate itself works; nothing here is a first run of the code path.
#
#   bash scripts/sva/launch/submit_mlpsae_replication.sh        # 12 jobs
#   DRY=1 bash scripts/sva/launch/submit_mlpsae_replication.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

TASK=${TASK:-addition}
DS=${DS:-arith}
NODES=${NODES:-mlp_sae_span}
ROOT=${ROOT:-results/mlpsae}
STEPS=${STEPS:-2000}
CHUNK=${CHUNK:-25}
COMMON=(--model llama3 --task "$TASK" --dataset "$DS" --nodes "$NODES" --loss logit_diff)
MATTR=(--method mattr --variant topk --mode sufficient --train-batch-size 1 --eval-examples 100
       --steps "$STEPS")

n=0
sub() { local name=$1 out=$2; shift 2
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $out :: $*"
  else mkdir -p "$out"; sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "${COMMON[@]}" "$@" --output "$out" >/dev/null; fi
  n=$((n+1)); }

# --- baselines: needed for dead@k's support AND for every rho ---
sub mlpsae_ig      "$ROOT/base" --method ig  --ig-steps 10 --eval-examples 100 --grad-batch "$CHUNK"
sub mlpsae_ixg     "$ROOT/base" --method ixg --eval-examples 100 --grad-batch "$CHUNK"
sub mlpsae_mcig    "$ROOT/base" --method mc_ig --ig-steps 2 --seed 42 --eval-examples 100 --grad-batch "$CHUNK"
sub mlpsae_rand    "$ROOT/base" --method random --seed 42 --eval-examples 100

# --- MAttr: shipped default (log-k, no clip), both optimizers ---
sub mlpsae_log_sgd  "$ROOT/log_sgd"      "${MATTR[@]}" --k-schedule log --optimizer sgd  --lr 1.0
sub mlpsae_log_ada  "$ROOT/log_adam"     "${MATTR[@]}" --k-schedule log --optimizer adam --lr 1.0 --adam-eps 1e-2

# --- the fix, and the cell that separates its two ingredients ---
sub mlpsae_logc_sgd "$ROOT/log_gc1_sgd"  "${MATTR[@]}" --k-schedule log     --optimizer sgd  --lr 1.0 --grad-norm 1.0
sub mlpsae_unic_sgd "$ROOT/uni_gc1_sgd"  "${MATTR[@]}" --k-schedule uniform --optimizer sgd  --lr 1.0 --grad-norm 1.0
sub mlpsae_unic_ada "$ROOT/uni_gc1_adam" "${MATTR[@]}" --k-schedule uniform --optimizer adam --lr 1.0 --adam-eps 1e-2 --grad-norm 1.0
sub mlpsae_uni_sgd  "$ROOT/uni_sgd"      "${MATTR[@]}" --k-schedule uniform --optimizer sgd  --lr 1.0

# --- path test: alpha pinned to IG's grid, so any residual gap is the PATH ---
sub mlpsae_fk05     "$ROOT/fk0.5"        "${MATTR[@]}" --fixed-k-frac 0.5 --optimizer sgd --lr 1.0 --grad-norm 1.0

# --- 60-step instrumented probe: is |grad| spiky here like resid, or flat like MLP neurons? ---
if [[ "${DRY:-0}" == 1 ]]; then echo "  mlpsae_probe -> $ROOT/probe (MATTR_LOG_EVERY=1)"; n=$((n+1))
else mkdir -p "$ROOT/probe"
  sbatch -J mlpsae_probe --export=ALL,MATTR_LOG_EVERY=1 scripts/sva/launch/sva_sweep.sbatch "${COMMON[@]}" \
    --method mattr --variant topk --mode sufficient --k-schedule log --train-batch-size 1 \
    --eval-examples 8 --steps 60 --optimizer sgd --lr 1.0 --output "$ROOT/probe" >/dev/null
  n=$((n+1)); fi

echo "== ${DRY:+DRY }submitted $n on $TASK/$NODES -> $ROOT =="
