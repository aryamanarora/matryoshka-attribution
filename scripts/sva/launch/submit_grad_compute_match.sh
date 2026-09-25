#!/bin/bash
# COMPUTE-MATCH the gradient baselines (IG, IxG, Expected Gradients, AttnLRP) to the REPORTED MAttr
# training budgets on SVA+, and deprecate the 100-example originals (2026-09-05, requested).
#
# BUDGETS. Coverage = model passes, matched to MAttr's steps x bs=1:
#   node substrates      2,000 passes  (MAttr trains 2,000 steps there)
#   mlp / mlp+attn / SAE 5,000 passes  (the 2026-09-03 5k bump)
# IG runs m=10 path points per example, so it gets 1/10 the EXAMPLES (200 / 500) for the same
# pass count. IxG / Expected Gradients (m=1) / AttnLRP are one pass per example (2,000 / 5,000 ex.).
#
# THE POOLS CAP THIS, AND THAT IS A CEILING, NOT A BUG. gradient_scores draws examples WITHOUT
# replacement, and for the deterministic methods repeating an example adds exactly nothing.
# Train pools: simple 480, addition ~1,267, nounpp 1,920, weekdays/months/hours ~2.8-3.1k,
# within_rc 9,600, rc 105,600 -- so e.g. simple/IxG saturates at 480 examples however large
# the budget. Coverage in the paper table reads "min(budget, full train pool)".
#
# STAGING, NOT IN-PLACE. run_tag() does not encode --grad-examples, so the new runs collide by
# filename with the deprecated ones. Every job writes into results/_gradcm_staging/<tree>/;
# finalize_grad_compute_match.sh (submitted here with --dependency=afterany on the whole wave)
# archives the live originals into results/_deprecated_grad_smallN/<tree>/ (symlinks into the
# 5k trees are just removed -- their targets in sva_sweep{,_ferr} ARE the archive) and moves
# the staged files live. Until finalize runs, the live trees still serve the old data, so
# figures rebuilt mid-wave stay complete instead of half-swapped.
#
# GATED ON a smoke job (the since-removed check_gradchunk.py): the whole wave is --dependency=afterok:<gate>, and the
# gate verifies the new non-SAE --grad-batch chunking against the full-batch path on a smoke
# pair. If the gate fails, every job here sits DependencyNeverSatisfied and nothing is written.
#
# TREES (the ones the reported figures read):
#   results/sva_sweep        node          10 tasks (SVA4 + ARITH4 + ioi/qwen2.5 + arc_easy)
#   results/sva_sweep_input  node +input   same 10
#   results/sva_sweep_5k     mlp, mlp+attn 8 tasks (SVA4 + ARITH4; ioi/arc are node-only)
#   results/sva_sweep_ferr5k SAE (MLP out) 8 tasks, IG + IxG only (AttnLRP/mc_ig not defined
#                                          there, matching what the figures draw)
# logit_diff ONLY -- the reported cut. The acc/ce arms stay at 100 ex. (appendix robustness
# grid; bump them the same way if that figure is ever promoted).
#
# GATE=<jobid> required (the smoke job's sbatch id). DRY=1 to preview.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs results/_gradcm_staging

[ "${DRY:-0}" = "1" ] || [ -n "${GATE:-}" ] || { echo "GATE=<check job id> required"; exit 1; }

STAGE=results/_gradcm_staging
IDS=""
n=0
sub() {  # sub <name> <tree> <task> <model> <dataset> <method-args...>
  local name=$1 tree=$2 task=$3 model=$4 ds=$5; shift 5
  local out="$STAGE/$tree"
  if [ "${DRY:-0}" = "1" ]; then echo "DRY $name -> $out :: $*"; n=$((n+1)); return; fi
  mkdir -p "$out"
  local id
  id=$(sbatch --parsable -J "$name" --dependency="afterok:${GATE}" scripts/sva/launch/sva_sweep.sbatch \
       --model "$model" --task "$task" --dataset "$ds" --loss logit_diff \
       --eval-examples 100 "$@" --output "$out")
  IDS="$IDS:$id"; n=$((n+1))
}

# method arg-sets, parameterised by the pass budget B (examples = B, except IG = B/10)
methods() {  # methods <tree> <task> <model> <ds> <B> <extra nodes args...>
  local tree=$1 task=$2 model=$3 ds=$4 B=$5; shift 5
  sub "gcm-ig-$tree-$task"  "$tree" "$task" "$model" "$ds" --method ig  --ig-steps 10 \
      --grad-examples $((B / 10)) "$@"
  sub "gcm-ixg-$tree-$task" "$tree" "$task" "$model" "$ds" --method ixg \
      --grad-examples "$B" "$@"
  if [ "${SAE_ONLY:-0}" = "0" ]; then
    sub "gcm-mcig-$tree-$task" "$tree" "$task" "$model" "$ds" --method mc_ig --ig-steps 1 \
        --seed 42 --grad-examples "$B" "$@"
    sub "gcm-lrp-$tree-$task" "$tree" "$task" "$model" "$ds" --method attnlrp \
        --grad-examples "$B" "$@"
  fi
}

model_of() { [ "$1" = ioi ] && echo qwen2.5 || echo llama3; }
ds_of() { case "$1" in nounpp|rc|simple|within_rc) echo sva;; ioi|arc_easy) echo mib;; *) echo arith;; esac; }

NODE_TASKS="nounpp rc simple within_rc addition months weekdays hours ioi arc_easy"
NEUR_TASKS="nounpp rc simple within_rc addition months weekdays hours"

for t in $NODE_TASKS; do
  methods sva_sweep       "$t" "$(model_of $t)" "$(ds_of $t)" 2000 --nodes node --grad-batch 64
  methods sva_sweep_input "$t" "$(model_of $t)" "$(ds_of $t)" 2000 --nodes node --grad-batch 64 --include-input
done
for t in $NEUR_TASKS; do
  methods sva_sweep_5k "$t" llama3 "$(ds_of $t)" 5000 --nodes mlp --grad-batch 64
  methods sva_sweep_5k "$t" llama3 "$(ds_of $t)" 5000 --nodes mlp+attn_head --grad-batch 64
done
# SAE column: IG + IxG only; the SAE branch has its own chunking, 25 is the tree's convention
for t in $NEUR_TASKS; do
  SAE_ONLY=1 methods sva_sweep_ferr5k "$t" llama3 "$(ds_of $t)" 5000 \
      --nodes mlp_sae_span --sae-error frozen --grad-batch 25
done

# mlp and mlp+attn write the SAME filenames into $STAGE/sva_sweep_5k -- distinct basenames
# (…_mlp_… vs …_mlp-attn_head_…), so one staging dir per TREE is collision-free.

if [ "${DRY:-0}" = "1" ]; then echo "== DRY total $n jobs =="; exit 0; fi
FIN=$(sbatch --parsable -J gcm-finalize --dependency="afterany${IDS}" -p main -c 2 --mem=8G \
      --time=0:30:00 -o logs/gcm_finalize_%j.out \
      --wrap "bash $(pwd)/scripts/sva/launch/finalize_grad_compute_match.sh")
echo "== total $n jobs, gated on $GATE; finalize job $FIN (afterany on the wave) =="
