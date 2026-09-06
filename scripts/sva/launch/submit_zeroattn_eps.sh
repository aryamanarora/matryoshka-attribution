#!/bin/bash
# Adam eps ladder on the ONE cell of the SVA+ grid where the eps=1e-2 arm is cooked:
# mlp+attn_head substrate, ZERO ablation. Everywhere else raising eps helps or is neutral.
#
# WHAT PROMPTED IT. On results/sva_zeroabl at mlp+attn_head, the k-independent training probe
# (train_eval_log, the full metric suite on a fixed train subset over the whole k grid) shows
# MAttr+Adam at eps=1e-2 STALLING rather than diverging -- on rc it goes 0.45 -> 0.60 by step
# 400 and then sits in a 0.54-0.65 band for the remaining 1600 steps, while the same config on
# the mlp-only substrate reaches 0.81. It is not overshoot: there is no rise-then-collapse and
# no oscillation, just no further progress.
#
# THE NUMBER THAT MAKES THIS WORTH RUNNING: on rc, DEFAULT eps=1e-8 scores 0.944 acc-AUC and
# eps=1e-2 scores 0.529, against an SGD(lr=1) 0.799 and a 0.510 random floor. That is the exact
# REVERSE of the neuron-substrate result the eps arm exists for (0.388 -> 0.490 on addition/mlp),
# so "bigger eps is better at fine granularity" is not a law and this cell is the counterexample.
# The ladder is what says whether the optimum is interior and where.
#
# TWO TASKS, and they answer different questions:
#   rc        THE discriminating cell -- methods span 0.51 (floor) to 0.94 there, so a change in
#             eps has room to show up.
#   addition  the cell the original eps sweep was run on (A_eps), included for continuity. Be
#             warned it is nearly useless here: at mlp+attn_head/zero EVERY method on disk sits
#             in 0.503-0.563 against a 0.507 floor, so this arm is expected to be flat and a
#             flat result is NOT evidence that eps does not matter -- it is evidence the cell
#             cannot measure anything.
#
# eps=1e-8 and eps=1e-2 are already on disk in results/sva_zeroabl for both tasks, so only the
# four interior/high rungs are submitted. Everything else is submit_sva_sweep.sh's zero-ablation
# recipe verbatim (topk/adam/log-k, lr 0.05, 2000 steps, bs 1, 100 eval examples), so the only
# thing varying across the ladder is eps.
#
# SEPARATE OUTPUT DIR, per the repo's probe-sweep convention. run_tag now encodes eps so these
# would not actually collide with the sweep dir, but they are a probe rather than grid cells and
# results/sva_zeroabl feeds a figure whose completeness check counts what is in it.
#
# NO SEED REPLICATES, and that is a real gap: --seed is not in run_tag for mattr, so replicates
# would overwrite each other and need a dir apiece. The rc/eps=1e-8 0.944 in particular is a
# single run and has not been shown to reproduce.
#
#   bash scripts/sva/launch/submit_zeroattn_eps.sh        # submit
#   DRY=1 bash scripts/sva/launch/submit_zeroattn_eps.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

OUT=${OUT:-results/adamsgd_mlp/N_zeroattn_eps}
n=0
for spec in "rc sva" "addition arith"; do
  read -r task ds <<<"$spec"
  for eps in 1e-6 1e-4 1e-1 1e0; do
    d="$OUT/${task}_eps_${eps}"
    f="$d/${task}_llama3_mlp-attn_head_sufficient_topk_adam_eps${eps}_zeroabl_bs1.json"
    if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then echo "skip $task eps=$eps"; continue; fi
    if [[ "${DRY:-0}" == 1 ]]; then echo "DRY zae-${task}-${eps} -> $f"; else
      mkdir -p "$d"
      sbatch -J "zae-${task}-${eps}" scripts/sva/launch/sva_sweep.sbatch \
        --model llama3 --task "$task" --dataset "$ds" --nodes mlp+attn_head \
        --method mattr --variant topk --optimizer adam --k-schedule log --mode sufficient \
        --loss logit_diff --lr 0.05 --adam-eps "$eps" \
        --train-batch-size 1 --steps 2000 --eval-examples 100 \
        --ablation zero --output "$d" >/dev/null
    fi
    n=$((n+1))
  done
done
echo "== ${DRY:+DRY }submitted $n -> $OUT =="
