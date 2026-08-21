#!/bin/bash
# Edge-level MIB, SOFT-fwd MAttr under SGD, both k-schedules. Validation only.
#
# The only SGD arm that existed at edge level was id-STE (submit_mib_edge_identity_sgd.sh),
# which is exactly LR-invariant and so needs no --lr -- that is why mib_edge.sbatch has no LR
# argument and why this script has to use the direct `sbatch --wrap` form of
# submit_edge_lr05.sh instead.
#
# LR IS IMPORTED FROM THE NODE-LEVEL OPTIMA, NOT SWEPT HERE:
#   log-k      lr=1.0   (node dir softlog_sgd_lr_1.0, peak 1.886 CPR AUC, bracketed)
#   uniform-k  lr=3.0   (node dir softuni_sgd_lr_3.0, peak 2.031 CPR AUC, bracketed)
# Those brackets are NODE brackets. Soft-fwd + SGD is LR-sensitive (Adam normalises the k/n
# gate-slope factor away, SGD does not), and edge n is 207-1507x node n -- gpt2 157 -> 32,491,
# llama3 1057 -> 1,592,881. So these two LRs are UNBRACKETED at edge scale. The rows are a
# same-LR transfer of the node setting, comparable to the node SGD rows; they are NOT evidence
# about where SGD's edge optimum lies, and must not be captioned as an optimiser claim without
# an edge LR sweep (cf. the optimum-ring convention in plot_optimizer_lr.NO_RING).
#
# 9 cells, the standard edge PAIRS (no llama3 arc_* at edge level), llama3 capped to
# eval-200/batch-2 per CLAUDE.md. GEMMA2 RUNS UNDER THE MIB VENV (TL 2.15.4) -- submit_edge_lr05.sh
# points every model at $ABS/.venv, whose TL 3.2.1 computes a wrong Gemma-2 forward (525673a);
# that is a defect of that script, not a pattern to copy.
#
# RESULT (2026-08-21, log-k complete 9/9 vs mib_edge_topk_log_lr05): the transfer FAILS.
# Mean area_under -1.221, and it is llama3-concentrated -- arith_sub -3.45, ioi -3.74, mcqa -2.42;
# gpt2 -1.84, qwen2.5 -0.34/-0.99, gemma2 actually +0.95/+0.83/+0.02 (but gemma2's headline edge
# numbers are the weakest on the board, ~2.85, so that is a low bar, not a win). uniform-k at
# lr=3.0 is at parity overall (+0.154) and loses only on llama3 (ioi -2.00). This is the
# unbracketed-LR hazard the header predicted, now observed -- do NOT report these rows as an
# optimiser claim. Use LR= below to bracket llama3 before drawing any conclusion.
#
# DRYRUN=1 to preview.  ONLY=gemma2 for one model.  SCHEDS="log" for one k-schedule.
# LR=0.3 overrides the imported node optimum for ALL selected schedules; the output dir follows
# the LR, so a probe never overwrites the lr=1.0/3.0 rows.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
PY_L2A=$ABS/.venv/bin/python                     # gpt2 / qwen2.5 / llama3
PY_TL2=$ABS/MIB-circuit-track/.venv/bin/python   # gemma2 ONLY (TL 2.15.4)
PP_TL2="PYTHONPATH=$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src "
DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-}
LR=${LR:-}
SCHEDS=${SCHEDS:-"log uniform"}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy"
)
n=0
for sched in $SCHEDS; do
  case $sched in
    log)     lr=${LR:-1.0}; out=mib_edge_softlog_sgd_lr_$lr; tag=slog ;;
    uniform) lr=${LR:-3.0}; out=mib_edge_softuni_sgd_lr_$lr; tag=suni ;;
    *) echo "unknown schedule $sched" >&2; exit 1 ;;
  esac
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then continue; fi
    py=$PY_L2A; pp=""
    case $model in
      gpt2|qwen2.5) cpus=3; mem=64G;  tlim=10:00:00; bs=5; ev=0 ;;
      gemma2)       cpus=4; mem=96G;  tlim=16:00:00; bs=5; ev=0; py=$PY_TL2; pp=$PP_TL2 ;;
      llama3)       cpus=5; mem=128G; tlim=24:00:00; bs=2; ev=200 ;;
    esac
    name="esgd-${tag}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$pp$py scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule $sched \
--masking topk --optimizer sgd --mode sufficient --lr $lr --split validation --train-split train \
--batch-size $bs --eval-examples $ev --output results/$out"
    if [ "$DRYRUN" = "1" ]; then echo "DRY $name lr=$lr -> $out [py=${py#$ABS/}]${pp:+ [+PYTHONPATH]}"; else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name lr=$lr"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n edge soft-fwd SGD validation jobs =="
