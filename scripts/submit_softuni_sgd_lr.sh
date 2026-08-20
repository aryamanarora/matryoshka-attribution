#!/bin/bash
# LR sweep for SOFT-fwd MAttr (--masking topk) + UNIFORM k-schedule + *SGD*, node level, train->val.
#
# Sister script to submit_softlog_sgd_lr.sh. That one moves the optimizer while holding the
# k-schedule at the headline's (log); this one moves BOTH, so the pair spans the
# k-schedule x optimizer square at a shared lr grid:
#
#   forward   k-sched   optimizer   sweep dir                paper role
#   soft      log       adam        topklog_lr_*             headline (\ourmethod{})
#   soft      uniform   adam        mib_node_topk_uniform_lr05 (lr=0.05 only) + final_node (0.01)
#   soft      log       sgd         softlog_sgd_lr_*         running now
#   soft      uniform   sgd         softuni_sgd_lr_*         <-- THIS SCRIPT
#
# WHY BOTHER, given "+ unif k" is an ablation and not the headline: the uniform-k Adam ablation
# scores *better* than the headline on area_under (mib_results.tex "+ unif k" 2.09 vs 1.88) and
# *worse* on acc-AUC, which is exactly the dense-end-vs-sparse-end split that CLAUDE.md warns
# makes uniform-k look deceptively good. The SGD arm at log-k reproduces that same split from
# the other direction (at lr=0.05 it is -0.005 on acc-AUC but -0.373 on area_under vs Adam), so
# whether the optimizer effect and the k-schedule effect are the SAME effect is currently
# unidentified. Crossing them at a shared grid is the only way to tell.
#
# GRID IS "0.05 0.1 0.3 1.0" -- byte-identical to the log-k SGD resubmission, on purpose. Do not
# "complete" it downward with 0.005/0.01: the log-k arm already ran those and they lose to Adam
# on 5/5 cells, and the SVA neuron substrate puts soft+SGD's optimum at lr=1.0. 1.0 is outside
# tabs/lr_sweep.tex's Adam grid and needs its own note if it wins.
#
# INHERITED TRAPS (see submit_softlog_sgd_lr.sh for the full write-up):
#  1. --lr is passed on the command line, NOT through mib_node_seed.sbatch, which reads only
#     $1..$8 and silently drops a 9th "lr" argument (every job would then run at the default).
#  2. gemma2 runs under MIB-circuit-track/.venv (TL 2.15.4) -- the L2A venv's Gemma-2 forward is
#     wrong (CLAUDE.md, proved in 525673a) -- and that venv needs an explicit PYTHONPATH because
#     learning_to_attribute is not installed in it. Note that submit_softuni_lr05.sh, which
#     produced the Adam uniform-k row this sweep is read against, does NOT do this; its gemma2
#     cells were fixed after the fact by reeval_gemma_mib.py.
#  3. eval_mib.py must be recent enough to carry the transformers dtype/torch_dtype version gate
#     (b65d931); without it every gemma2 job dies 28s in under the 4.46.3 venv.
#
# Protocol is otherwise byte-for-byte submit_softlog_sgd_lr.sh -- 500 steps, --include-input,
# --mode sufficient, train->validation, llama3/ioi capped at --eval-examples 200 -- because the
# comparison is only meaningful against the arm it mirrors.
#
# DRYRUN=1 to preview.  LRS="0.05 0.1" to override the grid.  ONLY=gemma2 to restrict to one model.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
PY_L2A=$ABS/.venv/bin/python                     # gpt2 / qwen2.5 / llama3
PY_TL2=$ABS/MIB-circuit-track/.venv/bin/python   # gemma2 ONLY (TL 2.15.4)
PP_TL2="PYTHONPATH=$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src "
DRYRUN=${DRYRUN:-0}
LRS=${LRS:-"0.05 0.1 0.3 1.0"}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
ONLY=${ONLY:-}
n=0
submit() { # lr model task
  local lr=$1 model=$2 task=$3
  if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then return 0; fi
  local py=$PY_L2A pp=""
  case $model in
    gpt2|qwen2.5) local cpus=2 mem=32G tlim=04:00:00 bs="" ;;
    gemma2)       local cpus=3 mem=64G tlim=08:00:00 bs="--batch-size 4"; py=$PY_TL2; pp=$PP_TL2 ;;
    llama3)       local cpus=4 mem=96G tlim=12:00:00 bs="--batch-size 2" ;;
  esac
  local ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
  local name="softunisgd-lr${lr}-${task}-${model}"
  local cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$pp$py scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform \
--masking topk --optimizer sgd --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output results/softuni_sgd_lr_$lr"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name ${ec:+[$ec]} [py=${py#$ABS/}]${pp:+ [+PYTHONPATH]}"; else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name ${ec:+[$ec]}"
  fi
  n=$((n+1))
}
for lr in $LRS; do
  for p in "${PAIRS[@]}"; do read -r model task <<< "$p"; submit "$lr" "$model" "$task"; done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n soft-fwd uniform-k SGD LR-sweep jobs =="
