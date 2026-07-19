#!/bin/bash
# Fill missing hard-fwd (masking=hard_topk) cells in the LR sweep table.
#  * log-k    (htklog_lr_*): auto-detect any missing (lr x cell) across the 11 cells.
#  * uniform-k (htk_lr_*):   sc-only dirs, so only the '---' cells (all llama/ioi + llama/arith
#                            at 0.05/0.1/0.3) are missing; run those locally. llama/ioi capped 200.
# 0.01 llama/ioi is run capped (htk*_lr_0.01) so that column is a fair n=200 comparison.
# DRYRUN=1 to preview.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
ALL_CELLS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0
submit() { # sched lr model task
  local sched=$1 lr=$2 model=$3 task=$4
  local dirpre; [ "$sched" = "log" ] && dirpre="htklog_lr" || dirpre="htk_lr"
  local pkl="results/${dirpre}_${lr}/${task}_${model}_validation.pkl"
  [ -f "$pkl" ] && return                 # already have it
  case $model in
    gpt2|qwen2.5) local cpus=2 mem=32G tlim=04:00:00 bs="" ;;
    gemma2)       local cpus=3 mem=64G tlim=08:00:00 bs="--batch-size 4" ;;
    llama3)       local cpus=4 mem=96G tlim=12:00:00 bs="--batch-size 2" ;;
  esac
  local ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
  local name="${dirpre/_lr/}-lr${lr}-${task}-${model}"
  local cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule $sched \
--masking hard_topk --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output results/${dirpre}_${lr}"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name ${ec:+[cap]}"; else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name ${ec:+[cap]}"
  fi
  n=$((n+1))
}
# --- log-k: any missing cell across all lrs (+ 0.01 llama/ioi anchor) ---
for lr in 0.005 0.05 0.1 0.3; do
  for c in "${ALL_CELLS[@]}"; do read -r m t <<< "$c"; submit log "$lr" "$m" "$t"; done
done
submit log 0.01 llama3 ioi
# --- uniform-k: only the '---' cells (llama/ioi all lrs incl 0.01 anchor; llama/arith 0.05/0.1/0.3) ---
for lr in 0.005 0.01 0.05 0.1 0.3; do submit uniform "$lr" llama3 ioi; done
for lr in 0.05 0.1 0.3;             do submit uniform "$lr" llama3 arithmetic_subtraction; done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n fill jobs =="
