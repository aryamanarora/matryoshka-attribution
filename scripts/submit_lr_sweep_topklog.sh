#!/bin/bash
# LR sweep for SOFT-fwd MAttr (masking=topk, sigmoid soft top-k) with LOG k-schedule, all 11 cells.
# Mirrors submit_lr_sweep_log.sh exactly (same steps/eval protocol/resources, llama3/ioi capped
# at 200) but --masking topk and output dirs topklog_lr_<lr>. lr=0.01 == existing mib_node_topk_log.
# DRYRUN=1 to preview.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0
submit() { # lr model task
  local lr=$1 model=$2 task=$3
  case $model in
    gpt2|qwen2.5) local cpus=2 mem=32G tlim=04:00:00 bs="" ;;
    gemma2)       local cpus=3 mem=64G tlim=08:00:00 bs="--batch-size 4" ;;
    llama3)       local cpus=4 mem=96G tlim=12:00:00 bs="--batch-size 2" ;;
  esac
  local ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
  local name="topklog-lr${lr}-${task}-${model}"
  local cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule log \
--masking topk --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output results/topklog_lr_$lr"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name ${ec:+[$ec]}"; else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name ${ec:+[$ec]}"
  fi
  n=$((n+1))
}
for lr in 0.005 0.05 0.1 0.3; do
  for p in "${PAIRS[@]}"; do read -r model task <<< "$p"; submit "$lr" "$model" "$task"; done
done
submit 0.01 llama3 ioi   # 0.01 anchor for llama3/ioi at n=200 (full-eval anchor = mib_node_topk_log)
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n soft-fwd log-k LR-sweep jobs =="
