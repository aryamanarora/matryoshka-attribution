#!/bin/bash
# LR sweep for hard_topk MAttr with LOG k-schedule (the headline default), all 11 cells.
# Mirrors submit_lr_sweep_all.sh (uniform-k) exactly — same steps/eval protocol/resources —
# but --k-schedule log and output dirs htklog_lr_<lr>. lr=0.01 == existing mib_node_hard_topk_log.
# Adapted to this cluster's sbatch (no nlprun). DRYRUN=1 to preview.
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
for lr in 0.005 0.05 0.1 0.3; do
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) cpus=2; mem=32G; tlim=04:00:00; bs="" ;;
      gemma2)       cpus=3; mem=64G; tlim=08:00:00; bs="--batch-size 4" ;;
      llama3)       cpus=4; mem=96G; tlim=12:00:00; bs="--batch-size 2" ;;
    esac
    name="htklog-lr${lr}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule log \
--masking hard_topk --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs --output results/htklog_lr_$lr"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
echo "== ${DRYRUN:+DRY }total $n log-k LR-sweep jobs =="
