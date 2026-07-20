#!/bin/bash
# TEST-set evals (train on train, eval on test) for the 3 headline MAttr variants at lr=0.05
# (the best LR from the sweep): hard-fwd log-k, soft-fwd log-k, hard-fwd uniform-k.
# Test split is small (ioi=1000), so full eval is fine everywhere -- no llama cap.
# 3 configs x 11 cells = 33 jobs. DRYRUN=1 to preview.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
# tag | masking | k-schedule | output-dir
CONFIGS=(
  "htklog|hard_topk|log|test_node_hard_topk_log_lr05"
  "tklog|topk|log|test_node_topk_log_lr05"
  "htkuni|hard_topk|uniform|test_node_hard_topk_uniform_lr05"
)
n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag mask sched out <<< "$c"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) cpus=2; mem=32G; tlim=04:00:00; bs="" ;;
      gemma2)       cpus=3; mem=64G; tlim=08:00:00; bs="--batch-size 4" ;;
      llama3)       cpus=4; mem=96G; tlim=12:00:00; bs="--batch-size 2" ;;
    esac
    name="t05-${tag}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule $sched \
--masking $mask --mode sufficient --lr 0.05 --split test --train-split train \
--include-input $bs --output results/$out"
    if [ "$DRYRUN" = "1" ]; then echo "DRY $name"; else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n test-set lr=0.05 jobs =="
