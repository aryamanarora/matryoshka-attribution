#!/bin/bash
# Edge-level MIB training at lr=0.05 for the 3 headline MAttr variants, on val AND test.
# hard-fwd log-k, soft-fwd log-k, hard-fwd uniform-k. Edge protocol: steps 5000, mode sufficient,
# no include-input, 9 cells (no llama arc), llama capped eval-200/batch-2 (daggered). DRYRUN=1 to preview.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy"
)
# tag | masking | k-schedule
CONFIGS=("htklog|hard_topk|log" "tklog|topk|log" "htkuni|hard_topk|uniform")
n=0
for split in validation test; do
  pre=$([ "$split" = validation ] && echo mib_edge || echo test_edge)
  for c in "${CONFIGS[@]}"; do
    IFS='|' read -r tag mask sched <<< "$c"
    out="${pre}_${mask}_${sched}_lr05"; [ "$mask" = topk ] && out="${pre}_topk_${sched}_lr05"
    for p in "${PAIRS[@]}"; do
      read -r model task <<< "$p"
      case $model in
        gpt2|qwen2.5) cpus=3; mem=64G; tlim=10:00:00; bs=5; ev=0 ;;
        gemma2)       cpus=4; mem=96G; tlim=16:00:00; bs=5; ev=0 ;;
        llama3)       cpus=5; mem=128G; tlim=24:00:00; bs=2; ev=200 ;;
      esac
      name="e05-${tag}-${sched}-${split:0:3}-${task}-${model}"
      cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule $sched \
--masking $mask --mode sufficient --lr 0.05 --split $split --train-split train \
--batch-size $bs --eval-examples $ev --output results/$out"
      if [ "$DRYRUN" = "1" ]; then echo "DRY $name -> $out"; else
        sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
          --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null && echo "submitted $name"
      fi
      n=$((n+1))
    done
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n edge lr=0.05 jobs =="
