#!/bin/bash
# Does effective batch (train-batch-size = grad-accum) fix the qwen/ioi collapse?
# 4 configs (2 known bs=1 collapsers + 2 working controls) x train-batch-size {1,4,16} x
# seeds {42,123,456} = 36 runs. Collapse = CPR ~0.25; eval capped at 1000 (floor is obvious).
# DRYRUN=1 to preview.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
# name | masking | optimizer | k-schedule | lr | (bs=1 behavior)
CONFIGS=(
  "htk_log|hard_topk|adam|log|0.05"          # collapses at bs=1
  "htk_uni|hard_topk|adam|uniform|0.05"       # works (control)
  "idste_uni|hard_topk_identity|sgd|uniform|0.01"  # collapses at bs=1
  "idste_log|hard_topk_identity|sgd|log|0.01"      # works (control)
)
n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name mask opt sched lr <<< "$c"
  for bs in 1 4 16; do
    for seed in 42 123 456; do
      out="results/bssweep_${name}_bs${bs}_s${seed}"
      jname="bss-${name}-bs${bs}-s${seed}"
      cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model qwen2.5 --task ioi --steps 500 --masking $mask \
--optimizer $opt --k-schedule $sched --mode sufficient --lr $lr --seed $seed \
--train-batch-size $bs --split validation --train-split train --include-input \
--eval-examples 1000 --output $out"
      if [ "$DRYRUN" = "1" ]; then echo "DRY $jname"; else
        sbatch --partition=main --gres=gpu:1 --cpus-per-task=2 --mem=32G --time=03:00:00 \
          --job-name="$jname" --output="$ABS/logs/${jname}.out" --wrap="$cmd" >/dev/null \
          && echo "submitted $jname"
      fi
      n=$((n+1))
    done
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n qwen/ioi batch-size sweep jobs =="
