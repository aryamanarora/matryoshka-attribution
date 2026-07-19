#!/bin/bash
# Does k-averaging (avg gradient over N k-draws/step) fix the qwen/ioi collapse that batching
# could NOT? Batch size only averages example noise; k is fixed within a batch, so k-avg is the
# knob that reduces k-schedule variance -- the suspected driver (tiny-k input-exclusion trap).
# 2 collapsers x k_avg {4,8} x seeds {42,123,456} = 12 runs (bs=1). DRYRUN=1 to preview.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
# name | masking | optimizer | k-schedule | lr
CONFIGS=(
  "htk_log|hard_topk|adam|log|0.05"                # collapses at k_avg=1
  "idste_uni|hard_topk_identity|sgd|uniform|0.01"  # collapses at k_avg=1 (deterministic)
)
n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r name mask opt sched lr <<< "$c"
  for ka in 4 8; do
    for seed in 42 123 456; do
      out="results/kavg_${name}_ka${ka}_s${seed}"
      jname="kavg-${name}-ka${ka}-s${seed}"
      cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model qwen2.5 --task ioi --steps 500 --masking $mask \
--optimizer $opt --k-schedule $sched --k-avg $ka --mode sufficient --lr $lr --seed $seed \
--train-batch-size 1 --split validation --train-split train --include-input \
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
echo "== ${pfx}total $n k-avg jobs =="
