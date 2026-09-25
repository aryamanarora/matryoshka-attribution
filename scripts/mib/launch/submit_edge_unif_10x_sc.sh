#!/bin/bash
# The edge-level headline (soft top-k, uniform k, Adam lr 0.05) trained 10x longer: 50000 steps
# instead of 5000, everything else submit_softuni_lr05.sh's edge protocol. Asks whether the dead
# sparse end of the uniform-k ranking (acc 0.05 at 0.1-0.5% on gpt2/ioi, hence acc-AUC 0.74 vs
# log-k's 0.96) is under-training -- k <= 0.5% of ~32k edges is drawn in ~0.5% of steps -- or the
# schedule itself. Small-model cells only (gpt2/ioi 889 s per 5k steps -> ~2.5 h; qwen ~8 h);
# llama3 would be a day per cell. Validation split. Output: results/mib_edge_topk_uniform_lr05_50k,
# listed in make_mib_table.OUR_METHODS as "$+$ $10\times$ steps" under the edge uniform-k group.
# cluster B / gpujob, run INSIDE tmux.
#   bash scripts/mib/launch/submit_edge_unif_10x_sc.sh
#   STEPS=50000 CELLS="gpt2 ioi" DRYRUN=1 bash scripts/mib/launch/submit_edge_unif_10x_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; STEPS=${STEPS:-50000}; SPLIT=${SPLIT:-validation}
OUT=${OUT:-results/mib_edge_topk_uniform_lr05_50k}
PAIRS=("gpt2 ioi" "qwen2.5 ioi" "qwen2.5 mcqa")
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
n=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  if [ -f "$OUT/${task}_${model}_scores.pt" ]; then echo "SKIP ${task}_${model}"; continue; fi
  name="eu10x-${SPLIT:0:3}-${task}-${model}"
  cmd="$EXP uv run python scripts/mib/eval_mib_edge.py --model $model --task $task --steps $STEPS --k-schedule uniform \
--masking topk --mode iso --lr 0.05 --split $SPLIT --train-split train --batch-size 5 --eval-examples 0 --output $OUT"
  n=$((n+1))
  if [ "$DRYRUN" = 1 ]; then echo "DRY gpujob -g 1 -q gpu -d a6000 -c 3 -r 64G -n $name"; echo "    $cmd"; else
    gpujob -g 1 -q gpu -d a6000 -c 3 -r 64G -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n edge uniform-k ${STEPS}-step jobs =="
