#!/bin/bash
# Single-node activation-patching baseline (scripts/mib/eval_mib_actpatch.py) on the small-model
# MIB node cells, validation + test. No learning: each node's score is its own interchange effect
# averaged over 200 train pairs, in both directions (denoise = restore i alone in the corrupted run,
# noise = corrupt i alone in the clean run). One job per cell writes BOTH
#   results/{mib,test}_node_actpatch_denoise/   results/{mib,test}_node_actpatch_noise/
# Only gpt2 and qwen2.5 to start (157 / 361 nodes; 2 + 2N forwards per batch of 20). The three
# cells are the only MIB cells those models have. sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_mib_actpatch_sc.sh
#   SPLITS=validation DRYRUN=1 bash scripts/mib/launch/submit_mib_actpatch_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; SPLITS=${SPLITS:-"validation test"}
PAIRS=("gpt2 ioi" "qwen2.5 ioi" "qwen2.5 mcqa")
n=0
for split in $SPLITS; do
  [ "$split" = validation ] && out=results/mib_node_actpatch || out=results/test_node_actpatch
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    name="actpatch-${split:0:3}-${task}-${model}"
    cmd="uv run python scripts/mib/eval_mib_actpatch.py --model $model --task $task \
--n-examples 200 --batch-size 20 --split $split --train-split train --output $out"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 -q jag -d a6000 -c 2 -r 32G -n $name"; echo "    $cmd"; else
      nlprun -g 1 -q jag -d a6000 -c 2 -r 32G -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n actpatch jobs =="
