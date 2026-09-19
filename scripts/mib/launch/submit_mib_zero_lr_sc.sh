#!/bin/bash
# LR probe for MAttr under ZERO ablation on the two small ioi cells (2026-09-18): at lr 0.05 the
# zero-trained Qwen ranking put `input` 79th of 361 (CPR 0.24 = Random) while the training loss
# reached -51 -- the soft mask keeps input partly on, the hard eval cuts it. Same protocol as
# submit_mib_zero_sc.sh's mattr arm, lr swept. Dirs: results/test_node_topk_uniform_lr<lr>_zero.
#   bash scripts/mib/launch/submit_mib_zero_lr_sc.sh
# PIN=1: pin the input node instead of learning it -- drop --include-input, so the input
# embedding is never ablated in training and eval_mib.py exports it at max_score (always kept).
# Dirs: results/test_node_topk_uniform_lr<lr>_zero_pininput; lr 0.05 (the headline) is in the
# default sweep here since the pinned arm has no other run at that lr.
#   PIN=1 bash scripts/mib/launch/submit_mib_zero_lr_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; PIN=${PIN:-0}; CELLS=${CELLS:-"gpt2:ioi qwen2.5:ioi"}
# BS=n: --train-batch-size n (gradient accumulated over n examples per step; default 1 as in
# every headline run). Dir suffix _bs<n>, job tag prefix b<n>.
BS=${BS:-1}; [ "$BS" = 1 ] && { bsarg=""; bsuf=""; btag=""; } || { bsarg="--train-batch-size $BS"; bsuf="_bs$BS"; btag="b$BS"; }
if [ "$PIN" = 1 ]; then LRS=${LRS:-"0.005 0.01 0.02 0.05 0.1 0.2"}; inp=""; suf="_pininput"; tag="zpin"
else LRS=${LRS:-"0.005 0.01 0.02 0.1 0.2"}; inp="--include-input"; suf=""; tag="zlr"; fi
n=0
for lr in $LRS; do for c in $CELLS; do
  IFS=: read -r model task <<< "$c"
  out=results/test_node_topk_uniform_lr${lr}_zero${suf}${bsuf}
  [ -f "$out/${task}_${model}_scores.pt" ] && continue
  name="$btag$tag$lr-$task-$model"
  cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/mib/eval_mib.py --model $model --task $task \
--steps 500 --k-schedule uniform --masking topk --optimizer adam --mode sufficient --lr $lr --split test --train-split train \
$inp $bsarg --ablation zero --output $out"
  n=$((n+1))
  if [ "$DRYRUN" = 1 ]; then echo "DRY $name"; else
    nlprun -g 1 -q jag -d a6000 -c 2 -r 32G -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
done; done
echo "== total $n zero-ablation lr-probe jobs =="
