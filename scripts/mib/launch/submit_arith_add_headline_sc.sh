#!/bin/bash
# sc (Stanford NLP) twin of submit_arith_add_headline.sh: same two arithmetic_addition/llama3
# cells for the headline MAttr config, submitted with nlprun instead of raw sbatch. Same
# protocol and output dirs -- see that script's header. Run INSIDE the tmux session (nlprun is
# not on PATH over plain ssh).
#
# GPU choice follows the sc precedent for llama3: node level fits a 48GB a6000 (CLAUDE.md:
# -d a6000 -c 4 -r 96G --batch-size 2); edge level at 5000 steps needs 80GB, i.e. sphinx h100
# (submit_sphinx_retry.sh -- the 40GB a100s OOM'd). nlprun's default time limit is 10 days.
#
#   bash scripts/mib/launch/submit_arith_add_headline_sc.sh              # test, node + edge
#   SPLITS="validation test" LEVELS=node DRYRUN=1 bash scripts/mib/launch/submit_arith_add_headline_sc.sh
set -u
ABS=/nlp/scr/aryaman/learning-to-attribute; cd "$ABS"; PY="uv run python"   # uv syncs the env inside the job
DRYRUN=${DRYRUN:-0}
SPLITS=${SPLITS:-test}
LEVELS=${LEVELS:-"node edge"}
MODEL=llama3; TASK=arithmetic_addition
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
n=0; skip=0
for split in $SPLITS; do
  for level in $LEVELS; do
    case $level in
      node)
        [ "$split" = test ] && out=test_node_softlog_sgd_lr_1.0 || out=softlog_sgd_lr_1.0
        res="-q jag -d a6000 -c 4 -r 96G"
        cmd="$EXP $PY scripts/mib/eval_mib.py --model $MODEL --task $TASK --steps 500 --k-schedule log \
--masking topk --optimizer sgd --mode sufficient --lr 1.0 --split $split --train-split train \
--include-input --batch-size 2 --output results/$out" ;;
      edge)
        [ "$split" = test ] && out=test_edge_softlog_sgd_lr_3.0 || out=mib_edge_softlog_sgd_lr_3.0
        res="-q sphinx -d h100 -r 128G"
        cmd="$EXP $PY scripts/mib/eval_mib_edge.py --model $MODEL --task $TASK --steps 5000 --k-schedule log \
--masking topk --optimizer sgd --mode sufficient --lr 3.0 --split $split --train-split train \
--batch-size 2 --eval-examples 200 --output results/$out" ;;
    esac
    if [ -f "$ABS/results/$out/${TASK}_${MODEL}_scores.pt" ]; then
      echo "SKIP $out/${TASK}_${MODEL}: already trained"; skip=$((skip+1)); continue
    fi
    name="arithadd-${level}-${split:0:3}-${MODEL}"
    if [ "$DRYRUN" = "1" ]; then echo "DRY nlprun -g 1 $res -n $name -> results/$out"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" && echo "submitted $name -> results/$out"
      sleep 1
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n arithmetic_addition headline jobs (sc), $skip skipped =="
