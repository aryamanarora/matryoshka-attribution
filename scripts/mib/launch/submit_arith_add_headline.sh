#!/bin/bash
# arithmetic_addition / llama3 for the HEADLINE MAttr config (soft fwd, log k, SGD), so the MIB
# leaderboard submission (make_mib_submission.py) covers every circuit-track task. The paper's
# 11 cells never included it (the validator warned "arithmetic-addition missing" -> Proceed
# Anyway); this fills that gap at both levels, TEST split by default, since the builder reads
# test_node_softlog_sgd_lr_1.0 and test_edge_softlog_sgd_lr_3.0.
#
# Protocol is byte-for-byte the llama3/arithmetic_subtraction cell of the dirs it lands in:
#   node  (submit_test_sgd.sh):      500 steps, --k-schedule log, --masking topk, sgd lr=1.0,
#                                    --include-input, --batch-size 2, full test split (no cap)
#   edge  (submit_edge_sgd_lr3.sh):  5000 steps, log, topk, sgd lr=3.0, no --include-input,
#                                    --batch-size 2 --eval-examples 200 (the $\dagger$ cap)
# Each level keeps ITS OWN LR (node 1.0, edge 3.0) -- the same own-best-LR policy every other
# headline dir follows; see CLAUDE.md's results-dir table.
#
#   bash scripts/mib/launch/submit_arith_add_headline.sh                 # test, node + edge
#   SPLITS="validation test" bash scripts/mib/launch/submit_arith_add_headline.sh
#   LEVELS=node DRYRUN=1 bash scripts/mib/launch/submit_arith_add_headline.sh
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
SPLITS=${SPLITS:-test}
LEVELS=${LEVELS:-"node edge"}
MODEL=llama3; TASK=arithmetic_addition
n=0; skip=0
for split in $SPLITS; do
  for level in $LEVELS; do
    case $level in
      node)
        [ "$split" = test ] && out=test_node_softlog_sgd_lr_1.0 || out=softlog_sgd_lr_1.0
        cpus=4; mem=96G; tlim=12:00:00
        # validation mirrors submit_softlog_sgd_lr.sh, which caps only ioi/llama3 -- not arith.
        cmd="$PY scripts/mib/eval_mib.py --model $MODEL --task $TASK --steps 500 --k-schedule log \
--masking topk --optimizer sgd --mode sufficient --lr 1.0 --split $split --train-split train \
--include-input --batch-size 2 --output results/$out" ;;
      edge)
        [ "$split" = test ] && out=test_edge_softlog_sgd_lr_3.0 || out=mib_edge_softlog_sgd_lr_3.0
        cpus=5; mem=128G; tlim=24:00:00   # 24h is the association's MaxWall
        cmd="$PY scripts/mib/eval_mib_edge.py --model $MODEL --task $TASK --steps 5000 --k-schedule log \
--masking topk --optimizer sgd --mode sufficient --lr 3.0 --split $split --train-split train \
--batch-size 2 --eval-examples 200 --output results/$out" ;;
    esac
    if [ -f "$ABS/results/$out/${TASK}_${MODEL}_scores.pt" ]; then
      echo "SKIP $out/${TASK}_${MODEL}: already trained"; skip=$((skip+1)); continue
    fi
    name="arithadd-${level}-${split:0:3}-${MODEL}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $cmd"
    if [ "$DRYRUN" = "1" ]; then echo "DRY $name -> results/$out"; echo "    $cmd"; else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name -> results/$out"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n arithmetic_addition headline jobs, $skip skipped =="
