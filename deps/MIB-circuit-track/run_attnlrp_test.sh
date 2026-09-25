#!/bin/bash
# TEST-set evals for the AttnLRP node baseline, so it can join the validation AttnLRP row
# (results/attnlrp_eval, 12/12 cells from run_attnlrp.sh) on the test split.
#
#   bash run_attnlrp_test.sh            # 11 jobs
#   DRYRUN=1 bash run_attnlrp_test.sh   # preview
#
# Sibling of run_gim_relpqk_test.sh and deliberately a SEPARATE script rather than a third
# entry in that file's METHODS array: that script has no per-method filter, so adding a row
# and re-running it would resubmit the 22 finished GIM / RelP-qkgrad test jobs and overwrite
# their pkls.
#
# EVAL ONLY -- run_attribution.py is deliberately absent. AttnLRP attributes on the TRAIN split
# (run_attnlrp.sh:20), so results/attnlrp/*/importances.json is already the circuit we owe the
# test set. Re-attributing would not merely waste GPU time, it would produce a *different*
# circuit and break the "same circuit, two splits" claim that lets the validation and test
# tables be read against each other.
#
# Two deliberate differences from the validation run, both matching run_gim_relpqk_test.sh:
#  1. NO --head. run_attnlrp.sh caps llama3 validation at 200 examples because 10k val crawls
#     on 8B; the test splits are <=1188 (ioi/arith 1000, arc-e 1188, arc-c 586, mcqa 50) and
#     every other row of the test table is full-split, so capping here would make these the
#     only subset-scored rows in that table.
#  2. --output-dir writes straight into the L2A results tree (absolute path), which is what
#     make_mib_test_table.py reads.
#
# 11 cells, not 12: arithmetic_addition is attributed but is not a column of the paper tables,
# so it is not evaluated here. Eval batch sizes mirror run_attnlrp.sh's per-cell values.
#
# Runs in THIS repo's .venv (TL 2.15.4) -- mandatory for the gemma2 cells, whose forward pass
# is wrong under the L2A venv's TL 3.2.1.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd $ABS; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}

# model task eval-batch
CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 10" "llama3 ioi 1"
 "llama3 arithmetic_subtraction 1"
 "qwen2.5 mcqa 10" "gemma2 mcqa 10" "llama3 mcqa 1"
 "gemma2 arc_easy 1" "llama3 arc_easy 1" "llama3 arc_challenge 1"
)
CDIR=attnlrp
ODIR=attnlrp_eval

n=0
for cell in "${CELLS[@]}"; do
  read -r model task ebatch <<< "$cell"
  cpath="$ABS/results/$CDIR/AttnLRP_patching_node/${task//_/-}_${model}/importances.json"
  if [ ! -f "$cpath" ]; then
    echo "SKIP ${task}-${model}: no circuit at $cpath"; continue
  fi
  if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00
  else mem=32G; tlim=02:00:00; fi
  name="t-attnlrp-${task}-${model}"
  cmd="$pp; \
$PY run_evaluation.py --models $model --tasks $task --method AttnLRP --level node \
--ablation patching --split test --batch-size $ebatch \
--circuit-dir results/$CDIR --output-dir $L2A/results/$ODIR"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY $name (mem=$mem t=$tlim batch=$ebatch)"
  else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n AttnLRP test-set eval jobs =="
