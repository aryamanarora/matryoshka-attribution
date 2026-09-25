#!/bin/bash
# TEST-set eval for CONDUCTANCE (`EAP-IG-inputs-local`, our fixed local-delta IG variant at
# --ig-steps 5; see run_variants.sh line 4). It was the one gradient baseline with a full
# validation row and NO test pass, so it could not appear in tabs/mib_test_results.tex or in
# plots/plot_mib_test_avg.py.
#
# EVAL ONLY, exactly like run_stepless_test.sh: the circuits in results/napig_local were
# attributed on the TRAIN split during the validation wave (run_variants.sh), so the test pass
# reuses them and the "same circuit, two splits" property holds. Re-attributing would spend GPU
# hours to produce the same file.
#
# NO --head: test splits are <=1188 and every other row of the test table is full-split.
# Runs in THIS repo's .venv (TL 2.15.4) -- mandatory for the gemma2 cells.
#
#   bash run_conductance_test.sh            # 11 jobs
#   DRYRUN=1 bash run_conductance_test.sh   # preview
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd $ABS; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}
CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 10" "llama3 ioi 1"
 "llama3 arithmetic_subtraction 1"
 "qwen2.5 mcqa 10" "gemma2 mcqa 10" "llama3 mcqa 1"
 "gemma2 arc_easy 1" "llama3 arc_easy 1" "llama3 arc_challenge 1"
)
# The method name decides the subfolder run_evaluation.py writes into, so it must stay
# EAP-IG-inputs-local: with EAP-IG-inputs it would collide with the IxG/IG arms' outputs.
cdir=napig_local; method=EAP-IG-inputs-local; odir=napig_local_test
n=0
for cell in "${CELLS[@]}"; do
  read -r model task ebatch <<< "$cell"
  cpath="$ABS/results/$cdir/${method}_patching_node/${task//_/-}_${model}/importances.json"
  if [ ! -f "$cpath" ]; then echo "SKIP ${task}-${model}: no circuit at $cpath"; continue; fi
  if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00
  else mem=32G; tlim=02:00:00; fi
  name="t-cond-${task}-${model}"
  cmd="$pp; \
$PY run_evaluation.py --models $model --tasks $task --method $method --level node \
--ablation patching --split test --batch-size $ebatch \
--circuit-dir results/$cdir --output-dir $L2A/results/$odir"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name (mem=$mem t=$tlim batch=$ebatch)"
  else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== total $n Conductance test-set eval jobs =="
