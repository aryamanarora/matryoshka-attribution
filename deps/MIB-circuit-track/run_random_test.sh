#!/bin/bash
# TEST-set eval of a RANDOM node ordering, 11 cells -- the x coordinate MIB's Table 1 does not
# give its Random control (the table reports CPR only; no IIA / acc-AUC for a random ranking
# exists anywhere on disk). plots/plot_accauc_vs_faithauc.py --cpr reads results/random_test.
#
# The circuits are results/random_s42/Random_patching_node/<cell>/importances.json: the ig1
# circuit of each cell with every node score replaced by U(0,1) (numpy default_rng(42 + cell
# index), edge scores zeroed). run_evaluation.py --level node ranks by node score, so the
# graph structure is the only thing borrowed from ig1.
#
# Everything else mirrors run_stepless_test.sh: no --head (test splits are full-size for every
# row of the test table), same per-cell eval batches, this repo's .venv (TL 2.15.4, mandatory
# for gemma2).
#   bash run_random_test.sh            # 11 jobs
#   DRYRUN=1 bash run_random_test.sh   # preview
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
cdir=random_s42; method=Random; odir=random_test
n=0
for cell in "${CELLS[@]}"; do
  read -r model task ebatch <<< "$cell"
  cpath="$ABS/results/$cdir/${method}_patching_node/${task//_/-}_${model}/importances.json"
  if [ ! -f "$cpath" ]; then echo "SKIP ${task}-${model}: no circuit at $cpath"; continue; fi
  if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00
  else mem=32G; tlim=02:00:00; fi
  name="t-rnd-${task}-${model}"
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
echo "== total $n random test-set eval jobs =="
