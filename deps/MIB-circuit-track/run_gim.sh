#!/bin/bash
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd $ABS; PY=$ABS/.venv/bin/python
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CELLS=(
 "gpt2 ioi 1000 20 0" "qwen2.5 ioi 1000 10 0" "qwen2.5 mcqa full 10 0"
 "gemma2 ioi 1000 10 0" "gemma2 mcqa full 10 0" "gemma2 arc_easy 100 1 0"
 "llama3 ioi 1000 1 200" "llama3 mcqa full 1 200" "llama3 arithmetic_addition 100 1 200"
 "llama3 arithmetic_subtraction 100 1 200" "llama3 arc_easy 100 1 200" "llama3 arc_challenge 100 1 200"
)
for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00
  else mem=32G; tlim=02:00:00; fi
  [ "$nex" = "full" ] && nf="" || nf="--num-examples $nex"
  [ "$ehead" = "0" ] && hf="" || hf="--head $ehead"
  name="v-gim-${task}-${model}"
  cmd="$pp; \
$PY run_attribution.py --models $model --tasks $task --method GIM --level node --ablation patching --split train --batch-size $abatch $nf --circuit-dir results/gim && \
$PY run_evaluation.py --models $model --tasks $task --method GIM --level node --ablation patching --split validation --batch-size $abatch $hf --circuit-dir results/gim --output-dir results/gim_eval"
  sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null && echo "submitted $name"
done
