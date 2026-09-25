#!/bin/bash
# NAP-IG (EAP-IG-inputs, node level) at --ig-steps 10 across the 12 MIB paper cells.
# Step-count control for run_variants.sh's `ref` row (identical in every other respect,
# ig-steps 5) -- the only knob that differs from SVA's `--method ig` default of 10.
# Cell sizing, venv, flags and dirs mirror run_variants.sh exactly so the pkls are
# directly comparable to results/napig_ref_eval/.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"
cd $ABS
PY=$ABS/.venv/bin/python
export_pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# cell: model task num_examples attr_batch eval_head(0=full)
CELLS=(
  "gpt2 ioi 1000 20 0"
  "qwen2.5 ioi 1000 10 0"
  "qwen2.5 mcqa full 10 0"
  "gemma2 ioi 1000 10 0"
  "gemma2 mcqa full 10 0"
  "gemma2 arc_easy 100 1 0"
  "llama3 ioi 1000 1 200"
  "llama3 mcqa full 1 200"
  "llama3 arithmetic_addition 100 1 200"
  "llama3 arithmetic_subtraction 100 1 200"
  "llama3 arc_easy 100 1 200"
  "llama3 arc_challenge 100 1 200"
)
# method: tag flag_method ig_steps circuit_dir output_dir
METHODS=(
  "ig10 EAP-IG-inputs 10 napig10 napig10_eval"
)

for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  # resources by model
  if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00; ebatch=1
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00; ebatch=$abatch
  else mem=32G; tlim=02:00:00; ebatch=$abatch; fi
  # flags that vary by cell
  if [ "$nex" = "full" ]; then nex_flag=""; else nex_flag="--num-examples $nex"; fi
  if [ "$ehead" = "0" ]; then head_flag=""; else head_flag="--head $ehead"; fi

  for m in "${METHODS[@]}"; do
    read -r tag method igs cdir odir <<< "$m"
    name="v-${tag}-${task}-${model}"
    cmd="$export_pp; \
$PY run_attribution.py --models $model --tasks $task --method $method --ig-steps $igs --level node --ablation patching --split train --batch-size $abatch $nex_flag --circuit-dir results/$cdir && \
$PY run_evaluation.py --models $model --tasks $task --method $method --level node --ablation patching --split validation --batch-size $ebatch $head_flag --circuit-dir results/$cdir --output-dir results/$odir"
    if [ "${DRYRUN:-0}" = "1" ]; then
      echo "[DRY] $name | mem=$mem t=$tlim | $method igs=$igs nex=$nex ehead=$ehead -> $odir"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" \
        --wrap="$cmd" >/dev/null && echo "submitted $name"
    fi
  done
done
