#!/bin/bash
# Submit attribution+evaluation for 3 node-level methods across the 12 MIB paper cells:
#   ref   = EAP-IG-inputs       --ig-steps 5   (standard NAP-IG)
#   local = EAP-IG-inputs-local --ig-steps 5   (our fixed local-delta variant)
#   ig1   = EAP-IG-inputs       --ig-steps 1   (input x grad / one-step IG)
# One SLURM job per (cell, method). Sizing matches replicate.sh; llama3 eval uses --head 200
# (full-val eval OOMs on 8B per the L2A notes).
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
  "ref EAP-IG-inputs 5 napig_ref napig_ref_eval"
  "local EAP-IG-inputs-local 5 napig_local napig_local_eval"
  "ig1 EAP-IG-inputs 1 ig1 ig1_eval"
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
