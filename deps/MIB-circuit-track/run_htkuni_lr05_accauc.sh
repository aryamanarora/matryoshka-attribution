#!/bin/bash
# acc-AUC (patched MIB eval) for the uniform-k lr=0.05 MAttr circuits (htk_lr_0.05), which were
# synced from sc with an older eval lacking acc_auc. Eval-only (circuits exist) -> doesn't touch
# the CPR/lr_sweep data. Writes to results/htkuni_lr05_accauc/. llama/ioi capped 200.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd "$ABS"; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"/results/htk_lr_0.05
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  cpath="$L2A/${task}_${model}_importances.json"
  [ -f "$cpath" ] || { echo "MISSING $cpath" >&2; continue; }
  case $model in
    llama3) mem=96G; cpus=4; tlim=04:00:00; abatch=1; head=200 ;;
    gemma2) mem=64G; cpus=3; tlim=03:00:00; head=0; case "$task" in arc_*|arithmetic_*) abatch=1;; ioi) abatch=4; head=200;; *) abatch=4;; esac ;;
    qwen2.5) mem=32G; cpus=4; tlim=02:00:00; abatch=10; head=0 ;;
    *) mem=32G; cpus=4; tlim=02:00:00; abatch=20; head=0 ;;
  esac
  [ "$head" = "0" ] && hf="" || hf="--head $head"
  name="hu05-${task}-${model}"
  cmd="$pp; $PY run_evaluation.py --models $model --tasks $task --method htkuni-lr05 \
--ablation patching --level node --split validation --batch-size $abatch $hf \
--circuit-files $cpath --output-dir results/htkuni_lr05_accauc"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name ${hf:+[cap]}"; else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== ${DRYRUN:+DRY }$n uniform-lr05 acc-AUC jobs =="
