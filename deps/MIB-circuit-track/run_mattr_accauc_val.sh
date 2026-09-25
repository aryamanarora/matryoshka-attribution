#!/bin/bash
# acc-AUC (patched MIB eval) for ALL MAttr node CPR-table variants — eval-only over existing
# circuits in ~/learning-to-attribute/results/<dir>/*_importances.json. Output:
#   results/mattr_accauc_val/<dir>_patching_node/<stask>_<model>_validation_abs-False.pkl
# llama/ioi capped 200. DRYRUN=1 to preview.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd "$ABS"; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"/results
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}
DIRS=(htk_lr_0.05 final_node # was: full list


)
PAIRS=("gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge")
n=0
for d in "${DIRS[@]}"; do
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    cpath="$L2A/$d/${task}_${model}_importances.json"; [ -f "$cpath" ] || continue
    out="results/mattr_accauc_val/${d}_patching_node/${task//_/-}_${model}_validation_abs-False.pkl"
    [ -f "$out" ] && continue
    case $model in
      llama3) mem=96G; cpus=4; tlim=04:00:00; abatch=1; head=200 ;;
      gemma2) mem=64G; cpus=3; tlim=03:00:00; case "$task" in ioi) abatch=4; head=200;; arc_*|arithmetic_*) abatch=1; head=0;; *) abatch=4; head=0;; esac ;;
      qwen2.5) mem=32G; cpus=4; tlim=02:00:00; abatch=10; head=0 ;;
      *) mem=32G; cpus=4; tlim=02:00:00; abatch=20; head=0 ;;
    esac
    [ "$head" = "0" ] && hf="" || hf="--head $head"
    name="mav-${d}-${task}-${model}"
    cmd="$pp; $PY run_evaluation.py --models $model --tasks $task --method $d --ablation patching --level node --split validation --batch-size $abatch $hf --circuit-files $cpath --output-dir results/mattr_accauc_val"
    if [ "$DRYRUN" = "1" ]; then echo "DRY $name"; else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}$n MAttr acc-AUC val jobs =="
