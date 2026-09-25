#!/bin/bash
# Re-evaluate every saved node circuit with the patched evaluate_area_under_curve
# so each run also gets log-sparsity-weighted acc-AUC (+ per-sparsity accuracy curve).
# No attribution: circuits already exist under results/<dir>/<msaveable>/<task>_<model>/.
# Writes acc-AUC pkls to results/<dir>_accauc/ (mirrors the *_eval layout).
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd "$ABS"; PY=$ABS/.venv/bin/python
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

DIRS=(napig_ref napig_local ig1 relp relp_qkgrad gim relpshapley attnlrp)
n=0
for dir in "${DIRS[@]}"; do
  while IFS= read -r cpath; do
    [ -z "$cpath" ] && continue
    msaveable=$(basename "$(dirname "$(dirname "$cpath")")")   # e.g. EAP-IG-inputs_patching_node
    method=${msaveable%_patching_node}                          # strip _patching_node
    tm=$(basename "$(dirname "$cpath")")                        # e.g. arc-challenge_llama3
    model=${tm##*_}                                             # llama3
    taskdash=${tm%_*}                                           # arc-challenge
    task=${taskdash//-/_}                                       # arc_challenge
    # resources + eval batch/head by model & task
    case "$model" in
      llama3) mem=96G; cpus=4; tlim=04:00:00; abatch=1; head=200 ;;
      gemma2) mem=64G; cpus=3; tlim=03:00:00
              # ioi val is ~2500 batches; full-val eval x12 passes blows the wall clock
              # on gemma2 (~1 it/s => ~8h), so cap ioi like llama. mcqa/arc finish uncapped.
              case "$task" in ioi) head=200 ;; *) head=0 ;; esac
              case "$task" in arc_*|arithmetic_*) abatch=1 ;; *) abatch=4 ;; esac ;;
      qwen2.5) mem=32G; cpus=4; tlim=02:00:00; abatch=10; head=0 ;;
      *)      mem=32G; cpus=4; tlim=02:00:00; abatch=20; head=0 ;;   # gpt2
    esac
    [ "$head" = "0" ] && hf="" || hf="--head $head"
    name="accauc-${method}-${task}-${model}"
    cmd="$pp; $PY run_evaluation.py --models $model --tasks $task --method $method \
--ablation patching --level node --split validation --batch-size $abatch $hf \
--circuit-files $cpath --output-dir results/${dir}_accauc"
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && { echo "submitted $name"; n=$((n+1)); }
  done < <(find "results/$dir" -path '*_patching_node/*/importances.json' 2>/dev/null)
done
echo "== submitted $n acc-AUC re-eval jobs =="
