#!/bin/bash
# acc-AUC eval for the MAttr (learned-attribution / L2A) MIB variants.
# Circuits live in ~/learning-to-attribute/results/mib_node_<variant>/{task}_{model}_importances.json
# (MIB-graph format, exported by eval_mib.py). We re-run the patched run_evaluation.py over
# them — same pipeline/eval as the baseline acc-AUC sweep — no retraining.
# Output: results/mattr_accauc/<variant>_patching_node/<task>-<model>_validation_abs-False.pkl
# DRYRUN=1 bash run_accauc_mattr.sh   -> print jobs without submitting.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd "$ABS"; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"/results
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}

# full-coverage variants only (11 circuits each); skip 1-2 circuit partial/seed dirs.
VARIANTS=(
  mib_node_hard_topk_log mib_node_hard_topk mib_node_hard_topk_gumbel mib_node_topk_log
  mib_node_bernoulli_reinforce mib_node_bernoulli_reinforce_log
  mib_node_detached_tau mib_node_detached_tau_log
  mib_node_identity_sgd mib_node_identity_sgd_log
  mib_node_identity_gumbel_sgd_log mib_node_identity_gumbel_sgd_uniform
)
n=0
for v in "${VARIANTS[@]}"; do
  [ -d "$L2A/$v" ] || { echo "MISSING dir $v" >&2; continue; }
  for cpath in "$L2A/$v"/*_importances.json; do
    [ -e "$cpath" ] || continue
    base=$(basename "$cpath" _importances.json)   # e.g. arithmetic_subtraction_llama3
    model=${base##*_}                             # llama3
    task=${base%_*}                               # arithmetic_subtraction
    case "$model" in
      llama3) mem=96G; cpus=4; tlim=04:00:00; abatch=1; head=200 ;;
      gemma2) mem=64G; cpus=3; tlim=03:00:00
              case "$task" in ioi) head=200 ;; *) head=0 ;; esac
              case "$task" in arc_*|arithmetic_*) abatch=1 ;; *) abatch=4 ;; esac ;;
      qwen2.5) mem=32G; cpus=4; tlim=02:00:00; abatch=10; head=0 ;;
      *)      mem=32G; cpus=4; tlim=02:00:00; abatch=20; head=0 ;;   # gpt2
    esac
    [ "$head" = "0" ] && hf="" || hf="--head $head"
    name="mattr-${v#mib_node_}-${task}-${model}"
    cmd="$pp; $PY run_evaluation.py --models $model --tasks $task --method $v \
--ablation patching --level node --split validation --batch-size $abatch $hf \
--circuit-files $cpath --output-dir results/mattr_accauc"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n MAttr acc-AUC jobs =="
