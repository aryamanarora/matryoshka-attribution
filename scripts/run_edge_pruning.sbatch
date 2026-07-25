#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eprun_%j.out
# Edge Pruning (Bhaskar et al., 2024) baseline on MIB: learn hard-concrete gates with the
# Lagrangian L0 recipe, dump the scored circuit, then score it with MIB's own harness.
#
#   sbatch scripts/run_edge_pruning.sbatch <gpt2|qwen2.5|gemma2|llama3> <task> [level] [steps] [split]
#
# Unlike UGS (docs/ugs_baseline.md), this runs for every MIB model and at BOTH levels: the
# gates live in our own trainer, so nothing is tied to a particular architecture.
# Writes results/eprun_eval/EdgePruning_patching_<level>/<task-dash>_<model>_<split>_abs-False.pkl,
# the layout make_mib_table.py / make_mib_accauc_table.py read for baselines.
set -euo pipefail
MODEL=${1:?model}
TASK=${2:?task}
LEVEL=${3:-node}
STEPS=${4:-3000}
SPLIT=${5:-validation}

L2A=/home/guests/aryaman/learning-to-attribute
MIB=/home/guests/aryaman/MIB-circuit-track
# TL 2.15.4 env: the L2A venv's TL 3.x has a Gemma-2 forward bug, and this script both
# trains and evaluates through TransformerLens.
PY=$MIB/.venv/bin/python

# Eval batch size follows the per-model guidance in CLAUDE.md.
case "$MODEL" in
  gpt2|qwen2.5) BS=20 ;;
  gemma2)       BS=4 ;;
  llama3)       BS=2 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

OUT=$L2A/results/eprun_${LEVEL}
GRAPH=$OUT/graph_${TASK}_${MODEL}.json
cd "$L2A"

PYTHONPATH=$MIB:$MIB/EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
  scripts/eval_mib_edge_pruning.py --mib-path "$MIB" \
  --model "$MODEL" --task "$TASK" --level "$LEVEL" --steps "$STEPS" \
  --split "$SPLIT" --output "$OUT" --skip-eval

cd "$MIB"
PYTHONPATH=.:EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
  run_evaluation.py --models "$MODEL" --tasks "$TASK" --level "$LEVEL" \
  --ablation patching --split "$SPLIT" --method EdgePruning \
  --circuit-files "$GRAPH" --batch-size "$BS" \
  --output-dir $L2A/results/eprun_eval
