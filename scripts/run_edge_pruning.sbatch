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
#   sbatch scripts/run_edge_pruning.sbatch <gpt2|qwen2.5|gemma2|llama3> <task> [level] [steps] [split] [target_sparsity]
#
# target_sparsity is the L0 budget the Lagrangian anneals to; it is THE knob that decides
# where the circuit switches on, so runs at different targets must not share output dirs.
# Omit it for the script default (0.9 node / 0.99 edge) and unsuffixed dirs; pass e.g. 0.95
# to get results/eprun_node_s0.95/ + results/eprun_eval_s0.95/.
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
SPARSITY=${6:-}

L2A=/home/guests/aryaman/learning-to-attribute
MIB=/home/guests/aryaman/MIB-circuit-track
# TL 2.15.4 env: the L2A venv's TL 3.x has a Gemma-2 forward bug, and this script both
# trains and evaluates through TransformerLens.
PY=$MIB/.venv/bin/python

# Eval batch size follows the per-model guidance in CLAUDE.md. HEAD must match the eval
# subset every other row in the table used (MIB-circuit-track/run_variants.sh): llama3
# cells are capped at 200 validation examples, everything else runs full validation.
# Without this the Edge Pruning llama3 cells are scored on a different (10k-example) set
# than the gradient baselines they sit next to -- not apples-to-apples, and ~50x slower.
case "$MODEL" in
  gpt2|qwen2.5) BS=20; HEAD="" ;;
  gemma2)       BS=4;  HEAD="" ;;
  llama3)       BS=2;  HEAD="--head 200" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac

if [ -n "$SPARSITY" ]; then
  SUF=_s${SPARSITY}
  SPARSITY_ARG="--target-sparsity $SPARSITY"
else
  SUF=""
  SPARSITY_ARG=""
fi
OUT=$L2A/results/eprun_${LEVEL}${SUF}
EVAL_OUT=$L2A/results/eprun_eval${SUF}
GRAPH=$OUT/graph_${TASK}_${MODEL}.json
cd "$L2A"

# EVAL_ONLY=1 reuses an already-trained $GRAPH and just rescores it (e.g. after fixing an
# eval flag), so a 40-minute llama3 training run is not repeated.
if [ "${EVAL_ONLY:-0}" = "1" ]; then
  [ -f "$GRAPH" ] || { echo "EVAL_ONLY=1 but $GRAPH is missing"; exit 1; }
  echo "EVAL_ONLY: reusing $GRAPH"
else
  PYTHONPATH=$MIB:$MIB/EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
    scripts/eval_mib_edge_pruning.py --mib-path "$MIB" \
    --model "$MODEL" --task "$TASK" --level "$LEVEL" --steps "$STEPS" \
    --split "$SPLIT" --output "$OUT" $SPARSITY_ARG --skip-eval
fi

cd "$MIB"
PYTHONPATH=.:EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
  run_evaluation.py --models "$MODEL" --tasks "$TASK" --level "$LEVEL" \
  --ablation patching --split "$SPLIT" --method EdgePruning \
  --circuit-files "$GRAPH" --batch-size "$BS" $HEAD \
  --output-dir "$EVAL_OUT"
