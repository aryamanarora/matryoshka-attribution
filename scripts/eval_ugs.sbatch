#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ugs_eval_%j.out
# Convert a trained UGS mask to a MIB graph and evaluate it with MIB's own harness.
#
#   sbatch scripts/eval_ugs.sbatch <gpt2-small|qwen> <ioi|mcqa> [lamb] [split] [ugs_results_root]
#
# Writes results/ugs_eval/UGS_patching_edge/<task-dash>_<mib_model>_<split>_abs-False.pkl,
# the layout make_mib_table.py reads for edge baselines. See docs/ugs_baseline.md.
set -euo pipefail
UGSMODEL=${1:?model}
TASK=${2:?task}
LAMB=${3:-0.001}
SPLIT=${4:-validation}
ROOT=${5:-/home/guests/aryaman/optimalablation/results/pruning}

case "$UGSMODEL" in
  gpt2-small) MIBMODEL=gpt2 ;;
  qwen)       MIBMODEL=qwen2.5 ;;
  *) echo "UGS supports only gpt2-small / qwen"; exit 1 ;;
esac

MIB=/home/guests/aryaman/MIB-circuit-track
PY=$MIB/.venv/bin/python
GRAPH="$ROOT/$TASK/cf/ugs_mib_${UGSMODEL}/${LAMB}/graph.json"
cd "$MIB"

PYTHONPATH=.:EAP-IG/src "$PY" convert_mask_to_graph.py \
  --path "$ROOT" --task "$TASK" --ablation cf --model "$UGSMODEL" --lambdas "$LAMB"

PYTHONPATH=.:EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" run_evaluation.py \
  --models "$MIBMODEL" --tasks "$TASK" --level edge --ablation patching --split "$SPLIT" \
  --method UGS --circuit-files "$GRAPH" --batch-size 20 \
  --output-dir /home/guests/aryaman/learning-to-attribute/results/ugs_eval
