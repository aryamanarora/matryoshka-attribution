#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/ugs_eval_%j.out
# Convert a trained UGS mask to a MIB graph and evaluate it with MIB's own harness.
#
#   sbatch scripts/mib/launch/eval_ugs.sbatch <gpt2-small|qwen> <ioi|mcqa> [lamb] [split] [ugs_results_root]
#
# Writes results/ugs_eval/UGS_patching_edge/<task-dash>_<mib_model>_<split>_abs-False.pkl,
# the layout make_mib_table.py reads for edge baselines. See docs/ugs_baseline.md.
set -euo pipefail
UGSMODEL=${1:?model}
TASK=${2:?task}
LAMB=${3:-0.001}
SPLIT=${4:-validation}
UGS=${UGS:-$HOME/optimalablation}          # alestolfo/optimalablation checkout (docs/ugs_baseline.md)
ROOT=${5:-$UGS/results/pruning}
L2A="${SLURM_SUBMIT_DIR:-$(pwd)}"          # submit from the repo root

case "$UGSMODEL" in
  gpt2-small) MIBMODEL=gpt2 ;;
  qwen)       MIBMODEL=qwen2.5 ;;
  *) echo "UGS supports only gpt2-small / qwen"; exit 1 ;;
esac

MIB=${L2A_MIB_PATH:-$L2A/deps/MIB-circuit-track}
PY="env UV_PROJECT_ENVIRONMENT=$L2A/.venv-tl2 uv run --project $L2A --no-default-groups --group tl2 python"
GRAPH="$ROOT/$TASK/cf/ugs_mib_${UGSMODEL}/${LAMB}/graph.json"
cd "$MIB"

PYTHONPATH=.:EAP-IG/src $PY convert_mask_to_graph.py \
  --path "$ROOT" --task "$TASK" --ablation cf --model "$UGSMODEL" --lambdas "$LAMB"

PYTHONPATH=.:EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY run_evaluation.py \
  --models "$MIBMODEL" --tasks "$TASK" --level edge --ablation patching --split "$SPLIT" \
  --method UGS --circuit-files "$GRAPH" --batch-size 20 \
  --output-dir "$L2A/results/ugs_eval"
