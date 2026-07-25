#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/ugs_%j.out
# Train a UGS (Uniform Gradient Sampling) edge mask -- the only mask-LEARNING baseline in MIB.
# UGS lives in a separate repo (alestolfo/optimalablation, cloned at ~/optimalablation) and only
# supports gpt2-small + qwen, so it covers 3 of our 11 MIB cells: ioi/gpt2, ioi/qwen, mcqa/qwen.
#
#   sbatch scripts/run_ugs.sbatch <gpt2-small|qwen> <ioi|mcqa|arc> [lamb] [max_seconds]
#
# Writes ~/optimalablation/results/pruning/<dataset>/cf/ugs_mib_<model>/<lamb>/snapshot.pth,
# which scripts/ugs_to_circuits.py then converts to MIB importances.json.
# Runs in the MIB venv (transformer_lens 2.15.4 -- the L2A venv's TL 3.x has a Gemma-2 bug and
# UGS was written against TL 2.x anyway).
set -euo pipefail
MODEL=${1:?model}
DATASET=${2:?dataset}
LAMB=${3:-0.001}          # MIB README: reg_lamb=0.001 for the reported UGS results
MAXSEC=${4:-0}            # >0 wraps the run in `timeout` (for smoke tests)

UGS=/home/guests/aryaman/optimalablation
PY=/home/guests/aryaman/MIB-circuit-track/.venv/bin/python
cd "$UGS"

CMD=("$PY" edge_pruning_unif_mib.py -d "$DATASET" -e cf -n "ugs_mib_${MODEL}" -l "$LAMB")
[ "$MAXSEC" -gt 0 ] && CMD=(timeout --signal=INT "$MAXSEC" "${CMD[@]}")

UGS_MODEL="$MODEL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${CMD[@]}"
