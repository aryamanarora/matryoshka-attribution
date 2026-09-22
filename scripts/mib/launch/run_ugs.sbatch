#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/ugs_%j.out
# Train a UGS (Uniform Gradient Sampling) edge mask -- the only mask-LEARNING baseline in MIB.
# UGS lives in a separate repo and only supports gpt2-small + qwen, so it covers 3 of our 11 MIB
# cells: ioi/gpt2, ioi/qwen, mcqa/qwen (gemma2/llama3 would need GQA handling in its converter).
#
# SETUP (once): clone alestolfo/optimalablation at $UGS (default ~/optimalablation) and apply
# baselines/optimalablation.patch, which (1) drops a `load_data` import that does not exist,
# (2) reads the model from the UGS_MODEL env var (the script has no --model flag and silently
# resets every arg on an unknown flag), (3) uses Qwen2.5-0.5B base, not -Instruct, to match what
# MIB evaluates, (4) skips the unused 100k-document OpenWebText download. It runs in the TL 2.x
# stack (the `tl2` uv group; its one extra dependency is seaborn).
#
#   sbatch scripts/mib/launch/run_ugs.sbatch <gpt2-small|qwen> <ioi|mcqa|arc> [lamb] [max_seconds]
#
# Writes $UGS/results/pruning/<dataset>/cf/ugs_mib_<model>/<lamb>/snapshot.pth; then
# scripts/mib/launch/eval_ugs.sbatch converts it to a MIB graph (the fork's
# convert_mask_to_graph.py, --lambdas) and scores it into results/ugs_eval/, the layout
# make_mib_table.py reads for the UGS row. reg_lamb=0.001 is the value MIB's README reports.
set -euo pipefail
MODEL=${1:?model}
DATASET=${2:?dataset}
LAMB=${3:-0.001}          # MIB README: reg_lamb=0.001 for the reported UGS results
MAXSEC=${4:-0}            # >0 wraps the run in `timeout` (for smoke tests)

UGS=${UGS:-$HOME/optimalablation}          # alestolfo/optimalablation checkout (see header)
L2A="${SLURM_SUBMIT_DIR:-$(pwd)}"          # submit from the repo root
PY="env UV_PROJECT_ENVIRONMENT=$L2A/.venv-tl2 uv run --project $L2A --no-default-groups --group tl2 python"   # TL 2.15.4 stack, same as the MIB fork
cd "$UGS"

CMD=($PY edge_pruning_unif_mib.py -d "$DATASET" -e cf -n "ugs_mib_${MODEL}" -l "$LAMB")
[ "$MAXSEC" -gt 0 ] && CMD=(timeout --signal=INT "$MAXSEC" "${CMD[@]}")

UGS_MODEL="$MODEL" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${CMD[@]}"
