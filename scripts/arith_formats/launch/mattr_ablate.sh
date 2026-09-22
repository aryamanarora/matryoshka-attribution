#!/bin/bash
# MAttr training-objective ablation: normalised loss (M in [0,1]) at SGD lr 1 and 10, and Adam
# (eps 1e-2, lr 0.05) on the nats loss. Reuses behav/AP from results/arith_formats (copied in).
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
MODEL=$1
run() { uv run python scripts/arith_formats/run_model.py --model $MODEL --stage circuits "$@"; }
run --out results/arith_formats_abl_norm_lr1  --loss-norm --lr 1
run --out results/arith_formats_abl_norm_lr10 --loss-norm --lr 10
run --out results/arith_formats_abl_adam --optimizer adam --lr 0.05 --adam-eps 1e-2
