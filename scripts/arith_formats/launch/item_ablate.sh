#!/bin/bash
# Per-item MAttr lr/steps ablation (English only, 400 items). MODEL env selects the model.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
MODEL=${MODEL:-gemma2-9b}
for cfg in "100 200 -" "10 200 lr10" "10 1000 lr10s1000" "1 1000 lr1s1000"; do
  set -- $cfg
  tag=$3; [ "$tag" = "-" ] && tag=""
  uv run python scripts/arith_formats/run_model.py --model $MODEL --stage itemmattr --formats english \
    --item-n 400 --item-steps $2 --item-bs 16 --item-lr $1 --item-tag "$tag"
done
