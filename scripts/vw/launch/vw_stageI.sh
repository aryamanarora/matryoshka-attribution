#!/bin/bash
# Stage I: refresh the FULL-FAMILY mean-ablation control so it covers the batch-32 and
# high-budget arms added in stages E/F/H.
#   sbatch -J vw_I --time=00:40:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageI.sh
#
# prune_mean.json was last written by stage D and holds 43 rankings against prune.json's 55.
# The arm that matters -- E_eps1e-8_lr0.01_s12000_b32 -- beats Fisher at EVERY density on the
# full family under ZERO ablation and is simply absent from the control, so the claim cannot
# be made either way until this runs.
set -uo pipefail
cd ~/learning-to-attribute
.venv/bin/python scripts/vw/vw_prune.py --run results/vw/base --seqs 1024 --ablation mean --out prune_mean.json
