#!/bin/bash
# Stage H: the bias-preserving control for the Features->Logits family, plus the batch-32
# arm that Result 15 showed is the setting that matters.
#   sbatch -J vw_H --time=01:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageH.sh
#
# WHY IT IS NOT OPTIONAL. On the zero-ablation sweep MAttr (Adam, eps 1e-8, lr 0.01) beats
# Fisher at EVERY density on this family (-0.001 / +0.017 / +0.074 / +0.139 against +0.001 /
# +0.033 / +0.107 / +0.197). Three separate MAttr advantages in this study have looked like
# that and then mostly dissolved under mean ablation, because the pruned family's average
# contribution is a constant that zero-ablation destroys and an optimised mask rebuilds. The
# claim does not get made without the control.
#
# For a continuous source the control is still exact: a dropped weight contributes
# W_ij * E[s_i] in expectation, and E[s_i] comes from the activation sums vw_scores_feat.py
# already stores.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY="uv run python"
R=results/vw/base
A="$PY scripts/vw/vw_feat.py attrib --run $R"

$A --method mattr --optimizer adam --adam-eps 1e-8 --lr 0.01 --steps 12000 --batch 32 \
   --tag H_eps1e-8_lr0.01_s12000_b32
$A --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 12000 --batch 32 \
   --tag H_eps1e-2_lr0.05_s12000_b32

$PY scripts/vw/vw_feat.py prune --run $R --seqs 256 --ablation mean --out prune_fl_mean.json
$PY scripts/vw/vw_feat.py prune --run $R --seqs 256 --out prune_fl.json
