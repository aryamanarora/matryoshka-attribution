#!/bin/bash
# Stage E: is MAttr's loss to Fisher a BUDGET/optimisation artifact rather than a real
# ranking gap?  Three candidate explanations, one block each.
#   sbatch -J vw_E --time=02:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageE.sh
#
# THE EVIDENCE THAT PROMPTED IT. On the clean cut (token rows, mean ablation) the 12,000-step
# arm beats the 3,000-step arm at every density (+0.155 vs +0.209 at d=0.55, +0.290 vs +0.327
# at d=0.002). A 4x budget bought about a third of the gap to Fisher, so the 3,000-step
# result was NOT converged and "MAttr loses" cannot be claimed from it alone.
#
# WHAT IS ALREADY COVERED, so this block does not repeat it: learning rate at {0.002, 0.005,
# 0.01, 0.02, 0.05, 0.2} x eps {1e-2, 1e-8} (stages B and D), k-schedule {log, log_both,
# uniform} (stage D), granularity {weight, row, col} (stages B and D), optimizer {adam, sgd}.
#
# WHAT IS NOT, and is the point of this stage:
#   1. STEPS. 48,000 steps -- 4x the long run, 16x the default. If the trend continues, the
#      3,000-step comparison was measuring convergence, not ranking quality.
#   2. BATCH. Every MAttr run so far used batch 8 (8,192 tokens/step), against Fisher and
#      helpfulness integrating over all 97.7M positions analytically. Batch 32 and 64 cut the
#      per-step gradient noise 4x and 8x at nearly the same wall clock, since batch 8 badly
#      underuses an H100. This is the axis most likely to matter for a 21M-unit mask and the
#      one we have not moved at all.
#   3. Both at once, on the arm that is currently best on the clean cut (lr 0.01, eps 1e-8).
#
# Tags are explicit: vw_mattr.py's automatic tag has no steps/batch field, so two runs that
# differ only in budget would overwrite each other.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base
M="$PY scripts/vw/vw_mattr.py --run $R --method mattr --optimizer adam"

$M --adam-eps 1e-2 --lr 0.05 --steps 48000 --batch 8  --tag E_eps1e-2_lr0.05_s48000_b8
$M --adam-eps 1e-8 --lr 0.01 --steps 48000 --batch 8  --tag E_eps1e-8_lr0.01_s48000_b8
$M --adam-eps 1e-2 --lr 0.05 --steps 12000 --batch 32 --tag E_eps1e-2_lr0.05_s12000_b32
$M --adam-eps 1e-8 --lr 0.01 --steps 12000 --batch 32 --tag E_eps1e-8_lr0.01_s12000_b32
$M --adam-eps 1e-2 --lr 0.05 --steps 3000  --batch 32 --tag E_eps1e-2_lr0.05_s3000_b32
$M --adam-eps 1e-2 --lr 0.05 --steps 6000  --batch 64 --tag E_eps1e-2_lr0.05_s6000_b64

# the decisive cut only: token rows, both ablations
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --out prune_tok.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --ablation mean --out prune_tok_mean.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --out prune.json
