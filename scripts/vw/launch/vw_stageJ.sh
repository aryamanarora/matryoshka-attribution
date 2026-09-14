#!/bin/bash
# Stage J: push the budget past where stage E stopped, on the arm and cut that matter.
#   sbatch -J vw_J --time=02:30:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageJ.sh
#
# WHY. Every "high budget" arm in stage E tops out at ~393M tokens (12,000 x 32, 48,000 x 8,
# 6,000 x 64 are all the same token count), and on the CONTROLLED token-rows cut the batch-32
# arm improves monotonically with steps and shows no saturation:
#     3,000 x b32 -> +0.036 / +0.040 / +0.052 / +0.083   (d = 0.55 / 0.15 / 0.05 / 0.02)
#    12,000 x b32 -> +0.030 / +0.033 / +0.041 / +0.059
#            Fisher +0.000 / +0.009 / +0.036 / +0.069
# It already beats Fisher at d=0.02 and is within 14% at d=0.05. Concluding "Fisher wins the
# token-rows cut" from an arm that was still descending is not a conclusion, it is where the
# compute ran out -- and Fisher gets all 97.7M positions, analytically and exactly.
#
# 4x and 8x the previous best, plus a lower-lr cell because lr and budget interact.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base
M="$PY scripts/vw/vw_mattr.py --run $R --method mattr --optimizer adam --adam-eps 1e-8"

$M --lr 0.01  --steps 48000 --batch 32 --tag J_eps1e-8_lr0.01_s48000_b32
$M --lr 0.005 --steps 48000 --batch 32 --tag J_eps1e-8_lr0.005_s48000_b32
$M --lr 0.01  --steps 24000 --batch 64 --tag J_eps1e-8_lr0.01_s24000_b64

$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --ablation mean --out prune_tok_mean.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --out prune_tok.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --ablation mean --out prune_mean.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --out prune.json
