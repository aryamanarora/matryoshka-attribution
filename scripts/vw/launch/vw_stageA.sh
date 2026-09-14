#!/bin/bash
# Stage A: train the note's transformer (LR sweep + pick) and score all 21M Tokens->Logits
# virtual weights.  One job rather than two because the cluster is saturated by other users
# and each additional job in a dependency chain costs another full queue wait.
#   sbatch -J vw_A --time=01:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageA.sh
set -euo pipefail
cd ~/learning-to-attribute
bash scripts/vw/launch/vw_stage1a.sh
bash scripts/vw/launch/vw_stage1b.sh
