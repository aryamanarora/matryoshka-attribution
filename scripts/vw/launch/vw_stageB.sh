#!/bin/bash
# Stage B: attribution methods, pruning sweeps, composition analysis and figures.
#   sbatch -J vw_B --time=02:30:00 --dependency=afterok:<A> scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageB.sh
#
# NOT `set -e` across the two stages: stage 2 is the long one and is ordered so that a
# truncated run still holds the headline comparison, so if it is cut short by the time limit
# we still want stage 3 to draw whatever landed rather than losing the figures with it.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
bash scripts/vw/launch/vw_stage2.sh || echo "WARNING: stage 2 exited non-zero; drawing what exists"
bash scripts/vw/launch/vw_stage3.sh
