#!/bin/bash
# Stage K2: redo the token-rows per-sign split with the stage-J best arm added.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python; R=results/vw/base; D="0.55 0.3 0.15 0.05 0.02 0.01 0.002"
for S in pos neg; do
  $PY scripts/vw/vw_prune.py --run $R --seqs 256 --rows token --sign $S --densities $D \
      --only fisher weight_abs helpfulness E_eps1e-8_lr0.01_s12000_b32 J_eps1e-8_lr0.01_s24000_b64 --out prune_tok_$S.json
done
echo K2_DONE
