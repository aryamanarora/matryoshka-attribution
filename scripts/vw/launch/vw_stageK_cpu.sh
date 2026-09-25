#!/bin/bash
# Stage K (CPU, login node): the note's per-SIGN pruning split, for the audit.
#   OMP_NUM_THREADS=8 nohup bash scripts/vw/launch/vw_stageK_cpu.sh > logs_vw/K_cpu.log 2>&1 &
# The note's appendix: Fisher beats |W| "for every density and individual weight family
# (except for negative Features->Logits weights)". Zero ablation (the note's), coarse grid,
# four rankings, 256 held-out sequences. Thread count capped: shared login node.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY="uv run python"
R=results/vw/base
D="0.55 0.3 0.15 0.05 0.02 0.01 0.002"
for S in pos neg; do
  $PY scripts/vw/vw_prune.py --run $R --seqs 256 --rows token --sign $S --densities $D \
      --only fisher weight_abs helpfulness E_eps1e-8_lr0.01_s12000_b32 --out prune_tok_$S.json
done
for S in pos neg; do
  $PY scripts/vw/vw_feat.py prune --run $R --seqs 256 --sign $S --densities $D \
      --only fisher weight_abs helpfulness H_eps1e-8_lr0.01_s12000_b32 --out prune_fl_$S.json
done
echo K_DONE
