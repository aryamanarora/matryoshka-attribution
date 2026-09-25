#!/bin/bash
# Stage 2: score the 21M Tokens->Logits virtual weights with MAttr and the gradient
# baselines, then run the pruning sweep and the report.  Needs stage 1.
#   sbatch -J vw_s2 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stage2.sh
#
# THIS IS MASK LEARNING OVER WEIGHTS, not over representations: the mask multiplies entries
# of the virtual weight matrix W_TL, so the lineage is the sibling repo's MaskedDelta work
# (mask over a frozen weight tensor) rather than MIB's node/edge activation masking. Two
# axes come with that framing and both are swept below: Adam's eps, and unit granularity.
#
# ADAM EPS IS THE FIRST AXIS, not a numerical guard. At 20,971,520 mask logits nearly every
# per-step gradient sits far below torch's default 1e-8, which turns Adam's update into
# sign(g) and reduces the learned score to a signed count of steps with all effect magnitude
# divided out. Both the default and the recommended 1e-2 are run so a figure can show it.
#
# ORDER MATTERS: the headline four arms run first, so a truncated job still answers the
# question. Everything after them is an ablation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY="uv run python"
R=results/vw/base
M="$PY scripts/vw/vw_mattr.py --run $R"

# --- headline: the requested MAttr+Adam, the broken-eps contrast, SGD, and the baseline ---
$M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 3000
$M --method mattr --optimizer adam --adam-eps 1e-8 --lr 0.05 --steps 3000
$M --method mattr --optimizer sgd                  --lr 1.0  --steps 3000
$M --method ig --steps 3000
$PY scripts/vw/vw_prune.py  --run $R --seqs 512 --out prune_early.json
$PY scripts/vw/vw_report.py --run $R

# --- ablations: learning rate at both eps, SGD's own LR, I x G ---
for eps in 1e-2 1e-8; do for lr in 0.01 0.2; do
  $M --method mattr --optimizer adam --adam-eps $eps --lr $lr --steps 3000
done; done
for lr in 0.3 3.0; do $M --method mattr --optimizer sgd --lr $lr --steps 3000; done
$M --method ixg --steps 3000

# --- budget control: MAttr sees 24.5M tokens at 3000x8 against the analytic scores' 97.8M.
#     12000x8 = 98M matches it, and also tests the toy-model finding that MAttr+Adam
#     DEGRADES past ~1000 steps as its interference scores diffuse. ---
$M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 12000 \
   --tag mattr_adam_lr0.05_eps1e-2_long
$M --method ig --steps 12000 --tag ig_long

# --- weight-mask GRANULARITY (masks/layout.py's axis): per-source-row and per-target-column ---
for gran in row col; do
  $M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 3000 --granularity $gran
done

$PY scripts/vw/vw_prune.py  --run $R --seqs 1024
# the bias-preserving control: same rankings, mean ablation instead of zero
$PY scripts/vw/vw_prune.py  --run $R --seqs 1024 --ablation mean --out prune_mean.json
$PY scripts/vw/vw_report.py --run $R
