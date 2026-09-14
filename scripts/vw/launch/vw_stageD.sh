#!/bin/bash
# Stage D: does the k-schedule explain MAttr's dense-end failure?
#   sbatch -J vw_D --time=00:40:00 --dependency=afterany:<C> scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageD.sh
#
# THE OBSERVATION. On the first pruning sweep MAttr (both optimizers) and Expected Gradients are the
# only rankings that beat "remove the whole family" at density <= 0.02, and are CATASTROPHIC
# above density 0.15 -- flat at dL ~ 1.35, i.e. worse than keeping nothing at all, where
# Fisher effectiveness sits at 0.001.
#
# THE HYPOTHESIS. `k_schedule="log"` draws k log-uniformly over [1, 20971520], so almost every
# step supervises the sparse end of the ordering and essentially none supervises the dense
# end. schedules.py already ships the two-sided fix: `log_both` spends half its draws on
# `total - log-uniform`, i.e. on which weights to exclude LAST. `uniform` is the other
# control -- it puts most of its mass at large k, so it should be the dense-end specialist
# and, if the hypothesis is right, should invert the pattern.
#
# This is a discriminator, not a sweep: if the dense-end failure is the schedule, log_both
# fixes it and uniform overshoots the other way. If it is not, all three look alike there.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base
M="$PY scripts/vw/vw_mattr.py --run $R --method mattr --optimizer adam --steps 3000"

for ks in log_both uniform; do
  $M --lr 0.05 --adam-eps 1e-2 --k-schedule $ks
  $M --lr 0.05 --adam-eps 1e-8 --k-schedule $ks
done

# SECOND BLOCK, added after the first sweep landed. `mattr_adam_lr0.01_eps1e-08` turned out
# to be the only arm good at BOTH ends (dL +0.034 at density 0.55 and +1.027 at 0.01, where
# lr 0.05 / eps 1e-2 is at +1.350 and +0.994). That inverts this project's standing guidance
# -- eps 1e-2 is the recommended neuron-scale setting because it restores magnitude
# sensitivity -- so the (lr, eps) corner it sits in gets resolved rather than quoted from one
# cell. Cheap: ~55 s each.
for lr in 0.002 0.005 0.02; do
  $M --adam-eps 1e-8 --lr $lr
  $M --adam-eps 1e-2 --lr $lr
done

# THIRD BLOCK: unit GRANULARITY, which the first sweep says is the biggest lever here.
# Per-WEIGHT MAttr is catastrophic at the dense end (dL +1.35 at density 0.55) while the same
# method at per-SOURCE-ROW granularity is at +0.057 and is the best method of any kind at
# densities 0.15 and 0.05. That is a 24x swing from changing only what one score governs, and
# it was one cell -- so the (granularity x lr x eps) block gets filled in before the claim is
# made. `row` = 5,120 units ("does this token's direct path survive at all"), `col` = 4,096
# ("does this logit receive a direct path"). Both are ~26 s per run.
for gran in row col; do
  for lr in 0.01 0.05 0.2; do
    $M --adam-eps 1e-2 --lr $lr --granularity $gran
    $M --adam-eps 1e-8 --lr $lr --granularity $gran
  done
done
$PY scripts/vw/vw_mattr.py --run $R --method mattr --optimizer sgd --lr 1.0 --steps 3000 --granularity row
$PY scripts/vw/vw_mattr.py --run $R --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 \
    --steps 12000 --granularity row --tag mattr_adam_lr0.05_eps1e-2_row_long

$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --out prune.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --ablation mean --out prune_mean.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --out prune_tok.json
$PY plots/plot_vw_prune.py --run $R
$PY plots/plot_vw_prune.py --run $R --prune prune_mean.json --out paper/figs/vw_prune_mean.pdf
$PY plots/plot_vw_prune.py --run $R --prune prune_tok.json --out paper/figs/vw_prune_token.pdf
$PY scripts/vw/vw_summary.py --run $R | tee $R/summary.md
