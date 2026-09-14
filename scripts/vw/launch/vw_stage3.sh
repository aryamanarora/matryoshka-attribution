#!/bin/bash
# Stage 3: figures, the worked examples, and the composition analysis.  Needs stages 1-2.
# Ordered cheapest-and-most-load-bearing first: the figures are seconds and are the
# deliverable, `vw_whatkept.py --refit-*` is minutes and writes its JSON incrementally, so a
# truncated run still leaves usable rows.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base

$PY plots/plot_vw_prune.py   --run $R
$PY plots/plot_vw_prune.py   --run $R --prune prune_mean.json --out paper/figs/vw_prune_mean.pdf
$PY plots/plot_vw_effhelp.py --run $R
$PY plots/plot_vw_source.py  --run $R --source IN

$PY scripts/vw/vw_examples.py --run $R --word ACETYLCHOLINE --source IN --top 8 | tee $R/examples.txt
$PY scripts/vw/vw_whatkept.py --run $R --density 0.5 0.15 0.05 0.02 0.005 --seqs 512 \
    --refit-density 0.05 0.02
$PY scripts/vw/vw_summary.py --run $R | tee $R/summary.md
