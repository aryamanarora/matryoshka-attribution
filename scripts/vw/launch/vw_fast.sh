#!/bin/bash
# Insurance run: the whole replication end to end at reduced budget, in one SHORT job.
#   sbatch -J vw_F --time=00:25:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_fast.sh
#
# WHY THIS EXISTS. The cluster is saturated by other accounts and slurm backfills short jobs
# into gaps that a 1-hour job cannot use, so this is the arm most likely to actually run.
# It writes to results/vw/fast, disjoint from the full run's results/vw/base, so the two
# never race and the full run supersedes this one wherever both exist.
#
# What is reduced, and nothing else: ONE learning rate instead of the sweep; the score pass
# over ~8M of the 97.8M training positions instead of all of them (the estimator is the same
# and still unbiased -- only its standard errors grow, and those are reported); MAttr and
# Expected Gradients at 1000 steps instead of 3000; the held-out eval on 256 sequences.
set -euo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/fast
M="$PY scripts/vw/vw_mattr.py --run $R"

$PY scripts/vw/vw_train.py --lr 2e-3 --decay-frac 0.2 --out $R
$PY scripts/vw/vw_scores.py --run $R --positions 8000000 --batch 32

$M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 1000
$M --method mattr --optimizer adam --adam-eps 1e-8 --lr 0.05 --steps 1000
$M --method mattr --optimizer sgd                  --lr 1.0  --steps 1000
$M --method ig --steps 1000
$M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 1000 --granularity row

$PY scripts/vw/vw_prune.py    --run $R --seqs 256
$PY scripts/vw/vw_prune.py    --run $R --seqs 256 --ablation mean --out prune_mean.json
$PY scripts/vw/vw_report.py   --run $R
$PY scripts/vw/vw_whatkept.py --run $R --density 0.5 0.15 0.05 0.02 0.005 --seqs 256 --refit-steps 150 --refit-density 0.02
$PY scripts/vw/vw_examples.py --run $R --word ACETYLCHOLINE --source IN --top 8 | tee $R/examples.txt
$PY plots/plot_vw_prune.py   --run $R --out paper/figs/vw_fast_prune.pdf
$PY plots/plot_vw_effhelp.py --run $R --out paper/figs/vw_fast_effhelp.pdf
$PY plots/plot_vw_source.py  --run $R --source IN --out paper/figs/vw_fast_source_IN.pdf
