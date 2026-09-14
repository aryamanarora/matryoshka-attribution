#!/bin/bash
# CPU fallback, run on the login node in parallel with the queued GPU jobs.
#   OMP_NUM_THREADS=16 nohup bash scripts/vw/launch/vw_cpu.sh > logs_vw/cpu.log 2>&1 &
#
# WHY. The cluster is saturated by other accounts; slurm's own estimate put the first link
# of the GPU chain hours out. This arm runs the SAME model at the SAME spec on CPU, writes
# to results/vw/cpu (disjoint from base/fast, so nothing races), and is superseded wherever
# the GPU run produces the same artifact. Thread count is capped deliberately: this is a
# shared login node.
#
# What is reduced: the score pass is capped by wall clock rather than run to completion, and
# the per-weight MAttr arms get 600 steps. The per-SOURCE-ROW granularity arm is NOT
# reduced -- with 5,120 mask units instead of 20,971,520, sigmoid_topk's bisection stops
# dominating and CPU is a perfectly reasonable machine for it.
set -euo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/cpu
M="$PY scripts/vw/vw_mattr.py --run $R"

$PY scripts/vw/vw_train.py --lr 2e-3 --decay-frac 0.2 --eval-every 4000 --eval-seqs 128 --out $R
$PY scripts/vw/vw_scores.py --run $R --batch 8 --time-budget 2100

$M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 600 --granularity row
$M --method mattr --optimizer sgd                  --lr 1.0  --steps 600 --granularity row
$M --method ig --steps 600
$M --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 600
$M --method mattr --optimizer adam --adam-eps 1e-8 --lr 0.05 --steps 600
$M --method mattr --optimizer sgd                  --lr 1.0  --steps 600

$PY scripts/vw/vw_prune.py    --run $R --seqs 128 --batch 8
$PY scripts/vw/vw_report.py   --run $R
$PY scripts/vw/vw_whatkept.py --run $R --density 0.5 0.15 0.05 0.02 0.005 --seqs 128 --batch 8 \
    --refit-steps 120 --refit-density 0.02
$PY scripts/vw/vw_examples.py --run $R --word ACETYLCHOLINE --source IN --top 8 | tee $R/examples.txt
