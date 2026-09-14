#!/bin/bash
# Stage C: the token-rows-only cut of the pruning comparison, and its figures.
#   sbatch -J vw_C --time=00:40:00 --dependency=afterany:<B> scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageC.sh
#
# The full family is dominated by its 1,024 fixed-sinusoid POSITION rows (20% of the
# weights, ~98% of the positive helpfulness mass), so an aggregate over it is largely a
# statement about the position embedding. This cut holds every position row permanently on
# and prunes only the 16,777,216 token->logit weights, which is where a ranking has to know
# something about language. Both ablations, same rankings, same held-out split.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base

$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --out prune_tok.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --ablation mean --out prune_tok_mean.json
$PY plots/plot_vw_prune.py --run $R --prune prune_tok.json --out paper/figs/vw_prune_token.pdf
$PY plots/plot_vw_prune.py --run $R --prune prune_tok_mean.json \
    --out paper/figs/vw_prune_token_mean.pdf
$PY scripts/vw/vw_summary.py --run $R --prune prune_tok.json | tee $R/summary_token.md
