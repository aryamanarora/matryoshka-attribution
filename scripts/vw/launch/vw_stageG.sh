#!/bin/bash
# Stage G: the SECOND weight family, Features->Logits, via the trained transcoder.
#   sbatch -J vw_G --time=02:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageG.sh
#
# W_FL = W_dec^T @ W_U is 4,096 x 4,096 = 16,777,216 virtual weights and, like
# Tokens->Logits, targets the logits -- so its helpfulness is also an exact closed form and
# the same oracle-quality comparison is available. What differs, and why this is a test
# rather than a repetition: the source activation `s` is a CONTINUOUS feature activation
# instead of a one-hot indicator, so neither `E[s^2 p(1-p)]` nor `e^{-sw}` factors out of the
# data, and the exponential asymmetry that drives Result 12 is much weaker (the smoke pass
# measures +2.20 / -1.78 positive/negative helpfulness mass here against the token family's
# 91.4 / 0.035). If the path-linear blind spot is really about large |sw|, Expected Gradients should
# do RELATIVELY better on this family than on the last one.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base
A="$PY scripts/vw/vw_feat.py attrib --run $R"

$PY scripts/vw/vw_scores_feat.py --run $R --batch 8 --chunk 16384 --time-budget 2400

$A --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 3000
$A --method mattr --optimizer adam --adam-eps 1e-8 --lr 0.01 --steps 3000
$A --method mattr --optimizer sgd                  --lr 1.0  --steps 3000
$A --method ig  --steps 3000
$A --method ixg --steps 3000
$A --method mattr --optimizer adam --adam-eps 1e-2 --lr 0.05 --steps 12000 \
   --tag mattr_adam_lr0.05_eps1e-2_long

$PY scripts/vw/vw_feat.py prune --run $R --seqs 256
