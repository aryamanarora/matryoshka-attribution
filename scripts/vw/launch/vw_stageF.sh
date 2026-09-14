#!/bin/bash
# Stage F: an independent k per BATCH ITEM.
#   sbatch -J vw_F --time=02:00:00 scripts/vw/launch/vw.sbatch scripts/vw/launch/vw_stageF.sh
#
# THE DEFECT IT TARGETS. `sample_k` draws one k per STEP, so a step supervises a single point
# on the density curve and eats that draw's noise whole. Every MAttr arm so far shows the
# signature you would predict from that: a good head and an almost flat tail (+0.209 at
# d=0.55 and +0.220 at d=0.02 on the clean cut -- the top 0.2% of the ranking already
# captures what the top 55% does).
#
# With a k per batch ITEM, a batch of 8 supervises 8 densities per step and a batch of 32
# supervises 32, for exactly the same number of forwards. It is the variance reduction
# `k_avg` buys, moved inside the batch so it costs nothing extra -- and it makes BATCH SIZE
# do double duty, which is why the two are crossed here rather than swept separately.
#
# Verified not to touch the frozen primitives: sigmoid_topk bisects on dim=-1 and compares
# `f_mid > k` elementwise, so (B, N) scores with (B, 1) k give per-item masks that match the
# scalar-k path to 3e-8. Only the outer loop is new.
set -uo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python
R=results/vw/base
M="$PY scripts/vw/vw_mattr.py --run $R --method mattr --optimizer adam --k-per-item"

# matched against the two arms that bracket current behaviour: the headline (eps 1e-2, worst
# tail: 21.5% of value in its bottom 45%) and the best (lr 0.01 / eps 1e-8, 0.1%)
$M --adam-eps 1e-2 --lr 0.05 --steps 3000  --batch 8  --tag F_eps1e-2_lr0.05_s3000_b8_kitem
$M --adam-eps 1e-8 --lr 0.01 --steps 3000  --batch 8  --tag F_eps1e-8_lr0.01_s3000_b8_kitem
$M --adam-eps 1e-2 --lr 0.05 --steps 3000  --batch 32 --tag F_eps1e-2_lr0.05_s3000_b32_kitem
$M --adam-eps 1e-8 --lr 0.01 --steps 3000  --batch 32 --tag F_eps1e-8_lr0.01_s3000_b32_kitem
$M --adam-eps 1e-2 --lr 0.05 --steps 12000 --batch 8  --tag F_eps1e-2_lr0.05_s12000_b8_kitem
$M --adam-eps 1e-2 --lr 0.05 --steps 3000  --batch 8  --granularity row \
   --tag F_eps1e-2_lr0.05_s3000_b8_row_kitem

$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --ablation mean --out prune_tok_mean.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --rows token --out prune_tok.json
$PY scripts/vw/vw_prune.py --run $R --seqs 1024 --out prune.json
