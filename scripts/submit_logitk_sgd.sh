#!/bin/bash
# LOGIT-k schedule, soft-fwd MAttr (--masking topk) + SGD, node level, train->validation.
#
# THE EXPERIMENT. At zero init sigmoid_topk's mask is uniform at alpha = k/total, the denoising
# intervention is h = alpha*clean + (1-alpha)*cf, and the implicit-diff backward multiplies
# dL/dmask by the gate slope alpha*(1-alpha). So the score a zero-init SGD run accumulates is a
# path integral of the attribution g.delta with weight w(alpha) ∝ alpha(1-alpha) p(alpha):
#
#     k-schedule   p(alpha)              w(alpha)          reading
#     uniform      const                 alpha(1-alpha)    IG, both endpoints killed
#     log          1/(alpha ln n)        1 - alpha         IG, clean end down-weighted
#     logit  <--   1/(2L alpha(1-alpha)) const             EXACT activation-path IG
#
# `logit` is the unique schedule that cancels the gate slope, so it turns MAttr+SGD's expected
# update into activation-path integrated gradients (up to a positive constant and a mean shift
# that is uniform across nodes, hence rank-irrelevant). Verified numerically in
# ~/.cache/probes/check_logit_schedule.py: the backward matches alpha(1-alpha)(g - mean g) to
# 1e-5, and the induced w is flat to +-6.5% MC noise where log's is (1-alpha) and uniform's is
# alpha(1-alpha), both matching the closed forms to 3 digits.
#
# WHAT IT TESTS. `mlp-substrate-mattr-collapses-to-ig` says MAttr+SGD IS IG at neuron
# granularity (top-10k overlap 0.457 = the same-solution floor) because the scores never escape
# the gate width T. If that account is right, the k-schedule is a dial on distance-from-IG, and
# at NODE substrate -- where MAttr genuinely beats IG -- moving to `logit` should HURT, because
# it moves the method toward the estimator it beats. If logit ~= log here, the "MAttr is
# weighted IG" story is wrong and the node-level win comes from somewhere other than the path
# weighting. Sharper than the --mattr-ig-steps null (`sva-sweep-mattr-ig-null`), which only
# re-estimated the same integral instead of changing its weight.
#
# WHY ONE LR IS ENOUGH HERE. Normally a schedule change would need its own LR sweep, but the
# expected per-step gradient scale is E[alpha(1-alpha)], and that is 1/(2 ln n) for BOTH log and
# logit -- measured 0.058851 vs 0.058722 at n=5000, 0.045017 vs 0.044678 at n=70000. The two
# schedules differ in WHERE they sample the path, not in how big a step they take, so lr=1.0
# transfers exactly. (uniform's is 1/6, ~3x larger -- that one WOULD need a re-sweep.)
# lr=1.0 is softlog_sgd's own optimum over {0.05,0.1,0.3,1.0,3.0,10.0}: acc-AUC 0.504 and
# area_under 1.886, both best of the grid, 11/11 cells.
#
# Protocol is otherwise byte-for-byte submit_softlog_sgd_lr.sh -- 500 steps, --include-input,
# --mode sufficient, train->validation, llama3/ioi capped at --eval-examples 200 -- so
# `softlogit_sgd_lr_1.0` vs `softlog_sgd_lr_1.0` isolates the k-schedule and nothing else.
# The two traps that script documents apply verbatim and are reproduced here:
#   1. --lr goes on the command line (mib_node_seed.sbatch drops a 9th positional arg).
#   2. gemma2 runs under MIB-circuit-track/.venv (TL 2.15.4) with an explicit PYTHONPATH; the
#      L2A venv's TL 3.2.1 computes a wrong Gemma-2 forward (CLAUDE.md, 525673a).
#
# DRYRUN=1 to preview.  LRS="0.3 1.0 3.0" to bracket.  ONLY=gemma2 to restrict to one model.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
PY_L2A=$ABS/.venv/bin/python                     # gpt2 / qwen2.5 / llama3
PY_TL2=$ABS/MIB-circuit-track/.venv/bin/python   # gemma2 ONLY (TL 2.15.4)
PP_TL2="PYTHONPATH=$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src "
DRYRUN=${DRYRUN:-0}
LRS=${LRS:-"1.0"}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
ONLY=${ONLY:-}
n=0; skip=0
submit() { # lr model task
  local lr=$1 model=$2 task=$3
  if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then return 0; fi
  local out=results/softlogit_sgd_lr_$lr
  # idempotent: a finished cell writes <task>_<model>_validation.pkl
  if [ "${FORCE:-0}" != 1 ] && [ -f "$out/${task}_${model}_validation.pkl" ]; then
    skip=$((skip+1)); return 0
  fi
  local py=$PY_L2A pp=""
  case $model in
    gpt2|qwen2.5) local cpus=2 mem=32G tlim=04:00:00 bs="" ;;
    gemma2)       local cpus=3 mem=64G tlim=08:00:00 bs="--batch-size 4"; py=$PY_TL2; pp=$PP_TL2 ;;
    llama3)       local cpus=4 mem=96G tlim=12:00:00 bs="--batch-size 2" ;;
  esac
  local ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
  local name="softlogit-lr${lr}-${task}-${model}"
  local cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$pp$py scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule logit \
--masking topk --optimizer sgd --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output $out"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name ${ec:+[$ec]} [py=${py#$ABS/}]${pp:+ [+PYTHONPATH]}"; else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name ${ec:+[$ec]}"
  fi
  n=$((n+1))
}
for lr in $LRS; do
  for p in "${PAIRS[@]}"; do read -r model task <<< "$p"; submit "$lr" "$model" "$task"; done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n logit-k SGD jobs, skipped $skip (done) =="
