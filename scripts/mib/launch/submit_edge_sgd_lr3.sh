#!/bin/bash
# EDGE-level MAttr+SGD, log-k, at SGD'S OWN EDGE OPTIMUM lr=3.0 -- validation AND test.
#
# WHY THIS EXISTS. Every other block in the paper follows an OWN-BEST-LR policy: each optimizer
# is reported at the LR it wins at, so a row label names a method and not an accident of tuning.
# The edge SGD rows are the one exception. They sit at lr=1.0, which is the NODE optimum carried
# over (submit_mib_edge_soft_sgd.sh:10-11 imported it, submit_test_edge_sgd.sh mirrored it), and
# submit_mib_edge_lr_sweep.sh later measured what that import costs. Paired over its 4 cells
# (ioi/gpt2, ioi/llama3, mcqa/llama3, arithmetic_subtraction/llama3), validation CPR AUC:
#
#     adam  0.005  0.01   0.05   0.1    0.3    1.0
#           5.98   6.90   7.58   7.65   7.37   6.81
#     sgd   0.3    1.0    3.0    10.0   30.0   100.0
#           3.55   4.72   6.89   6.53   (3/4)  (1/4)
#
# So lr=3.0 is SGD's edge optimum and it is BRACKETED (0.3 < 1.0 < 3.0 > 10.0). The rows we ship
# are at 4.72 -- detuned by 2.17 against SGD's own best, which is most of the 2.93 gap to Adam's
# 7.65. That is the difference between "SGD is worse at edge level" and "1.0 is the wrong LR at
# edge level", and the current rows cannot tell them apart. This wave makes them able to.
#
# WHAT IT DOES NOT DECIDE. Even at 3.0, Adam still leads on those 4 cells (7.65 vs 6.89), so this
# is not expected to overturn the edge default -- README.md's "edge stays Adam-default" stands
# unless the full 11 cells say otherwise. What it buys is that the `$+$ SGD` ablation row becomes
# an honest ablation (SGD at its best) rather than a detuned strawman, and that the choice of
# Adam at edge level rests on a tuned-vs-tuned comparison over 11 cells instead of 4.
#
# BOTH SPLITS, and that is the point rather than an extra. submit_test_edge_sgd.sh's header is
# explicit that test LR must match the validation dirs -- "using a different LR here than the
# validation dirs use would mean the two tables report different hyperparameters under one row
# label, which is worse than an untuned but consistent one". Shipping test-only at 3.0 would
# create exactly that defect. 11 validation + 11 test = 22 jobs.
#
# THE 4 EXISTING lr=3.0 CELLS ARE NOT REUSED. results/mib_edge_lrsweep_sgd_log_lr_3.0 has 4
# validation cells at this LR and this schedule, but the sweep ran EVERY model on the llama3
# protocol (batch 2, --eval-examples 200); ioi/gpt2 there is scored on 200 examples where every
# softlog/test_edge dir scores gpt2 on the full split. Folding it in would put one subset-scored
# cell in a row of full-split ones. Re-running all 11 here costs 1 extra small-model job and
# keeps the dir internally consistent.
#
# PROTOCOL IS READ OFF THE EXISTING lr=1.0 DIRS, verified cell-for-cell against the `args` dicts
# inside both results/mib_edge_softlog_sgd_lr_1.0/*_scores.pt and test_edge_softlog_sgd_lr_1.0/*:
#   steps 5000; --masking topk (soft forward); --k-schedule log; no --include-input (edge level)
#   gpt2 / qwen2.5 / gemma2 : --batch-size 5 --eval-examples 0   (full split)
#   llama3                  : --batch-size 2 --eval-examples 200
# The llama3 cap is what the $\dagger$ on every llama3 edge cell means
# (make_mib_test_table.EDGE_DAGGER); at edge level the Adam rows are capped too, so matching them
# is what keeps these rows comparable to the ones directly above them.
#
# GEMMA2 CELLS RUN IN THE MIB VENV (TL 2.15.4), per README.md -- the L2A venv is TL 3.2.1, whose
# Gemma-2 forward disagrees with HuggingFace (525673a). Same caveat submit_test_edge_sgd.sh
# documents applies unchanged: the Adam edge dirs were TRAINED through the broken forward and
# cannot be repaired by re-evaluation, so these gemma2 cells are correct and inconsistent with
# them rather than wrong and consistent. That discrepancy stays visible on purpose.
#
# 2 splits x 11 cells = 22 jobs. Safe to re-run: a cell with a scores.pt already on disk is
# skipped, and an edge llama3 cell is up to 24h, so re-training one is the most expensive
# possible no-op.
#
#   bash scripts/mib/launch/submit_edge_sgd_lr3.sh            # submit
#   DRYRUN=1 bash scripts/mib/launch/submit_edge_sgd_lr3.sh   # preview
#   ONLY=validation bash scripts/mib/launch/submit_edge_sgd_lr3.sh   # one split
set -u
ABS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "$ABS"; PY="uv run python"
PY_GEMMA="env UV_PROJECT_ENVIRONMENT=.venv-tl2 uv run --no-default-groups --group tl2 python"
PP_GEMMA="$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src"
DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-}
LR=${LR:-3.0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
# split | output-dir.  The dir name follows the LR, so a probe at another LR never overwrites
# these rows -- the same guard submit_mib_edge_soft_sgd.sh's LR= override uses.
CONFIGS=(
  "validation|mib_edge_softlog_sgd_lr_${LR}"
  "test|test_edge_softlog_sgd_lr_${LR}"
)
n=0; skip=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r split out <<< "$c"
  [ -n "$ONLY" ] && [ "$ONLY" != "$split" ] && continue
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    if [ -f "$ABS/results/$out/${task}_${model}_scores.pt" ]; then
      echo "SKIP $out/${task}_${model}: already trained"; skip=$((skip+1)); continue
    fi
    py=$PY; pre=""
    case $model in
      gpt2|qwen2.5) cpus=3; mem=64G;  tlim=08:00:00; bs="--batch-size 5 --eval-examples 0" ;;
      gemma2)       cpus=4; mem=96G;  tlim=16:00:00; bs="--batch-size 5 --eval-examples 0"
                    py=$PY_GEMMA; pre="export PYTHONPATH=$PP_GEMMA; " ;;
      # 24h is the association's MaxWall; 36h is rejected outright with
      # AssocMaxWallDurationPerJobLimit, so a longer request buys nothing but a failed submit.
      llama3)       cpus=5; mem=128G; tlim=24:00:00; bs="--batch-size 2 --eval-examples 200" ;;
    esac
    name="esgd3-${split:0:3}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${pre}\
$py scripts/mib/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule log \
--masking topk --optimizer sgd --mode sufficient --lr $LR --split $split --train-split train \
$bs --output results/$out"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name  (lr=$LR split=$split repo=$(basename $(dirname $(dirname $(dirname $py)))) -> $out)"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n edge-level SGD lr=$LR jobs, $skip skipped (already trained) =="
