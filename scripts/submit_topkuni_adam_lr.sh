#!/bin/bash
# LR sweep for SOFT-fwd MAttr with UNIFORM k-schedule under ADAM, all 11 node-level cells.
#
# Fills the one arm of the optimizer x k-schedule square that had no shape. The other three are
# swept over 5-8 LRs each; this one had TWO points, `final_node` (lr=0.01) and
# `mib_node_topk_uniform_lr05` (lr=0.05), and its maximum -- 2.092 CPR AUC, the highest of any
# arm in plots/optimizer_lr_cpr.pdf -- sat at the RIGHT EDGE of that two-point grid. The figure
# therefore draws it without the optimum ring the other three carry (plot_optimizer_lr.NO_RING),
# because "largest lr swept" is not "optimum". This closes that.
#
#   arm                     lr grid                              status before this script
#   soft log-k   Adam       0.005 0.05 0.1 0.3                   peak 1.881 @ 0.1, bracketed
#   soft log-k   SGD        0.005 0.01 0.05 0.1 0.3 1 3 10       peak 1.886 @ 1.0, bracketed
#   soft unif-k  SGD        0.05 0.1 0.3 1 3 10                  peak 2.031 @ 3.0, bracketed
#   soft unif-k  Adam       0.01 0.05                            <- 2 points, max at the edge
#
# GRID IS MATCHED TO THE ADAM LOG-K ARM, not to the SGD ones: {0.005, 0.01, 0.05, 0.1, 0.3}.
# 0.01 and 0.05 already exist under their own (older, inconsistent) dir names, so the default
# LRS here is the three that are missing. Matching Adam-log-k rather than SGD's decade-higher
# grid is the point -- the k-schedule is the variable being isolated, so the two Adam arms have
# to sit on one grid for their curves to be read against each other.
#
# NEW DIRS ARE results/topkuni_lr_<lr>, mirroring topklog_lr_<lr> exactly so the pair reads as
# one contrast. The two pre-existing points keep their names (renaming would orphan their pkls
# and every table that reads them); plot_optimizer_lr.OPT_SERIES lists all five explicitly.
#
# GEMMA2 RUNS UNDER THE MIB VENV. Copied from submit_softlog_sgd_lr.sh, NOT from
# submit_lr_sweep_topklog.sh -- that script points every model at $ABS/.venv, whose TL 3.2.1
# computes a wrong Gemma-2 forward (CLAUDE.md; proved in 525673a). Its gemma cells had to be
# repaired afterwards by scripts/reeval_gemma_mib.py. Getting it right at submit time is
# cheaper than re-evaluating 3 cells later.
#
# DRYRUN=1 to preview.  LRS="0.1" to override the grid.  ONLY=gemma2 for one model's cells.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
PY_L2A=$ABS/.venv/bin/python                     # gpt2 / qwen2.5 / llama3
PY_TL2=$ABS/MIB-circuit-track/.venv/bin/python   # gemma2 ONLY (TL 2.15.4)
PP_TL2="PYTHONPATH=$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src "
DRYRUN=${DRYRUN:-0}
LRS=${LRS:-"0.005 0.1 0.3"}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
ONLY=${ONLY:-}
n=0
submit() { # lr model task
  local lr=$1 model=$2 task=$3
  if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then return 0; fi
  local py=$PY_L2A pp=""
  case $model in
    gpt2|qwen2.5) local cpus=2 mem=32G tlim=04:00:00 bs="" ;;
    gemma2)       local cpus=3 mem=64G tlim=08:00:00 bs="--batch-size 4"; py=$PY_TL2; pp=$PP_TL2 ;;
    llama3)       local cpus=4 mem=96G tlim=12:00:00 bs="--batch-size 2" ;;
  esac
  # Validation-only llama3/ioi cap, matching every other arm's protocol (CLAUDE.md).
  local ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
  local name="topkuni-lr${lr}-${task}-${model}"
  local cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$pp$py scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform \
--masking topk --optimizer adam --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output results/topkuni_lr_$lr"
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
echo "== ${pfx}total $n soft-fwd uniform-k Adam LR-sweep jobs =="
