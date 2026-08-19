#!/bin/bash
# LR sweep for SOFT-fwd MAttr (--masking topk) + LOG k-schedule + *SGD*, node level, train->val.
#
# Fills the one empty cell of the forward x backward x optimizer grid. What exists today:
#
#   variant              forward        backward (dm/ds)      optimizer   sweep
#   topk                 soft top-k     sigmoid gate slope    adam        topklog_lr_*   (headline)
#   hard_topk            hard top-k     sigmoid gate slope    adam        htklog_lr_*
#   hard_topk_identity   hard top-k     identity (= 1)        sgd         (no real LR sweep -- see below)
#   topk                 soft top-k     sigmoid gate slope    SGD         <-- THIS SCRIPT
#
# Motivation: at the SVA neuron substrates (2.3M mask logits vs ~5.3k at node) the Adam arms
# collapse -- soft-topk/Adam scores acc-AUC 0.279 vs IG's 0.502 and recovers Feucht et al.'s
# published neurons in 3/12 cells, while the id-STE/SGD arm gets 0.479 and 11/12. But that arm
# flips BOTH the backward and the optimizer at once, so nothing on disk isolates either axis.
# This script holds the forward and backward fixed at the headline's and moves only the
# optimizer, so `softlog_sgd_lr_<lr>` vs `topklog_lr_<lr>` is a clean Adam-vs-SGD contrast.
#
# TWO TRAPS THIS SCRIPT DELIBERATELY AVOIDS -- do not "simplify" it back into either:
#
#  1. It passes --lr on the command line rather than going through mib_node_seed.sbatch.
#     That sbatch wrapper reads only $1..$8 and never passes --lr, so a 9th "lr" argument is
#     silently dropped and every job runs at eval_mib.py's default 0.01. submit_ident_adam_lr.sh
#     does exactly that, which is why results/ident_adam_lr_0.005 and .../ident_adam_lr_0.01
#     hold byte-identical pkls (verified by md5sum on ioi_gpt2_validation.pkl). That "sweep"
#     is three copies of one run, not three LRs.
#
#  2. It runs gemma2 under MIB-circuit-track/.venv (TL 2.15.4), not the L2A .venv (TL 3.2.1),
#     whose Gemma-2 forward is wrong (CLAUDE.md; proved against an HF reference in 525673a).
#     submit_lr_sweep_topklog.sh hardcodes $ABS/.venv/bin/python for every model, so re-running
#     it would silently reintroduce the bad Gemma numbers.
#     ...and that venv needs an explicit PYTHONPATH. It is the MIB repo's own environment and
#     does NOT have learning_to_attribute installed, so invoking its python directly dies at
#     `from learning_to_attribute import ...` with ModuleNotFoundError. The first four gemma2
#     jobs of this sweep failed exactly that way before $PP was added. `src` is ours; the other
#     two entries are the precedent from scripts/reeval_bern_gemma.sh:41, kept so eval_mib.py's
#     MIB-side imports resolve the same way they do there.
#
# Protocol is otherwise byte-for-byte submit_lr_sweep_topklog.sh -- 500 steps, --include-input,
# --mode sufficient, train->validation, llama3/ioi capped at --eval-examples 200 -- because the
# comparison is only meaningful against the arm it mirrors.
#
# DRYRUN=1 to preview.  LRS="0.05 0.1" to override the grid.  ONLY=gemma2 to restrict to one
# model's cells (used to resubmit just the gemma2 arm after the PYTHONPATH fix above).
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
PY_L2A=$ABS/.venv/bin/python                 # gpt2 / qwen2.5 / llama3
PY_TL2=$ABS/MIB-circuit-track/.venv/bin/python   # gemma2 ONLY (TL 2.15.4)
# Prefix for PY_TL2 only; PY_L2A has the package installed editable and needs none.
PP_TL2="PYTHONPATH=$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src "
DRYRUN=${DRYRUN:-0}
# Same grid as the paper's LR sweep (tabs/lr_sweep.tex) so the new row is directly comparable.
LRS=${LRS:-"0.005 0.01 0.05 0.1 0.3"}
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
  local ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
  local name="softsgd-lr${lr}-${task}-${model}"
  local cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$pp$py scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule log \
--masking topk --optimizer sgd --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output results/softlog_sgd_lr_$lr"
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
echo "== ${pfx}total $n soft-fwd log-k SGD LR-sweep jobs =="
