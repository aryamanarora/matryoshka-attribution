#!/bin/bash
# EDGE-level TEST-set runs (train on train, eval on test) for the two SGD optimizer ablations.
# The edge counterpart of submit_test_sgd.sh, and the reason it exists: the VALIDATION table
# (paper/tabs/mib_results.tex) carries a full \ourmethod{}-SGD block at edge level, while the
# test table carried only the two Adam rows -- so the two tables disagreed about which edge rows
# exist, which is the exact defect submit_test_sgd.sh was written to fix one level up.
#
# EACH CONFIG CARRIES ITS OWN LR (1.0 for log-k, 3.0 for uniform-k), matching the node SGD rows
# and the two edge validation dirs those LRs were imported into. MAttr+SGD is LR-invariant by
# construction (zero init, no momentum), so its useful LR scales like n/k, and uniform-k's
# E[alpha(1-alpha)] = 1/6 against log-k's 1/(2 ln n) moves the optimum by roughly 3x on its own.
# Forcing both to one LR would compare a tuned row against a detuned one.
#
# NOTE these two LRs are the NODE optima carried over, not an edge-level argmax -- exactly as
# submit_mib_edge_soft_sgd.sh:10-11 did for the validation dirs these mirror. That is a known
# limitation of the edge SGD rows, not something this script should silently re-tune: using a
# different LR here than the validation dirs use would mean the two tables report different
# hyperparameters under one row label, which is worse than an untuned but consistent one.
#
# PROTOCOL IS READ OFF THE EXISTING EDGE TEST DIRS, not re-chosen. Every arg below matches the
# `args` dict saved inside results/test_edge_topk_log_lr05/*_scores.pt:
#   steps 5000; --masking topk (soft forward, the headline); no --include-input (edge level)
#   gpt2 / qwen2.5 / gemma2 : --batch-size 5, --eval-examples 0  (full test split)
#   llama3                  : --batch-size 2, --eval-examples 200
# The llama3 cap is what the $\dagger$ on every llama3 edge cell of the test table means
# (make_mib_test_table.EDGE_DAGGER). It is NOT the validation-only cap that submit_test_sgd.sh
# deliberately drops -- at edge level the Adam rows are capped too, so matching them is what
# keeps the new rows comparable to the ones directly above them.
#
# GEMMA2 CELLS RUN IN THE MIB VENV (TL 2.15.4), per CLAUDE.md: the L2A venv is TL 3.2.1, whose
# Gemma-2 forward disagrees with HuggingFace (commit 525673a). Verified that eval_mib_edge.py
# imports and resolves cleanly under that venv given PYTHONPATH=src:MIB-circuit-track:EAP-IG/src.
#
#   *** THIS MAKES THE NEW ROWS' GEMMA2 CELLS INCONSISTENT WITH THE TWO ADAM ROWS ABOVE THEM. ***
# Every existing results/test_edge_* dir was trained in the L2A venv, i.e. through the broken
# Gemma-2 forward, and unlike the node dirs they cannot be repaired by reeval_gemma_mib.py --
# edge masks are TRAINED through the forward, so the circuit itself is wrong, not just its score.
# Retraining those Adam dirs is an open decision. Producing MORE wrong numbers to match them is
# not the way to keep a table consistent, so these run correctly and the discrepancy stays
# visible; when the Adam dirs are retrained the two gemma2 columns come back into agreement.
#
# 2 configs x 11 cells = 22 jobs. DRYRUN=1 to preview.
#
#   bash scripts/submit_test_edge_sgd.sh            # submit
#   DRYRUN=1 bash scripts/submit_test_edge_sgd.sh   # preview
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
PY_GEMMA=$ABS/MIB-circuit-track/.venv/bin/python
PP_GEMMA="$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src"
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
# tag | k-schedule | lr | output-dir
CONFIGS=(
  "eslog|log|1.0|test_edge_softlog_sgd_lr_1.0"
  "esuni|uniform|3.0|test_edge_softuni_sgd_lr_3.0"
)
n=0; skip=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag sched lr out <<< "$c"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    # Safe to re-run after a partial wave: an edge llama3 cell is up to 24h, so re-training one
    # that already landed is the most expensive possible no-op here.
    if [ -f "$ABS/results/$out/${task}_${model}_scores.pt" ]; then
      echo "SKIP $out/${task}_${model}: already trained"; skip=$((skip+1)); continue
    fi
    py=$PY; pre=""
    case $model in
      gpt2|qwen2.5) cpus=3; mem=64G;  tlim=08:00:00; bs="--batch-size 5 --eval-examples 0" ;;
      gemma2)       cpus=4; mem=96G;  tlim=16:00:00; bs="--batch-size 5 --eval-examples 0"
                    py=$PY_GEMMA; pre="export PYTHONPATH=$PP_GEMMA; " ;;
      # 24h is the association's MaxWall (sacctmgr show assoc); 36h is rejected outright with
      # AssocMaxWallDurationPerJobLimit, so a longer request buys nothing but a failed submit.
      llama3)       cpus=5; mem=128G; tlim=24:00:00; bs="--batch-size 2 --eval-examples 200" ;;
    esac
    name="tesgd-${tag}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${pre}\
$py scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule $sched \
--masking topk --optimizer sgd --mode sufficient --lr $lr --split test --train-split train \
$bs --output results/$out"
    if [ "$DRYRUN" = "1" ]; then
      # THREE dirnames, not two: both interpreters live in a directory called `.venv`, so
      # stripping only bin/ and .venv/ printed ".venv" for gemma2 and for everything else alike
      # and the one thing this line exists to show -- which venv a gemma2 cell will use -- was
      # invisible. Strip the .venv too and the repo name comes out (MIB-circuit-track vs
      # learning-to-attribute), which is the actual distinction.
      echo "DRY $name  (lr=$lr sched=$sched repo=$(basename $(dirname $(dirname $(dirname $py)))) -> $out)"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n edge-level test SGD jobs, $skip skipped (already trained) =="
