#!/bin/bash
# TEST-set evals (train on train, eval on test) for the two SGD optimizer ablations, so the
# test table can carry the same \ourmethod{}-SGD rows the validation table does. Until this
# ran, every results/test_* dir was Adam and the two tables disagreed about which rows exist.
#
# EACH CONFIG CARRIES ITS OWN LR, which is the one structural difference from
# submit_test_lr05.sh (that script shares lr=0.05 across all three of its configs). SGD's
# optimum moves with the k-schedule -- log-k peaks at lr=1.0 (validation CPR 1.886, vs 1.413 at
# the matched 0.05) and uniform-k at lr=3.0 (2.031) -- and make_mib_table.py's OUR_METHODS
# points the two SGD rows at those two dirs. The test runs MUST use the same LRs or the test
# table reports a different hyperparameter than the validation table under one row label.
# See the OUR_METHODS comments for why own-best-LR rather than matched-LR is the policy.
#
# Test split is small (ioi/arith 1000, arc-e 1188, arc-c 586, mcqa 50), so full eval everywhere
# -- NO llama cap, unlike the validation sweeps these mirror (submit_softlog_sgd_lr.sh:91 and
# submit_softuni_sgd_lr.sh:67 both pass --eval-examples 200 for ioi/llama3). That asymmetry is
# deliberate and matches submit_test_lr05.sh; it also means these test dirs must NOT be added
# to make_mib_table.IOI_LLAMA_CAPPED, which is validation-only.
#
# 2 configs x 11 cells = 22 jobs. DRYRUN=1 to preview.
#
# GEMMA2 CELLS RUN IN THE MIB VENV, and that is the one place this script deliberately does NOT
# copy submit_test_lr05.sh. The L2A venv is TL 3.2.1, whose Gemma-2 forward disagrees with
# HuggingFace (commit 525673a), so every gemma2 dir produced by the older test submitters had
# to be listed in make_mib_table.GEMMA_REEVAL_PENDING and fixed afterwards by
# reeval_gemma_mib.py. Verified that MIB-circuit-track/.venv (TL 2.15.4) runs eval_mib.py
# directly given PYTHONPATH=src, so the correct numbers can be produced on the first pass and
# these two dirs never need to enter that backlog. gpt2/qwen2.5/llama3 are version-stable and
# stay on the L2A venv, matching every other MAttr run.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
PY_GEMMA=$ABS/MIB-circuit-track/.venv/bin/python
PP_GEMMA="$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src"
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
# tag | k-schedule | lr | output-dir     (masking is `topk` = SOFT forward for both, matching
# the headline; the hard-forward variants are a separate ablation and are Adam-only on test)
CONFIGS=(
  "sgdlog|log|1.0|test_node_softlog_sgd_lr_1.0"
  "sgduni|uniform|3.0|test_node_softuni_sgd_lr_3.0"
)
n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r tag sched lr out <<< "$c"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    py=$PY; pre=""
    case $model in
      gpt2|qwen2.5) cpus=2; mem=32G; tlim=04:00:00; bs="" ;;
      gemma2)       cpus=3; mem=64G; tlim=08:00:00; bs="--batch-size 4"
                    py=$PY_GEMMA; pre="export PYTHONPATH=$PP_GEMMA; " ;;
      llama3)       cpus=4; mem=96G; tlim=12:00:00; bs="--batch-size 2" ;;
    esac
    name="tsgd-${tag}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; ${pre}\
$py scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule $sched \
--masking topk --optimizer sgd --mode sufficient --lr $lr --split test --train-split train \
--include-input $bs --output results/$out"
    if [ "$DRYRUN" = "1" ]; then echo "DRY $name  (lr=$lr sched=$sched py=$py -> $out)"; else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n test-set SGD jobs =="
