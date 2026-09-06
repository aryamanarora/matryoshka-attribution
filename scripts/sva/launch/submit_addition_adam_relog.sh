#!/bin/bash
# One-off backfill: re-run the 9 `addition` soft-top-k + Adam + log-k cells so they carry a
# `train_eval_log` trajectory.
#
# WHY. fig:optimiser-curves (plots/plot_train_curves.py --overlay) pairs on the CURVE -- a
# (substrate, task, loss) cell is drawn only if BOTH arms logged one -- and `train_eval_log` was
# added partway through the sweep. For `addition` all 9 SGD runs have it and all 9 Adam runs do
# not, so the pairing rule drops the task from every panel and each panel reads n=3
# (hours, months, weekdays) instead of n=4. Nothing is wrong with the Adam runs; they simply
# predate the logging. This fills that in and takes the figure to n=4.
#
# WHAT THIS IS NOT. It is not a re-scoring and not a new arm. Every flag below is copied from
# submit_sva_sweep.sh's emit_grid mattr branch for cfg="topk:adam", ks=log -- same LR (0.05),
# same 2000 steps, same --train-batch-size 1, same --eval-examples 100, same probe defaults
# (--train-eval-every 200, --train-eval-examples 20). If a re-run's endpoint acc_auc/faith_auc
# does NOT match the backup, that is a finding about reproducibility, not an expected drift:
# compare before overwriting anything downstream.
#
#   results/_backup_addition_adam_prelog_20260824/  holds the pre-run JSONs (eval_sva.py
#   overwrites in place at the same tag, so the originals would otherwise be gone).
#
#   bash scripts/sva/launch/submit_addition_adam_relog.sh          # submit
#   DRY=1 bash scripts/sva/launch/submit_addition_adam_relog.sh    # print, submit nothing
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs results/sva_sweep

MODEL=llama3
TASK=addition
OUT=results/sva_sweep
LR=0.05
STEPS=2000
NODES=(mlp "mlp+attn_head" node)
LOSSES=(ce acc logit_diff)

n=0
for nodes in "${NODES[@]}"; do
  nabbr=${nodes//+/-}
  for loss in "${LOSSES[@]}"; do
    # mattr_tag(topk, adam, $loss, log) -- logit_diff is the unmarked loss, log is the unmarked
    # k-schedule, so the logit_diff tag is bare "sufficient_topk_adam_bs1".
    tag="sufficient_topk_adam"
    [[ "$loss" != "logit_diff" ]] && tag="${tag}_${loss}"
    tag="${tag}_bs1"
    f="$OUT/${TASK}_${MODEL}_${nabbr}_${tag}.json"
    [[ -f "$f" ]] || { echo "MISSING (not a re-run target): $f" >&2; exit 1; }
    args=(--model "$MODEL" --task "$TASK" --dataset arith --nodes "$nodes"
          --method mattr --loss "$loss" --k-schedule log
          --variant topk --optimizer adam --lr "$LR"
          --mode sufficient --train-batch-size 1 --steps "$STEPS" --eval-examples 100
          --ablation patch --output "$OUT")
    name="sva_${TASK}_${nabbr}_mattr_stopk_adam_log_${loss}_relog"
    if [[ "${DRY:-0}" == "1" ]]; then echo "sbatch -J $name scripts/sva/launch/sva_sweep.sbatch ${args[*]}"
    else sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "${args[@]}"; fi
    n=$((n+1))
  done
done
echo "submitted $n -> $OUT"
