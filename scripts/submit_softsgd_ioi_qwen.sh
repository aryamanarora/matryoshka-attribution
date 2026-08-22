#!/bin/bash
# The 6 missing `MAttr (SGD)` runs for IOI/qwen2.5 in results/sva_sweep (2 k-schedules x 3 losses).
#
# WHY. IOI is a qwen2.5 cell everywhere in this sweep family -- submit_input_replication.sh:39,
# submit_sva_cause.sh:42 ("ioi is qwen2.5, everything else llama3"), submit_sva_dbm.sh:73 and
# submit_sva_node_pruning.sh:64 all pin it, and results/sva_sweep_input, sva_zeroabl and
# sva_zeroabl_input contain qwen2.5 IOI runs and nothing else. results/sva_sweep is the one dir
# that ALSO holds a set of llama3 IOI runs (64 files, 2026-08-21), from a different wave.
#
# That made plot_accauc_vs_faithauc.py's `Patched / Node, −input` panel compare across MODELS
# without saying so: its `raw` key is (method, loss, substrate, task) with no model in it, so two
# files for the same cell collided and glob order -- i.e. the filesystem -- decided the winner.
# It happened to keep qwen2.5 for every series that has both, but `softsgd-log` (MAttr (SGD),
# this figure's headline arm) has NO qwen2.5 IOI run, so that one series was drawn from llama3
# while the baselines beside it were drawn from qwen2.5. The gap is not cosmetic: on IOI the same
# method scores 0.443 (llama3) vs 0.495 (qwen2.5) for IG, and 0.022 vs 0.256 for I×G.
#
# The figure now pins task -> model (TASK_MODEL there) and drops off-model files, which is what
# makes this hole visible instead of silently filled by the wrong model. These jobs close it.
#
# BOTH k-schedules, unlike the +input backfill (submit_input_replication.sh:112, "LOG-k ONLY").
# There the uniform cells had no consumer; here they do -- fingerprint_node.tex's
# "$+$ SGD (soft top-$k$ fwd)" section has a unif-$k$ row, and with the pin applied its IOI
# columns render as `---` until these land.
#
# PROTOCOL is read off results/sva_sweep_input/ioi_qwen2.5_node_sufficient_topk_sgd_bs1.json's
# stored config, with include_input flipped back off: soft top-k forward, SGD, log-k, 2000 steps,
# train-batch-size 1, --eval-examples 100, --ablation patch. LR IS 1.0, not the sweep's shared
# 0.05 -- every topk:sgd cell in results/sva_sweep is lr=1.0 (same call, same reason, as
# submit_input_replication.sh:103). No --grad-examples: that flag caps the all-layer gradient
# capture and the mattr path takes none (the stored config confirms grad_examples: null).
#
#   bash scripts/submit_softsgd_ioi_qwen.sh          # submit missing
#   DRY=1 bash scripts/submit_softsgd_ioi_qwen.sh    # print only
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
OUT=results/sva_sweep

QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
n=0; skip=0; qskip=0
for loss in ce acc logit_diff; do
  ls=""; [[ "$loss" != logit_diff ]] && ls="_$loss"
  # log-k has no tag fragment; uniform-k appends `_uniformk` after the loss fragment (eval_sva
  # run_tag), which is why the two are spelled out rather than built from $sched.
  for sched in log uniform; do
    ks=""; [[ "$sched" == uniform ]] && ks="_uniformk"
    tag="sufficient_topk_sgd${ls}${ks}_bs1"
    name="sgdq_ioi_${sched}_${loss}"
    f="$OUT/ioi_qwen2.5_node_${tag}.json"
    if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); continue; fi
    if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); continue; fi
    args=(--model qwen2.5 --task ioi --dataset mib --nodes node --eval-examples 100
          --ablation patch --output "$OUT" --method mattr --loss "$loss" --variant topk
          --optimizer sgd --k-schedule "$sched" --mode sufficient --train-batch-size 1
          --steps 2000 --lr 1.0)
    if [[ "${DRY:-0}" == 1 ]]; then echo "sbatch -J $name ... $f"
    else sbatch -J "$name" sva_sweep.sbatch "${args[@]}" >/dev/null; fi
    n=$((n+1))
  done
done
echo "submitted $n, skipped $skip (done) + $qskip (already queued) -> $OUT"
