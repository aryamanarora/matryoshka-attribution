#!/bin/bash
# Cause-trained (--mode necessary: top-k PATCHED to source, complement clean, loss pushes the
# SOURCE label) and joint-trained MAttr on the NEURON substrates, where the cause metric is not
# saturated. Companion to submit_sva_cause.sh (node substrate, default-eps Adam, 2026-07-25).
#
# WHY THESE CONFIGS: scripts/sva/analyse_cause.py on the iso-trained sweep shows that at neuron
# scale the cause ranking (cause_accsrc_auc, task-group avg) is IG 0.849 ~ MAttr Adam eps1e-2
# 0.845 ~ MAttr SGD lr1 0.841 on mlp and 0.921 / 0.913 / 0.915 on mlp+attn_head; default-eps
# Adam is far behind (0.68 / 0.76). So the two TUNED MAttr configs are the ones worth training
# in the cause direction. Same lr protocol as the iso twins (submit_sva_sweep.sh: topk:sgd at
# 1.0, topk:adam eps1e-2 at 0.05) so each cause run has an iso twin differing only in --mode.
#
#   bash scripts/sva/launch/submit_sva_cause_neuron.sh                # default grid, skip existing
#   MODES=joint SUBS="mlp" bash scripts/sva/launch/submit_sva_cause_neuron.sh
#   DRY=1 ... / FORCE=1 ...
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs results/sva_sweep_cause
OUT=${OUT:-results/sva_sweep_cause}
read -ra MODES   <<< "${MODES:-necessary}"
read -ra SUBS    <<< "${SUBS:-mlp mlp+attn_head}"
read -ra LOSSES  <<< "${LOSSES:-ce acc logit_diff}"
read -ra TASKS   <<< "${TASKS:-nounpp rc simple within_rc addition months weekdays hours}"
# variant:optimizer:lr:eps   (eps only matters for adam; "-" = default)
read -ra CONFIGS <<< "${CONFIGS:-topk:sgd:1.0:- topk:adam:0.05:1e-2}"
STEPS=${STEPS:-2000}
SEED=${SEED:-42}      # non-default seeds MUST go with a separate OUT: run_tag does not encode the seed
KS=${KS:-log}
MODEL=llama3

tag_of() {   # $1=mode $2=variant $3=opt $4=eps $5=loss  -- mirrors eval_sva.run_tag
  local t="$1_$2_$3"
  [[ "$3" == adam && "$4" != "-" ]] && t="${t}_eps$4"
  [[ "$5" != logit_diff ]] && t="${t}_$5"
  [[ "$KS" == uniform ]] && t="${t}_uniformk"
  echo "${t}_bs1"
}
dataset_of() { case "$1" in addition|months|weekdays|hours) echo arith;; arc_easy|ioi) echo mib;; *) echo sva;; esac; }
model_of()   { case "$1" in ioi) echo qwen2.5;; *) echo llama3;; esac; }

n=0; skip=0
for mode in "${MODES[@]}"; do for task in "${TASKS[@]}"; do for sub in "${SUBS[@]}"; do
  ds=$(dataset_of "$task"); MODEL=$(model_of "$task")
  for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r variant opt lr eps <<< "$cfg"
    for loss in "${LOSSES[@]}"; do
      tag=$(tag_of "$mode" "$variant" "$opt" "$eps" "$loss")
      f="$OUT/${task}_${MODEL}_${sub//+/-}_${tag}.json"
      if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); continue; fi
      args=(--model $MODEL --task "$task" --dataset "$ds" --nodes "$sub" --method mattr
            --mode "$mode" --loss "$loss" --variant "$variant" --optimizer "$opt" --lr "$lr"
            --k-schedule "$KS" --train-batch-size 1 --steps "$STEPS" --eval-examples 100
            --seed "$SEED" --output "$OUT")
      [[ "$opt" == adam && "$eps" != "-" ]] && args+=(--adam-eps "$eps")
      [[ "$task" == hours || "$ds" == mib ]] && args+=(--grad-examples 32)
      name="${mode:0:4}_${task}_${sub//+/-}_${opt}_${loss}"
      if [[ "${DRY:-0}" == 1 ]]; then echo "sbatch -J $name scripts/sva/launch/sva_sweep.sbatch ${args[*]}"
      else sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "${args[@]}" >/dev/null; fi
      n=$((n+1))
    done
  done
done; done; done
echo "submitted $n, skipped $skip -> $OUT"
