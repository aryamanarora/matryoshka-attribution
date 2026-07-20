#!/bin/bash
# Paper-ready SVA sweep. Full grid, idempotent (skips configs whose JSON already exists,
# so re-running only fires missing jobs; FORCE=1 resubmits everything).
#
#   Methods x losses {ce, acc, logit_diff}:
#     IG, IxG                               -> loss = gradient-attribution target
#     MAttr, gate x optimizer:
#       hard_topk          + Adam           (soft sigmoid-top-k STE; L2A/MIB baseline)
#       hard_topk_identity + SGD            (identity-STE, lr-invariant accumulated ranking)
#     x k-schedule {log, uniform}
#
#   Substrates {mlp (MLP neurons only), mlp+attn_head (+ attn heads)}, per-token, bs=1.
#   Tasks: the four Sam-Marks feature-circuits SVA tasks.
#
# Results -> results/sva_sweep/ (a subdir of the MIB results root).
#   bash scripts/submit_sva_sweep.sh          # submit missing jobs
#   DRY=1 bash scripts/submit_sva_sweep.sh    # print, submit nothing
#   FORCE=1 bash scripts/submit_sva_sweep.sh  # resubmit even if output exists
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/sva_sweep

MODEL=${MODEL:-llama3}   # override for MIB tasks on other models, e.g. MODEL=qwen2.5
OUT=results/sva_sweep
read -ra TASKS <<< "${SVA_TASKS-nounpp rc simple within_rc}"   # SVA_TASKS="" (set, empty) runs only MIB_TASKS
NODES=(mlp "mlp+attn_head" node)   # node = MIB granularity (mlp block + attn head per layer)
LOSSES=(ce acc logit_diff)
GRAD=(ig ixg)
KS=(log uniform)
MATTR_CONFIGS=("hard_topk:adam" "hard_topk_identity:sgd" "topk:adam")   # gate:optimizer (topk = soft fwd)

STEPS=2000
MATTR_COMMON=(--mode sufficient --train-batch-size 1 --steps "$STEPS" --lr 0.05 --eval-examples 100)

# Reproduce eval_sva.py's output tag so we can skip already-finished configs.
grad_tag() {   # $1=method $2=loss
  local t=$1; [[ "$2" != "logit_diff" ]] && t="${t}_$2"; echo "$t"
}
mattr_tag() {  # $1=variant $2=optimizer $3=loss $4=kschedule $5=ig_steps(optional,>1)
  local t="sufficient_$1_$2"
  [[ "$3" != "logit_diff" ]] && t="${t}_$3"
  [[ "${5:-1}" -gt 1 ]] && t="${t}_ig${5}"
  [[ "$4" == "uniform" ]] && t="${t}_uniformk"
  echo "${t}_bs1"
}

# MAttr-IG variants (env-gated). IG_STEPS>1 enables them (integrate dL/dmask over that many
# CF->clean points); IG_KS restricts which k-schedules get an IG variant (default: log only).
IGS=${IG_STEPS:-0}
IG_KS_STR=${IG_KS:-log}

n=0; skip=0
submit() {   # $1=name $2=tag ; rest = eval_sva.py args
  local name=$1 tag=$2; shift 2
  local nabbr; nabbr=$(grep -oP '(?<=--nodes )\S+' <<<"$*" | tr '+' '-')
  local task; task=$(grep -oP '(?<=--task )\S+' <<<"$*")
  local f="$OUT/${task}_${MODEL}_${nabbr}_${tag}.json"
  if [[ "${FORCE:-0}" != "1" && -f "$f" ]]; then skip=$((skip+1)); return; fi
  if [[ "${DRY:-0}" == "1" ]]; then echo "sbatch -J $name sva_sweep.sbatch $*"
  else sbatch -J "$name" sva_sweep.sbatch "$@"; fi
  n=$((n+1))
}

emit_grid() {   # $1=task $2=nodes $3=dataset -- full method x loss grid for one (task, substrate)
  local task=$1 nodes=$2 dataset=$3 nabbr=${2//+/-} loss gm cfg variant opt vabbr ks
  # long-prompt MIB tasks: shrink the IG/IxG attribution batch (captured with grad over all
  # layers at once) to avoid OOM; eval sweep still uses 100 test pairs.
  local grad_extra=(); [[ "$dataset" == mib ]] && grad_extra=(--grad-examples 32)
  for loss in "${LOSSES[@]}"; do
    for gm in "${GRAD[@]}"; do
      submit "sva_${task}_${nabbr}_${gm}_${loss}" "$(grad_tag "$gm" "$loss")" \
        --model "$MODEL" --task "$task" --dataset "$dataset" --nodes "$nodes" \
        --method "$gm" --loss "$loss" --eval-examples 100 "${grad_extra[@]}" --output "$OUT"
    done
    for cfg in "${MATTR_CONFIGS[@]}"; do
      variant=${cfg%:*}; opt=${cfg#*:}
      vabbr=$(case "$variant" in hard_topk_identity) echo idste;; topk) echo stopk;; *) echo soft;; esac)
      for ks in "${KS[@]}"; do
        submit "sva_${task}_${nabbr}_mattr_${vabbr}_${opt}_${ks}_${loss}" \
          "$(mattr_tag "$variant" "$opt" "$loss" "$ks")" \
          --model "$MODEL" --task "$task" --dataset "$dataset" --nodes "$nodes" \
          --method mattr --loss "$loss" --k-schedule "$ks" \
          --variant "$variant" --optimizer "$opt" "${MATTR_COMMON[@]}" --output "$OUT"
      done
      if [[ "$IGS" -gt 1 ]]; then   # MAttr-IG variants (env-gated), IG_KS schedules only
        for ks in $IG_KS_STR; do
          submit "sva_${task}_${nabbr}_mattr_${vabbr}_${opt}_${ks}_ig${IGS}_${loss}" \
            "$(mattr_tag "$variant" "$opt" "$loss" "$ks" "$IGS")" \
            --model "$MODEL" --task "$task" --dataset "$dataset" --nodes "$nodes" \
            --method mattr --loss "$loss" --k-schedule "$ks" \
            --variant "$variant" --optimizer "$opt" --mattr-ig-steps "$IGS" \
            "${MATTR_COMMON[@]}" --output "$OUT"
        done
      fi
    done
  done
}

for task in "${TASKS[@]}"; do
  for nodes in "${NODES[@]}"; do emit_grid "$task" "$nodes" sva; done
done
# MIB tasks (e.g. arc_easy): variable-length -> node substrate only, --dataset mib.
#   MIB_TASKS="arc_easy" IG_STEPS=5 bash scripts/submit_sva_sweep.sh
read -ra MIB_TASKS_ARR <<< "${MIB_TASKS:-}"
for task in "${MIB_TASKS_ARR[@]}"; do
  [[ -n "$task" ]] && emit_grid "$task" node mib
done
echo "submitted $n, skipped $skip (already present) -> $OUT"
