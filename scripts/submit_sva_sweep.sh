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
OUT=${OUT:-results/sva_sweep}
# ABLATION=zero re-runs the whole grid with non-top-k units set to 0 instead of to the source
# activation. It is a second SETTING, not a re-scoring: MAttr retrains through it and the
# gradient baselines change estimator (IxG -> Gradient x Input, IG -> zero-baseline IG). Pair it
# with a separate OUT so the two settings' results dirs stay legible:
#   OUT=results/sva_zeroabl ABLATION=zero bash scripts/submit_sva_sweep.sh
ABLATION=${ABLATION:-patch}
ABL_ARG=(--ablation "$ABLATION")
ABL_SUF=""; [[ "$ABLATION" != "patch" ]] && ABL_SUF="_${ABLATION}abl"
read -ra TASKS <<< "${SVA_TASKS-nounpp rc simple within_rc}"   # SVA_TASKS="" (set, empty) runs only MIB_TASKS
NODES=(mlp "mlp+attn_head" node)   # node = MIB granularity (mlp block + attn head per layer)
LOSSES=(ce acc logit_diff)
GRAD=(ig ixg attnlrp mc_ig)
# "Stepless IG": alpha ~ U(0,1) per example instead of the fixed grid. MC_DRAWS=1 is the arm
# that matters -- it costs exactly what ixg costs (one fwd+bwd), so mc_ig vs ixg is a
# compute-matched contrast and mc_ig vs ig is a 1x-vs-10x one. eval_sva.py's --ig-steps default
# is 10, which for this method would silently buy a 10x run, so it is ALWAYS passed explicitly.
# MC_SEED defaults to eval_sva.py's own --seed default (42) so that the only thing differing
# between an mc_ig run and the ixg run beside it is the alpha placement -- not the global RNG
# that also seeds everything else. Replicates for the noise floor: MC_SEED=43, 44.
MC_DRAWS=${MC_DRAWS:-1}
MC_SEED=${MC_SEED:-42}
KS=(log uniform)
# gate:optimizer[:lr] (topk = soft fwd). LR field is OPTIONAL and defaults to $MATTR_LR.
# topk:sgd runs at lr=1.0, off the sweep's shared 0.05 protocol. CORRECTED 2026-08-20: the
# justification originally written here was that soft-fwd+SGD degenerates at 0.05 (gate stuck
# in its linear region, mask pinned at m=k/n, update collapsing to a mean-centred path integral
# = IG). That is a NODE-substrate fact imported from the MIB LR sweep (log-k CPR-AUC
# 1.41@0.05 -> 1.89@1.0, 1.0 bracketed: 3.0 and 10.0 both lose). It does NOT hold at the neuron
# substrate this sweep mostly runs on: results/sva_mlp_lr/topk_sgd (addition, mlp, 8 LRs) reads
# 0.490@0.05 vs 0.496@1.0 and stays in 0.44-0.50 from lr=0.05 all the way to 100. So lr=1.0
# buys ~0.006 here and costs one-protocol comparability. If this arm is re-run, prefer 0.05.
# SUPERSEDED FOR THE ZERO-ABLATION SETTING (2026-08-21): the 156 topk:sgd cells backfilled into
# results/sva_zeroabl were run at 1.0, NOT 0.05, because their whole purpose is the paired
# patch-vs-zero comparison in scripts/compare_ablation.py, which matches the two settings by
# filename. Dropping to 0.05 there would have compared a 0.05 zero run against a 1.0 patch run
# and charged the difference to the ablation. Same-LR beats better-LR when the axis under test
# is the ablation.
#
# What the deviation is NOT needed for: this arm's win over soft Adam is not an LR artifact.
# The same probe sweeps topk_adam over 8 LRs at the same substrate; its BEST is 0.346, below
# topk_sgd's WORST in-range 0.436. Adam cannot be tuned into that gap.
# The existing three configs stay at 0.05 -- they are Adam (invariant to this) or identity-STE
# (lr-invariant by construction), and 613 finished runs are already at that LR.
# NOTE: run_tag() does NOT encode lr, so an lr=1.0 run writes the filename an lr=0.05 run would.
# Safe only because zero topk_sgd outputs exist. If you ever sweep LR here, add it to the tag.
MATTR_CONFIGS=("hard_topk:adam" "hard_topk_identity:sgd" "topk:adam" "topk:sgd:1.0")

STEPS=2000
MATTR_LR=${MATTR_LR:-0.05}
MATTR_COMMON=(--mode sufficient --train-batch-size 1 --steps "$STEPS" --eval-examples 100)

# Reproduce eval_sva.py's output tag so we can skip already-finished configs.
# These two must stay byte-identical to eval_sva.run_tag() or the skip-if-exists check silently
# resubmits everything. run_tag appends the ablation suffix directly after the loss.
grad_tag() {   # $1=method $2=loss
  # mc_ig's identity carries its draw count and seed (run_tag encodes both, because --ig-steps
  # is otherwise absent from every tag and the seed IS this estimator's error bar). Order here
  # must match run_tag: method, then loss, then ablation.
  local t=$1
  [[ "$1" == "mc_ig" ]] && t="mc_ig_m${MC_DRAWS}_s${MC_SEED}"
  [[ "$2" != "logit_diff" ]] && t="${t}_$2"
  echo "${t}${ABL_SUF}"
}
mattr_tag() {  # $1=variant $2=optimizer $3=loss $4=kschedule $5=ig_steps(optional,>1)
  local t="sufficient_$1_$2"
  [[ "$3" != "logit_diff" ]] && t="${t}_$3"
  t="${t}${ABL_SUF}"
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
  local task=$1 nodes=$2 dataset=$3 nabbr=${2//+/-} loss gm cfg variant opt lr vabbr ks
  # long-prompt MIB tasks: shrink the IG/IxG attribution batch (captured with grad over all
  # layers at once) to avoid OOM; eval sweep still uses 100 test pairs.
  # long-prompt tasks: shrink the IG/IxG attribution batch (captured with grad over all
  # layers at once) to avoid OOM; eval sweep still uses 100 test pairs. arith/hours is 38
  # tokens -- arc-length, unlike the other three arith tasks (5-13).
  local grad_extra=()
  [[ "$dataset" == mib || "$task" == hours ]] && grad_extra=(--grad-examples 32)
  for loss in "${LOSSES[@]}"; do
    for gm in "${GRAD[@]}"; do
      local gm_extra=()
      [[ "$gm" == "mc_ig" ]] && gm_extra=(--ig-steps "$MC_DRAWS" --seed "$MC_SEED")
      submit "sva_${task}_${nabbr}_${gm}_${loss}" "$(grad_tag "$gm" "$loss")" \
        --model "$MODEL" --task "$task" --dataset "$dataset" --nodes "$nodes" \
        --method "$gm" --loss "$loss" --eval-examples 100 "${grad_extra[@]}" \
        "${gm_extra[@]}" "${ABL_ARG[@]}" --output "$OUT"
    done
    for cfg in "${MATTR_CONFIGS[@]}"; do
      IFS=: read -r variant opt lr <<< "$cfg"; lr=${lr:-$MATTR_LR}
      vabbr=$(case "$variant" in hard_topk_identity) echo idste;; topk) echo stopk;; *) echo soft;; esac)
      for ks in "${KS[@]}"; do
        submit "sva_${task}_${nabbr}_mattr_${vabbr}_${opt}_${ks}_${loss}" \
          "$(mattr_tag "$variant" "$opt" "$loss" "$ks")" \
          --model "$MODEL" --task "$task" --dataset "$dataset" --nodes "$nodes" \
          --method mattr --loss "$loss" --k-schedule "$ks" \
          --variant "$variant" --optimizer "$opt" --lr "$lr" "${MATTR_COMMON[@]}" \
          "${ABL_ARG[@]}" --output "$OUT"
      done
      if [[ "$IGS" -gt 1 ]]; then   # MAttr-IG variants (env-gated), IG_KS schedules only
        for ks in $IG_KS_STR; do
          submit "sva_${task}_${nabbr}_mattr_${vabbr}_${opt}_${ks}_ig${IGS}_${loss}" \
            "$(mattr_tag "$variant" "$opt" "$loss" "$ks" "$IGS")" \
            --model "$MODEL" --task "$task" --dataset "$dataset" --nodes "$nodes" \
            --method mattr --loss "$loss" --k-schedule "$ks" \
            --variant "$variant" --optimizer "$opt" --mattr-ig-steps "$IGS" --lr "$lr" \
            "${MATTR_COMMON[@]}" "${ABL_ARG[@]}" --output "$OUT"
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
# goodfire-ai/arithmetic-wild tasks. Unlike the MIB tasks these are (near-)fixed-length --
# addition 5 tokens, months/weekdays 13, hours 38 -- so all three substrates apply; eval_sva.py
# filters to the modal length for the per-position ones.
#   ARITH_TASKS="addition months weekdays hours" IG_STEPS=5 bash scripts/submit_sva_sweep.sh
read -ra ARITH_TASKS_ARR <<< "${ARITH_TASKS:-}"
for task in "${ARITH_TASKS_ARR[@]}"; do
  [[ -z "$task" ]] && continue
  for nodes in "${NODES[@]}"; do emit_grid "$task" "$nodes" arith; done
done
echo "submitted $n, skipped $skip (already present) -> $OUT"
