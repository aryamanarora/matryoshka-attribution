#!/bin/bash
# Replicate ALL node-substrate experiments WITH the input-embedding node scored + ablated
# (--include-input), matching MIB's graph which includes an input node. Per-token substrates
# can't score inputs (hooker supports it only for the node type), so this covers node only.
# Idempotent; separate output dir so nothing in results/sva_sweep/ is touched.
#   bash scripts/submit_input_replication.sh          # submit missing
#   DRY=1 bash scripts/submit_input_replication.sh    # print only
#
# ONLY / OUT / ABLATION fill the two holes in the `+input` column of fig:acc-faith
# (plots/plot_accauc_vs_faithauc.py), whose default method set is IG / I×G / MAttr (SGD):
#
#   ONLY=softsgd bash scripts/submit_input_replication.sh                      # 30 jobs
#   ONLY="ig ixg softsgd" OUT=results/sva_zeroabl_input ABLATION=zero \
#     bash scripts/submit_input_replication.sh                                 # 90 jobs
#
# ONLY is a space-separated list of ARM names (see `want` below); empty = every arm, the
# original behaviour. ABLATION=zero adds --ablation zero AND the `_zeroabl` tag fragment, so
# the skip-if-exists check still predicts the right filename -- eval_sva.run_tag inserts it
# right after the loss fragment (`ig_ce_zeroabl`, `sufficient_topk_sgd_zeroabl_bs1`), which is
# why AB is spliced into $ls rather than appended to the whole tag.
#
# A DIFFERENT OUT dir means a different experiment, so job names carry the dir: the QUEUED
# guard below matches on name, and `inp_ioi_ig_ce` submitted against sva_sweep_input would
# otherwise suppress the zero-ablation cell of the same name.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:-results/sva_sweep_input}
ABLATION=${ABLATION:-patch}
ONLY=${ONLY:-}
PFX=inp; [[ "$ABLATION" != patch ]] && PFX=inp0
AB=""; [[ "$ABLATION" != patch ]] && AB="_${ABLATION}abl"
want() { [[ -z "$ONLY" ]] || grep -qw -- "$1" <<<"$ONLY"; }
mkdir -p logs "$OUT"

LOSSES=(ce acc logit_diff)
# task -> "model dataset"
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
  # goodfire-ai/arithmetic-wild. Added so this dir carries the same four task-groups as
  # results/sva_sweep does at the node substrate -- plot_accauc_vs_faithauc.py requires every
  # node panel to average SVA+Arith+ARC-E+IOI, and without these the `Node, +input` column
  # would be a 3-group average sitting on the same axis as 4-group ones.
  [addition]="llama3 arith" [months]="llama3 arith"
  [weekdays]="llama3 arith" [hours]="llama3 arith"
)
# hours is 38 tokens (the others are 5-13), long enough that the all-layer grad capture OOMs at
# the default attribution batch -- same cap submit_sva_sweep.sh applies to it.
declare -A GRAD_EXAMPLES=([hours]=32)
MATTR_CFG=("hard_topk:adam:soft" "hard_topk_identity:sgd:idste")

# Names already in the queue. The skip-if-exists check below only sees FINISHED runs, so a
# partial submission -- sbatch dying midway, as "Failed to initialize plugin stack" did on
# 2026-08-19 after 102 of 168 -- would resubmit every queued-but-unfinished job on the next
# run, burning GPU on duplicates that race to write the same json. Job names are unique within
# this script (one OUT dir, one cell each), so matching on name is exact here.
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

n=0; skip=0; qskip=0
sub() {  # $1=jobname $2=predicted-filename-tag ; rest = eval_sva args
  local name=$1 tag=$2; shift 2
  local task model; task=$(grep -oP '(?<=--task )\S+' <<<"$*"); model=$(grep -oP '(?<=--model )\S+' <<<"$*")
  local f="$OUT/${task}_${model}_node_${tag}.json"
  if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); return; fi
  if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "sbatch -J $name ... $tag"; else sbatch -J "$name" sva_sweep.sbatch "$@" >/dev/null; fi
  n=$((n+1))
}

for task in nounpp rc simple within_rc arc_easy ioi addition months weekdays hours; do
  read -r model ds <<<"${CFG[$task]}"
  ge=(); [[ "$ds" == mib ]] && ge=(--grad-examples 32)
  [[ -n "${GRAD_EXAMPLES[$task]:-}" ]] && ge=(--grad-examples "${GRAD_EXAMPLES[$task]}")
  common=(--model "$model" --task "$task" --dataset "$ds" --nodes node --include-input
          --eval-examples 100 --output "$OUT" --ablation "$ABLATION")
  mattr_common=(--mode sufficient --train-batch-size 1 --steps 2000 --lr 0.05)
  for loss in "${LOSSES[@]}"; do
    ls=""; [[ "$loss" != logit_diff ]] && ls="_$loss"
    ls="${ls}${AB}"    # tag order is base + _loss + _zeroabl + ... ; see the header
    # gradient methods (loss = target)
    for gm in ig ixg attnlrp; do want "$gm" && sub "${PFX}_${task}_${gm}_${loss}" "${gm}${ls}" \
        "${common[@]}" --method "$gm" --loss "$loss" "${ge[@]}"; done
    want conductance && sub "${PFX}_${task}_cond_${loss}" "conductance${ls}" \
        "${common[@]}" --method conductance --loss "$loss" --ig-steps 10 "${ge[@]}"
    # MAttr: soft/idste x {log, uniform, fixed-10%, IG5}
    for cfg in "${MATTR_CFG[@]}"; do
      variant=${cfg%%:*}; opt=${cfg#*:}; opt=${opt%%:*}; ab=${cfg##*:}
      want "$ab" || continue
      base="sufficient_${variant}_${opt}${ls}"   # _bs1 suffix: train-batch-size=1 (eval_sva tag)
      sub "${PFX}_${task}_${ab}_log_${loss}"   "${base}_bs1"          "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule log     "${mattr_common[@]}"
      sub "${PFX}_${task}_${ab}_unif_${loss}"  "${base}_uniformk_bs1" "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule uniform "${mattr_common[@]}"
      sub "${PFX}_${task}_${ab}_fixed_${loss}" "${base}_fixedk10_bs1" "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule log --fixed-k-frac 0.1 "${mattr_common[@]}"
      sub "${PFX}_${task}_${ab}_ig5_${loss}"   "${base}_ig5_bs1"      "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule log --mattr-ig-steps 5 "${mattr_common[@]}"
    done
    # soft top-k forward (topk gate, Adam): log + uniform only (fixed/ig not used in fingerprints)
    if want stopk; then
    sub "${PFX}_${task}_stopk_log_${loss}"  "sufficient_topk_adam${ls}_bs1"          "${common[@]}" --method mattr --loss "$loss" --variant topk --optimizer adam --k-schedule log     "${mattr_common[@]}"
    sub "${PFX}_${task}_stopk_unif_${loss}" "sufficient_topk_adam${ls}_uniformk_bs1" "${common[@]}" --method mattr --loss "$loss" --variant topk --optimizer adam --k-schedule uniform "${mattr_common[@]}"
    fi
    # Same soft top-k forward, Adam -> SGD. This is the `MAttr (SGD)` series of fig:acc-faith,
    # and it is the arm the `+input` column was missing entirely.
    #
    # LR IS 1.0, NOT mattr_common's 0.05, and the trailing --lr wins over the earlier one. That
    # deviates from this script's shared protocol on purpose: results/sva_sweep's topk:sgd cells
    # are all lr=1.0 (60 node runs, verified), and the figure plots the -input and +input panels
    # side by side as the same method under a different substrate treatment. At 0.05 the two
    # panels would differ in LR as well as in --include-input, and the input effect would be
    # confounded by it. Soft-fwd + SGD is genuinely LR-sensitive (Adam normalises the k/n gate
    # slope away, SGD does not), so this is not a formality. Same call, same reason, as the
    # zero-ablation backfill documented in submit_sva_sweep.sh.
    #
    # LOG-k ONLY. results/sva_sweep carries both schedules for this arm, but nothing consumes
    # the +input uniform-k cells: fig:acc-faith uses `softsgd-log`, and
    # plot_kschedule_accauc_vs_faithauc.py -- the one figure that faces the schedules off across
    # both sweep dirs -- filters to `hard_topk` and never sees the soft forward at all. Adding
    # uniform would double the wave for a series no artifact reads.
    want softsgd && sub "${PFX}_${task}_softsgd_log_${loss}" "sufficient_topk_sgd${ls}_bs1" \
      "${common[@]}" --method mattr --loss "$loss" --variant topk --optimizer sgd \
      --k-schedule log "${mattr_common[@]}" --lr 1.0
  done
  # random baseline (MIB tasks only, 3 seeds)
  if [[ "$ds" == mib ]] && want random; then
    for seed in 42 43 44; do sub "${PFX}_${task}_random_s${seed}" "random_s${seed}" \
        "${common[@]}" --method random --seed "$seed"; done
  fi
done
echo "submitted $n, skipped $skip (done) + $qskip (already queued) -> $OUT"
