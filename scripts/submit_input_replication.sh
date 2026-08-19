#!/bin/bash
# Replicate ALL node-substrate experiments WITH the input-embedding node scored + ablated
# (--include-input), matching MIB's graph which includes an input node. Per-token substrates
# can't score inputs (hooker supports it only for the node type), so this covers node only.
# Idempotent; separate output dir so nothing in results/sva_sweep/ is touched.
#   bash scripts/submit_input_replication.sh          # submit missing
#   DRY=1 bash scripts/submit_input_replication.sh    # print only
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/sva_sweep_input
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
          --eval-examples 100 --output "$OUT")
  mattr_common=(--mode sufficient --train-batch-size 1 --steps 2000 --lr 0.05)
  for loss in "${LOSSES[@]}"; do
    ls=""; [[ "$loss" != logit_diff ]] && ls="_$loss"
    # gradient methods (loss = target)
    for gm in ig ixg attnlrp; do sub "inp_${task}_${gm}_${loss}" "${gm}${ls}" \
        "${common[@]}" --method "$gm" --loss "$loss" "${ge[@]}"; done
    sub "inp_${task}_cond_${loss}" "conductance${ls}" \
        "${common[@]}" --method conductance --loss "$loss" --ig-steps 10 "${ge[@]}"
    # MAttr: soft/idste x {log, uniform, fixed-10%, IG5}
    for cfg in "${MATTR_CFG[@]}"; do
      variant=${cfg%%:*}; opt=${cfg#*:}; opt=${opt%%:*}; ab=${cfg##*:}
      base="sufficient_${variant}_${opt}${ls}"   # _bs1 suffix: train-batch-size=1 (eval_sva tag)
      sub "inp_${task}_${ab}_log_${loss}"   "${base}_bs1"          "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule log     "${mattr_common[@]}"
      sub "inp_${task}_${ab}_unif_${loss}"  "${base}_uniformk_bs1" "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule uniform "${mattr_common[@]}"
      sub "inp_${task}_${ab}_fixed_${loss}" "${base}_fixedk10_bs1" "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule log --fixed-k-frac 0.1 "${mattr_common[@]}"
      sub "inp_${task}_${ab}_ig5_${loss}"   "${base}_ig5_bs1"      "${common[@]}" --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt" --k-schedule log --mattr-ig-steps 5 "${mattr_common[@]}"
    done
    # soft top-k forward (topk gate, Adam): log + uniform only (fixed/ig not used in fingerprints)
    sub "inp_${task}_stopk_log_${loss}"  "sufficient_topk_adam${ls}_bs1"          "${common[@]}" --method mattr --loss "$loss" --variant topk --optimizer adam --k-schedule log     "${mattr_common[@]}"
    sub "inp_${task}_stopk_unif_${loss}" "sufficient_topk_adam${ls}_uniformk_bs1" "${common[@]}" --method mattr --loss "$loss" --variant topk --optimizer adam --k-schedule uniform "${mattr_common[@]}"
  done
  # random baseline (MIB tasks only, 3 seeds)
  if [[ "$ds" == mib ]]; then
    for seed in 42 43 44; do sub "inp_${task}_random_s${seed}" "random_s${seed}" \
        "${common[@]}" --method random --seed "$seed"; done
  fi
done
echo "submitted $n, skipped $skip (done) + $qskip (already queued) -> $OUT"
