#!/bin/bash
# IG and I x G on the SAE RESIDUAL basis (--nodes resid_sae_span), at the MLP-neuron basis's
# settings, so the SVA+ acc-vs-faith figure can gain an "SAE" substrate facet.
#
# SETTINGS ARE submit_sva_sweep.sh's emit_grid() FOR GRADIENT METHODS, VERBATIM, with only
# --nodes changed: --eval-examples 100, all three losses, --grad-examples 32 for `hours` (38
# tokens against the other arith tasks' 5-13), and IG at eval_sva.py's default --ig-steps 10.
# Nothing else differs, which is the point -- an SAE facet is only comparable to the MLP one if
# the two were produced by the same recipe.
#
# SPAN SUBSTRATE, so SVA + Arith ONLY. eval_sva filters every pair to the modal clean-prompt
# token length for per-position layouts, and the two MIB tasks (arc_easy, ioi) are
# variable-length -- the same structural reason plot_accauc_vs_faithauc.REQUIRED gives for mlp
# and mlp+attn_head carrying no ARC-E/IOI cells.
#
# *** THE RISK THIS WAVE IS TESTING: no gradient method has EVER been run at an SAE substrate. ***
# Every SAE-substrate run on disk (4 of them, all on npi_any_subj-relc) is MAttr. resid_sae_span
# is 8,388,864 units against the mlp basis's 2,293,760 -- 3.7x -- and IG/IxG capture the gradient
# over all layers at once, which is exactly the thing submit_sva_sweep.sh caps with
# --grad-examples on long-prompt cells. So this may OOM at the shared 96G. SMOKE=1 runs the two
# nounpp/logit-diff cells first for that reason; check them before firing the other 46.
#
# ARMS selects which series to submit (default: the two gradient ones):
#   ARMS="ig ixg"            48 = 8 tasks x 2 methods x 3 losses
#   ARMS="random"            24 = 8 tasks x 3 seeds (lossless: a random ranking is not trained)
#   ARMS="mattr_sgd"         24 = 8 tasks x 3 losses, topk/SGD/log-k at lr 1.0
#   ARMS="mattr_adameps"      8 = 8 tasks, logit-diff ONLY -- matching its MLP-basis twin, which
#                                 was only ever run at logit-diff (see submit_sva_eps.sh)
#
#   SMOKE=1 bash scripts/sva/launch/submit_sae_grad.sh   # 2 jobs: ig + ixg, nounpp, logit_diff
#   bash scripts/sva/launch/submit_sae_grad.sh           # the full 48 (8 tasks x 2 methods x 3 losses)
#   DRY=1 ...                                 # print only
#   GRAD_EXAMPLES=32 ...                      # cap the attribution batch on EVERY cell, if the
#                                             # smoke OOMs. This IS a deviation from the MLP
#                                             # basis and must be recorded if it is used.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

OUT=${OUT:-results/sva_sweep}          # the patched, -input dir the figure already reads
NODES=${NODES:-resid_sae_span}
ARMS=${ARMS:-"ig ixg"}
# INPUT=1 scores/ablates the input-embedding node as an extra unit at index 0, matching MIB's
# graph. Supported on *_sae_span since 2026-08-30 (llama.py gates it to node + the two SAE
# layouts; on any OTHER per-token substrate the flag is silently dropped and you get a -input run
# under a +input label). Pair it with OUT=results/sva_sweep_input -- run_tag does NOT encode the
# input node, so the DIRECTORY is the only thing separating the two experiments, exactly as it is
# for the node substrate.
INP=(); [[ "${INPUT:-0}" == 1 ]] && INP=(--include-input)
# Job names must carry the SUBSTRATE and the +input flag, because the QUEUED guard below matches
# on name alone: without this, submitting resid_sae_span and then mlp_sae_span back-to-back makes
# the second wave see `saeg_ig_nounpp_ce` already in the queue and silently skip all 104 of its
# jobs. Same hazard submit_input_replication.sh's header documents for its OUT dirs.
SFX="$(printf %s "$NODES" | sed 's/_sae_span//; s/[+-]/_/g')"; [[ "${INPUT:-0}" == 1 ]] && SFX="${SFX}i"
# Env-overridable (space-separated), because both figures that read this dir filter to
# logit_diff -- `LOSSES="logit_diff"` fills a new substrate's facet in 56 jobs instead of 104.
read -r -a LOSSES <<<"${LOSSES:-ce acc logit_diff}"
# MAttr settings are submit_sva_sweep.sh's verbatim: topk gate, log k, 2000 steps, bs 1, 100 eval
# examples, SGD at lr 1.0 (its own off-protocol optimum, as on every other substrate) and Adam at
# lr 0.05. The eps arm adds --adam-eps 1e-2 and nothing else.
MATTR_COMMON=(--method mattr --variant topk --k-schedule log --mode sufficient
              --train-batch-size 1 --steps 2000 --eval-examples 100)
SEEDS=(42 43 44)
declare -A DS=( [nounpp]=sva [rc]=sva [simple]=sva [within_rc]=sva
                [addition]=arith [months]=arith [weekdays]=arith [hours]=arith )
TASKS=(nounpp rc simple within_rc addition months weekdays hours)
if [[ "${SMOKE:-0}" == 1 ]]; then TASKS=(nounpp); LOSSES=(logit_diff); fi

QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
n=0; skip=0; qskip=0
sub_one() {  # $1=jobname $2=predicted-filename ; rest = eval_sva args
  local name=$1 f=$2; shift 2
  if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); return; fi
  if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $f"; else
    sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "$@" --output "$OUT" >/dev/null; fi
  n=$((n+1))
}

for task in "${TASKS[@]}"; do
  # --- Random: lossless, 3 seeds, run_tag = random_s<seed>
  if grep -qw random <<<"$ARMS"; then
    for seed in "${SEEDS[@]}"; do
      sub_one "saer${SFX}_${task}_s${seed}" "$OUT/${task}_llama3_${NODES}_random_s${seed}.json" \
        --model llama3 --task "$task" --dataset "${DS[$task]}" --nodes "$NODES" "${INP[@]}" \
        --method random --seed "$seed" --loss logit_diff --eval-examples 100
    done
  fi
  # --- MAttr + SGD, all three losses
  if grep -qw mattr_sgd <<<"$ARMS"; then
    for loss in "${LOSSES[@]}"; do
      tg="sufficient_topk_sgd"; [[ "$loss" != logit_diff ]] && tg="${tg}_${loss}"
      sub_one "saems${SFX}_${task}_${loss}" "$OUT/${task}_llama3_${NODES}_${tg}_bs1.json" \
        --model llama3 --task "$task" --dataset "${DS[$task]}" --nodes "$NODES" "${INP[@]}" \
        --loss "$loss" --optimizer sgd --lr 1.0 "${MATTR_COMMON[@]}"
    done
  fi
  # --- MAttr + Adam at eps=1e-2, logit-diff only (matches its MLP-basis twin)
  if grep -qw mattr_adameps <<<"$ARMS"; then
    sub_one "saema${SFX}_${task}" "$OUT/${task}_llama3_${NODES}_sufficient_topk_adam_eps1e-2_bs1.json" \
      --model llama3 --task "$task" --dataset "${DS[$task]}" --nodes "$NODES" "${INP[@]}" \
      --loss logit_diff --optimizer adam --lr 0.05 --adam-eps 1e-2 "${MATTR_COMMON[@]}"
  fi
  for gm in ig ixg; do
    grep -qw "$gm" <<<"$ARMS" || continue
    for loss in "${LOSSES[@]}"; do
      # Mirror eval_sva.run_tag(): method, then _loss unless logit_diff. No ablation suffix
      # (patch), no bs suffix (gradient methods take no --train-batch-size).
      tag="$gm"; [[ "$loss" != logit_diff ]] && tag="${tag}_${loss}"
      f="$OUT/${task}_llama3_${NODES}_${tag}.json"
      name="saeg${SFX}_${gm}_${task}_${loss}"
      if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); continue; fi
      if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); continue; fi
      ge=()
      [[ -n "${GRAD_EXAMPLES:-}" ]] && ge=(--grad-examples "$GRAD_EXAMPLES")
      [[ -z "${GRAD_EXAMPLES:-}" && "$task" == hours ]] && ge=(--grad-examples 32)
      if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $f"; else
        sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch \
          --model llama3 --task "$task" --dataset "${DS[$task]}" --nodes "$NODES" "${INP[@]}" \
          --method "$gm" --loss "$loss" --eval-examples 100 "${ge[@]}" \
          --output "$OUT" >/dev/null
      fi
      n=$((n+1))
    done
  done
done
echo "== ${DRY:+DRY }${SMOKE:+SMOKE }submitted $n, skipped $skip (done) + $qskip (queued) -> $OUT/$NODES =="
