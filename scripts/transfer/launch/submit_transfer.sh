#!/bin/bash
# Cross-task transfer matrix for the headline MAttr (soft fwd, log-k, SGD, node, +input):
# evaluate every task's learned ranking on every OTHER task's eval harness, 13x13 over the
# same llama3 cells as paper/figs/task_corr_heatmap.pdf. 200 eval examples everywhere --
# including the diagonal, which is recomputed in-loop so every cell of the matrix (reference
# included) comes from the same eval on the same examples.
#
#   bash scripts/transfer/launch/submit_transfer.sh          # submit all 13 jobs (one per TARGET task)
#   DRY=1 bash scripts/transfer/launch/submit_transfer.sh    # print instead
#
# METHOD=<m> switches every source to results/transfer_src/<m>/{task}.pt (prepared by
# scripts/transfer/prep_transfer_sources.py -- e.g. adam / mc_ig / attnlrp) and suffixes the output
# dirs (results/transfer_{mib,sva}_<m>) and job names, so each method's round is its own
# resumable matrix. Unset = the original MAttr(SGD) round off the raw headline files.
#
# One job per TARGET task; the 13 SOURCE rankings are looped inside the job so the 8B model
# loads once. Both eval paths skip sources whose output already exists, so resubmitting after
# a preemption resumes rather than recomputing.
#
# MIB targets (ioi/arith-sub/mcqa/arc-e/arc-c) go through scripts/transfer/eval_transfer_mib.py
# (MIB's evaluate_area_under_curve -> results/transfer_mib/{task}_llama3_transfer.json).
# SVA/arith-wild targets go through eval_sva.py --scores-from
# (-> results/transfer_sva/{task}_llama3_node_xfer_{label}.json, the usual json schema).
# The two halves are different harnesses/metrics, so compare cells WITHIN a target row
# (normalise by the diagonal), never across the harness boundary.
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

MIB_DIR=results/softlog_sgd_lr_1.0                # headline MAttr, MIB harness (README.md)
SVA_DIR=results/sva_sweep_input                    # headline MAttr, eval_sva harness
SVA_TAG=node_iso_topk_sgd_bs1

MIB_TASKS=(ioi arithmetic_subtraction mcqa arc_easy arc_challenge)
SVA_TASKS=(simple nounpp rc within_rc)             # --dataset sva
ARITH_TASKS=(addition months weekdays hours)       # --dataset arith (arithmetic in the wild)

# The 13 sources, LABEL:PATH. Labels are the task names; they become filename fragments.
METHOD=${METHOD:-}
SUF=""; [[ -n "$METHOD" ]] && SUF="_${METHOD}"
SOURCES=()
HAVE=" "   # tasks with a source; a SVA-only method (ixg) legitimately lacks the MIB five,
           # and a target without its own source has no diagonal, so it gets no job either.
if [[ -n "$METHOD" ]]; then
  [[ -d "results/transfer_src/$METHOD" ]] || { echo "no results/transfer_src/$METHOD (run scripts/transfer/prep_transfer_sources.py)" >&2; exit 1; }
  for t in "${MIB_TASKS[@]}" "${SVA_TASKS[@]}" "${ARITH_TASKS[@]}"; do
    f="results/transfer_src/$METHOD/${t}.pt"
    [[ -f $f ]] || { echo "skip source $t (no $f)"; continue; }
    SOURCES+=("$t:$f"); HAVE+="$t "
  done
else
  HAVE=" ${MIB_TASKS[*]} ${SVA_TASKS[*]} ${ARITH_TASKS[*]} "
  for t in "${MIB_TASKS[@]}"; do
    f="$MIB_DIR/${t}_llama3_scores.pt"; [[ -f $f ]] || { echo "MISSING $f" >&2; exit 1; }
    SOURCES+=("$t:$f")
  done
  for t in "${SVA_TASKS[@]}" "${ARITH_TASKS[@]}"; do
    f="$SVA_DIR/${t}_llama3_${SVA_TAG}.scores.pt"; [[ -f $f ]] || { echo "MISSING $f" >&2; exit 1; }
    SOURCES+=("$t:$f")
  done
fi

QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
sub() {  # $1=jobname, rest = sbatch args
  local name=$1; shift
  if grep -qxF "$name" <<<"$QUEUED"; then echo "queued  $name"; return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "DRY sbatch -J $name $*"; else
    sbatch -J "$name" "$@" >/dev/null && echo "submit  $name"; fi
}

# --- MIB harness targets ---------------------------------------------------------------
for t in "${MIB_TASKS[@]}"; do
  [[ "$HAVE" == *" $t "* ]] || continue
  sub "xferm${SUF}_${t}" scripts/transfer/launch/transfer_mib.sbatch \
    --model llama3 --task "$t" --split validation --batch-size 2 --eval-examples 200 \
    --output "results/transfer_mib${SUF}" --sources "${SOURCES[@]}"
done

# --- eval_sva harness targets ----------------------------------------------------------
# Recipe flags match the headline runs (they only feed the config record -- nothing trains);
# --eval-examples 200 is the one deliberate difference from the originals' 100, which is why
# the diagonal is recomputed here rather than read from $SVA_DIR.
for t in "${SVA_TASKS[@]}" "${ARITH_TASKS[@]}"; do
  [[ "$HAVE" == *" $t "* ]] || continue
  ds=sva; [[ " ${ARITH_TASKS[*]} " == *" $t "* ]] && ds=arith
  sub "xfers${SUF}_${t}" --time=24:00:00 scripts/sva/launch/sva_sweep.sbatch \
    --model llama3 --task "$t" --dataset "$ds" --nodes node --include-input \
    --method mattr --variant topk --optimizer sgd --k-schedule log --mode iso \
    --train-batch-size 1 --lr 1.0 --loss logit_diff --eval-examples 200 --no-wandb \
    --output "results/transfer_sva${SUF}" --scores-from "${SOURCES[@]}"
done
