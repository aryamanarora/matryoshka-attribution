#!/bin/bash
# Sparsity-penalty sweep for the DBM (pyvene sigmoid-mask) baseline, at its best LR.
#
# WHY THIS EXISTS: the DBM rows in the paper are trained with NO sparsity term, and the defence
# for that was "we faithfully reproduce pyvene." That defence is weaker than it looked. The
# pyvene *library* adds no penalty, but pyvene's own tutorial for this exact class
# (tutorials/advanced_tutorials/IOI_with_Mask_Intervention.ipynb) trains it with
#   loss + 1.0 * torch.norm(v.mask, 1)
# and its Boundless DAS tutorial does the same with 2.0 * intervention_boundaries.sum().
# So an unpenalised mask is a choice we made, not one the method forces, and it is exactly the
# choice that produces DBM's 38--54% achieved density at every LR. This block gives the
# baseline the penalty it is normally trained with.
#
# WHICH PENALTY: --l1-target gate, i.e. coeff * z.mean(). That is Boundless DAS's term, not an
# invention of ours. Its `intervention_boundaries` is a scalar in [1e-3, 1] that multiplies
# embed_dim to place the mask boundary, so the mask is ~1 below index b*d and b IS the density;
# penalising b == penalising mean gate value == the paper's "L1 encouraging the smallest
# sufficient subspace" (Wu et al. 2023).
#
# NOT the mask tutorial's ||mask||_1 (--l1-target logit). That penalises the pre-sigmoid logits,
# which init at 0, so it drives every gate toward z=0.5 -- toward the ~50% density the
# unpenalised runs already sit at. Measured on a synthetic task: density 0.525 -> 0.500 as the
# coefficient grows, vs 0.999 -> 0.031 for the gate form. It regularises magnitude, it does not
# sparsify. The flag exists (L1_TARGET=logit) if we ever want that row for completeness.
#
# WHICH COEFFICIENTS: log-spaced around 2.0, Boundless DAS's published value. z.mean() and their
# boundary b have the same range and the same meaning, so their constant transfers directly --
# this block gets a published default anchor just like "0.001 (pyvene)" anchors the DBM LR block
# and "0.8 (default)" anchors Node Pruning's. 0.2--20 also brackets the transition measured on
# the synthetic task (density starts moving between 1 and 30).
#
# WHICH LR: 0.3, the best of the five DBM LR points by Avg CPR over all 11 cells
# (1.31; 0.1 -> 1.10 on 10 cells, 1.0 -> 1.06 on 3, pyvene's own 0.001 -> 0.75). Sweeping the
# penalty at a bad LR would confound the two knobs.
#
# The runner handles what is easy to get wrong: trains and evaluates in the MIB venv
# (TL 2.15.4, so the gemma2 cells dodge the Gemma-2 forward bug) and applies --head 200 to
# llama3 validation only. Nothing here overrides either.
#
# Cost: 5 coefficients x 11 cells = 55 jobs. Submitted behind the llama3 ARC edge wave, which is
# filling actual "---" holes in the paper; this sharpens a baseline row that already has numbers.
#
#   bash scripts/submit_dbm_l1.sh              # submit all 55
#   bash scripts/submit_dbm_l1.sh 2.0          # just one coefficient
#   DRYRUN=1 bash scripts/submit_dbm_l1.sh     # preview
#   L1_TARGET=logit bash scripts/submit_dbm_l1.sh   # the mask tutorial's term instead
#
# Register new dirs in scripts/make_lr_table.py:METHODS to get their table rows.
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
NICE=${NICE:-3000}
LR=${LR:-0.3}
L1_TARGET=${L1_TARGET:-gate}

COEFFS=("${@:-}")
[ -z "${1:-}" ] && COEFFS=(0.2 0.6 2.0 6.0 20.0)

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

case "$L1_TARGET" in
  gate)  TAG="_l1" ;;
  logit) TAG="_l1logit" ;;
  *) echo "L1_TARGET must be gate or logit, got $L1_TARGET"; exit 1 ;;
esac

n=0; skipped=0
for c in "${COEFFS[@]}"; do
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    # Suffix must mirror run_edge_pruning.sbatch exactly (_ld then _sig then _lr<LR> then
    # _l1<C>), or the skip check reads a dir that never gets written and every job reruns.
    dir="eprun_node_ld_sig_lr${LR}${TAG}${c}"
    if [ -f "$L2A/results/$dir/graph_${task}_${model}.json" ]; then
      echo "SKIP $task/$model l1=$c: already trained"
      skipped=$((skipped+1))
      continue
    fi
    name="dbml1-${c}-${task}-${model}"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name -> results/$dir"
    else
      GATE=sigmoid LOSS=logit_diff LR="$LR" L1="$c" L1_TARGET="$L1_TARGET" \
        sbatch --nice="$NICE" --job-name="$name" \
        scripts/run_edge_pruning.sbatch "$model" "$task" node 3000 validation >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n DBM+L1 jobs (${skipped} skipped), target=$L1_TARGET lr=$LR, coeffs ${COEFFS[*]} =="
