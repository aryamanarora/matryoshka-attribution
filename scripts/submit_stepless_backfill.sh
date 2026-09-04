#!/bin/bash
# Backfill Stepless IG so it can be drawn in the MAIN acc-vs-faith figure
# (plots/plot_accauc_vs_faithauc.py, default cut).
#
# WHY 24 RUNS. That figure's default cut is patched / logit-diff over four panels, and
# group_avg is all-or-nothing per task-group -- a method missing ANY subtask of a required
# group is dropped from the panel entirely. Stepless IG had 12/36 of those cells: the four SVA
# tasks in results/sva_sweep at each substrate, and nothing else. So it was silently absent
# from all four panels, which is why it has its own narrowed `--stepless` cut. Missing:
#
#   results/sva_sweep        node           addition months weekdays hours arc_easy ioi   (6)
#   results/sva_sweep        mlp            addition months weekdays hours                (4)
#   results/sva_sweep        mlp+attn_head  addition months weekdays hours                (4)
#   results/sva_sweep_input  node           all ten tasks                                 (10)
#
# LOGIT-DIFF ONLY, because that is all the default cut draws. This does NOT make Stepless IG
# complete for the multi-loss artifacts: `--all` / `--adam` and plots/plot_sva_robustness_grid.py
# want ce and acc too, which is a further ~66 runs. Those stay out of scope until someone wants
# that figure to carry the row.
#
# --ig-steps 1 IS ALWAYS PASSED. eval_sva.py defaults it to 10, which for this method silently
# buys a 10x-cost run under a filename that claims m=1 -- and m=1 is the whole point of the
# arm, since at one draw it is compute-matched to I x G. --seed 42 matches MC_SEED in
# submit_sva_sweep.sh / submit_input_replication.sh, so these cells pair 1:1 by filename with
# the twelve already on disk. run_tag writes `mc_ig_m1_s42`.
#
#   bash scripts/submit_stepless_backfill.sh        # submit missing
#   DRY=1 bash scripts/submit_stepless_backfill.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

M=1; SEED=42; TAG="mc_ig_m${M}_s${SEED}"
# task -> "model dataset". Same pins as every submitter in this family: ioi is qwen2.5,
# everything else llama3. plot_accauc_vs_faithauc.on_model drops a run on the wrong model.
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith" [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
n=0; skip=0; qskip=0

sub() {  # $1=out $2=task $3=nodes $4=include-input(0|1)
  local out=$1 task=$2 nodes=$3 inp=$4
  local model ds; read -r model ds <<<"${CFG[$task]}"
  local nabbr=${nodes//+/-}
  local f="$out/${task}_${model}_${nabbr}_${TAG}.json"
  local name="sless_$( [[ $inp == 1 ]] && echo i)${task}_${nabbr}"
  if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); return; fi
  if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); return; fi
  # Long-prompt cells shrink the attribution batch: the gradient is captured over all layers at
  # once and OOMs at the default. Same cap submit_sva_sweep.sh applies -- the MIB tasks, plus
  # `hours` at 38 tokens against the other arith tasks' 5-13.
  local ge=(); [[ "$ds" == mib || "$task" == hours ]] && ge=(--grad-examples 32)
  local extra=(); [[ "$inp" == 1 ]] && extra=(--include-input)
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $f"; else
    mkdir -p "$out"
    sbatch -J "$name" sva_sweep.sbatch \
      --model "$model" --task "$task" --dataset "$ds" --nodes "$nodes" \
      --method mc_ig --ig-steps "$M" --seed "$SEED" --loss logit_diff \
      --eval-examples 100 "${ge[@]}" "${extra[@]}" --output "$out" >/dev/null
  fi
  n=$((n+1))
}

ARITH=(addition months weekdays hours)
for t in "${ARITH[@]}"; do
  for nd in node mlp "mlp+attn_head"; do sub results/sva_sweep "$t" "$nd" 0; done
done
for t in arc_easy ioi; do sub results/sva_sweep "$t" node 0; done
# +input: node only -- the hooker can score/ablate the input embedding for that type alone.
for t in nounpp rc simple within_rc "${ARITH[@]}" arc_easy ioi; do
  sub results/sva_sweep_input "$t" node 1
done
echo "== ${DRY:+DRY }submitted $n, skipped $skip (done) + $qskip (queued) =="
