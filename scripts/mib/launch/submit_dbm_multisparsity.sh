#!/bin/bash
# Evaluate the DBM sparsity ladder at each run's OWN empirical L0, one job per MIB node cell.
#
# See scripts/mib/eval_dbm_multisparsity.py for what this measures and why it is not the same
# object as the other rows of the table. In short: MIB sweeps one ranking over ten fixed
# proportions, which scores a DBM run mostly on the part of its score vector that stopped being
# trained; this scores each run only at the sparsity it was trained for.
#
# ONE JOB PER CELL, not per (cell, lambda): the script loads the model once and shares the clean
# and fully-ablated references across the whole ladder, so a cell is (n_rungs + 2) graph
# evaluations rather than n_rungs * 12. Cost is a bit below one ordinary MIB eval per cell.
#
# Runs in the MIB venv (TL 2.15.4) -- mandatory for gemma2 (Gemma-2 forward bug under TL 3.x).
# --head 200 on llama3 VALIDATION only, matching run_variants.sh and every other row of the
# validation table; test splits are uncapped everywhere in this repo.
#
#   bash scripts/mib/launch/submit_dbm_multisparsity.sh              # 11 validation jobs
#   SPLIT=test bash scripts/mib/launch/submit_dbm_multisparsity.sh   # needs a test-split ladder
#   DRYRUN=1 bash scripts/mib/launch/submit_dbm_multisparsity.sh
set -u
L2A=/home/guests/aryaman/learning-to-attribute
MIB=/home/guests/aryaman/MIB-circuit-track
cd "$L2A"
DRYRUN=${DRYRUN:-0}
SPLIT=${SPLIT:-validation}
PY=$MIB/.venv/bin/python

CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 10" "llama3 ioi 2"
 "llama3 arithmetic_subtraction 2"
 "qwen2.5 mcqa 10" "gemma2 mcqa 10" "llama3 mcqa 2"
 "gemma2 arc_easy 4" "llama3 arc_easy 2" "llama3 arc_challenge 2"
)
n=0
for cell in "${CELLS[@]}"; do
  read -r model task bs <<< "$cell"
  # ONLY is a grep -E pattern on "task/model", for relaunching a subset -- a cell whose ladder
  # is still training should NOT be submitted, because a missing rung is silently SKIPped and
  # the cell lands on a shorter ladder than its neighbours (which dbm_multisparsity.check_
  # consistent then warns about, after the GPU time is already spent).
  if [ -n "${ONLY:-}" ] && ! echo "$task/$model" | grep -qE "$ONLY"; then continue; fi
  HEAD=""
  [ "$model" = "llama3" ] && [ "$SPLIT" = "validation" ] && HEAD="--head 200"
  # 16h on llama3, not 8: the ARC test splits at batch 2 are ~20 min per dataset pass, and both
  # ARC cells hit the old 8h wall on the seven-rung ladder. The reference cache took the cost
  # from 4 passes per rung to 1, so a 9-rung ladder is now ~10 passes rather than ~40, but
  # arc_easy (1188 examples) still lands near 7h and there is no reason to run it that close.
  if [ "$model" = "llama3" ]; then mem=96G; tlim=16:00:00
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=04:00:00
  else mem=32G; tlim=02:00:00; fi
  # Split goes in the NAME, so a validation wave does not overwrite the test wave's logs. The
  # logs are the only place a rung's fate is recorded -- "degenerate run, skipped" vs "SKIP
  # l1=...: no graph/scores" -- and dbm_multisparsity.check_consistent points readers at them.
  name="dbmL0$([ "$SPLIT" = validation ] && echo v)-${task}-${model}"
  cmd="cd $L2A; export PYTHONPATH=$MIB:$MIB/EAP-IG/src; \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/mib/eval_dbm_multisparsity.py --model $model --task $task --split $SPLIT \
--batch-size $bs --mib-path $MIB $HEAD"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name (mem=$mem t=$tlim bs=$bs) $HEAD"
  else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
      --job-name="$name" --output="$L2A/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== $([ "$DRYRUN" = 1 ] && echo "DRY ")total $n DBM multi-sparsity eval jobs (split=$SPLIT) =="
