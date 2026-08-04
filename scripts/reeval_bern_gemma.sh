#!/bin/bash
# Re-evaluate the four gemma2 cells that submit_bern_fill.sh produced, under TL 2.15.4.
#
# submit_bern_fill.sh trains AND evaluates through the L2A venv (TL 3.2.1), whose Gemma-2
# forward is wrong (CLAUDE.md; proved against an HF reference in 525673a). Training there is
# what every sibling cell of these rows did, so it stays -- swapping venvs for the new cells
# only would make them the odd ones out, and eval_mib.py cannot run in the MIB venv anyway
# (no learning_to_attribute package). But the EVAL has to be redone, which is exactly what
# reeval_gemma_mib.py did for the rest of the sweep on 2026-07-24.
#
# --clobber-cpr is REQUIRED here and is the opposite of the backfill's default: for these cells
# the area_under sitting in the pkl was computed under the buggy forward minutes ago, so it is
# the thing being fixed, not a published number being protected.
#
# Jobs chain off the training jobs with --dependency=afterok, so a training failure leaves the
# bad pkl visibly un-re-evaluated instead of quietly re-blessing a half-written circuit.
#
#   bash scripts/reeval_bern_gemma.sh              # submit (auto-detects running trainers)
#   DRYRUN=1 bash scripts/reeval_bern_gemma.sh     # preview
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
MIBV=/home/guests/aryaman/MIB-circuit-track/.venv/bin/python
DRYRUN=${DRYRUN:-0}

# dir | task   -- the gemma2 holes submit_bern_fill.sh was filling
CELLS=("bern_lr_0.05|ioi" "bern_lr_0.1_2k|ioi" "bern_lr_0.1_2k|mcqa" "bern_lr_0.1_2k|arc_easy")

n=0
for c in "${CELLS[@]}"; do
  IFS='|' read -r out task <<< "$c"
  # find the trainer for this cell if it is still queued/running, and chain off it
  jid=$(squeue -u "$USER" -h -o "%i %j" \
        | awk -v p="bf-${out#bern_lr_}-${task}-gemma2" '$2==p {print $1; exit}')
  dep=""; [ -n "$jid" ] && dep="--dependency=afterok:$jid"
  if [ -z "$jid" ] && [ ! -f "$ABS/results/$out/${task}_gemma2_scores.pt" ]; then
    echo "SKIP $out/$task: no circuit and no queued trainer"
    continue
  fi
  bs=4; [ "$task" = arc_easy ] && bs=1
  name="rg-${out#bern_lr_}-${task}-gemma2"
  cmd="cd $ABS && PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $MIBV scripts/reeval_mib_accauc.py \
--level node --split validation --model gemma2 --task $task --dirs $out \
--batch-size $bs --clobber-cpr"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY $name ${dep:-(no dep; circuit already on disk)}"
  else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=3 --mem=64G --time=16:00:00 $dep \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name ${dep:+after $jid}"
  fi
  n=$((n+1))
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n gemma2 bern re-evals =="
