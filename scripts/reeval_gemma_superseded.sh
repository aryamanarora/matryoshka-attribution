#!/bin/bash
# Re-evaluate the gemma2 cells of the two superseded test dirs under TL 2.15.4, CPR included.
#
# test_node_hard_topk_uniform and test_edge_hard_topk_uniform are the pre-lr05 test dirs.
# make_mib_test_table.py reads only their *_lr05 twins and nothing else references them, which
# is why reeval_gemma_mib.py's 2026-07-24 pass skipped them -- correctly, at the time.
#
# But submit_accauc_backfill.py adds acc-AUC to them, and its preserve-CPR default is exactly
# wrong here: their gemma2 CPR was computed in the L2A venv (TL 3.2.1, wrong Gemma-2 forward),
# so preserving it leaves one pkl holding a TL 3.2.1 CPR next to a TL 2.15.4 acc-AUC, with
# nothing on disk to say the two halves disagree. The gap is not academic -- the backfill logged
# "CPR kept 6.410; fresh was 5.067" for ioi/gemma2, a 26% inflation.
#
# So: --clobber-cpr. No published number moves (nothing reads these dirs); what moves is a
# stale number in an unread dir, replaced by one from the same forward as its acc-AUC.
#
#   bash scripts/reeval_gemma_superseded.sh            # submit
#   DRYRUN=1 bash scripts/reeval_gemma_superseded.sh   # preview
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
MIBV=/home/guests/aryaman/MIB-circuit-track/.venv/bin/python
DRYRUN=${DRYRUN:-0}

# the gemma2 columns of MIB: ioi, mcqa, arc_easy
n=0
for lvl in node edge; do
  out="test_${lvl}_hard_topk_uniform"
  for task in ioi mcqa arc_easy; do
    [ -f "$ABS/results/$out/${task}_gemma2_$( [ $lvl = node ] && echo importances.json \
        || echo scores.pt)" ] || { echo "SKIP $out/$task: no circuit"; continue; }
    bs=4; [ "$task" = arc_easy ] && bs=1
    name="gs-${lvl}-${task}-gemma2"
    cmd="cd $ABS && PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $MIBV scripts/reeval_mib_accauc.py \
--level $lvl --split test --model gemma2 --task $task --dirs $out \
--batch-size $bs --clobber-cpr"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name -> $out"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=3 --mem=64G --time=16:00:00 \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n superseded-dir gemma2 re-evals =="
