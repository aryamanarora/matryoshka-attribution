#!/bin/bash
# Fill the missing cells of the REINFORCE ($+$ hard bwd) LR-sweep rows in paper/tabs/lr_sweep.tex.
#
# Those rows are the only ones in that table with holes: bern_lr_0.01 is 10/11, bern_lr_0.05 9/11,
# bern_lr_0.3 7/11 and bern_lr_0.1_2k 1/11 (it was a single-cell probe of "does 4x the step budget
# rescue REINFORCE?" on ioi/gpt2, never extended). A row averaged over 7 cells is not comparable to
# one averaged over 11, so make_lr_table.py prints "---" and the row average is quietly computed
# over a different subset than its neighbours -- filling is the only way to make the LR curve for
# this method mean the same thing as the others.
#
# Hyperparameters copied from the `args` of each dir's existing scores.pt: masking
# bernoulli_reinforce, k-schedule uniform, steps 500 (2000 for the _2k row), --include-input, and
# the per-model batch sizes the sweep already used. lr comes from the dir name.
#
# GEMMA2 CELLS NEED A FOLLOW-UP: this runs eval_mib.py, which only exists in the L2A venv
# (TL 3.2.1), whose Gemma-2 forward is wrong. Training there matches every sibling cell of
# these rows, but the eval does not -- run scripts/reeval_bern_gemma.sh afterwards to redo it
# under TL 2.15.4, the same fix reeval_gemma_mib.py applied to the rest of the sweep.
#
#   bash scripts/submit_bern_fill.sh            # submit
#   DRYRUN=1 bash scripts/submit_bern_fill.sh   # preview
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy"
  "llama3 arc_challenge"
)
# dir | lr | steps
CONFIGS=("bern_lr_0.01|0.01|500" "bern_lr_0.05|0.05|500" "bern_lr_0.3|0.3|500"
         "bern_lr_0.1_2k|0.1|2000")
n=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r out lr steps <<< "$c"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    [ -f "$ABS/results/$out/${task}_${model}_scores.pt" ] && continue
    case $model in
      gpt2|qwen2.5) cpus=2; mem=32G; tlim=04:00:00; bs="" ;;
      gemma2)       cpus=3; mem=64G; tlim=08:00:00; bs="--batch-size 4" ;;
      llama3)       cpus=4; mem=96G; tlim=12:00:00; bs="--batch-size 2" ;;
    esac
    # llama3/ioi is the one cell the whole sweep caps at 200 eval examples (the dagger in
    # lr_sweep.tex); matching it here keeps the new cell in the same column as the old ones.
    ec=""; [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && { ec="--eval-examples 200"; tlim=06:00:00; }
    [ "$steps" = "2000" ] && tlim=$(echo "$tlim" | awk -F: '{printf "%02d:%s:%s", $1*2, $2, $3}')
    name="bf-${out#bern_lr_}-${task}-${model}"
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model $model --task $task --steps $steps --k-schedule uniform \
--masking bernoulli_reinforce --mode sufficient --lr $lr --split validation --train-split train \
--include-input $bs $ec --output results/$out"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name -> $out ${ec:+[$ec]}"
    else
      mkdir -p "$ABS/logs"
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n REINFORCE fill jobs =="
