#!/bin/bash
# Soft-forward MAttr with the UNIFORM k-schedule at lr=0.05, on node AND edge, val AND test.
#
#   bash scripts/submit_softuni_lr05.sh            # 40 jobs
#   DRYRUN=1 bash scripts/submit_softuni_lr05.sh   # preview
#
# WHY THIS EXISTS: the paper's uniform-k ablation was only ever run with the HARD forward at
# lr=0.05 (htk_lr_0.05 / mib_edge_hard_topk_uniform_lr05). The soft forward -- which is the
# headline MAttr -- was run with uniform k only at node level and only at the DEFAULT lr=0.01
# (results/final_node, from submit_mib_node_ablations.sh, which passes no --lr), and at edge
# level not at all ("No soft-fwd uniform edge run", make_mib_table.py). So there was no
# soft-uniform row that could go in the test table next to the other lr=0.05 rows.
#
# lr=0.05 for all four waves, including the validation reruns, so the validation and test
# "+ unif k" rows are the SAME config. Using final_node (lr=0.01) for validation and a fresh
# lr=0.05 run for test would put two different configs under one row label across two tables.
#
# Protocols are copied from the scripts that produced the neighbouring rows, NOT re-derived:
#   node  -- submit_lr_sweep_topklog.sh / submit_test_lr05.sh: steps 500, --include-input,
#            11 cells; llama3/ioi capped at --eval-examples 200 on VALIDATION only (that is the
#            dagger in the val tables); test splits are <=1188 so nothing is capped there.
#   edge  -- submit_edge_lr05.sh: steps 5000, no --include-input, 9 cells (no llama3 ARC, there
#            are no edge circuits there), llama3 capped at 200 on BOTH splits.
#
# GEMMA2 CAVEAT: like every other script here this runs the L2A venv (TL 3.2.1), whose Gemma-2
# forward is wrong. The four new dirs are registered in reeval_gemma_mib.py's DIRS, so after
# this wave lands the gemma2 cells MUST be re-evaluated under the MIB venv (TL 2.15.4):
#   PYTHONPATH=MIB-circuit-track:MIB-circuit-track/EAP-IG/src \
#     MIB-circuit-track/.venv/bin/python scripts/reeval_gemma_mib.py --level node --split test \
#       --task <ioi|mcqa|arc_easy> --dirs test_node_topk_uniform_lr05
# Skipping that step silently ships wrong Gemma numbers in three cells per dir.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"; PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}

NODE_PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
# Edge is 11 cells, same as node. An earlier version of this list had 9, copied from
# submit_edge_lr05.sh's header ("no llama arc") -- that is WRONG for us and the header was
# wrong too. The 9-cell restriction belongs to the MIB *baselines* (EAP-IG and friends read a
# circuit file, and none is published for llama3 ARC at edge level); we learn the mask
# ourselves and need no circuit, which is why results/{mib,test}_edge_topk_log_lr05 each hold
# 11 pkls. Dropping the two llama3 ARC cells would have left "+ unif k" permanently 9/11 and
# thus never comparable to the \ourmethod{} row it exists to be read against.
EDGE_PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy"
  "llama3 arc_easy" "llama3 arc_challenge"
)

n=0
run() {  # level split model task
  local level=$1 split=$2 model=$3 task=$4 out cmd name cpus mem tlim
  if [ "$level" = node ]; then
    out=$([ "$split" = validation ] && echo mib_node_topk_uniform_lr05 || echo test_node_topk_uniform_lr05)
    case $model in
      gpt2|qwen2.5) cpus=2; mem=32G;  tlim=04:00:00; bs="" ;;
      gemma2)       cpus=3; mem=64G;  tlim=08:00:00; bs="--batch-size 4" ;;
      llama3)       cpus=4; mem=96G;  tlim=12:00:00; bs="--batch-size 2" ;;
    esac
    # The val-only llama3/ioi cap: 10k validation examples crawl on 8B. Test is <=1188.
    local ec=""
    [ "$split" = validation ] && [ "$model" = llama3 ] && [ "$task" = ioi ] && { ec="--eval-examples 200"; tlim=06:00:00; }
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform \
--masking topk --mode sufficient --lr 0.05 --split $split --train-split train \
--include-input $bs $ec --output results/$out"
  else
    out=$([ "$split" = validation ] && echo mib_edge_topk_uniform_lr05 || echo test_edge_topk_uniform_lr05)
    case $model in
      gpt2|qwen2.5) cpus=3; mem=64G;  tlim=10:00:00; bs=5; ev=0 ;;
      gemma2)       cpus=4; mem=96G;  tlim=16:00:00; bs=5; ev=0 ;;
      llama3)       cpus=5; mem=128G; tlim=24:00:00; bs=2; ev=200 ;;
    esac
    cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$PY scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule uniform \
--masking topk --mode sufficient --lr 0.05 --split $split --train-split train \
--batch-size $bs --eval-examples $ev --output results/$out"
  fi
  name="su05-${level:0:1}-${split:0:3}-${task}-${model}"
  if [ "$DRYRUN" = "1" ]; then echo "DRY $name -> $out"; else
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name -> $out"
  fi
  n=$((n+1))
}

for split in validation test; do
  for p in "${NODE_PAIRS[@]}"; do read -r model task <<< "$p"; run node "$split" "$model" "$task"; done
  for p in "${EDGE_PAIRS[@]}"; do read -r model task <<< "$p"; run edge "$split" "$model" "$task"; done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n soft-fwd uniform-k lr=0.05 jobs =="
