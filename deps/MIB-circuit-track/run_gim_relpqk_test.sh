#!/bin/bash
# TEST-set evals for the GIM and RelP+QK node baselines, so they can join
# paper/tabs/mib_test_results.tex alongside \ourmethod{} and Node Pruning.
#
#   bash run_gim_relpqk_test.sh            # 22 jobs (11 cells x 2 methods)
#   DRYRUN=1 bash run_gim_relpqk_test.sh   # preview
#   ARMS=gim bash run_gim_relpqk_test.sh   # just one method
#
# WHY THE ARMS FILTER EXISTS. The Aug 6 run of this script wrote 11 GIM test pkls from the
# PRE-FIX nomlp circuits; the GIM bug was fixed on Aug 8 and results/gim rebuilt, validation was
# re-run, but test was not -- so the buggy test outputs were quarantined to _stale_gim_nomlp and
# the paper has no GIM test row. Re-running the whole script to fix that would also recompute
# RelP+QK, whose 11/11 test pkls are already correct: ~6 llama3 cells at up to 10h each, thrown
# away. The failure would be silent (identical numbers rewritten), which is exactly the kind of
# waste that never shows up in a diff.
#
# EVAL ONLY -- run_attribution.py is deliberately absent. Both methods attribute on the TRAIN
# split (see run_gim.sh / run_relp_qkgrad.sh), so results/{gim,relp_qkgrad}/*.json is already
# the circuit we owe the test set. Re-attributing would not merely waste GPU time, it would
# produce a *different* circuit and break the "same circuit, two splits" claim that lets the
# validation and test tables be read against each other.
#
# Two deliberate differences from the validation runs:
#  1. NO --head. run_gim.sh caps llama3 validation at 200 examples because 10k val crawls on 8B;
#     the test splits are <=1188 (ioi/arith 1000, arc-e 1188, arc-c 586, mcqa 50) and every
#     other row of the test table is full-split, so capping here would make these the only
#     subset-scored rows in that table.
#  2. --output-dir writes straight into the L2A results tree (absolute path), which is what
#     make_mib_test_table.py reads. The validation pkls exist in both trees because they were
#     copied by hand afterwards; writing once removes that drift.
#
# Runs in THIS repo's .venv (TL 2.15.4) -- mandatory for the gemma2 cells, whose forward pass
# is wrong under the L2A venv's TL 3.2.1.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd $ABS; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}

# model task eval-batch -- the 11 cells of the paper tables (arithmetic_addition is attributed
# but is not a column, so it is not evaluated here). Batch sizes mirror the validation runs.
CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 10" "llama3 ioi 1"
 "llama3 arithmetic_subtraction 1"
 "qwen2.5 mcqa 10" "gemma2 mcqa 10" "llama3 mcqa 1"
 "gemma2 arc_easy 1" "llama3 arc_easy 1" "llama3 arc_challenge 1"
)
# tag  method-name-for-run_evaluation  circuit-dir  eval-out-dir
METHODS=(
 "gim GIM gim gim_eval"
 "relpqk RelP-qkgrad relp_qkgrad relp_qkgrad_eval"
 "relp RelP relp relp_eval"
)
# NOTE the default deliberately does NOT include relp. Both other arms are complete at 11/11,
# so a bare `bash run_gim_relpqk_test.sh` must stay a no-op-sized job rather than silently
# recomputing ~12 llama3 cells at up to 10h each -- the exact waste the ARMS filter was added
# to prevent, described at the top of this file. Run the new arm explicitly:
#   ARMS=relp bash run_gim_relpqk_test.sh
ARMS=${ARMS:-"gim relpqk"}

n=0
for m in "${METHODS[@]}"; do
  read -r tag method cdir odir <<< "$m"
  case " $ARMS " in *" $tag "*) ;; *) continue ;; esac
  for cell in "${CELLS[@]}"; do
    read -r model task ebatch <<< "$cell"
    if [ ! -d "$ABS/results/$cdir" ]; then
      echo "SKIP $tag: no circuit dir at $ABS/results/$cdir"; continue
    fi
    if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00
    elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00
    else mem=32G; tlim=02:00:00; fi
    name="t-${tag}-${task}-${model}"
    cmd="$pp; \
$PY run_evaluation.py --models $model --tasks $task --method $method --level node \
--ablation patching --split test --batch-size $ebatch \
--circuit-dir results/$cdir --output-dir $L2A/results/$odir"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name: $cmd"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n test-set eval jobs =="
