#!/bin/bash
# TEST-set evals for the stepless-IG contrast: MC(m=1) against its compute-matched control
# IxG(m=1 grid), plus the m=5 and m=30 grid arms as cost markers.
#
#   bash run_stepless_test.sh                  # 44 jobs (4 arms x 11 cells)
#   DRYRUN=1 bash run_stepless_test.sh         # preview
#   ARMS="mc ig1" bash run_stepless_test.sh    # just the compute-matched pair
#
# EVAL ONLY -- run_attribution.py is deliberately absent, exactly as in run_attnlrp_test.sh.
# Every arm here attributed on the TRAIN split during the validation wave (run_napig_mc.sh,
# run_variants.sh, run_napig30.sh), so results/<cdir>/*/importances.json is already the circuit
# we owe the test set. Re-attributing would not merely cost GPU time: for the MC arm it would
# draw a DIFFERENT alpha stream and produce a different circuit, breaking the "same circuit,
# two splits" property that lets the validation and test numbers be read against each other --
# and it would do so silently, since the output filename does not encode the circuit.
#
# NO --head, matching run_attnlrp_test.sh / run_gim_relpqk_test.sh: the llama3 validation cap of
# 200 exists because 10k val crawls on 8B, but the test splits are <=1188 (ioi/arith 1000,
# arc-e 1188, arc-c 586, mcqa 50) and every other row of the test table is full-split.
#
# THIS IS WHY THE TEST NUMBERS MAY BE STEADIER THAN THE VALIDATION ONES. compare_stepless_ig.py
# measures a seed spread of 0.78 area_under on ioi/llama3 against 0.02-0.05 on the gpt2/qwen2.5/
# gemma2 cells; the llama3 cells are the ones scored on only 200 validation examples, so part of
# that spread is eval-set size rather than estimator variance. Full-split test eval separates
# the two -- if the llama3 spread collapses here, the validation floor was measurement noise.
# Do not merge test and validation numbers into one mean; they are different eval sizes.
#
# 11 cells, not 12: arithmetic_addition is attributed but is not a column of the paper tables.
# Output goes to the L2A results tree, which is what make_mib_test_table.py reads; validation
# results stay in this repo's results/. That split is the existing convention, not a choice.
#
# Runs in THIS repo's .venv (TL 2.15.4) -- mandatory for the gemma2 cells, whose forward pass is
# wrong under the L2A venv's TL 3.2.1.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"; cd $ABS; PY=$ABS/.venv/bin/python
L2A="$(cd "$ABS/../.." && pwd)"
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}

# model task eval-batch   (batches mirror the validation wave's per-cell values)
CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 10" "llama3 ioi 1"
 "llama3 arithmetic_subtraction 1"
 "qwen2.5 mcqa 10" "gemma2 mcqa 10" "llama3 mcqa 1"
 "gemma2 arc_easy 1" "llama3 arc_easy 1" "llama3 arc_challenge 1"
)

# arm: key circuit-dir method output-dir
# The method name decides the subfolder run_evaluation.py writes into, so the MC arm MUST keep
# --method EAP-IG-inputs-mc even though the evaluation itself is method-agnostic: with
# EAP-IG-inputs it would write into the grid arms' subfolder and collide with them.
ARM_SPECS=(
 "mc   napig_mc   EAP-IG-inputs-mc  napig_mc_test"
 "ig1  ig1        EAP-IG-inputs     ig1_test"
 "ref  napig_ref  EAP-IG-inputs     napig_ref_test"
 "m30  napig30    EAP-IG-inputs     napig30_test"
 # m=10, added 2026-08-24. Its absence from the test split was an OVERSIGHT, not a decision --
 # nothing in this file or in make_mib_test_table ever justified skipping it, and its 12
 # train-split circuits have sat in results/napig10 since the validation wave. The gap only
 # became visible when the figure and the table were put side by side: the scatter plots the
 # ladder as 5 -> 10 (10 is where the integral converges on validation, rho 0.994 against 30),
 # while the test table could only show 5 -> 30. Same method, different rungs, for no reason.
 "m10  napig10    EAP-IG-inputs     napig10_test"
)
# m10 stays OUT of the default: the other four arms are complete at 11/11, and re-running them
# would throw away ~24 llama3 cells at up to 10h each. Run it as ARMS=m10.
ARMS=${ARMS:-"mc ig1 ref m30"}

n=0; skipped=0
for spec in "${ARM_SPECS[@]}"; do
  read -r key cdir method odir <<< "$spec"
  case " $ARMS " in *" $key "*) ;; *) continue ;; esac
  for cell in "${CELLS[@]}"; do
    read -r model task ebatch <<< "$cell"
    cpath="$ABS/results/$cdir/${method}_patching_node/${task//_/-}_${model}/importances.json"
    if [ ! -f "$cpath" ]; then
      echo "SKIP $key ${task}-${model}: no circuit at $cpath"; skipped=$((skipped+1)); continue
    fi
    if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00
    elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00
    else mem=32G; tlim=02:00:00; fi
    name="t-${key}-${task}-${model}"
    cmd="$pp; \
$PY run_evaluation.py --models $model --tasks $task --method $method --level node \
--ablation patching --split test --batch-size $ebatch \
--circuit-dir results/$cdir --output-dir $L2A/results/$odir"
    if [ "$DRYRUN" = "1" ]; then
      echo "DRY $name (mem=$mem t=$tlim batch=$ebatch) -> $odir"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n stepless-IG test-set eval jobs ($skipped skipped for missing circuits) =="
