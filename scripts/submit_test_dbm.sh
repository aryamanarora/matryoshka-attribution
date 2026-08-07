#!/bin/bash
# TEST-set evals for the DBM baseline, for ONE (lr, L1) config.
#
#   bash scripts/submit_test_dbm.sh 0.3 6.0     # results/eprun_node_ld_sig_lr0.3_l16.0
#   bash scripts/submit_test_dbm.sh 0.3         # unpenalised lr=0.3
#   DRYRUN=1 bash scripts/submit_test_dbm.sh 0.3 6.0
#
# This is submit_test_node_pruning.sh for the sigmoid gate. That script builds its dir suffix
# from (sparsity, loss) only, so it cannot name a _sig/_lr/_l1 dir at all -- hence a separate
# script rather than another env var on that one.
#
# WHICH CONFIG: lr=0.3, L1=6.0 is the best DBM setting on validation -- best Avg CPR (1.50 vs
# 1.31 unpenalised) and second-best acc-AUC, and both sweeps peak in the interior (lr 0.3 of
# {0.001..1.0}, lambda 6.0 of {0..20}), so neither is an edge-of-grid maximum. Note the L1 path
# also drops density 0.56 -> 0.30, so its CPR gain is confounded with the sparsity change; the
# tuned setting is still the right one to put in the test table, but "L1 makes DBM better" is
# not something this sweep alone establishes.
#
# EVAL_ONLY=1 is the point: the mask was trained on the TRAIN split (--train-split train,
# independent of --split), so the graph already on disk from the validation pass is the same
# circuit we owe the test set. Retraining would waste GPU hours AND produce a different
# circuit, quietly breaking the "same circuit, two splits" claim. Nothing here touches
# graph_*.json.
#
# The sbatch handles the two things that are easy to get wrong, and this script overrides
# neither: it runs everything under the MIB venv (TL 2.15.4, so the gemma2 cells dodge the
# Gemma-2 forward bug), and it drops the llama3 --head 200 cap on non-validation splits
# ([ "$SPLIT" = validation ] || HEAD=""), because the MIB paper's test numbers and
# submit_test_lr05.sh are both full-split.
#
# 11 cells, one job each -> results/eprun_eval_ld_sig_lr<LR>[_l1<L1>]/
#   EdgePruning_patching_node/<task-dash>_<model>_test_abs-False.pkl
set -u
L2A=/home/guests/aryaman/learning-to-attribute
cd "$L2A"
DRYRUN=${DRYRUN:-0}
LR=${1:-0.3}
L1=${2:-}

# Mirror run_edge_pruning.sbatch's own SUF construction rather than reimplementing it: the
# sigmoid gate always carries _ld_sig, then _lr<x>, then _l1<x>.
SUF="_ld_sig_lr${LR}"
[ -n "$L1" ] && SUF="${SUF}_l1${L1}"
echo "== config: gate=sigmoid loss=logit_diff lr=$LR l1=${L1:-0} -> results/eprun_node${SUF} =="

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

n=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  graph="$L2A/results/eprun_node${SUF}/graph_${task}_${model}.json"
  if [ ! -f "$graph" ]; then
    echo "SKIP $task/$model: no trained graph at $graph"
    continue
  fi
  name="dbm-test${SUF}-${task}-${model}"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY $name"
  else
    # `env`, not a bare assignment prefix: ${L1:+L1=$L1} is expanded AFTER the shell has
    # decided which leading words are assignments, so as a prefix it is parsed as a command
    # ("L1=6.0: command not found") and the sbatch never runs. As an argument to env it is
    # just another word, so the optional variable can stay optional without duplicating the
    # whole submit line for the penalised and unpenalised cases.
    env EVAL_ONLY=1 GATE=sigmoid LOSS=logit_diff LR="$LR" ${L1:+L1="$L1"} \
      sbatch --job-name="$name" \
      scripts/run_edge_pruning.sbatch "$model" "$task" node 3000 test >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n DBM test-set eval jobs =="
