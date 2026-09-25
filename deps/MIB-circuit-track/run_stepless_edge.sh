#!/bin/bash
# "Stepless" IG at EDGE level: alpha ~ U(0,1) per example at m=1, against the m=5 grid already
# on disk. The edge twin of run_napig_mc.sh, which did this at node level.
#
#   bash run_stepless_edge.sh                 # 12 jobs, submitted HELD (see HOLD below)
#   DRYRUN=1 bash run_stepless_edge.sh        # preview
#   HOLD=0 bash run_stepless_edge.sh          # submit runnable
#   ARMS="mc ig1" bash run_stepless_edge.sh   # add the compute-matched grid control
#   SEED=1 bash run_stepless_edge.sh          # a replicate; writes to results/eapig_mc_s1
#
# WHAT THE INTERESTING NUMBER IS. The edge row of every MIB table is EAP-IG-inputs at m=5, and
# Hanna et al. defend m=5 as sufficient AT EDGE LEVEL specifically (COLM'24 App. C). So the
# question this wave asks is not "does more integration help" -- it is whether ONE Monte-Carlo
# draw matches that defended 5-step grid at a fifth of the cost. results/eapig_clean_eval (m=5)
# and results/eapig_clean10_eval (m=10) are the comparands and are already scored, so the mc arm
# alone answers it; nothing here re-runs them.
#
# THE m=1 CONTROL IS A DIFFERENT CONTROL HERE THAN AT NODE LEVEL, which is why it is not in the
# default ARMS. attribute.get_scores_eap_ig runs alpha = 0, 1/m, ..., (m-1)/m (LEFT endpoint),
# while attribute_node's grid is 1/m, ..., 1 (RIGHT endpoint). So node `ig1` is the gradient at
# the CLEAN input, i.e. input x gradient -- a method with a name -- whereas edge `ig1` is the
# gradient at the CORRUPTED input, which is not one. Both are one backward pass and either is a
# fair compute match, but do not report an edge mc-vs-ig1 gap as though it were the node one.
# See the docstring on get_scores_eap_ig_mc in EAP-IG/src/eap/attribute.py.
#
# ATTRIBUTION + VALIDATION EVAL, one SLURM job per cell, exactly like run_eapig_edge.sh -- the
# test split is a separate eval-only wave over the SAME circuits, once these numbers say whether
# there is anything to carry over. Re-attributing for the test split would draw a different alpha
# stream and silently produce a different circuit; run_stepless_test.sh says the same thing.
#
# CELLS and resource sizing are copied verbatim from run_eapig_edge.sh, which is the comparand:
# a different --num-examples or a different llama3 --head would make the mc-vs-m5 gap partly an
# eval-set difference. Edge graphs are far heavier than node, so the node scripts' sizing (and
# run_napig_mc.sh's) is not enough -- llama3 edge OOM'd at 96G and needs 128G / 24h.
#
# Runs in THIS repo's .venv (TL 2.15.4) -- mandatory for the gemma2 cells, whose forward pass is
# wrong under the L2A venv's TL 3.2.1.
#
# HELD BY DEFAULT. The cluster is reserved for other work; these jobs sit in JobHeldUser until
#   scontrol release $(squeue -u $USER -h -t PD -o "%i %j" | awk '$2 ~ /^mcedge-/{print $1}')
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"
cd $ABS
PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
HOLD=${HOLD:-1}
SEED=${SEED:-0}
ARMS=${ARMS:-"mc"}
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# Seed 0 is the headline run and keeps the bare dir name; replicates are suffixed, matching the
# node-level napig_mc / napig_mc_s1 / napig_mc_s2 convention.
[ "$SEED" = "0" ] && sfx="" || sfx="_s$SEED"

# arm: key method ig-steps circuit-dir output-dir
ARM_SPECS=(
 "mc   EAP-IG-inputs-mc  1  results/eapig_mc$sfx   results/eapig_mc${sfx}_eval"
 "ig1  EAP-IG-inputs     1  results/eapig_ig1      results/eapig_ig1_eval"
)

# cell: model task num_examples attr_batch eval_head(0=full)   -- verbatim from run_eapig_edge.sh
CELLS=(
  "gpt2 ioi 1000 20 0"
  "qwen2.5 ioi 1000 10 0"
  "qwen2.5 mcqa full 10 0"
  "gemma2 ioi 1000 10 0"
  "gemma2 mcqa full 10 0"
  "gemma2 arc_easy 100 1 0"
  "llama3 ioi 1000 1 200"
  "llama3 mcqa full 1 200"
  "llama3 arithmetic_addition 100 1 200"
  "llama3 arithmetic_subtraction 100 1 200"
  "llama3 arc_easy 100 1 200"
  "llama3 arc_challenge 100 1 200"
)

n=0; skip=0
for spec in "${ARM_SPECS[@]}"; do
  read -r key method msteps CDIR OUT <<< "$spec"
  case " $ARMS " in *" $key "*) ;; *) continue ;; esac
  # The seed only means anything for the mc arm; the grid arm is deterministic, so it keeps one
  # dir across seeds rather than writing identical circuits to eapig_ig1_s1, _s2, ...
  for cell in "${CELLS[@]}"; do
    read -r model task nex abatch ehead <<< "$cell"
    tdash=${task//_/-}
    if [ -f "$OUT/${method}_patching_edge/${tdash}_${model}_validation_abs-False.pkl" ]; then
      echo "SKIP $key $task/$model: already scored"; skip=$((skip+1)); continue
    fi
    case $model in
      llama3)  cpus=5; mem=128G; tlim=24:00:00; ebatch=1 ;;
      gemma2)  cpus=4; mem=96G;  tlim=16:00:00; ebatch=1 ;;
      qwen2.5) cpus=4; mem=64G;  tlim=12:00:00; ebatch=5 ;;
      *)       cpus=3; mem=48G;  tlim=12:00:00; ebatch=10 ;;   # gpt2
    esac
    [ "$nex"   = "full" ] && nex_flag=""  || nex_flag="--num-examples $nex"
    [ "$ehead" = "0"    ] && head_flag="" || head_flag="--head $ehead"

    name="${key}edge${sfx}-${task}-${model}"
    cmd="$pp; \
$PY run_attribution.py --models $model --tasks $task --method $method --ig-steps $msteps \
--mc-seed $SEED --level edge --ablation patching --split train --batch-size $abatch $nex_flag \
--circuit-dir $CDIR && \
$PY run_evaluation.py --models $model --tasks $task --method $method \
--level edge --ablation patching --split validation --batch-size $ebatch $head_flag \
--circuit-dir $CDIR --output-dir $OUT"
    if [ "$DRYRUN" = "1" ]; then
      echo "[DRY] $name | mem=$mem cpus=$cpus t=$tlim | nex=$nex abatch=$abatch ehead=$ehead -> $OUT"
    else
      mkdir -p "$ABS/logs"
      [ "$HOLD" = "1" ] && hold_flag="--hold" || hold_flag=""
      sbatch $hold_flag --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem \
        --time=$tlim --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
        && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
[ "$HOLD" = "1" ] && held=" (HELD)" || held=""
echo "== ${pfx}total $n edge stepless-IG jobs${held}, $skip skipped =="
