#!/bin/bash
# "Stepless IG": EAP-IG-inputs-mc at --ig-steps 1, node level, across the 12 MIB paper cells.
#
# THE CONTRAST. run_variants.sh already runs the arm this is matched against:
#     ig1  = EAP-IG-inputs --ig-steps 1   -> results/ig1_eval        "input x grad / one-step IG"
# One forward+backward per batch, with the input embedding interpolated at alpha = 1/1 = 1, i.e.
# the clean input. That is IxG. This wave runs the SAME budget -- one forward+backward per batch --
# with alpha ~ U(0,1) drawn per example instead. Identical compute, identical data, identical
# metric; the only difference in the whole pipeline is where alpha is placed. In expectation the
# mc arm is IG at m -> infinity, so if it beats ig1 the gain is free, and the two 5x/10x-cost
# reference arms are already on disk to say how much of IG's advantage it recovers:
#
#     results/ig1_eval        m=1 grid  (alpha=1)          <- COMPUTE-MATCHED CONTROL
#     results/napig_mc_eval   m=1 mc    (alpha~U(0,1))     <- this script
#     results/napig_ref_eval  m=5 grid                     run_variants.sh `ref`
#     results/napig10_eval    m=10 grid                    run_napig10.sh
#     results/napig30_eval    m=30 grid                    run_napig30.sh
#
# SEEDS ARE NOT OPTIONAL AND THEY ARE NOT UNIFORM. A one-sample-per-example MC estimator has real
# variance, so "mc beats ig1 by 0.2 CPR AUC" is meaningless without knowing what two seeds of mc
# differ by. But replicating all 12 cells triples the wave for a number that only needs measuring
# once, and evaluation -- not attribution -- is what costs here. So: seed 0 everywhere, plus seeds
# 1 and 2 on the three CHEAPEST cells only. Those three give the noise floor; the other nine are
# read against it. If the floor turns out to be large, come back and replicate more.
#
# WHY VARIANCE IS SMALLER THAN IT LOOKS: alpha is drawn PER EXAMPLE inside the batch (see
# get_scores_eap_ig_mc's docstring), not per batch, and the score sums over examples before
# anything else. So a 1000-example cell averages 1000 independent alphas, not 50, and the
# batch-size-1 llama3 cells are not penalised relative to the batch-size-20 gpt2 one.
#
# Everything else -- cell list, sizing, venv, flags, --head 200 on llama3 eval -- mirrors
# run_napig10.sh / run_variants.sh exactly, so the pkls drop straight into the same comparisons.
#
# DRYRUN=1 to preview.  SEEDS="0 1 2" / CHEAP_SEEDS="" / ONLY=gpt2 to narrow.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"
cd $ABS
PY=$ABS/.venv/bin/python
export_pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-}
# ${SEEDS-...} not ${SEEDS:-...}: the colon form treats an explicitly empty SEEDS as unset and
# silently restores "0", which makes "top up the replicates without resubmitting the base run"
# -- SEEDS= CHEAP_SEEDS="1 2" -- impossible to express, and quietly duplicates finished jobs.
SEEDS=${SEEDS-"0"}                  # run on every cell
CHEAP_SEEDS=${CHEAP_SEEDS:-"1 2"}   # extra replicates, cheap cells only -> the noise floor
# gemma2/arc_easy is NOT here to be cheap -- it is the cheapest cell with num_examples=100.
# The other three all attribute over 1000 examples, and since alpha is drawn per example the MC
# error scales like 1/sqrt(n_examples): a floor measured only on 1000-example cells understates
# the floor on the 100-example ones (arc/arithmetic, gemma2 and llama3) by about sqrt(10) = 3.2x.
# Reading those nine cells against a 1000-example floor would call noise a result. One 100-example
# replicate pins the other end of that scaling; the remaining cells interpolate between the two.
# Overridable so a cell that turns out anomalous can be replicated without editing this list:
#   CHEAP="llama3/ioi" SEEDS="" CHEAP_SEEDS="1 2" ONLY=llama3 bash run_napig_mc.sh
CHEAP=${CHEAP:-"gpt2/ioi qwen2.5/ioi qwen2.5/mcqa gemma2/arc_easy"}

# cell: model task num_examples attr_batch eval_head(0=full)   [identical to run_napig10.sh]
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

n=0
for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then continue; fi
  if [ "$model" = "llama3" ]; then mem=96G; tlim=10:00:00; ebatch=1
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=05:00:00; ebatch=$abatch
  else mem=32G; tlim=02:00:00; ebatch=$abatch; fi
  if [ "$nex" = "full" ]; then nex_flag=""; else nex_flag="--num-examples $nex"; fi
  if [ "$ehead" = "0" ]; then head_flag=""; else head_flag="--head $ehead"; fi

  seeds="$SEEDS"
  case " $CHEAP " in *" $model/$task "*) seeds="$SEEDS $CHEAP_SEEDS" ;; esac

  for s in $seeds; do
    # SEED IN THE DIR NAME IS LOAD-BEARING. run_attribution.py's save path is
    # "<method>_<ablation>_<level>/<task>_<model>" -- it encodes neither the seed nor ig-steps, so
    # two seeds written to one --circuit-dir would overwrite each other and the replicate that
    # exists to measure MC noise would silently become a single run. Seed 0 keeps the bare name so
    # it reads as the headline arm.
    if [ "$s" = "0" ]; then cdir=napig_mc; odir=napig_mc_eval
    else cdir=napig_mc_s$s; odir=napig_mc_s${s}_eval; fi
    name="mc-s${s}-${task}-${model}"
    cmd="$export_pp; \
$PY run_attribution.py --models $model --tasks $task --method EAP-IG-inputs-mc --ig-steps 1 --mc-seed $s --level node --ablation patching --split train --batch-size $abatch $nex_flag --circuit-dir results/$cdir && \
$PY run_evaluation.py --models $model --tasks $task --method EAP-IG-inputs-mc --level node --ablation patching --split validation --batch-size $ebatch $head_flag --circuit-dir results/$cdir --output-dir results/$odir"
    if [ "$DRYRUN" = "1" ]; then
      echo "[DRY] $name | mem=$mem t=$tlim | nex=$nex ehead=$ehead -> $odir"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" \
        --wrap="$cmd" >/dev/null && echo "submitted $name"
    fi
    n=$((n+1))
  done
done
# ${DRYRUN:+...} would expand for DRYRUN=0 too (it is set, just false), printing "DRY" on a real
# submission. Test the value, not whether the variable exists.
if [ "$DRYRUN" = "1" ]; then echo "== DRY total $n stepless-IG jobs =="
else echo "== submitted $n stepless-IG jobs =="; fi

# THE ACCOUNT IS CAPPED AT 8 CONCURRENT GPUs and the edge LR sweep (submit_mib_edge_lr_sweep.sh)
# is holding all of them, so this wave queues behind it -- SLURM runs this QOS roughly FIFO.
# ONLY=gpt2 / ONLY=qwen2.5 first if you want the noise floor before the 8B cells.
