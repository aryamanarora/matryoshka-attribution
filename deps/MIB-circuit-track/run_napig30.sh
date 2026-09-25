#!/bin/bash
# NAP-IG (EAP-IG-inputs, node level) at --ig-steps 30 across the 12 MIB paper cells.
#
# Five is not MIB cutting a corner -- it is the EAP-IG paper's DELIBERATE, DEFENDED choice.
# Hanna et al. (COLM 2024) Appendix C is titled "EAP-IG Requires Few Steps": they sweep
# m in {2,3,5,7,10,15,20,30,50} and conclude "although our results suggest even m = 3 steps
# would suffice, we use m = 5, to leave a margin for error". They even pre-empt the objection --
# "integrated gradients typically requires a higher number of steps; Sundararajan et al. (2017)
# recommends between 20 and 300" -- and explain it away with two hypotheses, that approximation
# accuracy does not matter for EAP-IG's purposes, and that clean and corrupted activations sit
# close enough together that the path is short.
#
# So do NOT write "upstream defaults to 30". 30 is a vestigial function-signature default
# (get_scores_eap_ig(steps=30), hannamw 3a60a25, 2024-03-02); the string `ig_steps=30` appears
# NOWHERE in that repo's history, while its notebooks (since ed020fd, 2024-07-09) and README
# (ad8331e, 2025-02-24) all pass 5. MIB passes 5 explicitly, so the 30 never executes.
#
# What this ladder actually shows is a SCOPE LIMIT on Appendix C, not a contradiction of it.
# Their sweep was run on EDGES, on three GPT-2 tasks, scored by faithfulness at fixed edge
# counts. Our edge ladder reproduces them (10 steps changes nothing, if anything slightly
# worse). At NODE level across four models it fails: CPR-AUC 0.85 -> 1.31, acc-AUC 0.30 ->
# 0.46, and 5-step MLP rankings correlate rho +0.04 with their own converged selves -- closer
# to 1-step I*G (+0.67) than to 10-step IG. Their hypothesis (b) is the likely reason it
# splits that way: whatever makes the clean/corrupted path short for an edge score need not
# make it short for an MLP's summed node total.
#
# The 5-vs-10 control is what showed 5 to be under-resolved at node level: the top of the
# ranking is still moving, and on mcqa/qwen2.5 the largest-magnitude node (m0) flips SIGN
# between 5 and 10 steps. Since CPR-AUC is probed at 0.1-1% sparsity, that unstable top-5 is
# exactly what the metric is reading.
#
# The paper reports both this row and the 5-step row, so results/napig_ref* must stay intact.
#
# Cell list, venv, --num-examples, eval batch sizes and the llama3 --head 200 cap mirror
# run_variants.sh / run_napig10.sh exactly, so --ig-steps is the ONLY difference between the
# three rows and the pkls are directly comparable.
#
# WALL CLOCKS ARE RAISED, and deliberately do not mirror run_variants.sh. Attribution cost is
# linear in ig-steps while eval cost is unchanged (same circuit, same sparsity sweep), so only
# the attribution half triples vs the 10-step run. Measured at 10 steps: ioi/llama3 5.56 s/it
# x 1000 batches = ~93 min attribution, ioi/gemma2 ~13 min. At 30 steps that is ~4.6 h for
# ioi/llama3 alone, which does not fit the inherited 10 h limit once eval is added.
#   llama3 10:00:00 -> 16:00:00 ; gemma2 05:00:00 -> 08:00:00
# Cheap insurance: a wall-clock kill would discard hours of attribution and leave a partial dir.
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"
cd $ABS
PY=$ABS/.venv/bin/python
export_pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
DRYRUN=${DRYRUN:-0}

# cell: model task num_examples attr_batch eval_head(0=full)
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
# method: tag flag_method ig_steps circuit_dir output_dir
METHODS=(
  "ig30 EAP-IG-inputs 30 napig30 napig30_eval"
)

for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  # resources by model -- time limits raised for the 30-step attribution (see header)
  if [ "$model" = "llama3" ]; then mem=96G; tlim=16:00:00; ebatch=1
  elif [ "$model" = "gemma2" ]; then mem=64G; tlim=08:00:00; ebatch=$abatch
  else mem=32G; tlim=04:00:00; ebatch=$abatch; fi
  # flags that vary by cell
  if [ "$nex" = "full" ]; then nex_flag=""; else nex_flag="--num-examples $nex"; fi
  if [ "$ehead" = "0" ]; then head_flag=""; else head_flag="--head $ehead"; fi

  for m in "${METHODS[@]}"; do
    read -r tag method igs cdir odir <<< "$m"
    name="v-${tag}-${task}-${model}"
    cmd="$export_pp; \
$PY run_attribution.py --models $model --tasks $task --method $method --ig-steps $igs --level node --ablation patching --split train --batch-size $abatch $nex_flag --circuit-dir results/$cdir && \
$PY run_evaluation.py --models $model --tasks $task --method $method --level node --ablation patching --split validation --batch-size $ebatch $head_flag --circuit-dir results/$cdir --output-dir results/$odir"
    if [ "$DRYRUN" = "1" ]; then
      echo "[DRY] $name | mem=$mem t=$tlim | $method igs=$igs nex=$nex ehead=$ehead -> $odir"
    else
      sbatch --partition=main --gres=gpu:1 --cpus-per-task=4 --mem=$mem --time=$tlim \
        --job-name="$name" --output="$ABS/logs/${name}.out" \
        --wrap="$cmd" >/dev/null && echo "submitted $name"
    fi
  done
done
