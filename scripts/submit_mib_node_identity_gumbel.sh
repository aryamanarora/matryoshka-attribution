#!/bin/bash
# Node-level id-STE + Gumbel-forward + SGD sweep, train->val, both k-schedules.
# Reproduces jobs from 2026-06-30. Run from repo root on tilde (SLURM).
#
#   masking  = hard_topk_identity_gumbel  (Gumbel(0,1)-perturbed hard top-k selection in
#              the forward, identity STE backward dm/ds=1 on the clean scores)
#   optimizer= sgd
#   schedules= uniform + log  -> results/mib_node_identity_gumbel_sgd_{uniform,log}
#
# Why Gumbel: deterministic uniform-k SGD buries the always-on upstream node (m0 / input
# embedding) because it never leaves the kept set to receive a "keep me" gradient, which
# pins CPR at the 0.25 floor (ioi/qwen, arithmetic/llama). Stochastic selection lets it
# flip out and recover. NOTE: it is a targeted rescue, not a general win — it degrades the
# already-healthy cells (esp. llama3). See memory uniform-k-sgd-starves-upstream-node.
set -u
cd ~/learning-to-attribute

# Standard 11 node pairs.
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "qwen2.5 mcqa"
  "gemma2 ioi" "gemma2 mcqa" "gemma2 arc_easy"
  "llama3 ioi" "llama3 mcqa" "llama3 arc_easy" "llama3 arc_challenge" "llama3 arithmetic_subtraction"
)

for SCHED in uniform log; do
  case $SCHED in
    uniform) tag=uni; out=mib_node_identity_gumbel_sgd_uniform;;
    log)     tag=log; out=mib_node_identity_gumbel_sgd_log;;
  esac
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) RES="--cpus-per-task=2 --mem=32G --time=4:00:00";  bs=20;;
      gemma2)       RES="--cpus-per-task=3 --mem=64G --time=8:00:00";  bs=4;;
      llama3)       RES="--cpus-per-task=4 --mem=96G --time=12:00:00"; bs=2;;  # ioi eval is slow
    esac
    # mib_node_seed.sbatch args: <model> <task> <masking> <kschedule> <outdir> <opt> <seed> <bs>
    sbatch $RES -J idgumb_${tag}_${model}_${task} mib_node_seed.sbatch \
      "$model" "$task" hard_topk_identity_gumbel "$SCHED" "$out" sgd 42 "$bs"
    sleep 1
  done
done
echo "ALL NODE id-STE+Gumbel+SGD JOBS SUBMITTED (11 pairs x 2 schedules = 22)"
