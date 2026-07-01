#!/bin/bash
# Edge-level id-STE + SGD sweep (deterministic hard_topk_identity), train->val, both
# k-schedules. Reproduces jobs from 2026-07-01. Run from repo root on tilde (SLURM).
#
#   masking  = hard_topk_identity   (hard 0/1 forward, identity STE backward dm/ds=1)
#   optimizer= sgd
#   schedules= uniform + log   -> results/mib_edge_identity_sgd_{uniform,log}
#
# Eval-set convention (matches the mib_edge_hard_topk baseline exactly, see
# submit_mib_edge_ablations.sh vs submit_sphinx_llama_edge.sh):
#   - gpt2/qwen2.5/gemma2 : --eval-examples 0  (ALL val examples), batch-size 5
#   - llama3              : --eval-examples 200 (subset),           batch-size 2
# Getting this wrong makes llama3 non-comparable to the baseline table AND runs ~forever.
set -u
cd ~/learning-to-attribute

# Standard 9 edge pairs (no llama3 arc_* at edge level).
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy"
)

for SCHED in uniform log; do
  case $SCHED in
    uniform) tag=uni; out=mib_edge_identity_sgd_uniform;;
    log)     tag=log; out=mib_edge_identity_sgd_log;;
  esac
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case $model in
      gpt2|qwen2.5) RES="--cpus-per-task=3 --mem=64G  --time=12:00:00"; bs=5; eval=0;;
      gemma2)       RES="--cpus-per-task=4 --mem=96G  --time=18:00:00"; bs=5; eval=0;;
      llama3)       RES="--cpus-per-task=5 --mem=128G --time=24:00:00"; bs=2; eval=200;;
    esac
    # mib_edge.sbatch args: <model> <task> <masking> <kschedule> <outdir> <opt> <seed> <bs> <eval-examples>
    sbatch $RES -J eidsgd_${tag}_${model}_${task} mib_edge.sbatch \
      "$model" "$task" hard_topk_identity "$SCHED" "$out" sgd 42 "$bs" "$eval"
    sleep 1
  done
done
echo "ALL EDGE id-STE+SGD JOBS SUBMITTED (9 pairs x 2 schedules = 18)"
