#!/bin/bash
# LR sweep for id-STE (hard_topk_identity) + Adam, uniform-k, node-level, train->val.
# Mirrors the paper's LR sweep (submit_htk_lr.sh -> tabs/lr_sweep.tex): uniform-k,
# --include-input, 500 steps, LRs {0.005,0.01,0.05,0.1,0.3}. All 11 pairs at LR=0.01;
# all pairs EXCEPT llama3/ioi for the other LRs (matches the paper's --- for that cell).
# Purpose: pick the right LR for id-STE+Adam before filling the full node grid, since the
# paper notes hard-backward variants need a higher LR than 0.01 with Adam.
set -u
cd ~/learning-to-attribute

ALL=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)

for lr in 0.005 0.01 0.05 0.1 0.3; do
  for p in "${ALL[@]}"; do
    read -r model task <<< "$p"
    # skip the expensive llama3/ioi cell except at the main LR=0.01 (matches paper coverage)
    if [ "$model" = "llama3" ] && [ "$task" = "ioi" ] && [ "$lr" != "0.01" ]; then continue; fi
    case $model in
      gpt2|qwen2.5) RES="--cpus-per-task=2 --mem=32G --time=4:00:00";  bs=20;;
      gemma2)       RES="--cpus-per-task=3 --mem=64G --time=8:00:00";  bs=4;;
      llama3)       RES="--cpus-per-task=4 --mem=96G --time=12:00:00"; bs=2;;
    esac
    # mib_node_seed.sbatch: <model> <task> <masking> <ks> <outdir> <opt> <seed> <bs> <lr>
    sbatch $RES -J iadamlr_${lr}_${model}_${task} mib_node_seed.sbatch \
      "$model" "$task" hard_topk_identity uniform "ident_adam_lr_$lr" adam 42 "$bs" "$lr"
    sleep 1
  done
done
echo "id-STE+Adam LR SWEEP SUBMITTED (LR=0.01: 11 pairs; others: 10 pairs each = 51 jobs)"
