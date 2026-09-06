#!/bin/bash
# LR bracket for the global-KL MAttr arms.
#
# WHY: round 1 ran SGD at lr=1.0 and Adam at lr=0.05, both IMPORTED from the MIB node-level
# sweeps, where the objective is a logit diff. This objective is a KL averaged over 128
# positions -- a different gradient scale entirely -- and the round-1 gap (Adam KL-AUC 7.00 vs
# SGD 9.66) is therefore a comparison of two arbitrary points, not of two optimizers. Adam is
# scale-free and lands anywhere in a wide LR band; zero-init SGD is not. Bracket both before
# writing a single sentence about which optimizer wins here.
#
#   bash scripts/global_kl/launch/submit_global_kl_lr.sh          # submit
#   DRY=1 bash scripts/global_kl/launch/submit_global_kl_lr.sh    # print
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

DATA=${DATA:-data/fineweb_edu_200.jsonl}
STEPS=${STEPS:-1000}
OUT=${OUT:-results/global_kl_lr}
SGD_LRS=${SGD_LRS:-"0.1 1.0 10.0 100.0"}
ADAM_LRS=${ADAM_LRS:-"0.01 0.05 0.2"}

COMMON=(--model llama3 --data "$DATA" --n-train 100 --n-eval 50 --seq-len 128
        --train-batch-size 4 --eval-batch-size 4 --output "$OUT" --reuse-scores
        --method mattr --steps "$STEPS")

for lr in $SGD_LRS; do
  cmd=(sbatch -J "gklr_sgd$lr" scripts/global_kl/launch/global_kl.sbatch "${COMMON[@]}" --optimizer sgd --lr "$lr")
  [[ -n ${DRY:-} ]] && echo "${cmd[*]}" || "${cmd[@]}"
done
for lr in $ADAM_LRS; do
  cmd=(sbatch -J "gklr_adam$lr" scripts/global_kl/launch/global_kl.sbatch "${COMMON[@]}" --optimizer adam --lr "$lr")
  [[ -n ${DRY:-} ]] && echo "${cmd[*]}" || "${cmd[@]}"
done
