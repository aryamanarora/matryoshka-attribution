#!/bin/bash
# Global (task-free) circuit on Llama-3.1-8B: zero-ablate MLP neurons + attention heads on
# FineWeb-Edu and rank them by KL to the unintervened next-token distribution.
#
#   bash scripts/submit_global_kl.sh          # submit all arms
#   DRY=1 bash scripts/submit_global_kl.sh    # print instead
#
# Four arms, one job each (each loads the 8B model once):
#   mattr + sgd   (lr 1.0)   -- the CLAUDE.md headline recipe's optimizer/LR
#   mattr + adam  (lr 0.05)  -- the "$+$ Adam" ablation's optimizer/LR
#   mc_ig m=1                -- Stepless IG, compute-matched to ONE forward+backward per doc
#   random                   -- ranking-free control (the sweep also reports its own random
#                               ordering, so this is a second seed of the same control)
#
# Both LRs are IMPORTED from the MIB node-level sweeps; nothing has been tuned for the KL
# objective, whose gradient scale is not the logit-diff scale. Read the first round as
# "does the recipe transfer at all", and sweep --lr before comparing SGD to Adam as optimizers.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

DATA=${DATA:-data/fineweb_edu_200.jsonl}
STEPS=${STEPS:-1000}
NTRAIN=${NTRAIN:-100}
NEVAL=${NEVAL:-50}
SEQ=${SEQ:-128}
BS=${BS:-4}
OUT=${OUT:-results/global_kl}

# --reuse-scores is passed to every arm: resubmitting after a preempted job, or after
# changing only the sweep (e.g. --sparsity-grid), reuses the learned scores on disk instead of
# retraining. Delete "$OUT"/*_scores.pt to force a real re-run.
COMMON=(--model llama3 --data "$DATA" --n-train "$NTRAIN" --n-eval "$NEVAL"
        --seq-len "$SEQ" --train-batch-size "$BS" --eval-batch-size "$BS" --output "$OUT")

submit () {   # submit <jobname> <args...>
  local name=$1; shift
  if [[ -n ${DRY:-} ]]; then
    echo "sbatch -J $name global_kl.sbatch ${COMMON[*]} $*"
  else
    sbatch -J "$name" global_kl.sbatch "${COMMON[@]}" --reuse-scores "$@"
  fi
}

submit gkl_sgd    --method mattr --optimizer sgd  --lr 1.0  --steps "$STEPS"
submit gkl_adam   --method mattr --optimizer adam --lr 0.05 --steps "$STEPS"
submit gkl_mcig   --method mc_ig --ig-steps 1
submit gkl_random --method random
