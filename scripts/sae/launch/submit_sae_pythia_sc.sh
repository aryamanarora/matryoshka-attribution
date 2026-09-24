#!/bin/bash
# Cheapest SAEBench cell for the sigmoid top-k SAE: Pythia-160M-deduped, resid_post layer 8, 4096
# latents, 500M Pile tokens (SAEBench's baseline recipe; scripts/sae/train_sae.py lists the
# deliberate changes), then SAEBench core at L0 = 20..640 against the released TopK and
# Matryoshka BatchTopK SAEs of the same cell. Both stages run in the `sae` dependency group's own
# env (.venv-sae), synced from uv.lock by the job. sc / nlprun, run INSIDE tmux.
#   STAGE=train bash scripts/sae/launch/submit_sae_pythia_sc.sh
#   STAGE=eval  bash scripts/sae/launch/submit_sae_pythia_sc.sh      # after training finished
#   SMOKE=1 STAGE=train bash ...  then  SMOKE=1 STAGE=eval bash ...  # 2M tokens, 1 k, 1 baseline
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
STAGE=${STAGE:-train}; SMOKE=${SMOKE:-0}; DRYRUN=${DRYRUN:-0}
PY="UV_PROJECT_ENVIRONMENT=$ABS/.venv-sae uv run --no-default-groups --group sae python"
RES="-q jag -d a6000 -c 4 -r 32G"
TAG=pythia160m_l8_4k_sigtopk_uniform
if [ "$SMOKE" = 1 ]; then
  TAG=smoke_$TAG
  TRAIN_EXTRA="--tokens 2048000 --warmup-steps 100 --log-every 10 --diag-every 100 --ckpt-every 500"
  EVAL_EXTRA="--ks 20 --baselines TopK --n-trainers 1"
else
  TRAIN_EXTRA=""; EVAL_EXTRA=""
fi
OUT=results/sae/$TAG
case $STAGE in
  train) name="sae-train-$TAG"; cmd="$PY scripts/sae/train_sae.py --out $OUT $TRAIN_EXTRA" ;;
  eval)  name="sae-core-$TAG";  cmd="$PY scripts/sae/eval_core.py --sae-dir $OUT $EVAL_EXTRA" ;;
  *) echo "STAGE must be train or eval"; exit 1 ;;
esac
if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $RES -n $name"; echo "    $cmd"; else
  nlprun -g 1 $RES -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"
fi
