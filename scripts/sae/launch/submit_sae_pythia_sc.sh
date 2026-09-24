#!/bin/bash
# The 2x2 of forward (hard top-k / sigmoid top-k) x k (fixed 20 / log-uniform on [1, 640]) on the
# cheapest SAEBench cell: Pythia-160M-deduped, resid_post layer 8, 4096 latents, 100M Pile tokens,
# MSE loss only (scripts/sae/train_sae.py). Then SAEBench core at L0 = 20..640, next to the
# released TopK and Matryoshka BatchTopK SAEs of the same cell.
#   A  hard, k=20      does our pipeline reproduce SAEBench's TopK k=20?
#   B  soft, k=20      the sigmoid top-k operator alone
#   C  hard, log k     random k alone
#   D  soft, log k     both
#   E  ste, log k      C's hard forward, sigmoid top-k backward (D's soft forward was a loose
#                      relaxation: latents rescale to absorb the mask)
# Both stages run in the `sae` dependency group's env (.venv-sae), synced from uv.lock by the job.
# sc / nlprun, run INSIDE tmux.
#   STAGE=train RUNS="A B C D" bash scripts/sae/launch/submit_sae_pythia_sc.sh
#   STAGE=eval  RUNS="A B C D" bash scripts/sae/launch/submit_sae_pythia_sc.sh   # after training
#   SMOKE=1 ... # 2M tokens; eval at k=20 against one TopK baseline
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
STAGE=${STAGE:-train}; RUNS=${RUNS:-"A B C D"}; SMOKE=${SMOKE:-0}; DRYRUN=${DRYRUN:-0}
PY="UV_PROJECT_ENVIRONMENT=$ABS/.venv-sae uv run --no-default-groups --group sae python"
RES="-q jag -d a6000 -c 4 -r 32G"
for run in $RUNS; do
  case $run in
    A) flags="--forward hard --k-schedule fixed --k 20";    tag=hard_k20 ;;
    B) flags="--forward soft --k-schedule fixed --k 20";    tag=soft_k20 ;;
    C) flags="--forward hard --k-schedule log --k-max 640"; tag=hard_logk640 ;;
    D) flags="--forward soft --k-schedule log --k-max 640"; tag=soft_logk640 ;;
    E) flags="--forward ste --k-schedule log --k-max 640";  tag=ste_logk640 ;;
    *) echo "unknown run $run"; exit 1 ;;
  esac
  TAG=pythia160m_l8_4k_${tag}_100M; extra=""
  if [ "$SMOKE" = 1 ]; then
    TAG=smoke_$TAG
    [ "$STAGE" = train ] && extra="--tokens 2048000 --warmup-steps 100 --log-every 10 --diag-every 100 --ckpt-every 500"
    [ "$STAGE" = eval ] && extra="--ks 20 --baselines TopK --n-trainers 1"
  fi
  OUT=results/sae/$TAG
  case $STAGE in
    train) name="sae-train-$TAG"; cmd="$PY scripts/sae/train_sae.py $flags --out $OUT $extra" ;;
    eval)  name="sae-core-$TAG";  cmd="$PY scripts/sae/eval_core.py --sae-dir $OUT $extra" ;;
    *) echo "STAGE must be train or eval"; exit 1 ;;
  esac
  if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $RES -n $name"; echo "    $cmd"; else
    nlprun -g 1 $RES -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
done
