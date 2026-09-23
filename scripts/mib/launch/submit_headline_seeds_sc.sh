#!/bin/bash
# Seed replicates of the HEADLINE MAttr node-level runs on the MIB TEST split (2026-09-21):
# results/test_node_topk_uniform_lr05 is seed 42 (eval_mib.py's default); this submits the same
# 12 cells at seeds 43 and 44 into results/test_node_topk_uniform_lr05_s<seed>, so the paper can
# report seed variance for the main result rather than a single run. Every flag other than
# --seed and --output is the headline's (submit_softuni_lr05.sh, node branch): 500 steps, uniform
# k, soft top-k, Adam lr 0.05 (eps default 1e-8), sufficient, train on train, --include-input,
# full test split (no llama cap: test splits are <= 1188 examples).
#
# gemma2 trains AND evaluates in the TL 2.15.4 stack (CLAUDE.md: the L2A venv's Gemma-2 forward
# is wrong), i.e. the `tl2` dependency group in its own env, exactly as submit_mib_zero_sc.sh.
#
# sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_headline_seeds_sc.sh
#   SEEDS="43" TASKS="ioi" DRYRUN=1 bash scripts/mib/launch/submit_headline_seeds_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}
SEEDS=${SEEDS:-"43 44"}
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}
MODELS=${MODELS:-"gpt2 qwen2.5 gemma2 llama3"}
# The two ARC/llama3 cells OOM a 48 GB a6000 at TEST-split evaluation (an 11 GB allocation in
# the full-split eval; all four seed-43/44 attempts died there on 2026-09-21), so they default
# to the 141 GB h200s. BIG="" puts everything on the a6000 slot.
BIG=${BIG-"-q sphinx -d h200 -r 128G"}; BIG_TASKS=${BIG_TASKS-"arc_easy arc_challenge"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY_L2A="uv run python"
PY_G="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
CELLS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_addition" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0; skip=0
for seed in $SEEDS; do
  out=results/test_node_topk_uniform_lr05_s${seed}
  for cell in "${CELLS[@]}"; do
    read -r model task <<< "$cell"
    case " $MODELS " in *" $model "*) ;; *) continue ;; esac
    case " $TASKS " in *" $task "*) ;; *) continue ;; esac
    if [ -f "$out/${task}_${model}_test.pkl" ]; then
      echo "SKIP $out/${task}_${model}: already evaluated"; skip=$((skip+1)); continue
    fi
    py=$PY_L2A; bs=""
    case $model in
      gpt2|qwen2.5) res="-q jag -d a6000 -c 2 -r 32G" ;;
      gemma2)       res="-q jag -d a6000 -c 3 -r 64G"; py=$PY_G; bs="--batch-size 4" ;;
      llama3)       res="-q jag -d a6000 -c 4 -r 96G"; bs="--batch-size 2" ;;
    esac
    [ -n "$BIG" ] && [ "$model" = llama3 ] && case " $BIG_TASKS " in *" $task "*) res="$BIG" ;; esac
    name="hseed${seed}-${task}-${model}"
    cmd="$EXP $py scripts/mib/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform \
--masking topk --optimizer adam --mode sufficient --lr 0.05 --split test --train-split train \
--include-input --seed $seed $bs --output $out"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
[ "$DRYRUN" = 1 ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n headline seed-replicate jobs, $skip skipped =="
