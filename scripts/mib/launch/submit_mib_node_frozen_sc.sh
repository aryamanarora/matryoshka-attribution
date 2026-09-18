#!/bin/bash
# MAttr WITHOUT LEARNING on the MIB node cells: the headline protocol (soft top-k, uniform k,
# 500 steps, --include-input) with --optimizer none -- scores stay 0, the mask is the same
# every step, and the attribution is the mean negated gradient (trainer.learn_scores: the
# first-step update of the gradient appendix, Monte-Carlo over k and data = centred
# path-reweighted IG in mask space). Same cost per step as learning. Proposed 2026-09-18 as the
# no-learning control for the ablation tables. lr is passed but unused (the dir keeps the
# headline's name pattern so the tables' lr column reads it off the saved args as usual).
#   mib_node_topk_uniform_frozen   (validation)   test_node_topk_uniform_frozen  (test)
# LOGK=1 runs the log-k twin into *_topk_log_frozen. gemma2 in the tl2 env, opt-in via MODELS.
# sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_mib_node_frozen_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_mib_node_frozen_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; SPLITS=${SPLITS:-"validation test"}; LOGK=${LOGK:-0}
# MASK=topk_identity (default): soft forward, identity backward = IG along the mask path, dirs
# *_topkid_*; MASK=topk: the soft top-k Jacobian (sigma' + centring), dirs *_topk_*.
MASK=${MASK:-topk_identity}; [ "$MASK" = topk_identity ] && mtag=topkid || mtag=topk
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY_GEMMA="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
[ "$LOGK" = 1 ] && { ks=log; tag=log; } || { ks=uniform; tag=uniform; }
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_addition" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0; skip=0
for split in $SPLITS; do
  [ "$split" = validation ] && out="mib_node_${mtag}_${tag}_frozen" || out="test_node_${mtag}_${tag}_frozen"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case " $MODELS " in *" $model "*) ;; *) continue ;; esac
    case " $TASKS " in *" $task "*) ;; *) continue ;; esac
    if [ -f "results/$out/${task}_${model}_scores.pt" ]; then echo "SKIP $out/${task}_${model}"; skip=$((skip+1)); continue; fi
    py="uv run python"; bs=""; ec=""
    case $model in
      gpt2|qwen2.5) res="-q jag -d a6000 -c 2 -r 32G" ;;
      gemma2)       res="-q jag -d a6000 -c 3 -r 64G"; bs="--batch-size 4"; py=$PY_GEMMA ;;
      llama3)       res="-q jag -d a6000 -c 4 -r 96G"; bs="--batch-size 2" ;;
    esac
    [ "$model" = llama3 ] && [[ "$task" == arc_* ]] && res="-q sphinx -d h100 -r 128G"
    [ "$split" = validation ] && [ "$model" = llama3 ] && [ "$task" = ioi ] && ec="--eval-examples 200"
    name="frz-$mtag-$tag-${split:0:3}-${task}-${model}"
    cmd="$EXP $py scripts/mib/eval_mib.py --model $model --task $task --steps 500 --k-schedule $ks \
--masking $MASK --optimizer none --mode sufficient --lr 0.05 --split $split --train-split train \
--include-input $bs $ec --output results/$out"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n no-learning ($tag k) jobs, $skip skipped =="
