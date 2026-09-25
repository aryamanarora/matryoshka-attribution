#!/bin/bash
# The node-level MAttr headline (soft top-k, uniform k, Adam lr 0.05, --include-input) trained 10x
# longer: 5000 steps instead of 500, the node twin of submit_edge_unif_10x_sc.sh (does more training
# fix the sparse end of the uniform-k ranking, or is it the schedule?). Protocol is
# submit_mib_node_mode_ablation_sc.sh's cell for cell (which copies the headline's saved args);
# the ONLY change is --steps. Output dirs mirror the headline's with the step count spliced in:
#   mib_node_topk_uniform_lr05_5k   (validation)
#   test_node_topk_uniform_lr05_5k  (test)
# gemma2 cells run under TL 2.15.4 (the `tl2` group, own .venv-tl2) and are opt-in via MODELS.
# cluster B / gpujob, run INSIDE tmux.
#   bash scripts/mib/launch/submit_node_unif_10x_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_node_unif_10x_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; SPLITS=${SPLITS:-"validation"}; STEPS=${STEPS:-5000}
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY_GEMMA="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_addition" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0; skip=0
for split in $SPLITS; do
  [ "$split" = validation ] && out="mib_node_topk_uniform_lr05_5k" || out="test_node_topk_uniform_lr05_5k"
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case " $MODELS " in *" $model "*) ;; *) continue ;; esac
    case " $TASKS " in *" $task "*) ;; *) continue ;; esac
    if [ -f "results/$out/${task}_${model}_scores.pt" ]; then echo "SKIP $out/${task}_${model}"; skip=$((skip+1)); continue; fi
    py="uv run python"; bs=""; ec=""
    case $model in
      gpt2|qwen2.5) res="-q gpu -d a6000 -c 2 -r 32G" ;;
      gemma2)       res="-q gpu -d a6000 -c 3 -r 64G"; bs="--batch-size 4"; py=$PY_GEMMA ;;
      llama3)       res="-q gpu -d a6000 -c 4 -r 96G"; bs="--batch-size 2" ;;
    esac
    [ "$model" = llama3 ] && [[ "$task" == arc_* ]] && res="-q gpu-big -d h100 -r 128G"
    [ "$split" = validation ] && [ "$model" = llama3 ] && [ "$task" = ioi ] && ec="--eval-examples 200"
    name="nu10x-${split:0:3}-${task}-${model}"
    cmd="$EXP $py scripts/mib/eval_mib.py --model $model --task $task --steps $STEPS --k-schedule uniform \
--masking topk --optimizer adam --mode iso --lr 0.05 --split $split --train-split train \
--include-input $bs $ec --output results/$out"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY gpujob -g 1 $res -n $name"; echo "    $cmd"; else
      gpujob -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n node uniform-k ${STEPS}-step jobs, $skip skipped =="
