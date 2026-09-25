#!/bin/bash
# Noising (necessity) MIB sweep for the three objective-ablation runs, VALIDATION split, one
# gpujob job per (objective, cell): scripts/mib/eval_mib_noising.py -> <dir>/<task>_<model>_
# validation_noising.pkl. cluster B, run INSIDE tmux. Eval only (the rankings exist).
#
#   bash scripts/mib/launch/submit_mib_noising_val_sc.sh            # 36 jobs
#   OBJS="cause" ONLY='ioi/gpt2' DRYRUN=1 bash ...                  # OBJS: iso cause joint; ONLY: grep -E on task/model
#
# gemma2 in the tl2 group (TL 2.15.4); qwen2.5 pinned to a6000; llama3 --head 200 (the
# validation cap every other validation row uses), batch 2, on a6000 like the validation
# rows of the ablation itself.
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; ONLY=${ONLY:-}; OBJS=${OBJS:-"iso cause joint"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY="uv run python"
PY_GEMMA="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 4" "llama3 ioi 2"
 "llama3 arithmetic_addition 2" "llama3 arithmetic_subtraction 2"
 "qwen2.5 mcqa 10" "gemma2 mcqa 4" "llama3 mcqa 2"
 "gemma2 arc_easy 4" "llama3 arc_easy 2" "llama3 arc_challenge 2"
)
dir_of() { case "$1" in iso) echo mib_node_topk_uniform_lr05 ;; cause) echo mib_node_cause_topk_uniform_lr05 ;; joint) echo mib_node_joint_topk_uniform_lr05 ;; *) echo "bad objective $1" >&2; exit 1 ;; esac; }
n=0; skip=0
for obj in $OBJS; do
  d=$(dir_of "$obj")
  for cell in "${CELLS[@]}"; do
    read -r model task bs <<< "$cell"
    if [ -n "$ONLY" ] && ! echo "$task/$model" | grep -qE "$ONLY"; then continue; fi
    if [ -f "results/$d/${task}_${model}_validation_noising.pkl" ]; then skip=$((skip+1)); continue; fi
    if [ ! -f "results/$d/${task}_${model}_scores.pt" ]; then echo "MISSING scores: results/$d/${task}_${model}_scores.pt" >&2; continue; fi
    py=$PY; head=""
    case "$model" in
      gpt2|qwen2.5) res="-q gpu -d a6000 -c 2 -r 32G" ;;
      gemma2)       res="-q gpu -d a6000 -c 3 -r 64G"; py=$PY_GEMMA ;;
      llama3)       res="-q gpu -d a6000 -c 4 -r 96G"; head="--head 200" ;;
    esac
    name="noise-$obj-${task}-${model}"
    cmd="cd $ABS && $EXP $py scripts/mib/eval_mib_noising.py --model $model --task $task --dir $d --split validation --batch-size $bs $head"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY gpujob -g 1 $res -n $name"; else
      gpujob -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n noising validation jobs, $skip skipped =="
