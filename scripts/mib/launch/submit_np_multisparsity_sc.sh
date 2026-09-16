#!/bin/bash
# Node Pruning's target-s ladder read as a FRONTIER: each of the nine s-sweep masks
# (results/eprun_node_s<S>_ld, tabs/sparsity_sweep.tex) evaluated at the size it emits, one job
# per MIB node cell, through eval_dbm_multisparsity.py --ladder np -> results/np_multisparsity.
# The analogue of submit_dbm_multisparsity.sh (Tilde/sbatch) for sc/nlprun; run INSIDE tmux.
# Eval only: the graphs exist for all 12 cells x 9 rungs (validation sweep, split-agnostic).
#
#   bash scripts/mib/launch/submit_np_multisparsity_sc.sh                  # 12 test-split jobs
#   SPLIT=validation ONLY='ioi/gpt2' DRYRUN=1 bash ...                     # ONLY: grep -E on task/model
#
# gemma2 runs in the tl2 group (TL 2.15.4; the L2A venv's TL 3.x Gemma-2 forward is wrong);
# llama3 on sphinx h100 (the ARC test splits at batch 2 are the long ones -- ~7 h on Tilde);
# qwen2.5 pinned to a6000 (CUDA "no kernel image" on older cards). --head 200 on llama3
# VALIDATION only, like every other row of the validation table.
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; SPLIT=${SPLIT:-test}; ONLY=${ONLY:-}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PY="uv run python"
PY_GEMMA="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
CELLS=(
 "gpt2 ioi 20" "qwen2.5 ioi 10" "gemma2 ioi 10" "llama3 ioi 2"
 "llama3 arithmetic_addition 2" "llama3 arithmetic_subtraction 2"
 "qwen2.5 mcqa 10" "gemma2 mcqa 10" "llama3 mcqa 2"
 "gemma2 arc_easy 4" "llama3 arc_easy 2" "llama3 arc_challenge 2"
)
n=0; skip=0
for cell in "${CELLS[@]}"; do
  read -r model task bs <<< "$cell"
  if [ -n "$ONLY" ] && ! echo "$task/$model" | grep -qE "$ONLY"; then continue; fi
  if [ -f "results/np_multisparsity/${task}_${model}_${SPLIT}.json" ]; then echo "SKIP $task/$model ($SPLIT json exists)"; skip=$((skip+1)); continue; fi
  HEAD=""; [ "$model" = llama3 ] && [ "$SPLIT" = validation ] && HEAD="--head 200"
  py=$PY
  case "$model" in
    gpt2|qwen2.5) res="-q jag -d a6000 -c 2 -r 32G" ;;
    gemma2)       res="-q jag -d a6000 -c 3 -r 64G"; py=$PY_GEMMA ;;
    llama3)       res="-q sphinx -d h100 -r 128G" ;;
  esac
  name="npL0$([ "$SPLIT" = validation ] && echo v)-${task}-${model}"
  cmd="cd $ABS && $EXP $py scripts/mib/eval_dbm_multisparsity.py --ladder np --model $model --task $task --split $SPLIT --batch-size $bs $HEAD"
  n=$((n+1))
  if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
    nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n Node Pruning multi-sparsity eval jobs (split=$SPLIT), $skip skipped =="
