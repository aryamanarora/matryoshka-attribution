#!/bin/bash
# Single-node activation-patching baseline (scripts/mib/eval_mib_actpatch.py) on EVERY MIB node
# cell, validation + test. No learning: each node's score is its own interchange effect averaged
# over 200 train pairs, in both directions (denoise = restore i alone in the corrupted run, noise =
# corrupt i alone in the clean run). One job per cell writes BOTH
#   results/{mib,test}_node_actpatch_denoise/   results/{mib,test}_node_actpatch_noise/
# Cost is 3 + 2N forwards per batch: N = 157 (gpt2), 361 (qwen2.5-0.5B), 235 (gemma-2-2b: 26x8+26+1),
# 1057 (llama3-8B: 32x32+32+1) -- ~1 min / ~4 min / ~15 min / ~2-3 h of forwards per cell. The
# phase-1 batch is 10 on the big models (ARC prompts are long), and the MIB-eval batch follows the
# headline launchers (4 gemma2, 2 llama3); llama3 ARC evals take the h100 like the other launchers
# (a6000 OOMs in MIB's eval), and validation ioi/llama3 is capped at 200 examples like every other
# validation row. gemma2 runs in the TL 2.15.4 `tl2` env and is opt-in via MODELS. A cell whose
# noise pkl exists is skipped, so the wave is resumable. sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_mib_actpatch_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_mib_actpatch_sc.sh
#   SPLITS=validation DRYRUN=1 bash scripts/mib/launch/submit_mib_actpatch_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; SPLITS=${SPLITS:-"validation test"}
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
  [ "$split" = validation ] && out=results/mib_node_actpatch || out=results/test_node_actpatch
  for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case " $MODELS " in *" $model "*) ;; *) continue ;; esac
    case " $TASKS " in *" $task "*) ;; *) continue ;; esac
    if [ -f "${out}_noise/${task}_${model}_${split}.pkl" ]; then echo "SKIP ${split} ${task}_${model}"; skip=$((skip+1)); continue; fi
    py="uv run python"; ec=""
    case $model in
      gpt2|qwen2.5) res="-q jag -d a6000 -c 2 -r 32G"; bs=20; ebs=20 ;;
      gemma2)       res="-q jag -d a6000 -c 3 -r 64G"; bs=10; ebs=4; py=$PY_GEMMA ;;
      llama3)       res="-q jag -d a6000 -c 4 -r 96G"; bs=10; ebs=2 ;;
    esac
    [ "$model" = llama3 ] && [[ "$task" == arc_* ]] && res="-q sphinx -d h100 -r 128G"
    [ "$split" = validation ] && [ "$model" = llama3 ] && [ "$task" = ioi ] && ec="--eval-examples 200"
    name="actpatch-${split:0:3}-${task}-${model}"
    cmd="$EXP $py scripts/mib/eval_mib_actpatch.py --model $model --task $task \
--n-examples 200 --batch-size $bs --eval-batch-size $ebs --split $split --train-split train $ec --output $out"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n actpatch jobs, $skip skipped =="
