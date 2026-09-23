#!/bin/bash
# OBJECTIVE ABLATION of the headline MAttr (soft top-k, uniform k, Adam lr 0.05, 500 steps) at
# node level: train under the CAUSE objective (noising: top-k corrupted, complement clean --
# legacy `necessary`) and under JOINT (a coin flip between iso and cause every step), then
# score with MIB's CPR exactly as the headline is scored. The headline itself is ISO (denoising,
# legacy `sufficient`), which is also what CPR measures, so this asks whether a mask trained for
# necessity (or both) transfers to the sufficiency metric. Never run on MIB before 2026-09-16:
# every results dir whose saved args say `necessary` predates the 2026-06-15 label flip and is an
# iso run under the old name (README.md). The cause objective has only ever been run on SVA+.
#
# Protocol is the headline's, cell for cell (submit_softuni_lr05.sh / the saved args of
# mib_node_topk_uniform_lr05 and test_node_topk_uniform_lr05): --steps 500 --k-schedule uniform
# --masking topk --optimizer adam --lr 0.05 --include-input; batch 2 on llama3, 4 on gemma2,
# default otherwise; validation caps ioi/llama3 at --eval-examples 200, test is uncapped. The ONLY
# change is --mode. Output dirs mirror the headline's names with the objective spliced in:
#   mib_node_{cause,joint}_topk_uniform_lr05     (validation)   -> make_mib_table OUR_METHODS
#   test_node_{cause,joint}_topk_uniform_lr05    (test)
#
# gemma2 cells must run under TL 2.15.4 (README.md): the `tl2` dependency group, in its own
# uv-managed env (.venv-tl2) that the job syncs from uv.lock. They are opt-in via MODELS so the
# two stacks can be submitted separately: 2 modes x 2 splits x 9 cells = 36 jobs by default,
# 12 more with MODELS=gemma2. sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_mib_node_mode_ablation_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_mib_node_mode_ablation_sc.sh
#   MODES=cause SPLITS=validation DRYRUN=1 bash ...
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRYRUN=${DRYRUN:-0}; MODES=${MODES:-"cause joint"}; SPLITS=${SPLITS:-"validation test"}
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}   # gemma2 is opt-in: MODELS=gemma2 (or add it to the list)
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}   # subset for resubmits
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
# gemma2 runs in the TL 2.15.4 environment, the `tl2` dependency group of pyproject.toml, synced
# from uv.lock by the job itself (same arrangement as the ViT group). The fork's modules come in
# through deps.find_mib_path, so no PYTHONPATH and no venv inside the fork checkout.
PY_GEMMA="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_addition" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0; skip=0
for mode in $MODES; do for split in $SPLITS; do
  [ "$split" = validation ] && out="mib_node_${mode}_topk_uniform_lr05" || out="test_node_${mode}_topk_uniform_lr05"
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
    # ARC prompts are long: llama3 ARC cells OOM in MIB's eval on the 48 GB a6000 (6 of 8 did on
    # 2026-09-16, after training completed), so they take the 80 GB h100 like the edge jobs.
    [ "$model" = llama3 ] && [[ "$task" == arc_* ]] && res="-q sphinx -d h100 -r 128G"
    [ "$split" = validation ] && [ "$model" = llama3 ] && [ "$task" = ioi ] && ec="--eval-examples 200"
    name="mode-${mode}-${split:0:3}-${task}-${model}"
    cmd="$EXP $py scripts/mib/eval_mib.py --model $model --task $task --steps 500 --k-schedule uniform \
--masking topk --optimizer adam --mode $mode --lr 0.05 --split $split --train-split train \
--include-input $bs $ec --output results/$out"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done; done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n objective-ablation jobs, $skip skipped =="
