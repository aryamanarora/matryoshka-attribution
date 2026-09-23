#!/bin/bash
# Edge Pruning (Bhaskar et al., 2024) at EDGE level on MIB, validation + test -- the baseline the
# edge tables never had (every eprun_* dir on disk is node level; run_edge_pruning.sbatch is the
# Tilde-era launcher this copies, paths and env swapped for sc). Hard-concrete gates, the original
# Lagrangian L0 recipe, at MAttr's edge step count (5000, submit_softuni_lr05.sh) and, like the
# "Node Pruning" row (make_mib_table.EPRUN_BEST_SPARSITY = s0.5_ld), on the logit-diff loss.
# Target sparsity defaults to the script's edge default 0.99; SPARSITIES="0.99 0.5" adds twins.
# Dirs follow the sbatch's suffix rule so make_mib_table's eprun rows resolve them:
#   results/eprun_edge_s<s>_ld/graph_<task>_<model>.json                      (learned gates)
#   results/eprun_eval_s<s>_ld/EdgePruning_patching_edge/<task>_<model>_<split>_abs-False.pkl
# llama3 is capped at --head 200 on BOTH splits, matching the MAttr edge cells it sits next to
# (submit_softuni_lr05.sh: ev=200 for llama3 edge on validation and test); the node sbatch capped
# validation only, which is the node convention. gemma2 runs in the TL 2.15.4 `tl2` env and is
# opt-in via MODELS. sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_eprun_edge_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_eprun_edge_sc.sh
#   SPARSITIES=0.5 SPLITS=validation DRYRUN=1 bash scripts/mib/launch/submit_eprun_edge_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
MIB=$ABS/deps/MIB-circuit-track
DRYRUN=${DRYRUN:-0}; SPLITS=${SPLITS:-"validation test"}; SPARSITIES=${SPARSITIES:-"0.99"}
STEPS=${STEPS:-5000}; LOSS=${LOSS:-logit_diff}
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_addition" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0; skip=0
for s in $SPARSITIES; do
  suf="_s${s}"; [ "$LOSS" = kl ] || suf="${suf}_ld"
  out=results/eprun_edge$suf; evalout=$ABS/results/eprun_eval$suf
  for split in $SPLITS; do for p in "${PAIRS[@]}"; do
    read -r model task <<< "$p"
    case " $MODELS " in *" $model "*) ;; *) continue ;; esac
    case " $TASKS " in *" $task "*) ;; *) continue ;; esac
    tdash=${task//_/-}
    if [ -f "$evalout/EdgePruning_patching_edge/${tdash}_${model}_${split}_abs-False.pkl" ]; then
      echo "SKIP eprun_eval$suf/edge/${task}_${model}_${split}"; skip=$((skip+1)); continue; fi
    py="uv run python"; pye="uv run --project $ABS python"; head=""
    case $model in
      gpt2|qwen2.5) res="-q jag -d a6000 -c 3 -r 64G"; bs=20 ;;
      gemma2)       res="-q jag -d a6000 -c 4 -r 96G"; bs=4
                    py="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
                    pye="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 --project $ABS python" ;;
      llama3)       res="-q sphinx -d h100 -r 128G"; bs=2; head="--head 200" ;;
    esac
    # The graph is shared by the two splits of a cell: train it once (validation job), and let the
    # test job reuse it when it already exists so the two rows score the SAME circuit.
    graph=$out/graph_${task}_${model}.json
    train="$EXP $py scripts/mib/eval_mib_edge_pruning.py --model $model --task $task --level edge \
--steps $STEPS --split $split --output $out --loss $LOSS --target-sparsity $s --skip-eval"
    evalc="cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP $pye run_evaluation.py --models $model --tasks $task \
--level edge --ablation patching --split $split --method EdgePruning --circuit-files $ABS/$graph \
--batch-size $bs $head --output-dir $evalout"
    cmd="if [ -f $graph ]; then echo REUSING $graph; else $train; fi && $evalc"
    name="epedge-s$s-${split:0:3}-${task}-${model}"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done; done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n edge Edge-Pruning jobs ($STEPS steps, $LOSS), $skip skipped =="
