#!/bin/bash
# AtP and AtP* (Kramar et al. 2024, arXiv:2403.00745) as node-level MIB baselines on the
# VALIDATION split, all 12 cells. Method ids are the EAP-IG fork's (attribute_node.py,
# get_scores_atp):
#   AtP      = per-example |(noise - clean) . grad|, one backward per batch -- the I x G estimate
#              with the absolute value taken inside the expectation (their Eq. 5)
#   AtP-star = AtP + GradDrop: one gradient-dropped backward per layer (L per batch), |.| per term,
#              summed and divided by L-1 (their Eq. 12). The QK-fix half of AtP* is defined for
#              query/key nodes and MIB's node set (head outputs, MLP outputs, input) has none, so on
#              this node set AtP* IS AtP + GradDrop; the docstring says so at length.
# Protocol matches the other gradient baselines (fork run_variants.sh CELLS): attribute on the TRAIN
# split with the fork's per-cell --num-examples / --batch-size, evaluate on VALIDATION with llama3
# capped at --head 200 (the daggered cells). Circuits land in the fork's results/ like every other
# fork-side method; pkls on the L2A side, where make_mib_table.EXTRA_NODE_BASELINES reads them:
#   deps/MIB-circuit-track/results/atp/<Method>_patching_node/<task-dash>_<model>/importances.json
#   results/atp_eval/<Method>_patching_node/<task-dash>_<model>_validation_abs-False.pkl
# An existing importances.json is reused (eval-only rerun); an existing pkl skips the cell.
# gemma2 runs in the TL 2.15.4 `tl2` env (opt-in via MODELS -- TL 3.2.1's Gemma-2 forward is
# wrong, README.md). llama3 ARC on the h100. sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_atp_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_atp_sc.sh
#   METHODS=AtP-star DRYRUN=1 bash scripts/mib/launch/submit_atp_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
MIB=$ABS/deps/MIB-circuit-track
DRYRUN=${DRYRUN:-0}; SPLIT=${SPLIT:-validation}
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}
METHODS=${METHODS:-"AtP AtP-star"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
PYE_L2A="uv run --project $ABS python"
PYE_G="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 --project $ABS python"
CDIR=results/atp                 # relative to $MIB
OUT=$ABS/results/atp_eval
# model task n_attr_examples attr_batch eval_head(0=full)   (fork run_variants.sh CELLS)
CELLS=(
  "gpt2 ioi 1000 20 0" "qwen2.5 ioi 1000 10 0" "qwen2.5 mcqa full 10 0"
  "gemma2 ioi 1000 10 0" "gemma2 mcqa full 10 0" "gemma2 arc_easy 100 1 0"
  "llama3 ioi 1000 1 200" "llama3 mcqa full 1 200" "llama3 arithmetic_addition 100 1 200"
  "llama3 arithmetic_subtraction 100 1 200" "llama3 arc_easy 100 1 200" "llama3 arc_challenge 100 1 200"
)
n=0; skip=0
for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  case " $MODELS " in *" $model "*) ;; *) continue ;; esac
  tdash=${task//_/-}
  pye=$PYE_L2A
  case $model in   # eval batch sizes as submit_mib_zero_sc.sh (the last sc wave through this runner)
    gpt2|qwen2.5) res="-q jag -d a6000 -c 2 -r 32G"; ebs=20 ;;
    gemma2)       res="-q jag -d a6000 -c 3 -r 64G"; ebs=4; pye=$PYE_G ;;
    llama3)       res="-q jag -d a6000 -c 4 -r 96G"; ebs=2 ;;
  esac
  # 80 GB slot for the llama3 cells that OOM a 48 GB a6000: ARC, and mcqa (attribution over the
  # full train split; both methods died there 2026-09-18). BIG as in submit_mib_zero_sc.sh.
  [ "$model" = llama3 ] && { [[ "$task" == arc_* ]] || [ "$task" = mcqa ]; } && res="${BIG:--q sphinx -d h100 -r 128G}"
  [ "$nex" = full ] && nf="" || nf="--num-examples $nex"
  [ "$ehead" = 0 ] && hf="" || hf="--head $ehead"
  for meth in $METHODS; do
    pkl=$OUT/${meth}_patching_node/${tdash}_${model}_${SPLIT}_abs-False.pkl
    [ -f "$pkl" ] && { echo "SKIP $meth $task/$model"; skip=$((skip+1)); continue; }
    circ=$MIB/$CDIR/${meth}_patching_node/${tdash}_${model}/importances.json
    attr="PYTHONPATH=.:EAP-IG/src $EXP $pye run_attribution.py --models $model --tasks $task --method $meth \
--level node --ablation patching --split train --batch-size $abatch $nf --circuit-dir $CDIR"
    cmd="cd $MIB && if [ -f $circ ]; then echo REUSING $circ; else $attr; fi && \
PYTHONPATH=.:EAP-IG/src $EXP $pye run_evaluation.py --models $model --tasks $task --method $meth --level node \
--ablation patching --split $SPLIT --batch-size $ebs $hf --circuit-dir $CDIR --output-dir $OUT"
    name="atp-${meth//-star/s}-${task}-${model}"
    n=$((n+1))
    if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n AtP/AtP* $SPLIT jobs, $skip skipped =="
