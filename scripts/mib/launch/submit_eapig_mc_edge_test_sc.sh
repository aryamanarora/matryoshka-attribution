#!/bin/bash
# Expected Gradients (EAP-IG-inputs-mc: alpha ~ U(0,1) per example, --ig-steps 1) at EDGE level,
# TEST split. The validation arm is done -- 12/12 circuits in the fork's
# deps/MIB-circuit-track/results/eapig_mc/EAP-IG-inputs-mc_patching_edge/ and 12/12 pkls in
# results/eapig_mc_eval/ there (the "Expected Gradients" edge row of make_mib_accauc_table.py) --
# so this is eval-only: run_evaluation.py --split test on the EXISTING circuits, the same way the
# node arm got its test row (submit_arith_add_fill_sc.sh -> napig_mc_test). Output on the L2A
# side, mirroring that dir name:
#   results/eapig_mc_test/EAP-IG-inputs-mc_patching_edge/<task-dash>_<model>_test_abs-False.pkl
# llama3 is capped at --head 200, matching the MAttr edge test cells it sits next to
# (submit_softuni_lr05.sh: ev=200 for llama3 edge on both splits). Eval batch sizes are the fork's
# run_eapig_edge.sh's (gpt2 10, qwen2.5 5, gemma2 1, llama3 1). gemma2 runs in the TL 2.15.4
# `tl2` env and is opt-in via MODELS. sc / nlprun, run INSIDE tmux.
# CDIR/METHOD generalise it to the other fork-side edge circuit sets that only ever had a
# validation eval (eapig_clean = EAP-IG-inp m=5, the "EAP-IG-inp (CF, repro)" edge row;
# eapig_clean10 = m=10, the "+ 10 IG steps" row); output is results/<CDIR>_<SPLIT>/ either way.
#   bash scripts/mib/launch/submit_eapig_mc_edge_test_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_eapig_mc_edge_test_sc.sh
#   CDIR=eapig_clean METHOD=EAP-IG-inputs bash scripts/mib/launch/submit_eapig_mc_edge_test_sc.sh
#   CDIR=eapig_clean10 METHOD=EAP-IG-inputs bash scripts/mib/launch/submit_eapig_mc_edge_test_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
MIB=$ABS/deps/MIB-circuit-track
DRYRUN=${DRYRUN:-0}; SPLIT=${SPLIT:-test}
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CDIR=${CDIR:-eapig_mc}; METHOD=${METHOD:-EAP-IG-inputs-mc}
CPATH=results/$CDIR                         # relative to $MIB (where the circuits live)
OUT=$ABS/results/${CDIR}_$SPLIT
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi"
  "llama3 arithmetic_addition" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa"
  "gemma2 arc_easy" "llama3 arc_easy" "llama3 arc_challenge"
)
n=0; skip=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  case " $MODELS " in *" $model "*) ;; *) continue ;; esac
  case " $TASKS " in *" $task "*) ;; *) continue ;; esac
  tdash=${task//_/-}
  [ -d "$MIB/$CPATH/${METHOD}_patching_edge/${tdash}_${model}" ] || { echo "MISSING circuit ${tdash}_${model}"; continue; }
  if [ -f "$OUT/${METHOD}_patching_edge/${tdash}_${model}_${SPLIT}_abs-False.pkl" ]; then
    echo "SKIP ${task}_${model}"; skip=$((skip+1)); continue; fi
  pye="uv run --project $ABS python"; head=""
  case $model in
    gpt2)    res="-q jag -d a6000 -c 3 -r 64G"; bs=10 ;;
    qwen2.5) res="-q jag -d a6000 -c 3 -r 64G"; bs=5 ;;
    gemma2)  res="-q jag -d a6000 -c 4 -r 96G"; bs=1
             pye="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 --project $ABS python" ;;
    llama3)  res="-q sphinx -d h100 -r 128G"; bs=1; head="--head 200" ;;
  esac
  cmd="cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP $pye run_evaluation.py --models $model --tasks $task \
--method $METHOD --level edge --ablation patching --split $SPLIT --batch-size $bs $head \
--circuit-dir $CPATH --output-dir $OUT"
  name="${CDIR}-${SPLIT:0:3}-${task}-${model}"
  n=$((n+1))
  if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
    nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n edge $METHOD ($CDIR) $SPLIT evals, $skip skipped =="
