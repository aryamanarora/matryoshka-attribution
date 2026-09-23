#!/bin/bash
# ZERO ABLATION on the MIB node cells -- the SVA+ zero-ablation method set (submit_zero_5k_sc.sh)
# applied to MIB: MAttr (headline: uniform k, Adam lr 0.05, 500 steps; and the log-k twin), Node
# Pruning and DBM (the tables' tuned recipes, node level), IG (m=10), I x G (m=1), Expected
# Gradients (mc, m=1), AttnLRP, Random. Every method is trained/attributed under zero and scored
# by MIB's harness with --ablation zero, on the TEST split (full split; llama3 uncapped, as every
# test row is). Attribution examples per cell follow the fork's CELLS block (run_napig10.sh) so
# the gradient rows cost what their patching twins cost. Random is the existing ranking
# (results/random_s42, ablation-independent) re-scored under zero via --circuit-files.
# Outputs, all with _zero in the name so nothing patching-side is touched:
#   MAttr    results/test_node_topk_{uniform,log}_lr05_zero/         (eval_mib.py layout)
#   NP, DBM  results/eprun_node_{s0.5_ld,ld_sig_lr0.3_l16.0}_zero/graph_*.json
#            -> results/eprun_eval_{...}_zero/EdgePruning_zero_node/
#   grads    deps/MIB-circuit-track/results/zero/<Method>_zero_node/   (circuits)
#            results/mib_zero_test/<Method>_zero_node/                 (pkls)
# gemma2 in the tl2 env (opt-in via MODELS); llama3 ARC on h100. sc / nlprun, run INSIDE tmux.
#   bash scripts/mib/launch/submit_mib_zero_sc.sh
#   MODELS=gemma2 bash scripts/mib/launch/submit_mib_zero_sc.sh
#   METHODS="mattr ig" DRYRUN=1 bash scripts/mib/launch/submit_mib_zero_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
MIB=$ABS/deps/MIB-circuit-track
DRYRUN=${DRYRUN:-0}; SPLIT=test
MODELS=${MODELS:-"gpt2 qwen2.5 llama3"}
TASKS=${TASKS:-"ioi arithmetic_addition arithmetic_subtraction mcqa arc_easy arc_challenge"}
METHODS=${METHODS:-"mattr mattr-log np dbm ig ixg eg attnlrp random"}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
# 80 GB slot for the llama3 cells that OOM a 48 GB a6000: ARC (all methods), mcqa attribution
# (IG/I×G/EG over the full train split), and the NP/DBM gate trainers. Default is the h100; pass
# BIG="-q sphinx -d h200 -r 128G" for the 141 GB h200s (sphinx10/11); note "-d a100" can land on
# sphinx1/2, which are 40 GB and OOM these cells -- prefer the h200 override (sphinx9, the only h100,
# killed every job step with "Unable to satisfy cpu bind request" on 2026-09-18).
BIG=${BIG:-"-q sphinx -d h100 -r 128G"}
PY_L2A="uv run python"; PYE_L2A="uv run --project $ABS python"
PY_G="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 python"
PYE_G="UV_PROJECT_ENVIRONMENT=$ABS/.venv-tl2 uv run --no-default-groups --group tl2 --project $ABS python"
# model task n_attr_examples attr_batch  (run_napig10.sh's CELLS; eval batch per model below)
CELLS=(
  "gpt2 ioi 1000 20" "qwen2.5 ioi 1000 10" "gemma2 ioi 1000 10" "llama3 ioi 1000 1"
  "llama3 arithmetic_addition 100 1" "llama3 arithmetic_subtraction 100 1"
  "qwen2.5 mcqa full 10" "gemma2 mcqa full 10" "llama3 mcqa full 1"
  "gemma2 arc_easy 100 1" "llama3 arc_easy 100 1" "llama3 arc_challenge 100 1"
)
n=0; skip=0
sub() {  # name res cmd
  n=$((n+1))
  if [ "$DRYRUN" = 1 ]; then echo "DRY nlprun -g 1 $2 -n $1"; echo "    $3"; else
    nlprun -g 1 $2 -n "$1" -o "$ABS/logs/$1.out" "$3" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$1: /"; sleep 1
  fi
}
for cell in "${CELLS[@]}"; do
  read -r model task nex abatch <<< "$cell"
  case " $MODELS " in *" $model "*) ;; *) continue ;; esac
  case " $TASKS " in *" $task "*) ;; *) continue ;; esac
  tdash=${task//_/-}
  py=$PY_L2A; pye=$PYE_L2A; tbs=""; ebs=20
  case $model in
    gpt2|qwen2.5) res="-q jag -d a6000 -c 2 -r 32G" ;;
    gemma2)       res="-q jag -d a6000 -c 3 -r 64G"; py=$PY_G; pye=$PYE_G; tbs="--batch-size 4"; ebs=4 ;;
    llama3)       res="-q jag -d a6000 -c 4 -r 96G"; tbs="--batch-size 2"; ebs=2 ;;
  esac
  [ "$model" = llama3 ] && [[ "$task" == arc_* ]] && res="$BIG"
  gres=$res; [ "$model" = llama3 ] && [ "$task" = mcqa ] && gres="$BIG"   # mcqa/llama3 IG-family attribution OOMs a6000
  [ "$nex" = full ] && nf="" || nf="--num-examples $nex"
  for m in $METHODS; do
    case $m in
      mattr|mattr-log)
        [ $m = mattr ] && ks=uniform || ks=log
        out=results/test_node_topk_${ks}_lr05_zero
        [ -f "$out/${task}_${model}_scores.pt" ] && { skip=$((skip+1)); continue; }
        sub "z-$m-$task-$model" "$res" "$EXP $py scripts/mib/eval_mib.py --model $model --task $task --steps 500 --k-schedule $ks \
--masking topk --optimizer adam --mode sufficient --lr 0.05 --split $SPLIT --train-split train --include-input \
--ablation zero $tbs --output $out" ;;
      np|dbm)
        if [ $m = np ]; then out=results/eprun_node_s0.5_ld_zero; ev=results/eprun_eval_s0.5_ld_zero
             a="--target-sparsity 0.5 --steps 3000"
        else out=results/eprun_node_ld_sig_lr0.3_l16.0_zero; ev=results/eprun_eval_ld_sig_lr0.3_l16.0_zero
             a="--gate sigmoid --lr 0.3 --l1-coeff 6.0 --l1-target gate --steps 3000"; fi
        [ -f "$ev/EdgePruning_zero_node/${tdash}_${model}_${SPLIT}_abs-False.pkl" ] && { skip=$((skip+1)); continue; }
        graph=$out/graph_${task}_${model}.json
        # The gate trainers keep the live activations of every node in the graph: llama3 OOMs a
        # 48 GB a6000 (ioi cell, 2026-09-18), as the patching-side NP runs also needed the h100.
        mres=$res; [ "$model" = llama3 ] && mres="$BIG"
        sub "z-$m-$task-$model" "$mres" "if [ -f $graph ]; then echo REUSING $graph; else $EXP $py scripts/mib/eval_mib_edge_pruning.py \
--model $model --task $task --level node --split $SPLIT --output $out --loss logit_diff --ablation zero $a --skip-eval; fi && \
cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP $pye run_evaluation.py --models $model --tasks $task --level node --ablation zero \
--split $SPLIT --method EdgePruning --circuit-files $ABS/$graph --batch-size $ebs --output-dir $ABS/$ev" ;;
      ig|ixg|eg|attnlrp)
        case $m in ig) meth=EAP-IG-inputs; a="--ig-steps 10";; ixg) meth=EAP-IG-inputs; a="--ig-steps 1";;
                   eg) meth=EAP-IG-inputs-mc; a="--ig-steps 1";; attnlrp) meth=AttnLRP; a="";; esac
        cdir=results/zero_$m; odir=$ABS/results/mib_zero_test/$m
        [ -f "$odir/${meth}_zero_node/${tdash}_${model}_${SPLIT}_abs-False.pkl" ] && { skip=$((skip+1)); continue; }
        sub "z-$m-$task-$model" "$gres" "cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP $pye run_attribution.py --models $model --tasks $task \
--method $meth $a --level node --ablation zero --split train --batch-size $abatch $nf --circuit-dir $cdir && \
PYTHONPATH=.:EAP-IG/src $EXP $pye run_evaluation.py --models $model --tasks $task --method $meth --level node --ablation zero \
--split $SPLIT --batch-size $ebs --circuit-dir $cdir --output-dir $odir" ;;
      random)
        odir=$ABS/results/mib_zero_test/random
        [ -f "$odir/Random_zero_node/${tdash}_${model}_${SPLIT}_abs-False.pkl" ] && { skip=$((skip+1)); continue; }
        cf=$MIB/results/random_s42/Random_patching_node/${tdash}_${model}/importances.json
        [ -f "$cf" ] || { echo "MISSING random circuit $cf"; continue; }
        sub "z-$m-$task-$model" "$res" "cd $MIB && PYTHONPATH=.:EAP-IG/src $EXP $pye run_evaluation.py --models $model --tasks $task \
--method Random --level node --ablation zero --split $SPLIT --batch-size $ebs --circuit-files $cf --output-dir $odir" ;;
    esac
  done
done
echo "== $([ "$DRYRUN" = 1 ] && echo 'DRY ')total $n MIB zero-ablation jobs, $skip skipped =="
