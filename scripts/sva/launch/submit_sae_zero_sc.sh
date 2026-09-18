#!/bin/bash
# ZERO ablation on the SAE (MLP out) substrate: the substrate's full method set for the
# zero-ablation SVA+ tables (scripts/sva/make_sva_table.py --zero). Zero ablation on SAE latents
# is defined as of 2026-09-18 (models/llama.py _sae_interchange: ablated latent -> 0, ablated
# error node -> 0). Flags are the PATCHED SAE runs' (results/sva_sweep_ferr5k, frozen error term),
# at the zero tree's 2000-step budget for every trained method -- results/sva_zeroabl is 2k
# throughout, and the tables read one budget per tree. Same output tree as the other zero
# substrates (the _ferr tag keeps the files distinct; no absorb-error SAE runs live there).
#   MAttr   uniform k, Adam eps 1e-2, lr 0.5   (the headline SAE arm)
#   IG      ig-steps 10, grad-batch 25         IxG   grad-batch 25      Random  seed 42
#   NP      eprun s=0.90, lr 0.05              DBM   sig lr 0.3, l1 6.0
# All eight SVA+Arith tasks, llama3, sphinx h100 (the SAE encode OOMs a 48 GB a6000). sc / nlprun,
# run INSIDE tmux.
#   bash scripts/sva/launch/submit_sae_zero_sc.sh
#   METHODS="mattr ig" DRY=1 bash scripts/sva/launch/submit_sae_zero_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
OUT=results/sva_zeroabl; DRY=${DRY:-0}
METHODS=${METHODS:-"mattr ig ixg random np dbm"}
TASKS=${TASKS:-"nounpp rc simple within_rc addition months weekdays hours"}
declare -A DS=( [nounpp]=sva [rc]=sva [simple]=sva [within_rc]=sva
                [addition]=arith [months]=arith [weekdays]=arith [hours]=arith )
C="--model llama3 --nodes mlp_sae_span --sae-error frozen --loss logit_diff --ablation zero \
--eval-examples 100 --seed 42 --output $OUT"
RES="-q sphinx -d h100 -r 128G"
n=0; skip=0
for t in $TASKS; do for m in $METHODS; do
  case $m in
    mattr) a="--method mattr --variant topk --mode sufficient --k-schedule uniform --optimizer adam --adam-eps 1e-2 --lr 0.5 --T 0.5 --steps 2000 --train-batch-size 1 --train-eval-every 200 --train-eval-examples 20"
           f="${t}_llama3_mlp_sae_span_sufficient_topk_adam_eps1e-2_zeroabl_uniformk_ferr_bs1.json" ;;
    ig)    a="--method ig --ig-steps 10 --grad-batch 25";  f="${t}_llama3_mlp_sae_span_ig_zeroabl_ferr.json" ;;
    ixg)   a="--method ixg --grad-batch 25";               f="${t}_llama3_mlp_sae_span_ixg_zeroabl_ferr.json" ;;
    random) a="--method random";                           f="${t}_llama3_mlp_sae_span_random_s42_zeroabl_ferr.json" ;;
    np)    a="--method edge_pruning --target-sparsity 0.9 --lr 0.05 --steps 2000 --train-batch-size 1"
           f="${t}_llama3_mlp_sae_span_eprun_s090_zeroabl_ferr.json" ;;
    dbm)   a="--method sigmoid_mask --lr 0.3 --l1-coeff 6.0 --steps 2000 --train-batch-size 1"
           f="${t}_llama3_mlp_sae_span_sig_lr0.3_l16.0_zeroabl_ferr.json" ;;
  esac
  if [ -f "$OUT/$f" ]; then echo "SKIP $f"; skip=$((skip+1)); continue; fi
  name="zsae-$m-$t"
  cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py --task $t --dataset ${DS[$t]} $a $C"
  n=$((n+1))
  if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $RES -n $name"; echo "    $cmd"; else
    nlprun -g 1 $RES -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
  fi
done; done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n SAE zero-ablation jobs, $skip skipped =="
