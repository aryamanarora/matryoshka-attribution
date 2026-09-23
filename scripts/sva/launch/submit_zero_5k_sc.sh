#!/bin/bash
# ZERO ABLATION at the patched trees' 5k protocol, on all four SVA+ substrates, into ONE tree:
#   results/sva_zeroabl_5k
# results/sva_zeroabl (the older zero tree) is 2000 steps for every trained method and its
# gradient baselines attribute on 32 examples, so nothing in it is comparable to the patched
# tables (5000 steps; IG 500 ex x 10, I x G / EG / AttnLRP 5000 ex). This wave re-runs the
# zero-ablation setting at exactly those budgets (2026-09-18, "everything should be 5k steps"),
# node included (patched node is 2k; here every substrate is 5k so the table has one budget).
# Zero ablation on SAE latents: models/llama.py _sae_interchange (ablated latent -> 0, ablated
# error -> 0); the SAE cells carry the frozen error term and lr 0.5, as the patched SAE runs do.
#   mattr   uniform k, Adam eps 1e-2 (the headline)     mattr-log  its log-k twin
#   np      eprun s=0.90, lr 0.05                         dbm        sig lr 0.3, l1 6.0
#   ig      500 ex x 10 steps      ixg / eg / attnlrp   5000 ex     random  seed 42
# EG and AttnLRP: not on SAE (no patched runs either). Node has ten tasks (arc_easy on llama3,
# ioi on qwen2.5), the other substrates the eight SVA+Arith ones. Resumable (a cell whose json
# exists is skipped). SAE on sphinx h100 (the encode OOMs a 48 GB a6000). sc / nlprun, in tmux.
#   bash scripts/sva/launch/submit_zero_5k_sc.sh
#   NODES=node METHODS="mattr ig" DRY=1 bash scripts/sva/launch/submit_zero_5k_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
OUT=results/sva_zeroabl_5k; DRY=${DRY:-0}
NODES=${NODES:-"node mlp mlp+attn_head mlp_sae_span"}
METHODS=${METHODS:-"mattr mattr-log np dbm ig ixg eg attnlrp random"}
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith" [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
TR="--train-batch-size 1 --steps 5000 --train-eval-every 200 --train-eval-examples 20"
n=0; skip=0
for nodes in $NODES; do
  [ "$nodes" = node ] && tasks="nounpp rc simple within_rc addition months weekdays hours arc_easy ioi" \
                      || tasks="nounpp rc simple within_rc addition months weekdays hours"
  sae=0; [ "$nodes" = mlp_sae_span ] && sae=1
  # --grad-batch is what let the patched 5000-example gradient runs fit (64; SAE 25): without it
  # the whole example set is one batch and the first wave of these OOM'd on the a6000 (2026-09-18).
  nd=${nodes//+/-}; ferr=""; saeflags=""; mlr=0.05; gb="--grad-batch 64"
  [ $sae = 1 ] && { ferr="_ferr"; saeflags="--sae-error frozen"; mlr=0.5; gb="--grad-batch 25"; }
  for task in $tasks; do
    read -r model ds <<< "${CFG[$task]}"
    # arc_easy prompts are long: the patched node runs attribute it at --grad-batch 8, and 64 OOMs.
    gbt=$gb; [ "$task" = arc_easy ] && [ $sae = 0 ] && gbt="--grad-batch 8"
    for m in $METHODS; do
      case $m in
        mattr)     a="--method mattr --variant topk --mode sufficient --k-schedule uniform --optimizer adam --adam-eps 1e-2 --lr $mlr --T 0.5 $TR"
                   f="${task}_${model}_${nd}_sufficient_topk_adam_eps1e-2_zeroabl_uniformk${ferr}_bs1_s5000.json" ;;
        mattr-log) a="--method mattr --variant topk --mode sufficient --k-schedule log --optimizer adam --adam-eps 1e-2 --lr $mlr --T 0.5 $TR"
                   f="${task}_${model}_${nd}_sufficient_topk_adam_eps1e-2_zeroabl${ferr}_bs1_s5000.json" ;;
        np)        a="--method edge_pruning --target-sparsity 0.9 --lr 0.05 $TR"
                   f="${task}_${model}_${nd}_eprun_s090_zeroabl${ferr}_s5000.json" ;;
        dbm)       a="--method sigmoid_mask --lr 0.3 --l1-coeff 6.0 $TR"
                   f="${task}_${model}_${nd}_sig_lr0.3_l16.0_zeroabl${ferr}_s5000.json" ;;
        ig)        a="--method ig --ig-steps 10 --grad-examples 500 $gbt";   f="${task}_${model}_${nd}_ig_zeroabl${ferr}.json" ;;
        ixg)       a="--method ixg --grad-examples 5000 $gbt";               f="${task}_${model}_${nd}_ixg_zeroabl${ferr}.json" ;;
        eg)        [ $sae = 1 ] && continue
                   a="--method mc_ig --ig-steps 1 --grad-examples 5000 $gbt"; f="${task}_${model}_${nd}_mc_ig_m1_s42_zeroabl.json" ;;
        attnlrp)   [ $sae = 1 ] && continue
                   a="--method attnlrp --grad-examples 5000 $gbt";               f="${task}_${model}_${nd}_attnlrp_zeroabl.json" ;;
        random)    a="--method random";                                     f="${task}_${model}_${nd}_random_s42_zeroabl${ferr}.json" ;;
      esac
      if [ -f "$OUT/$f" ]; then skip=$((skip+1)); continue; fi
      if [ $sae = 1 ]; then res="-q sphinx -d h100 -r 128G"
      elif [ "$model" = qwen2.5 ]; then res="-q jag -d a6000 -c 2 -r 32G"
      else res="-q jag -d a6000 -c 4 -r 96G"; fi
      name="z5k-$m-$nd-$task"
      cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py \
--model $model --task $task --dataset $ds --nodes $nodes $saeflags --loss logit_diff --ablation zero \
--eval-examples 100 --seed 42 $a --output $OUT"
      n=$((n+1))
      if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
        nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
      fi
    done
  done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n zero-ablation 5k jobs, $skip skipped =="
