#!/bin/bash
# The SVA+ headline arm (uniform k, Adam eps=1e-2, logit-diff) under ZERO ablation, the one arm
# results/sva_zeroabl never had -- its MAttr rows are the log-k eps=1e-2 and default-eps uniform
# runs, so plots/plot_accauc_vs_faithauc.py --cpr --zero drew "+log k" and no "MAttr" star
# (2026-09-18). Flags are the zero tree's (read off its log-k eps=1e-2 json: 2000 steps, lr 0.05,
# T 0.5, batch 1, eval 100, probe every 200 on 20, seed 42); --k-schedule uniform is the only
# change. Node has all ten tasks, the neuron substrates the eight SVA+Arith ones, as in the tree.
# Output tag: <task>_<model>_<nodes>_sufficient_topk_adam_eps1e-2_zeroabl_uniformk_bs1.json.
# sc / nlprun, run INSIDE tmux.
#   bash scripts/sva/launch/submit_unifk_eps_zero_sc.sh
#   NODES=node DRY=1 bash scripts/sva/launch/submit_unifk_eps_zero_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
OUT=results/sva_zeroabl; DRY=${DRY:-0}
NODES=${NODES:-"node mlp mlp+attn_head"}
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith" [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
COMMON="--method mattr --variant topk --mode sufficient --loss logit_diff --optimizer adam --adam-eps 1e-2 \
--k-schedule uniform --ablation zero --steps 2000 --lr 0.05 --T 0.5 --train-batch-size 1 --eval-examples 100 \
--train-eval-every 200 --train-eval-examples 20 --seed 42 --output $OUT"
n=0; skip=0
for nodes in $NODES; do
  [ "$nodes" = node ] && tasks="nounpp rc simple within_rc addition months weekdays hours arc_easy ioi" \
                      || tasks="nounpp rc simple within_rc addition months weekdays hours"
  for task in $tasks; do
    read -r model ds <<< "${CFG[$task]}"
    f="$OUT/${task}_${model}_${nodes//+/-}_sufficient_topk_adam_eps1e-2_zeroabl_uniformk_bs1.json"
    if [ -f "$f" ]; then echo "SKIP $(basename $f)"; skip=$((skip+1)); continue; fi
    name="zunif-${nodes//+/-}-${task}"
    [ "$model" = qwen2.5 ] && res="-q jag -d a6000 -c 2 -r 32G" || res="-q jag -d a6000 -c 4 -r 96G"
    cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py \
--model $model --task $task --dataset $ds --nodes $nodes $COMMON"
    n=$((n+1))
    if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n zero-ablation uniform-k eps=1e-2 jobs, $skip skipped =="
