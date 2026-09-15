#!/bin/bash
# NODE-substrate runs of the SVA+ HEADLINE arm (uniform k, Adam eps=1e-2; `stopk-unif-eps1e-2`
# in plot_accauc_vs_faithauc.parse_method), which existed on the three neuron substrates
# (submit_unifk_eps_5k.sh) but never on `--nodes node` -- so the node-level section of
# tabs/sva_results.tex had no headline row once the headline became uniform-k Adam
# (scripts/mib/mattr_variants.py, 2026-09-15).
#
# Every flag is read off `config` in the landed node-tree json of the LOG-k eps=1e-2 twin
# (results/sva_sweep/<task>_<model>_node_sufficient_topk_adam_eps1e-2[_<loss>]_bs1.json):
#   2000 steps, lr 0.05, T 0.5, eps 1e-2, train-batch-size 1, eval-examples 100,
#   train-eval-every 200, train-eval-examples 20, seed 42, ablation patch, no --include-input.
# The ONLY difference is --k-schedule uniform. All three losses, like every other node-tree
# arm (logit_diff / ce / acc): 10 tasks x 3 losses = 30 jobs. Output tag (eval_sva.py):
#   <task>_<model>_node_sufficient_topk_adam_eps1e-2[_<loss>]_uniformk_bs1.json
#
# sc / nlprun, run INSIDE tmux. ioi is qwen2.5 (V.on_model), everything else llama3.
#   bash scripts/sva/launch/submit_unifk_eps_node_sc.sh            # submit
#   DRY=1 LOSSES=logit_diff bash scripts/sva/launch/submit_unifk_eps_node_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
OUT=results/sva_sweep
LOSSES=${LOSSES:-"logit_diff ce acc"}
TASKS=${TASKS:-"nounpp rc simple within_rc addition months weekdays hours arc_easy ioi"}
DRY=${DRY:-0}
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith" [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
COMMON="--nodes node --method mattr --variant topk --mode sufficient --optimizer adam --adam-eps 1e-2 \
--k-schedule uniform --steps 2000 --lr 0.05 --T 0.5 --train-batch-size 1 --eval-examples 100 \
--train-eval-every 200 --train-eval-examples 20 --seed 42 --output $OUT"
n=0; skip=0
for task in $TASKS; do
  read -r model ds <<< "${CFG[$task]}"
  for loss in $LOSSES; do
    ls=""; [ "$loss" != logit_diff ] && ls="_$loss"
    f="$OUT/${task}_${model}_node_sufficient_topk_adam_eps1e-2${ls}_uniformk_bs1.json"
    if [ -f "$f" ]; then echo "SKIP $(basename $f)"; skip=$((skip+1)); continue; fi
    name="unifeps-node-${task}-${loss}"
    [ "$model" = qwen2.5 ] && res="-q jag -c 2 -r 32G" || res="-q jag -d a6000 -c 4 -r 96G"
    cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py \
--model $model --task $task --dataset $ds --loss $loss $COMMON"
    n=$((n+1))
    if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n node-substrate uniform-k eps=1e-2 jobs, $skip skipped =="
