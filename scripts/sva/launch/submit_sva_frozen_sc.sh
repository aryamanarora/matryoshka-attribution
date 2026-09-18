#!/bin/bash
# MAttr WITHOUT LEARNING (--optimizer none; see submit_mib_node_frozen_sc.sh) on the SVA+
# substrates, at each tree's budget so the rows sit next to the headline in the tables:
#   node           results/sva_sweep         2000 steps            (10 tasks)
#   mlp, mlp+attn  results/sva_sweep_5k      5000 steps            (8 tasks)
#   mlp_sae_span   results/sva_sweep_ferr5k  5000 steps, frozen err (8 tasks, h100)
# Both k-schedules (uniform = headline twin, log). Tags carry no eps (no optimizer):
#   <task>_<model>_<nodes>_sufficient_topk_none[_uniformk][_ferr]_bs1[_s5000].json
# sc / nlprun, run INSIDE tmux.
#   bash scripts/sva/launch/submit_sva_frozen_sc.sh
#   NODES=node KS=uniform DRY=1 bash scripts/sva/launch/submit_sva_frozen_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRY=${DRY:-0}; NODES=${NODES:-"node mlp mlp+attn_head mlp_sae_span"}; KS=${KS:-"uniform"}
# MASK=topk_identity (default): identity backward = IG along the mask path; MASK=topk: soft Jacobian.
MASK=${MASK:-topk_identity}
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith" [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
n=0; skip=0
for nodes in $NODES; do
  nd=${nodes//+/-}; ferr=""; extra=""; steps=5000; out=results/sva_sweep_5k; ssuf="_s5000"
  tasks="nounpp rc simple within_rc addition months weekdays hours"
  case $nodes in
    node) steps=2000; out=results/sva_sweep; ssuf=""; tasks="$tasks arc_easy ioi" ;;
    mlp_sae_span) ferr="_ferr"; extra="--sae-error frozen --lr 0.5"; out=results/sva_sweep_ferr5k ;;
  esac
  for ks in $KS; do
    [ "$ks" = uniform ] && ktag="_uniformk" || ktag=""
    for task in $tasks; do
      read -r model ds <<< "${CFG[$task]}"
      f="$out/${task}_${model}_${nd}_sufficient_${MASK}_none${ktag}${ferr}_bs1${ssuf}.json"
      if [ -f "$f" ]; then skip=$((skip+1)); continue; fi
      if [ "$nodes" = mlp_sae_span ]; then res="-q sphinx -d h100 -r 128G"
      elif [ "$model" = qwen2.5 ]; then res="-q jag -d a6000 -c 2 -r 32G"
      else res="-q jag -d a6000 -c 4 -r 96G"; fi
      name="frz-$MASK-$ks-$nd-$task"
      cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py \
--model $model --task $task --dataset $ds --nodes $nodes --method mattr --variant $MASK --mode sufficient \
--loss logit_diff --optimizer none --k-schedule $ks --steps $steps --T 0.5 --train-batch-size 1 \
--eval-examples 100 --train-eval-every 200 --train-eval-examples 20 --seed 42 $extra --output $out"
      n=$((n+1))
      if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; echo "    $cmd"; else
        nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
      fi
    done
  done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n SVA+ no-learning jobs, $skip skipped =="
