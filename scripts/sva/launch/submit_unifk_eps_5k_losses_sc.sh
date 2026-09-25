#!/bin/bash
# The SVA+ headline arm (uniform k, Adam eps=1e-2, 5000 steps; submit_unifk_eps_5k.sh) under the
# OTHER attribution losses on the two neuron substrates (2026-09-21). The headline's CE / soft-acc
# twins existed only on `--nodes node` (submit_unifk_eps_node_sc.sh); every baseline and the
# log-k eps=1e-2 arm have all three losses on mlp / mlp+attn_head in results/sva_sweep_5k, so
# these 32 jobs complete the grid for the loss-by-substrate CPR-vs-Compactness figure
# (plot_accauc_vs_faithauc.py --cpr --by-loss). Every flag other than --loss is the 5k
# headline's (read off the landed logit_diff json's `config`): lr 0.05, T 0.5, eps 1e-2,
# 5000 steps, bs 1, eval-examples 100, train-eval-every 200, seed 42. Output tag:
#   results/sva_sweep_5k/<task>_llama3_<mlp|mlp-attn_head>_iso_topk_adam_eps1e-2_<loss>_uniformk_bs1_s5000.json
#
# cluster B / gpujob, run INSIDE tmux. LOSSES defaults to the two missing ones; pass LOSSES="kl cmd"
# for the newer losses once they are wanted here too.
#   bash scripts/sva/launch/submit_unifk_eps_5k_losses_sc.sh
#   DRY=1 NODES=mlp bash scripts/sva/launch/submit_unifk_eps_5k_losses_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRY=${DRY:-0}
LOSSES=${LOSSES:-"ce acc"}
TASKS=${TASKS:-"nounpp rc simple within_rc addition months weekdays hours"}
NODES=${NODES:-"mlp mlp+attn_head"}
OUT=results/sva_sweep_5k
declare -A DS=( [nounpp]=sva [rc]=sva [simple]=sva [within_rc]=sva
                [addition]=arith [months]=arith [weekdays]=arith [hours]=arith )
COMMON="--model llama3 --method mattr --variant topk --k-schedule uniform --mode iso \
--optimizer adam --adam-eps 1e-2 --lr 0.05 --T 0.5 --steps 5000 --train-batch-size 1 \
--eval-examples 100 --train-eval-every 200 --train-eval-examples 20 --seed 42 --output $OUT"
RES="-q gpu -d a6000 -c 4 -r 96G"
n=0; skip=0
for t in $TASKS; do
  for nodes in $NODES; do
    for loss in $LOSSES; do
      ls=""; [ "$loss" != logit_diff ] && ls="_$loss"
      f="$OUT/${t}_llama3_${nodes//+/-}_iso_topk_adam_eps1e-2${ls}_uniformk_bs1_s5000.json"
      if [ -f "$f" ]; then echo "SKIP $(basename $f)"; skip=$((skip+1)); continue; fi
      name="u5k-${loss}-${nodes//+/-}-${t}"
      cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py \
--task $t --dataset ${DS[$t]} --nodes $nodes --loss $loss $COMMON"
      n=$((n+1))
      if [ "$DRY" = 1 ]; then echo "DRY gpujob -g 1 $RES -n $name"; echo "    $cmd"; else
        gpujob -g 1 $RES -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
      fi
    done
  done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n SVA+ 5k headline loss-twin jobs, $skip skipped =="
