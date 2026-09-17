#!/bin/bash
# The SVA+ headline arm (uniform k, Adam eps=1e-2; submit_unifk_eps_5k.sh) trained 10x longer:
# 50000 steps instead of 5000, on the three trained substrates the paper's scatter draws (MLP,
# MLP+Attn, SAE (MLP out) with the frozen error term). Twin of the MIB 10x runs
# (submit_node_unif_10x_sc.sh / submit_edge_unif_10x_sc.sh, 2026-09-17), which raised every cell.
# Every other flag is the headline's; probes every 1000 steps (50, vs the headline's 25 at 200)
# so the train-curve overlay stays affordable. SEPARATE trees, one budget per tree, because
# parse_method strips the `_s\d{4,}` budget suffix and a `_s50000` file next to the `_s5000` one
# would collide under the same key (plot_accauc_vs_faithauc.SUBSTRATE_RES explains the rule):
#   results/sva_sweep_50k      mlp, mlp+attn_head   (lr 0.05)
#   results/sva_sweep_ferr50k  mlp_sae_span         (lr 0.5, --sae-error frozen)
# The 5k runs took 7-10 min (MLP), 8-10 min (MLP+Attn), 15-35 min (SAE) on Tilde, so expect
# ~1.5 h / ~1.5 h / 3-6 h per cell here. sc / nlprun, run INSIDE tmux.
#   bash scripts/sva/launch/submit_unifk_eps_50k_sc.sh
#   NODES="mlp" DRY=1 bash scripts/sva/launch/submit_unifk_eps_50k_sc.sh
set -u
ABS=${ABS:-/juice3/scr3/nlp/interp/learning-to-attribute}; cd "$ABS"; mkdir -p logs
DRY=${DRY:-0}; STEPS=${STEPS:-50000}
TASKS=${TASKS:-"nounpp rc simple within_rc addition months weekdays hours"}
NODES=${NODES:-"mlp mlp+attn_head mlp_sae_span"}
declare -A DS=( [nounpp]=sva [rc]=sva [simple]=sva [within_rc]=sva
                [addition]=arith [months]=arith [weekdays]=arith [hours]=arith )
COMMON="--model llama3 --method mattr --variant topk --k-schedule uniform --mode sufficient \
--loss logit_diff --optimizer adam --adam-eps 1e-2 --T 0.5 --steps $STEPS --train-batch-size 1 \
--eval-examples 100 --train-eval-every 1000 --train-eval-examples 20 --seed 42"
RES="-q jag -d a6000 -c 4 -r 96G"
n=0; skip=0
for t in $TASKS; do
  for nodes in $NODES; do
    if [ "$nodes" = mlp_sae_span ]; then
      out=results/sva_sweep_ferr50k; extra="--lr 0.5 --sae-error frozen"
      f="$out/${t}_llama3_mlp_sae_span_sufficient_topk_adam_eps1e-2_uniformk_ferr_bs1_s${STEPS}.json"
    else
      out=results/sva_sweep_50k; extra="--lr 0.05"
      f="$out/${t}_llama3_${nodes//+/-}_sufficient_topk_adam_eps1e-2_uniformk_bs1_s${STEPS}.json"
    fi
    if [ -f "$f" ]; then echo "SKIP $(basename $f)"; skip=$((skip+1)); continue; fi
    name="u50k-${nodes//+/-}-${t}"
    cmd="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python scripts/sva/eval_sva.py \
--task $t --dataset ${DS[$t]} --nodes $nodes $extra $COMMON --output $out"
    n=$((n+1))
    if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $RES -n $name"; echo "    $cmd"; else
      nlprun -g 1 $RES -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1
    fi
  done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n SVA+ uniform-k eps=1e-2 ${STEPS}-step jobs, $skip skipped =="
