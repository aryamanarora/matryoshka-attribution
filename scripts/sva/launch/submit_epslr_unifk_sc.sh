#!/bin/bash
# UNIFORM-k twin of the Adam eps x lr grid behind figs/epsgrid_facets.pdf (whose runs are all
# log-k: submit_epslr.sh / submit_adam_vs_sgd_mlp.sh ARM A). Same four substrates, same six eps
# and four lr, same everything else (topk / logit_diff / iso / bs 1 / 2000 steps / 100
# eval examples / probe every 250 on 64); the ONLY change is --k-schedule uniform, the headline
# schedule since 2026-09-15 (scripts/mib/mattr_variants.py). Separate trees (<tree>_unifk) and
# a separate figure (plots/plot_epsgrid_facets.py --tag unifk) -- the log-k figure stays as is.
#
#   rows (tree)                                   model    nodes          jobs
#   results/epslr_node_unifk                      llama3   node           24
#   results/epslr_mlpn_gemma2_unifk               gemma2   mlp            24
#   results/epslr_mlpn_unifk                      llama3   mlp            24   (log-k twin: results/adamsgd_mlp/A_eps)
#   results/saefrozen_epslr_unifk                 llama3   mlp_sae_span   24   (--sae-error frozen)
# plus one uniform-k MAttr+SGD reference per row in <tree>/refs (the rho-vs-SGD panel must
# correlate against the same k-schedule; the IG references are k-independent and reused).
#
# sc / nlprun, run INSIDE tmux. gemma2 is fine in the L2A env here (eval_sva loads through
# AutoModelForCausalLM, no transformer_lens). Cells with a landed json are skipped.
#   ROWS="node mlpn_gemma2" DRY=1 bash scripts/sva/launch/submit_epslr_unifk_sc.sh
set -u
ABS=${ABS:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}; cd "$ABS"; mkdir -p logs
DRY=${DRY:-0}; ROWS=${ROWS:-"node mlpn_gemma2 mlpn sae"}; REFS=${REFS:-1}
EPSES=${EPSES:-"1e-8 1e-6 1e-4 1e-2 1e-1 1e0"}
LRS=${LRS:-"0.005 0.05 0.5 5.0"}
STEPS=${STEPS:-2000}
EXP="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
row_spec() {  # -> "model nodes errarg outbase prefix res"
  case "$1" in
    node)        echo "llama3 node - results/epslr_node_unifk ukn -q_jag_-d_a6000_-c_4_-r_96G" ;;
    mlpn_gemma2) echo "gemma2 mlp - results/epslr_mlpn_gemma2_unifk ukmg -q_jag_-d_a6000_-c_3_-r_64G" ;;
    mlpn)        echo "llama3 mlp - results/epslr_mlpn_unifk ukm -q_jag_-d_a6000_-c_4_-r_96G" ;;
    # sphinx h100, not jag a6000: Llama-8B + 32 Llama-Scope SAEs (5.2M mask logits) OOMs a 48 GB
    # card at the first training step (2026-09-16); the log-k twin ran on Tilde's 80 GB H100s.
    sae)         echo "llama3 mlp_sae_span frozen results/saefrozen_epslr_unifk uks -q_sphinx_-d_h100_-r_128G" ;;
    *) echo "bad row $1" >&2; exit 1 ;;
  esac
}
n=0; skip=0
for row in $ROWS; do
  read -r model nodes err outbase prefix res <<< "$(row_spec "$row")"
  res=${res//_/ }
  errarg=""; [ "$err" != "-" ] && errarg="--sae-error $err"
  common="--model $model --task addition --dataset arith --nodes $nodes $errarg \
--method mattr --variant topk --k-schedule uniform --mode iso --loss logit_diff \
--optimizer adam --train-batch-size 1 --steps $STEPS --eval-examples 100 \
--train-eval-every 250 --train-eval-examples 64"
  if [ "$REFS" = 1 ]; then
    out="$outbase/refs"
    if ls "$out"/*sgd*.json >/dev/null 2>&1; then skip=$((skip+1)); else
      name="${prefix}-ref-sgd"
      cmd="cd $ABS && mkdir -p $out && $EXP uv run python scripts/sva/eval_sva.py --model $model --task addition --dataset arith --nodes $nodes $errarg \
--loss logit_diff --method mattr --variant topk --mode iso --k-schedule uniform --optimizer sgd --lr 1.0 \
--train-batch-size 1 --steps $STEPS --eval-examples 100 --output $out"
      n=$((n+1))
      if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; else
        nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1; fi
    fi
  fi
  for eps in $EPSES; do for lr in $LRS; do
    out="$outbase/eps_${eps}_lr_${lr}"
    if ls "$out"/*.json >/dev/null 2>&1; then skip=$((skip+1)); continue; fi
    name="${prefix}-${eps}-lr${lr}"
    cmd="cd $ABS && mkdir -p $out && $EXP uv run python scripts/sva/eval_sva.py $common --lr $lr --adam-eps $eps --output $out"
    n=$((n+1))
    if [ "$DRY" = 1 ]; then echo "DRY nlprun -g 1 $res -n $name"; else
      nlprun -g 1 $res -n "$name" -o "$ABS/logs/${name}.out" "$cmd" 2>&1 | grep -E 'Submitted batch job' | sed "s/^/$name: /"; sleep 1; fi
  done; done
done
echo "== $([ "$DRY" = 1 ] && echo 'DRY ')total $n uniform-k eps x lr jobs, $skip skipped =="
