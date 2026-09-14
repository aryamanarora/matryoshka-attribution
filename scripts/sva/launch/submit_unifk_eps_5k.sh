#!/bin/bash
# UNIFORM-k twin of the 5k-step, eps=1e-2 Adam log-k arm (`stopk-log-eps1e-2`), on the three
# fine-grained SVA+ substrates. Every other flag is copied from the run on disk it pairs with
# (read off `config` in the landed json), so the ONLY difference is --k-schedule uniform:
#
#   mlp, mlp+attn_head : lr 0.05, T 0.5, eps 1e-2, 5000 steps      -> results/sva_sweep_5k
#   mlp_sae_span       : lr 0.5,  T 0.5, eps 1e-2, 5000 steps, --sae-error frozen
#                                                                  -> results/sva_sweep_ferr_lr0.5
#                        (results/sva_sweep_ferr5k symlinks to that dir; add the links for
#                         these files once they land -- see the note in the figure script)
#
# The pre-existing uniform-k Adam runs (`stopk-unif`, results/sva_sweep) are 2k steps at the
# DEFAULT eps, so they are not this arm -- eps is what fixes Adam at neuron scale
# (submit_adam_eps_followup.sh), and the comparison the figure wants is uniform vs log at the
# same budget and eps. run_tag spells this arm `..._topk_adam_eps1e-2_uniformk_bs1_s5000`, which
# plot_accauc_vs_faithauc.parse_method keys as `stopk-unif-eps1e-2`.
#
# logit-diff only, seed 42, 8 tasks x 3 substrates = 24 jobs.
#   bash scripts/sva/launch/submit_unifk_eps_5k.sh        # submit
#   DRY=1 bash scripts/sva/launch/submit_unifk_eps_5k.sh  # print only
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs
declare -A DS=( [nounpp]=sva [rc]=sva [simple]=sva [within_rc]=sva
                [addition]=arith [months]=arith [weekdays]=arith [hours]=arith )
TASKS=(nounpp rc simple within_rc addition months weekdays hours)
COMMON=(--model llama3 --method mattr --variant topk --k-schedule uniform --mode sufficient
        --loss logit_diff --optimizer adam --adam-eps 1e-2 --T 0.5 --steps 5000
        --train-batch-size 1 --eval-examples 100 --train-eval-every 200
        --train-eval-examples 20 --seed 42)
n=0
sub() { local name=$1; shift
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name :: $*"
  else sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "$@" >/dev/null; echo "submitted $name"; fi
  n=$((n+1)); }
for t in "${TASKS[@]}"; do
  d=${DS[$t]}
  for nodes in mlp mlp+attn_head; do
    sub "unifeps_${nodes}_${t}" "${COMMON[@]}" --task "$t" --dataset "$d" --nodes "$nodes" \
        --lr 0.05 --output results/sva_sweep_5k
  done
  sub "unifeps_sae_${t}" "${COMMON[@]}" --task "$t" --dataset "$d" --nodes mlp_sae_span \
      --lr 0.5 --sae-error frozen --output results/sva_sweep_ferr_lr0.5
done
echo "== ${DRY:+DRY }total $n uniform-k eps=1e-2 5k jobs =="
