#!/bin/bash
# The frozen-error MLP-output SAE column of figs/accauc_vs_faithauc.pdf.
# 8 tasks x the 5 series that figure draws on an SAE panel (eprun/DBM have no SAE runs at all).
# Settings are copied from the -input Patched cut in results/sva_sweep so the column differs from
# its neighbours ONLY in the error intervention.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=${ROOT:-results/sva_sweep_ferr}
mkdir -p logs "$ROOT"
n=0
sub() { local name=$1; shift
  if squeue -h -u "$USER" -o '%j' 2>/dev/null | grep -qx "$name"; then echo "  SKIP $name"; return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name :: $*"
  else sbatch -J "$name" sva_sweep.sbatch --model llama3 --nodes mlp_sae_span \
       --sae-error frozen --loss logit_diff "$@" --output "$ROOT" >/dev/null; fi
  n=$((n+1)); }
for spec in "sva nounpp rc simple within_rc" "arith addition months weekdays hours"; do
  read -ra p <<< "$spec"; ds=${p[0]}
  for t in "${p[@]:1}"; do
    C=(--task "$t" --dataset "$ds")
    sub "fx_${t}_ig"   "${C[@]}" --method ig --ig-steps 10 --eval-examples 100 --grad-batch 25
    sub "fx_${t}_ixg"  "${C[@]}" --method ixg --eval-examples 100 --grad-batch 25
    sub "fx_${t}_rand" "${C[@]}" --method random --seed 42 --eval-examples 100
    sub "fx_${t}_adam" "${C[@]}" --method mattr --variant topk --mode sufficient --k-schedule log \
        --optimizer adam --lr 1.0 --adam-eps 1e-2 --train-batch-size 1 --steps 2000 --eval-examples 100
    sub "fx_${t}_sgd"  "${C[@]}" --method mattr --variant topk --mode sufficient --k-schedule log \
        --optimizer sgd --lr 1.0 --train-batch-size 1 --steps 2000 --eval-examples 100
  done
done
echo "== ${DRY:+DRY }$n jobs -> $ROOT =="
