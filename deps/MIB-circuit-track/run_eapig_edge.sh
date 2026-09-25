#!/bin/bash
# EAP-IG-inputs at EDGE level, all 12 MIB paper cells, under THIS repo's venv (TL 2.15.4).
#
# Why this exists: the edge row of paper/tabs/mib_results.tex read results/eapig_repro_eval/,
# produced in June under the L2A venv (TL 3.2.1), whose Gemma-2 forward is wrong (L2A commit
# 525673a). Those pkls are now quarantined in L2A under results/_stale_tl321/. The only other
# clean edge dir on disk, eapig_repro_accauc, was attributed with --num-examples 1000 for EVERY
# cell, which does not match the CELLS convention below that run_variants.sh / run_relp.sh /
# run_gim.sh / run_attnlrp.sh all share -- so it is left alone as the acc-AUC source and this
# script regenerates the CPR row instead.
#
# Attribution + evaluation, train -> validation, one SLURM job per cell. CELLS is copied verbatim
# from run_variants.sh; resource sizing is copied from L2A's submit_eapig_edge_accauc.sh, which
# is the only edge-level wave that has actually completed on this cluster (edge graphs are far
# heavier than node, so the node scripts' sizing is not enough).
#
#   bash run_eapig_edge.sh            # submit
#   DRYRUN=1 bash run_eapig_edge.sh   # preview
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"
cd $ABS
PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CDIR=results/eapig_clean
OUT=results/eapig_clean_eval

# cell: model task num_examples attr_batch eval_head(0=full)   -- verbatim from run_variants.sh
CELLS=(
  "gpt2 ioi 1000 20 0"
  "qwen2.5 ioi 1000 10 0"
  "qwen2.5 mcqa full 10 0"
  "gemma2 ioi 1000 10 0"
  "gemma2 mcqa full 10 0"
  "gemma2 arc_easy 100 1 0"
  "llama3 ioi 1000 1 200"
  "llama3 mcqa full 1 200"
  "llama3 arithmetic_addition 100 1 200"
  "llama3 arithmetic_subtraction 100 1 200"
  "llama3 arc_easy 100 1 200"
  "llama3 arc_challenge 100 1 200"
)

n=0; skip=0
for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  tdash=${task//_/-}
  if [ -f "$OUT/EAP-IG-inputs_patching_edge/${tdash}_${model}_validation_abs-False.pkl" ]; then
    echo "SKIP $task/$model: already scored"; skip=$((skip+1)); continue
  fi
  # Edge-level footprint, not the node scripts' sizing: llama3 edge OOM'd at 96G in the June
  # wave and only landed once submit_eapig_edge_accauc.sh raised it to 128G / 24h.
  case $model in
    llama3)  cpus=5; mem=128G; tlim=24:00:00; ebatch=1 ;;
    gemma2)  cpus=4; mem=96G;  tlim=16:00:00; ebatch=1 ;;
    qwen2.5) cpus=4; mem=64G;  tlim=12:00:00; ebatch=5 ;;
    *)       cpus=3; mem=48G;  tlim=12:00:00; ebatch=10 ;;   # gpt2
  esac
  [ "$nex"   = "full" ] && nex_flag=""      || nex_flag="--num-examples $nex"
  [ "$ehead" = "0"    ] && head_flag=""     || head_flag="--head $ehead"

  name="eapedge-${task}-${model}"
  cmd="$pp; \
$PY run_attribution.py --models $model --tasks $task --method EAP-IG-inputs --ig-steps 5 \
--level edge --ablation patching --split train --batch-size $abatch $nex_flag \
--circuit-dir $CDIR && \
$PY run_evaluation.py --models $model --tasks $task --method EAP-IG-inputs \
--level edge --ablation patching --split validation --batch-size $ebatch $head_flag \
--circuit-dir $CDIR --output-dir $OUT"
  if [ "$DRYRUN" = "1" ]; then
    echo "[DRY] $name | mem=$mem cpus=$cpus t=$tlim | nex=$nex abatch=$abatch ehead=$ehead"
  else
    mkdir -p "$ABS/logs"
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== submitted $n, skipped $skip -> $OUT =="
