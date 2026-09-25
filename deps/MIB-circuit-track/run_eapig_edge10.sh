#!/bin/bash
# EAP-IG-inputs at EDGE level with --ig-steps 10, all 12 MIB paper cells (TL 2.15.4 venv).
#
# Edge twin of run_napig10.sh. At NODE level the step count turned out to matter a lot: MIB
# ships --ig-steps 5, and moving to 10 lifted the CPR-AUC row average 0.85 -> 1.31 and acc-AUC
# 0.30 -> 0.46, because at 5 the integral is not converged -- 27 nodes change SIGN while sitting
# in the top 10 by |score| between the two rungs (napig_step_convergence.py). 10 -> 30 then gave
# rho 0.994 with ZERO sign flips, so 10 is the converged setting and 30 merely confirms it.
# This script asks whether the same under-integration is inflating our win over the EDGE row of
# paper/tabs/mib_results.tex, which currently reads results/eapig_clean_eval/ at 5 steps.
#
# Everything except --ig-steps and the two dirs is copied verbatim from run_eapig_edge.sh --
# same CELLS, same venv, same --head 200 llama3 eval cap, same train -> validation direction --
# so the gap between this and the eapig_clean_eval row is the integration grid and nothing else.
# Do NOT "fix" the sizing to match the node scripts: edge graphs are far heavier, and llama3
# edge OOM'd at 96G in the June wave. Attribution cost is exactly linear in --ig-steps, so the
# smaller models get more headroom than run_eapig_edge.sh gave them -- but 24h is a HARD ceiling,
# not a choice: partition main is untimed, yet our account association caps MaxWall at
# 1-00:00:00 (`sacctmgr show assoc user=$USER format=MaxWall`). Asking for 36h does not queue
# and wait, it is REJECTED at submit time with AssocMaxWallDurationPerJobLimit -- which is how
# the first run of this script silently dropped all six llama3 cells while the other six
# started. That is also why run_eapig_edge.sh's llama3 row says 24h: it was already at the cap.
# The Aug 8 5-step wave's slowest cell was ioi/gemma2 at ~11h, and the llama3 edge cells all
# landed in under ~2h, so 24h holds at 10 steps with room to spare.
#
#   bash run_eapig_edge10.sh            # submit
#   DRYRUN=1 bash run_eapig_edge10.sh   # preview
set -u
ABS="$(cd "$(dirname "$0")" && pwd)"
cd $ABS
PY=$ABS/.venv/bin/python
DRYRUN=${DRYRUN:-0}
pp="export PYTHONPATH=EAP-IG/src:.; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
IGS=10
CDIR=results/eapig_clean10
OUT=results/eapig_clean10_eval

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

# Names already queued/running, so a re-run tops up the missing cells instead of duplicating the
# live ones. The pkl check alone is not enough: a cell that is mid-flight has no output yet, and
# this script HAS to be re-runnable partway through -- the first submission lost six cells to the
# wall-clock cap while the other six were already running.
INFLIGHT=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

n=0; skip=0
for cell in "${CELLS[@]}"; do
  read -r model task nex abatch ehead <<< "$cell"
  tdash=${task//_/-}
  if [ -f "$OUT/EAP-IG-inputs_patching_edge/${tdash}_${model}_validation_abs-False.pkl" ]; then
    echo "SKIP $task/$model: already scored"; skip=$((skip+1)); continue
  fi
  if grep -qx "eapedge10-${task}-${model}" <<< "$INFLIGHT"; then
    echo "SKIP $task/$model: already queued/running"; skip=$((skip+1)); continue
  fi
  case $model in
    llama3)  cpus=5; mem=128G; tlim=24:00:00; ebatch=1 ;;   # 24h = association cap, see above
    gemma2)  cpus=4; mem=96G;  tlim=24:00:00; ebatch=1 ;;
    qwen2.5) cpus=4; mem=64G;  tlim=16:00:00; ebatch=5 ;;
    *)       cpus=3; mem=48G;  tlim=16:00:00; ebatch=10 ;;   # gpt2
  esac
  [ "$nex"   = "full" ] && nex_flag=""      || nex_flag="--num-examples $nex"
  [ "$ehead" = "0"    ] && head_flag=""     || head_flag="--head $ehead"

  name="eapedge10-${task}-${model}"
  cmd="$pp; \
$PY run_attribution.py --models $model --tasks $task --method EAP-IG-inputs --ig-steps $IGS \
--level edge --ablation patching --split train --batch-size $abatch $nex_flag \
--circuit-dir $CDIR && \
$PY run_evaluation.py --models $model --tasks $task --method EAP-IG-inputs \
--level edge --ablation patching --split validation --batch-size $ebatch $head_flag \
--circuit-dir $CDIR --output-dir $OUT"
  if [ "$DRYRUN" = "1" ]; then
    echo "[DRY] $name | mem=$mem cpus=$cpus t=$tlim | igs=$IGS nex=$nex abatch=$abatch ehead=$ehead"
  else
    mkdir -p "$ABS/logs"
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
echo "== submitted $n, skipped $skip -> $OUT =="
