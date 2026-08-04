#!/bin/bash
# acc-AUC for the EAP-IG-inputs EDGE baseline -- the one thing blocking an edge-level version of
# plots/mib_accauc_cpr_scatter_full.pdf.
#
# The existing results/eapig_repro_eval/ pkls predate acc_auc (their keys are area_under,
# area_from_1, average, faithfulnesses, weighted_edge_counts and nothing else), and unlike every
# other backfill this one CANNOT be a re-evaluation: results/eapig_repro/ -- the circuit dir those
# pkls were scored from -- no longer exists on disk. So attribution has to be re-run before the
# eval, which is why this is a separate script from submit_accauc_backfill.py.
#
# Output goes to a NEW dir, results/eapig_repro_accauc/, NOT back into eapig_repro_eval. The CPR
# numbers in the "EAP-IG-inp (CF, repro)" row of mib_results.tex came from the ORIGINAL circuits;
# re-attribution reproduces them only up to data-order and numerics, so overwriting would quietly
# move a published row as a side effect of adding a column. Splitting acc-AUC into a *_accauc dir
# is also exactly what every node-level baseline already does (results/*_accauc vs results/*_eval
# in make_mib_accauc_table.py).
#
# Batch/head per model follow MIB-circuit-track/run_accauc.sh: llama3 -> --head 200 (the dagger),
# gemma2 -> 200 on ioi only, gpt2/qwen -> full validation.
#
#   bash scripts/submit_eapig_edge_accauc.sh            # submit
#   DRYRUN=1 bash scripts/submit_eapig_edge_accauc.sh   # preview
set -u
ABS=/home/guests/aryaman/learning-to-attribute
MIB=/home/guests/aryaman/MIB-circuit-track
PY=$MIB/.venv/bin/python
DRYRUN=${DRYRUN:-0}
CDIR=$ABS/results/eapig_repro_accauc_circuits
OUT=$ABS/results/eapig_repro_accauc

PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy" "llama3 arc_easy"
  "llama3 arc_challenge"
)
n=0
for p in "${PAIRS[@]}"; do
  read -r model task <<< "$p"
  tdash=${task//_/-}
  if [ -f "$OUT/EAP-IG-inputs_patching_edge/${tdash}_${model}_validation_abs-False.pkl" ]; then
    echo "SKIP $task/$model: already scored"
    continue
  fi
  case $model in
    llama3)  cpus=5; mem=128G; tlim=24:00:00; bs=1; head="--head 200" ;;
    gemma2)  cpus=4; mem=96G;  tlim=16:00:00; bs=1
             [ "$task" = ioi ] && head="--head 200" || head="" ;;
    qwen2.5) cpus=4; mem=64G;  tlim=12:00:00; bs=5;  head="" ;;
    *)       cpus=3; mem=48G;  tlim=12:00:00; bs=10; head="" ;;   # gpt2
  esac
  name="eapacc-${task}-${model}"
  cmd="cd $MIB && export PYTHONPATH=EAP-IG/src:. && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
$PY run_attribution.py --models $model --tasks $task --method EAP-IG-inputs --level edge \
--ablation patching --split train --batch-size $bs --num-examples 1000 --circuit-dir $CDIR && \
$PY run_evaluation.py --models $model --tasks $task --method EAP-IG-inputs --level edge \
--ablation patching --split validation --batch-size $bs $head --circuit-dir $CDIR \
--output-dir $OUT"
  if [ "$DRYRUN" = "1" ]; then
    echo "DRY $name"
  else
    mkdir -p "$ABS/logs"
    sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
      --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
      && echo "submitted $name"
  fi
  n=$((n+1))
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n EAP-IG edge acc-AUC jobs =="
