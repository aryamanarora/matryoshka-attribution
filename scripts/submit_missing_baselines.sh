#!/bin/bash
# Fill missing MIB baseline cells: NAP-IG node (mcqa/llama3) + EAP-IG-inputs edge
# (arc_easy/gemma2, arc_challenge/llama3). Original runs failed on missing eap import;
# fix = PYTHONPATH=EAP-IG/src. Train->val, matching original flags (--num-examples 1000).
set -u
ABS=/juice2/scr2/aryaman/learning-to-attribute
PY=$ABS/.venv/bin/python
MIB=$ABS/MIB-circuit-track
cd $ABS

run() {  # name model task level circuit_dir out_dir res
  local name=$1 model=$2 task=$3 level=$4 cdir=$5 odir=$6 res=$7
  nlprun -g 1 -q jag -d a6000 $res -n "$name" \
    "cd $MIB && PYTHONPATH=EAP-IG/src $PY run_attribution.py --models $model --tasks $task --method EAP-IG-inputs --level $level --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/$cdir && PYTHONPATH=EAP-IG/src $PY run_evaluation.py --models $model --tasks $task --method EAP-IG-inputs --level $level --ablation patching --split validation --batch-size 1 --circuit-dir $ABS/results/$cdir --output-dir $ABS/results/$odir"
  sleep 1
}

run "nap-mcqa-llama3"   llama3 mcqa          node napig_repro napig_repro_eval "-c 4 -r 96G"
run "eap-arce-gemma2"   gemma2 arc_easy      edge eapig_repro eapig_repro_eval "-c 4 -r 96G"
run "eap-arcc-llama3"   llama3 arc_challenge edge eapig_repro eapig_repro_eval "-c 5 -r 128G"
echo "MISSING BASELINE JOBS SUBMITTED"
