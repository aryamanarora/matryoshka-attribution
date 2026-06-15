#!/bin/bash
# Resubmit OOM'd llama3 jobs on a REDUCED eval subset (--head 50 / --eval-examples 50)
# + expandable_segments. Cells are daggered in the tables.
set -u
ABS=/juice2/scr2/aryaman/learning-to-attribute
PY=$ABS/.venv/bin/python
MIB=$ABS/MIB-circuit-track
cd $ABS
EXP=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NAP-IG node baseline: mcqa/llama3 (eval head 200)
nlprun -g 1 -q jag -d a6000 -c 4 -r 96G -n nap-mcqa-llama3 \
  "cd $MIB && PYTHONPATH=EAP-IG/src $EXP $PY run_attribution.py --models llama3 --tasks mcqa --method EAP-IG-inputs --level node --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/napig_repro && PYTHONPATH=EAP-IG/src $EXP $PY run_evaluation.py --models llama3 --tasks mcqa --method EAP-IG-inputs --level node --ablation patching --split validation --batch-size 1 --head 50 --circuit-dir $ABS/results/napig_repro --output-dir $ABS/results/napig_repro_eval"
sleep 1
# EAP-IG edge baseline: arc_challenge/llama3 (eval head 200)
nlprun -g 1 -q jag -d a6000 -c 5 -r 128G -n eap-arcc-llama3 \
  "cd $MIB && PYTHONPATH=EAP-IG/src $EXP $PY run_attribution.py --models llama3 --tasks arc_challenge --method EAP-IG-inputs --level edge --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/eapig_repro && PYTHONPATH=EAP-IG/src $EXP $PY run_evaluation.py --models llama3 --tasks arc_challenge --method EAP-IG-inputs --level edge --ablation patching --split validation --batch-size 1 --head 50 --circuit-dir $ABS/results/eapig_repro --output-dir $ABS/results/eapig_repro_eval"
sleep 1
# Test-edge uniform (our method): llama3 ioi/arith/mcqa (eval subset 200, batch 2)
for t in ioi arithmetic_subtraction mcqa; do
  nlprun -g 1 -q jag -d a6000 -c 5 -r 128G -n teu-${t}-llama3 \
    "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split test --train-split train --batch-size 1 --eval-examples 50 --output results/test_edge_hard_topk_uniform"
  sleep 1
done
echo "OOM REDUCED RESUBMITS DONE"
