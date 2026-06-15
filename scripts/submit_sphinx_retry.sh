#!/bin/bash
# Retry the 5 sphinx jobs that hit 40GB a100s (OOM) / IndexError — pin to h100 (80GB+).
set -u
ABS=/juice2/scr2/aryaman/learning-to-attribute
PY=$ABS/.venv/bin/python
MIB=$ABS/MIB-circuit-track
cd $ABS
EXP=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# edge L2A (eval_mib_edge), subset 200, batch 2
nlprun -g 1 -q sphinx -d h100 -r 128G -n se-log-ioi-llama3 \
  "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task ioi --steps 5000 --k-schedule log --masking hard_topk --mode sufficient --split validation --train-split train --batch-size 2 --eval-examples 200 --output results/mib_edge_hard_topk"
sleep 1
nlprun -g 1 -q sphinx -d h100 -r 128G -n se-uni-ioi-llama3 \
  "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task ioi --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split validation --train-split train --batch-size 2 --eval-examples 200 --output results/mib_edge_hard_topk_uniform"
sleep 1
nlprun -g 1 -q sphinx -d h100 -r 128G -n se-tst-mcqa-llama3 \
  "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task mcqa --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split test --train-split train --batch-size 2 --eval-examples 200 --output results/test_edge_hard_topk_uniform"
sleep 1
# baselines (head 200)
nlprun -g 1 -q sphinx -d h100 -r 128G -n se-eap-arcc-llama3 \
  "cd $MIB && PYTHONPATH=EAP-IG/src $EXP $PY run_attribution.py --models llama3 --tasks arc_challenge --method EAP-IG-inputs --level edge --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/eapig_repro && PYTHONPATH=EAP-IG/src $EXP $PY run_evaluation.py --models llama3 --tasks arc_challenge --method EAP-IG-inputs --level edge --ablation patching --split validation --batch-size 1 --head 200 --circuit-dir $ABS/results/eapig_repro --output-dir $ABS/results/eapig_repro_eval"
sleep 1
nlprun -g 1 -q sphinx -d h100 -r 128G -n se-nap-mcqa-llama3 \
  "cd $MIB && PYTHONPATH=EAP-IG/src $EXP $PY run_attribution.py --models llama3 --tasks mcqa --method EAP-IG-inputs --level node --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/napig_repro && PYTHONPATH=EAP-IG/src $EXP $PY run_evaluation.py --models llama3 --tasks mcqa --method EAP-IG-inputs --level node --ablation patching --split validation --batch-size 1 --head 200 --circuit-dir $ABS/results/napig_repro --output-dir $ABS/results/napig_repro_eval"
echo "SPHINX H100 RETRY SUBMITTED"
