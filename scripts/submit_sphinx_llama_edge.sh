#!/bin/bash
# Rerun the 48GB-blocked llama3 cells on SPHINX (a100/h100 80GB+) with a reduced eval
# subset. Covers: val edge (log+uniform), test edge (uniform), and the 2 MIB baselines.
set -u
ABS=/juice2/scr2/aryaman/learning-to-attribute
PY=$ABS/.venv/bin/python
MIB=$ABS/MIB-circuit-track
cd $ABS
EXP=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# L2A val/test edge (our eval_mib_edge), llama3, eval subset 200, batch 2
for t in ioi arithmetic_subtraction mcqa; do
  # val log
  nlprun -g 1 -q sphinx -r 128G -n se-log-${t}-llama3 \
    "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule log --masking hard_topk --mode sufficient --split validation --train-split train --batch-size 2 --eval-examples 200 --output results/mib_edge_hard_topk"
  sleep 1
  # val uniform
  nlprun -g 1 -q sphinx -r 128G -n se-uni-${t}-llama3 \
    "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split validation --train-split train --batch-size 2 --eval-examples 200 --output results/mib_edge_hard_topk_uniform"
  sleep 1
  # test uniform
  nlprun -g 1 -q sphinx -r 128G -n se-tst-${t}-llama3 \
    "$EXP uv run python scripts/eval_mib_edge.py --model llama3 --task $t --steps 5000 --k-schedule uniform --masking hard_topk --mode sufficient --split test --train-split train --batch-size 2 --eval-examples 200 --output results/test_edge_hard_topk_uniform"
  sleep 1
done

# MIB baselines (head 200): NAP-IG node mcqa/llama3, EAP-IG edge arc_challenge/llama3
nlprun -g 1 -q sphinx -r 128G -n se-nap-mcqa-llama3 \
  "cd $MIB && PYTHONPATH=EAP-IG/src $EXP $PY run_attribution.py --models llama3 --tasks mcqa --method EAP-IG-inputs --level node --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/napig_repro && PYTHONPATH=EAP-IG/src $EXP $PY run_evaluation.py --models llama3 --tasks mcqa --method EAP-IG-inputs --level node --ablation patching --split validation --batch-size 1 --head 200 --circuit-dir $ABS/results/napig_repro --output-dir $ABS/results/napig_repro_eval"
sleep 1
nlprun -g 1 -q sphinx -r 128G -n se-eap-arcc-llama3 \
  "cd $MIB && PYTHONPATH=EAP-IG/src $EXP $PY run_attribution.py --models llama3 --tasks arc_challenge --method EAP-IG-inputs --level edge --ablation patching --split train --batch-size 1 --num-examples 1000 --circuit-dir $ABS/results/eapig_repro && PYTHONPATH=EAP-IG/src $EXP $PY run_evaluation.py --models llama3 --tasks arc_challenge --method EAP-IG-inputs --level edge --ablation patching --split validation --batch-size 1 --head 200 --circuit-dir $ABS/results/eapig_repro --output-dir $ABS/results/eapig_repro_eval"
echo "SPHINX LLAMA EDGE JOBS SUBMITTED"
