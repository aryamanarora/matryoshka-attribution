#!/bin/bash
# EDGE-level MIB learning-rate sweep, validation split, BOTH optimizers, soft-fwd top-k, log-k.
#
# WHY THIS EXISTS. Every edge-level MAttr LR on disk is an IMPORT from the node-level optimum,
# never a bracket measured at edge scale:
#   adam  lr=0.05  results/mib_edge_topk_log_lr05        (node argmax, carried over)
#   sgd   lr=1.0   results/mib_edge_softlog_sgd_lr_1.0   (node argmax, carried over)
# submit_mib_edge_soft_sgd.sh's header says so in as many words and ends "Use LR= below to
# bracket llama3 before drawing any conclusion" -- because that transfer FAILED (mean area_under
# -1.221 vs the adam row, llama3-concentrated: arith_sub -3.45, ioi -3.74, mcqa -2.42). An
# unbracketed LR cannot distinguish "SGD is worse at edge level" from "1.0 is the wrong LR at
# edge level", and the two have opposite implications for the paper. This script measures it.
#
# WHY THE TWO GRIDS ARE NOT THE SAME. The soft top-k backward carries dm/ds ~ k/n. Adam
# normalises that factor away (its update is ~lr regardless of gradient scale), SGD does not --
# so SGD's useful LR should scale with n/k while Adam's should not move. Edge n is 207-1507x
# node n (gpt2 157 -> 32,491; llama3 1057 -> 1,592,881), so the SGD grid is pushed UP to 100 and
# the Adam grid is centred on its node optimum. A shared grid would spend most of its jobs in a
# region that is dead for one of the two arms -- the exact trap submit_sva_mlp_lr.sh's header
# describes ("a grid that stops at the node-level optimum returns no effect for the same reason
# a thermometer that stops at 40C does").
#
#   adam  0.005 0.01 0.05 0.1 0.3 1.0
#   sgd   0.3 1.0 3.0 10.0 30.0 100.0
#
# BOTH ANCHORS ARE RE-RUN RATHER THAN REUSED, deliberately.
#  - adam lr=0.05: results/mib_edge_topk_log_lr05's gemma2 SCORES are dated 2026-07-21 and were
#    trained by submit_edge_lr05.sh, which points EVERY model at $ABS/.venv (TL 3.2.1, the wrong
#    Gemma-2 forward, 525673a). scripts/reeval_gemma_mib.py re-EVALUATED those cells on 07-24,
#    which fixes the pkl but NOT the circuit -- the search itself ran through the bad forward.
#    Dropping that point into the middle of a curve would put one venv's answer among five
#    others'. (This also means the paper's gemma2 edge MAttr rows rest on circuits trained
#    through the bad forward; that is a separate finding, not something this sweep fixes.)
#  - sgd lr=1.0: results/mib_edge_softlog_sgd_lr_1.0 IS clean (correct venv). It is re-run anyway
#    so the whole curve comes from one submission, which makes the duplicate a free
#    reproducibility check rather than wasted compute.
#
# SCOPE. log-k ONLY -- that is the headline schedule (CLAUDE.md). The uniform-k SGD arm at lr=3.0
# is the more interesting *outlier* (it transferred at parity, +0.154, where log-k lost 1.221),
# so `SCHEDS=uniform` is worth a second wave; it is not bundled here to keep one night's fan-out
# at 108 jobs rather than 216.
#
# GEMMA2 RUNS UNDER THE MIB VENV (TL 2.15.4) for TRAINING and eval both. Copied from
# submit_mib_edge_soft_sgd.sh; do NOT copy submit_edge_lr05.sh's single-venv form.
#
# Cost, from sacct on the 2026-08-21 esgd wave: gpt2/ioi 0:31, qwen2.5/ioi 1:42, qwen2.5/mcqa
# 0:52, gemma2/mcqa 1:02, gemma2/arc_easy 2:01, llama3 {ioi,arith,mcqa} ~2:40, gemma2/ioi 6:22.
# That is ~245 GPU-hours for the full 108-job wave. It does NOT land in 6.5h: this account is
# capped at 8 concurrent GPUs, so wall-clock is ~31h and the SUBMISSION ORDER decides what you
# have by morning. See the footer.
#
# DRYRUN=1 to preview.  OPTS="sgd" / SCHEDS="uniform" / ONLY=llama3 / LRS_SGD=... to narrow.
set -u
ABS=/home/guests/aryaman/learning-to-attribute; cd "$ABS"
PY_L2A=$ABS/.venv/bin/python                     # gpt2 / qwen2.5 / llama3
PY_TL2=$ABS/MIB-circuit-track/.venv/bin/python   # gemma2 ONLY (TL 2.15.4)
PP_TL2="PYTHONPATH=$ABS/src:$ABS/MIB-circuit-track:$ABS/MIB-circuit-track/EAP-IG/src "
DRYRUN=${DRYRUN:-0}
ONLY=${ONLY:-}
OPTS=${OPTS:-"adam sgd"}
SCHEDS=${SCHEDS:-"log"}
LRS_ADAM=${LRS_ADAM:-"0.005 0.01 0.05 0.1 0.3 1.0"}
LRS_SGD=${LRS_SGD:-"0.3 1.0 3.0 10.0 30.0 100.0"}
# The standard edge PAIRS: 9 cells, no llama3 arc_* (never run at edge level).
PAIRS=(
  "gpt2 ioi" "qwen2.5 ioi" "gemma2 ioi" "llama3 ioi" "llama3 arithmetic_subtraction"
  "qwen2.5 mcqa" "gemma2 mcqa" "llama3 mcqa" "gemma2 arc_easy"
)
n=0
for opt in $OPTS; do
  case $opt in
    adam) lrs=$LRS_ADAM ;;
    sgd)  lrs=$LRS_SGD ;;
    *) echo "unknown optimizer $opt" >&2; exit 1 ;;
  esac
  for sched in $SCHEDS; do
    for lr in $lrs; do
      # The LR and the optimizer are BOTH in the dir name. eval_mib_edge.py's output filenames
      # encode neither, so two points of this sweep written to one --output would overwrite each
      # other and the sweep would look complete while holding one run -- the same failure
      # submit_sva_mlp_lr.sh's "OUTPUT DIR PER LR IS LOAD-BEARING" note documents.
      out="mib_edge_lrsweep_${opt}_${sched}_lr_${lr}"
      for p in "${PAIRS[@]}"; do
        read -r model task <<< "$p"
        if [ -n "$ONLY" ] && [ "$model" != "$ONLY" ]; then continue; fi
        py=$PY_L2A; pp=""
        case $model in
          gpt2|qwen2.5) cpus=3; mem=64G;  tlim=10:00:00; bs=5; ev=0 ;;
          gemma2)       cpus=4; mem=96G;  tlim=16:00:00; bs=5; ev=0; py=$PY_TL2; pp=$PP_TL2 ;;
          # llama3 validation is capped to 200 examples per CLAUDE.md -- the dagger in the
          # appendix tables. A sweep point scored on full validation would not be comparable
          # to the rows it is meant to inform.
          llama3)       cpus=5; mem=128G; tlim=24:00:00; bs=2; ev=200 ;;
        esac
        name="elr-${opt}-${sched}-lr${lr}-${task}-${model}"
        cmd="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
$pp$py scripts/eval_mib_edge.py --model $model --task $task --steps 5000 --k-schedule $sched \
--masking topk --optimizer $opt --mode sufficient --lr $lr --split validation --train-split train \
--batch-size $bs --eval-examples $ev --output results/$out"
        if [ "$DRYRUN" = "1" ]; then
          echo "DRY $name -> $out [py=${py#$ABS/}]${pp:+ +PP}"
        else
          sbatch --partition=main --gres=gpu:1 --cpus-per-task=$cpus --mem=$mem --time=$tlim \
            --job-name="$name" --output="$ABS/logs/${name}.out" --wrap="$cmd" >/dev/null \
            && echo "submitted $name"
        fi
        n=$((n+1))
      done
    done
  done
done
[ "$DRYRUN" = "1" ] && pfx="DRY " || pfx=""
echo "== ${pfx}total $n edge LR-sweep validation jobs =="

# SUBMIT BY MODEL, IN COST ORDER -- the account is capped at 8 concurrent GPUs
# (guests/guest-dev, GrpTRES gres/gpu=8), so all 108 jobs at once is ~31 h wall-clock, not one
# night. SLURM runs this QOS roughly FIFO, so submission order IS priority order:
#
#   for m in gpt2 llama3 qwen2.5 gemma2; do ONLY=$m bash scripts/submit_mib_edge_lr_sweep.sh; done
#
# cumulative completion at 8-wide:  gpt2 0.8 h | +llama3 12.7 h | +qwen2.5 16.5 h | +gemma2 30.6 h
# llama3 is the cell the sweep exists to answer (soft_sgd's transfer failure is llama3-concentrated:
# arith_sub -3.45, ioi -3.74, mcqa -2.42), and gemma2/ioi alone is 76 GPU-h -- 31% of the budget
# for the one cell where the SGD transfer already WON (+0.95). So gemma2 goes last, not first.
