#!/bin/bash
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eprun_%j.out
# Edge Pruning (Bhaskar et al., 2024) baseline on MIB: learn hard-concrete gates with the
# Lagrangian L0 recipe, dump the scored circuit, then score it with MIB's own harness.
#
#   sbatch scripts/run_edge_pruning.sbatch <gpt2|qwen2.5|gemma2|llama3> <task> [level] [steps] [split] [target_sparsity]
#
# target_sparsity is the L0 budget the Lagrangian anneals to; it is THE knob that decides
# where the circuit switches on, so runs at different targets must not share output dirs.
# Omit it for the script default (0.9 node / 0.99 edge) and unsuffixed dirs; pass e.g. 0.95
# to get results/eprun_node_s0.95/ + results/eprun_eval_s0.95/.
#
# LOSS=logit_diff (env var) trains on MAttr's objective instead of Edge Pruning's KL and adds
# an _ld suffix to both dirs, e.g. LOSS=logit_diff ... 0.9 -> results/eprun_node_s0.9_ld/.
#
# GATE=sigmoid (env var) swaps the hard-concrete gates for pyvene's SigmoidMaskIntervention
# (deterministic sigmoid, temperature 50->0.1, no sparsity term) and adds a _sig suffix. It
# ignores target_sparsity, so pass none: GATE=sigmoid LOSS=logit_diff ... -> eprun_node_ld_sig/.
#
# LR=<x> (env var) overrides the mask learning rate and adds an _lr<x> suffix, e.g.
# GATE=sigmoid LOSS=logit_diff LR=0.1 ... -> results/eprun_node_ld_sig_lr0.1/.
#
# GATE=dcm is DCM (Prakash et al. 2024 / roonbug/belief_dynamics): a raw coefficient clamped
# to [0,1], no sigmoid/temperature/lr-schedule, with a PID controller driving the sparsity
# term until round(mask) hits DENSITY. It has no ranking (clamp piles scores on 0 and 1), so
# it is only apples-to-apples at that density. DENSITY=<d> is REQUIRED and goes in the dir
# name: GATE=dcm LOSS=logit_diff DENSITY=0.05 LR=0.1 ... -> results/eprun_node_ld_dcm_d0.05_lr0.1/.
#
# L1=<x> (env var, GATE=sigmoid only) adds an L1 sparsity penalty of weight <x> on the gate
# and an _l1<x> suffix; L1_TARGET=logit switches to pyvene's ||mask||_1 form. See
# eval_mib_edge_pruning.py --l1-coeff. GATE=sigmoid LOSS=logit_diff LR=0.3 L1=0.5 ...
# -> results/eprun_node_ld_sig_lr0.3_l10.5/.
#
# Unlike UGS (docs/ugs_baseline.md), this runs for every MIB model and at BOTH levels: the
# gates live in our own trainer, so nothing is tied to a particular architecture.
# Writes results/eprun_eval/EdgePruning_patching_<level>/<task-dash>_<model>_<split>_abs-False.pkl,
# the layout make_mib_table.py / make_mib_accauc_table.py read for baselines.
set -euo pipefail
MODEL=${1:?model}
TASK=${2:?task}
LEVEL=${3:-node}
STEPS=${4:-3000}
SPLIT=${5:-validation}
SPARSITY=${6:-}
# LOSS=logit_diff trains the gates on MAttr's objective instead of Edge Pruning's KL. Every
# run before 2026-08-02 was KL, so the MIB tables compare Node Pruning to MAttr across BOTH a
# different objective and a different mask parameterization; this isolates the second.
# It gets its own dirs (suffix _ld) -- a different objective is a different circuit.
LOSS=${LOSS:-kl}
# GATE=sigmoid is pyvene's SigmoidMaskIntervention recipe instead of the hard-concrete gates;
# same environment, same task loss, same step count, different mask parameterization.
GATE=${GATE:-hard_concrete}
# LR overrides the mask learning rate (default 0.8 hard_concrete / 1e-3 sigmoid) and adds an
# _lr<LR> suffix. Different lr = different circuit, so it must not share a dir.
LR=${LR:-}
# L1 is the sigmoid gate's sparsity penalty weight. pyvene's library trains this mask with no
# penalty at all, but its own tutorial for the class uses loss + 1.0*||mask||_1 -- so the
# unpenalised runs are a choice we made, and this is the variant that closes that gap.
L1=${L1:-}
L1_TARGET=${L1_TARGET:-gate}
# DENSITY is the fraction of units GATE=dcm's PID controller pins the circuit to. Required
# for that gate and inert otherwise; different density = different circuit = different dir.
DENSITY=${DENSITY:-}
DENSITY_ARG=""
if [ "$GATE" = "dcm" ]; then
  [ -n "$DENSITY" ] || { echo "GATE=dcm requires DENSITY=<fraction>"; exit 1; }
elif [ -n "$DENSITY" ]; then
  echo "DENSITY=$DENSITY requires GATE=dcm"; exit 1
fi

L2A=/home/guests/aryaman/learning-to-attribute
MIB=/home/guests/aryaman/MIB-circuit-track
# TL 2.15.4 env: the L2A venv's TL 3.x has a Gemma-2 forward bug, and this script both
# trains and evaluates through TransformerLens.
PY=$MIB/.venv/bin/python

# Eval batch size follows the per-model guidance in CLAUDE.md. HEAD must match the eval
# subset every other row in the table used (MIB-circuit-track/run_variants.sh): llama3
# cells are capped at 200 validation examples, everything else runs full validation.
# Without this the llama3 cells are scored on a different (10k-example) set than the
# gradient baselines they sit next to -- not apples-to-apples, and ~50x slower.
#
# The cap is VALIDATION-only. Test splits are small (ioi/arith 1000, arc-e 1188, arc-c 586,
# mcqa 50), the MIB paper's test numbers are full-split, and scripts/submit_test_lr05.sh runs
# MAttr's test cells uncapped -- so capping here would make this the only subset-scored row in
# the test table. Full llama3 test eval is ~6x the 200-example cost, not the 50x validation is.
case "$MODEL" in
  gpt2|qwen2.5) BS=20; HEAD="" ;;
  gemma2)       BS=4;  HEAD="" ;;
  llama3)       BS=2;  HEAD="--head 200" ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
[ "$SPLIT" = "validation" ] || HEAD=""

if [ -n "$SPARSITY" ]; then
  SUF=_s${SPARSITY}
  SPARSITY_ARG="--target-sparsity $SPARSITY"
else
  SUF=""
  SPARSITY_ARG=""
fi
[ "$LOSS" = "kl" ] || SUF="${SUF}_ld"
# GATE=sigmoid has no sparsity target at all; a _s<S> dir name would be a lie. GATE=dcm has
# one, but it is a DENSITY the PID lands on rather than an L0 budget, so it gets its own knob.
if [ "$GATE" != "hard_concrete" ]; then
  [ -z "$SPARSITY" ] || { echo "GATE=$GATE ignores target_sparsity; drop arg 6"; exit 1; }
  case "$GATE" in
    sigmoid) SUF="${SUF}_sig" ;;
    dcm)     SUF="${SUF}_dcm_d${DENSITY}"; DENSITY_ARG="--target-density $DENSITY" ;;
    *) echo "unknown GATE=$GATE"; exit 1 ;;
  esac
fi
if [ -n "$LR" ]; then
  SUF="${SUF}_lr${LR}"
  LR_ARG="--lr $LR"
else
  LR_ARG=""
fi
# The penalty only exists for the sigmoid gate; hard_concrete's Lagrangian owns sparsity.
if [ -n "$L1" ]; then
  [ "$GATE" = "sigmoid" ] || { echo "L1=$L1 requires GATE=sigmoid (hard_concrete uses target_sparsity)"; exit 1; }
  case "$L1_TARGET" in
    gate)  SUF="${SUF}_l1${L1}" ;;
    logit) SUF="${SUF}_l1logit${L1}" ;;
    *) echo "L1_TARGET must be gate or logit, got $L1_TARGET"; exit 1 ;;
  esac
  L1_ARG="--l1-coeff $L1 --l1-target $L1_TARGET"
else
  L1_ARG=""
fi
OUT=$L2A/results/eprun_${LEVEL}${SUF}
EVAL_OUT=$L2A/results/eprun_eval${SUF}
GRAPH=$OUT/graph_${TASK}_${MODEL}.json
cd "$L2A"

# EVAL_ONLY=1 reuses an already-trained $GRAPH and just rescores it (e.g. after fixing an
# eval flag), so a 40-minute llama3 training run is not repeated.
if [ "${EVAL_ONLY:-0}" = "1" ]; then
  [ -f "$GRAPH" ] || { echo "EVAL_ONLY=1 but $GRAPH is missing"; exit 1; }
  echo "EVAL_ONLY: reusing $GRAPH"
else
  PYTHONPATH=$MIB:$MIB/EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
    scripts/eval_mib_edge_pruning.py --mib-path "$MIB" \
    --model "$MODEL" --task "$TASK" --level "$LEVEL" --steps "$STEPS" \
    --split "$SPLIT" --output "$OUT" --loss "$LOSS" --gate "$GATE" $LR_ARG $L1_ARG $DENSITY_ARG $SPARSITY_ARG --skip-eval
fi

cd "$MIB"
PYTHONPATH=.:EAP-IG/src PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" \
  run_evaluation.py --models "$MODEL" --tasks "$TASK" --level "$LEVEL" \
  --ablation patching --split "$SPLIT" --method EdgePruning \
  --circuit-files "$GRAPH" --batch-size "$BS" $HEAD \
  --output-dir "$EVAL_OUT"
