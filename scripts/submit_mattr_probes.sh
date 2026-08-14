#!/bin/bash
# Probes into "why is MAttr weak at the neuron (mlp) substrate on arithmetic?".
#
#   A) steps  -- is it simply under-converged? 8k and 16k steps against the 2k headline.
#   B) lr     -- or is 0.05 just too small a step for a 2.3M-unit score vector? lr 0.1/0.3/1.0.
#   C) lr_low -- B came back monotonically NEGATIVE, so 0.05 is bracketed only from above.
#                lr 0.01/0.02, and at 8000 steps too (a smaller step needs a longer budget,
#                so the two knobs interact and the 2k grid alone could hide the optimum).
#
# Usage: scripts/submit_mattr_probes.sh C          # one or more arm letters, no default.
# Arms A and B are DONE (see the results table in git log / wandb); re-running one overwrites
# its results dir, so the arm is an explicit argument rather than "submit everything".
#
# All carry a control at the far end of the effect (nounpp, where MAttr already matches IG):
# if the SVA cell moves too, the knob is a global under-tuning and every existing
# fingerprint_mlp*.tex number is understated -- not an arithmetic-specific finding.
#
# WHY EACH RUN GETS ITS OWN --output DIR: the filename tag encodes only knobs that change the
# circuit's identity (see eval_sva.run_tag), and --lr is not one of them. All three lr values
# of a variant therefore write the SAME filename; the first submission of this sweep did
# exactly that and the lr=0.1 results were overwritten before anyone read them. --steps IS in
# the tag (as _s8000), so the steps arm could share a dir -- it gets per-arm dirs anyway so the
# two arms read alike. eval_sva.py now also records the full argv under out["config"], so a
# future collision is at least detectable after the fact.
#
# Every run gets the k-independent training probe (--train-eval-every) -- the whole question is
# "has it converged", and the train LOSS cannot answer that because it is measured at a k that
# moves over training. Read probe/acc_auc in wandb (project l2a-arith / l2a-sva), not the loss.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--nodes mlp --loss acc --train-batch-size 1 --train-eval-every 250 --train-eval-examples 16"
sub() { sbatch -J "$1" sva_sweep.sbatch "${@:2}"; }

# ---- A) steps: headline lr, 4x and 8x the step budget --------------------------------------
arm_A() {
for S in 8000 16000; do
  sub "conv_add_stopk_$S"  --dataset arith --task addition $COMMON --variant topk --optimizer adam \
      --lr 0.05 --steps $S --output "results/probe_steps/add_stopk_$S"
  sub "conv_add_idste_$S"  --dataset arith --task addition $COMMON --variant hard_topk_identity \
      --optimizer sgd --lr 0.05 --steps $S --output "results/probe_steps/add_idste_$S"
done
# SVA control: does the agreement cell gain from 4x steps too?
sub "conv_nounpp_stopk_8000" --task nounpp $COMMON --variant topk --optimizer adam \
    --lr 0.05 --steps 8000 --output "results/probe_steps/nounpp_stopk_8000"
}

# ---- B) lr: headline step budget, 2x/6x/20x the learning rate ------------------------------
arm_B() {
for LR in 0.1 0.3 1.0; do
  sub "lr_add_stopk_$LR" --dataset arith --task addition $COMMON --variant topk --optimizer adam \
      --lr $LR --output "results/probe_lr/add_stopk_$LR"
  sub "lr_add_soft_$LR"  --dataset arith --task addition $COMMON --variant hard_topk --optimizer adam \
      --lr $LR --output "results/probe_lr/add_soft_$LR"
done
# idSTE+SGD is the documented lr-INVARIANT variant (accumulated ranking): it should not move.
# If it does, that invariance claim is wrong and the ablation table's framing needs revisiting.
sub "lr_add_idste_0.3" --dataset arith --task addition $COMMON --variant hard_topk_identity \
    --optimizer sgd --lr 0.3 --output "results/probe_lr/add_idste_0.3"
# SVA control, as above.
sub "lr_nounpp_stopk_0.3" --task nounpp $COMMON --variant topk --optimizer adam \
    --lr 0.3 --output "results/probe_lr/nounpp_stopk_0.3"
}

# ---- C) lr BELOW the headline, crossed with the step budget --------------------------------
# Arm B's outcome (test acc-AUC on addition/mlp/acc, vs 0.366 at the headline lr=0.05):
# stopk 0.1 -> 0.282, 0.3 -> 0.280, 1.0 -> 0.175. Monotone down. Nothing there rules out the
# optimum sitting BELOW 0.05, and arm A showed the same cell is still climbing at 16k steps --
# so a smaller step may simply need more of them. Hence the 2k x 8k crossing rather than a
# flat lr grid: 8 jobs, exactly the guests account's gres/gpu=8 ceiling, so the arm runs as
# one wave.
arm_C() {
for LR in 0.01 0.02; do
  # 2000 steps: directly comparable to the headline grid (stopk 0.366, soft 0.372).
  sub "lrlo_add_stopk_$LR" --dataset arith --task addition $COMMON --variant topk --optimizer adam \
      --lr $LR --output "results/probe_lr_low/add_stopk_${LR}_s2000"
  sub "lrlo_add_soft_$LR"  --dataset arith --task addition $COMMON --variant hard_topk --optimizer adam \
      --lr $LR --output "results/probe_lr_low/add_soft_${LR}_s2000"
  # 8000 steps: the interaction test, against 0.05@8000 = 0.425 (and 0.05@16000 = 0.453).
  sub "lrlo_add_stopk_${LR}_8k" --dataset arith --task addition $COMMON --variant topk \
      --optimizer adam --lr $LR --steps 8000 --output "results/probe_lr_low/add_stopk_${LR}_s8000"
done
# idSTE+SGD invariance control, this time in the DOWNWARD direction (0.3 already checked, +0.009).
sub "lrlo_add_idste_0.01" --dataset arith --task addition $COMMON --variant hard_topk_identity \
    --optimizer sgd --lr 0.01 --output "results/probe_lr_low/add_idste_0.01"
# SVA control, as in arms A and B (0.05 -> 0.663).
sub "lrlo_nounpp_stopk_0.01" --task nounpp $COMMON --variant topk --optimizer adam \
    --lr 0.01 --output "results/probe_lr_low/nounpp_stopk_0.01"
}

# ---- D) confirm arm C's winner ------------------------------------------------------------
# Arm C's result: lr 0.02 x 8000 steps = 0.465, beating BOTH 0.05@8000 (0.425) and 0.05@16000
# (0.453) -- the latter at half the budget -- and its probe is flat over the last 1000 steps
# where 0.05's was still climbing. Three things that single number does not establish:
#   1. is +0.040 above seed noise? -> paired seed-1 replicates of the winner AND its reference.
#   2. does the winner transfer to the other variants, or is 0.02 a topk-specific fluke?
#   3. arm C's SVA control FELL hard at 0.01 (0.663 -> 0.529) with a probe still climbing at
#      the buzzer -- budget-starved, or does low lr genuinely hurt the cell where MAttr
#      already wins? Only a longer budget separates those.
arm_D() {
sub "d_add_stopk_0.02_16k" --dataset arith --task addition $COMMON --variant topk --optimizer adam \
    --lr 0.02 --steps 16000 --output "results/probe_lr_low/add_stopk_0.02_s16000"
# (1) paired replicates: winner and reference, same new seed, 8k both.
for L in 0.02 0.05; do
  sub "d_add_stopk_${L}_8k_s1" --dataset arith --task addition $COMMON --variant topk \
      --optimizer adam --lr $L --steps 8000 --seed 1 \
      --output "results/probe_lr_low/add_stopk_${L}_s8000_seed1"
done
# (2) the other two variants at the winning lr, same 8k budget. The +hard ablation has to be
# retuned alongside the headline or the ablation table compares tuned against untuned.
sub "d_add_soft_0.02_8k"  --dataset arith --task addition $COMMON --variant hard_topk \
    --optimizer adam --lr 0.02 --steps 8000 --output "results/probe_lr_low/add_soft_0.02_s8000"
sub "d_add_idste_0.01_8k" --dataset arith --task addition $COMMON --variant hard_topk_identity \
    --optimizer sgd --lr 0.01 --steps 8000 --output "results/probe_lr_low/add_idste_0.01_s8000"
sub "d_add_idste_0.02_8k" --dataset arith --task addition $COMMON --variant hard_topk_identity \
    --optimizer sgd --lr 0.02 --steps 8000 --output "results/probe_lr_low/add_idste_0.02_s8000"
# (3) the SVA control at 8k, at both low lrs.
for L in 0.01 0.02; do
  sub "d_nounpp_stopk_${L}_8k" --task nounpp $COMMON --variant topk --optimizer adam \
      --lr $L --steps 8000 --output "results/probe_lr_low/nounpp_stopk_${L}_s8000"
done
}

[ $# -gt 0 ] || { echo "usage: $0 <arm>...   (arms: A steps, B lr, C lr_low, D confirm)" >&2; exit 2; }
for arm in "$@"; do
  case "$arm" in
    A|B|C|D) echo "== arm $arm =="; "arm_$arm" ;;
    *) echo "unknown arm '$arm' (expected A, B, C or D)" >&2; exit 2 ;;
  esac
done
