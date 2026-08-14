#!/bin/bash
# Two probes into "why is MAttr weak at the neuron (mlp) substrate on arithmetic?".
#
#   A) steps  -- is it simply under-converged? 8k and 16k steps against the 2k headline.
#   B) lr     -- or is 0.05 just too small a step for a 2.3M-unit score vector? lr 0.1/0.3/1.0.
#
# Both carry a control at the far end of the effect (nounpp, where MAttr already matches IG):
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
for S in 8000 16000; do
  sub "conv_add_stopk_$S"  --dataset arith --task addition $COMMON --variant topk --optimizer adam \
      --lr 0.05 --steps $S --output "results/probe_steps/add_stopk_$S"
  sub "conv_add_idste_$S"  --dataset arith --task addition $COMMON --variant hard_topk_identity \
      --optimizer sgd --lr 0.05 --steps $S --output "results/probe_steps/add_idste_$S"
done
# SVA control: does the agreement cell gain from 4x steps too?
sub "conv_nounpp_stopk_8000" --task nounpp $COMMON --variant topk --optimizer adam \
    --lr 0.05 --steps 8000 --output "results/probe_steps/nounpp_stopk_8000"

# ---- B) lr: headline step budget, 2x/6x/20x the learning rate ------------------------------
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
