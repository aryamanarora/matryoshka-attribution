#!/bin/bash
# MAttr + Adam at eps=1e-2, logit_diff, across every cell plots/plot_accauc_vs_faithauc.py
# (the SVA+ acc-vs-faith figure) needs, so the eps arm can be drawn as a series there.
#
# WHY THIS ARM EXISTS. scripts/sva/launch/submit_adam_eps_followup.sh established that Adam's eps is an
# IDENTITY knob at neuron scale, not a numerical guard: at 2.29M mask logits nearly every
# per-step |grad| exceeds the default 1e-8, so Adam's update degenerates to ~sign(g)*lr, every
# neuron takes the same size step regardless of effect size, and the learned score becomes a
# signed COUNT of steps. Raising eps above the typical |g| restores magnitude weighting. On
# addition/llama3/mlp that is worth acc-AUC 0.388 -> 0.490 and moves the top-k overlap with IG
# from 0.08 to 0.73. This wave asks whether that holds across the whole SVA+ grid, in the
# figure that the SVA+ optimiser claim is actually made on.
#
# ONE KNOB. Every flag below is the existing `topk:adam` arm of submit_sva_sweep.sh verbatim --
# same variant, same optimizer, same lr, same steps, same batch size, same k-schedule -- with
# `--adam-eps 1e-2` added and nothing else changed. So the new series and the "MAttr (log)"
# series already in the figure differ in exactly one number, which is the only reason a gap
# between them can be attributed to eps.
#
# LR IS 0.05, THE SWEEP'S SHARED PROTOCOL, and that is also this arm's own argmax on the cells
# where both LRs have been run -- eps=1e-2 at lr 0.05 vs 0.5 reads 0.702/0.698 (nounpp),
# 0.701/0.681 (rc), 0.676/0.675 (simple), 0.700/0.697 (within_rc) on the mlp substrate. So
# unlike the `topk:sgd` arm (which runs at 1.0, off-protocol, and carries that as a documented
# confound) this one needs no deviation: on-protocol and at its own optimum are the same choice.
# addition/mlp is the one measured cell where lr=0.5 wins, by 0.490 vs 0.483.
#
# logit_diff ONLY -- one point per panel, not the three-loss trajectory the other MAttr series
# draw. The eps result was established at logit_diff and the figure's dashed guide is skipped
# for a single-point series (see plot_accauc_vs_faithauc.SINGLE_LOSS). Add ce/acc here if the
# loss axis is ever wanted for this arm; it triples the wave.
#
# DIRS. All four of the figure's SOURCES, so the series is complete in every panel rather than
# appearing in some and being named in main()'s MISSING column in the rest:
#   sva_sweep        patched, -input   node + mlp + mlp+attn_head
#   sva_zeroabl      zero,    -input   node + mlp + mlp+attn_head
#   sva_sweep_input  patched, +input   node only (--include-input is node-only in the hooker)
#   sva_zeroabl_input zero,   +input   node only
# The per-position substrates carry SVA+Arith only; ARC-E/IOI are node-only and structurally so
# (variable length -> eval_sva's modal-length filter keeps 3.8% of ARC-E). That is exactly
# plot_accauc_vs_faithauc.REQUIRED, which drops any series missing a required task-group.
#
# THIS WAVE DEPENDS ON THE 2026-08-28 run_tag CHANGE. eval_sva.run_tag now encodes a non-default
# --adam-eps (`iso_topk_adam_eps1e-2_bs1`); before it did not, and these runs would have
# OVERWRITTEN the default-eps ones in the same dir. If the predicted filenames below come out
# without `_eps1e-2`, stop -- the tag change is missing and this script will destroy data.
#
# SIX OF THESE 72 CELLS WERE ALREADY ON DISK when this wave was first submitted, and the
# skip-if-exists check below could not see them -- they sat in per-config --output dirs under
# results/adamsgd_mlp/, which is the workaround that existed precisely BECAUSE run_tag did not
# encode eps. Same config in every field (eps 0.01, lr 0.05, logit_diff, patch, no
# include-input, 2000 steps, bs1, log k, topk/adam/iso, 100 eval examples), verified
# against each json's stored `config`:
#   A_eps/eps_1e-2_lr_0.05         addition / mlp
#   G_nounpp/eps_1e-2_lr_0.05      nounpp   / mlp
#   H_node/eps_1e-2_lr_0.05        addition / node
#   M_svaplus/{rc,simple,within_rc}_ld_lr_0.05      rc, simple, within_rc / mlp
# They were re-run rather than copied, which cost ~30 GPU-minutes and bought a check worth more
# than that: six independent repeats of the same config through the NEW tag path, so the wave
# validates itself end to end. Do not "optimise" this by teaching the script to hoover up the
# quarantine dirs -- those runs carry --train-eval-every probes and other per-arm flags this
# grid does not, and matching them by config field would be a second, weaker identity check
# living alongside the filename one.
#
#   bash scripts/sva/launch/submit_sva_eps.sh          # submit missing
#   DRY=1 bash scripts/sva/launch/submit_sva_eps.sh    # print, submit nothing
#   FORCE=1 bash scripts/sva/launch/submit_sva_eps.sh  # resubmit even if the json exists
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p logs

EPS=${EPS:-1e-2}            # spelled as eval_sva.eps_tag() normalises it, so the tag below is exact
LR=${LR:-0.05}
LOSS=${LOSS:-logit_diff}
# Job names must carry the loss: the QUEUED guard below matches on NAME, so submitting
# LOSS=ce and then LOSS=acc back-to-back would make the second wave see every name already in
# the queue and silently skip all 72 of its jobs. Empty for logit_diff so existing names are
# unchanged (and so the skip-if-already-queued check still works against runs in flight).
LTAG=""; [[ "$LOSS" != logit_diff ]] && LTAG="_$LOSS"

# task -> "model dataset". Same pins as submit_sva_sweep.sh / submit_input_replication.sh:
# ioi is qwen2.5, everything else llama3. plot_accauc_vs_faithauc.on_model enforces it when
# reading, so a run on the wrong model is silently dropped from the figure rather than averaged.
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith" [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
PERPOS=(nounpp rc simple within_rc addition months weekdays hours)   # SVA + Arith
NODEONLY=(arc_easy ioi)                                              # + the two MIB tasks

# Job names must carry the dir: the queued-name guard below matches on name, so the same cell
# in the patched and the zero dir would otherwise suppress each other. Same reason
# submit_input_replication.sh prefixes inp / inp0.
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
n=0; skip=0; qskip=0

sub() {  # $1=jobname $2=outdir $3=predicted-filename ; rest = eval_sva.py args
  local name=$1 out=$2 f=$3; shift 3
  if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); return; fi
  if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "  $name -> $f"
  else mkdir -p "$out"; sbatch -J "$name" scripts/sva/launch/sva_sweep.sbatch "$@" --output "$out" >/dev/null; fi
  n=$((n+1))
}

emit() {  # $1=task $2=nodes $3=outdir $4=ablation $5=include-input(0|1) $6=name-prefix
  local task=$1 nodes=$2 out=$3 abl=$4 inp=$5 pfx=$6
  local model ds; read -r model ds <<<"${CFG[$task]}"
  local nabbr=${nodes//+/-}
  # Mirror eval_sva.run_tag() EXACTLY: base, then _eps, then _loss (omitted for logit_diff),
  # then _zeroabl, then _bs1. A drift here does not fail loudly -- it makes the skip-if-exists
  # check miss and the wave resubmit everything, so it is written in run_tag's own order.
  local tag="iso_topk_adam_eps${EPS}"
  [[ "$LOSS" != logit_diff ]] && tag="${tag}_${LOSS}"
  [[ "$abl" != patch ]] && tag="${tag}_${abl}abl"
  tag="${tag}_bs1"
  local extra=(); [[ "$inp" == 1 ]] && extra=(--include-input)
  sub "${pfx}${LTAG}_${task}_${nabbr}" "$out" "$out/${task}_${model}_${nabbr}_${tag}.json" \
      --model "$model" --task "$task" --dataset "$ds" --nodes "$nodes" \
      --method mattr --variant topk --optimizer adam --k-schedule log --mode iso \
      --loss "$LOSS" --lr "$LR" --adam-eps "$EPS" \
      --train-batch-size 1 --steps 2000 --eval-examples 100 \
      --ablation "$abl" "${extra[@]}"
}

# -input: all three substrates. The two MIB tasks are node-only (see the header).
for spec in "results/sva_sweep patch eps" "results/sva_zeroabl zero eps0"; do
  read -r out abl pfx <<<"$spec"
  for task in "${PERPOS[@]}"; do
    for nodes in node mlp "mlp+attn_head"; do emit "$task" "$nodes" "$out" "$abl" 0 "$pfx"; done
  done
  for task in "${NODEONLY[@]}"; do emit "$task" node "$out" "$abl" 0 "$pfx"; done
done

# +input: node substrate only -- the hooker can score/ablate the input embedding only for the
# node type, which is why submit_input_replication.sh is node-only too.
for spec in "results/sva_sweep_input patch epsi" "results/sva_zeroabl_input zero epsi0"; do
  read -r out abl pfx <<<"$spec"
  for task in "${PERPOS[@]}" "${NODEONLY[@]}"; do emit "$task" node "$out" "$abl" 1 "$pfx"; done
done

echo "== ${DRY:+DRY }submitted $n, skipped $skip (done) + $qskip (queued) :: eps=$EPS lr=$LR loss=$LOSS =="
