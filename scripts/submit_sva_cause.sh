#!/bin/bash
# Cause-trained (noising) MAttr on the SVA-sweep tasks -- the missing control for every
# cause-direction number we report.
#
# WHY: all 720 runs in results/sva_sweep{,_input} are --mode sufficient, i.e. the mask is
# TRAINED to keep the top-k clean (denoising) and then evaluated in BOTH directions. So the
# published "cause"/noising comparison (where IG beats MAttr on cause AUC, 0.067 vs 0.093)
# pits a sufficiency-trained mask against gradient scores in a direction it never trained
# for. This script trains the SAME method with --mode necessary so the comparison is
# like-for-like.
#
# MAttr ONLY, on purpose: --mode never reaches gradient_scores (it is consumed at
# eval_sva.py:418 -> resolve_direction inside the MAttr training loop), so IG / IxG /
# conductance rankings are byte-identical under --mode necessary and re-running them would
# only produce differently-named copies. Evaluation is direction-independent too: summarize()
# sets hooker.sufficient explicitly per call, and F_clean/F_patch always use sufficient=False.
#
# Headline variant only (soft top-k forward + log-k, = the MAttr headline per CLAUDE.md).
# 6 tasks x 3 losses = 18 jobs. The "+hard" ablation would be another 18 via VARIANTS=...
#
# Results -> results/sva_sweep_cause/ (a SEPARATE dir: the mode is in the json `mode` field and
# in the filename tag `necessary_...`, but a separate dir also keeps these out of every glob
# that feeds the sufficiency figures. plot_accauc_vs_faithauc.parse_method now returns None for
# `necessary_*` rather than silently labelling it "IG" -- do not undo that.)
#
#   bash scripts/submit_sva_cause.sh          # submit missing jobs
#   DRY=1 bash scripts/submit_sva_cause.sh    # print, submit nothing
#   FORCE=1 bash scripts/submit_sva_cause.sh  # resubmit even if output exists
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/sva_sweep_cause

OUT=results/sva_sweep_cause
LOSSES=(ce acc logit_diff)
# variant:optimizer. topk = soft top-k forward (headline). Override to add the ablation:
#   VARIANTS="topk:adam hard_topk:adam" bash scripts/submit_sva_cause.sh
read -ra VARIANTS <<< "${VARIANTS:-topk:adam}"
STEPS=${STEPS:-2000}
COMMON=(--mode necessary --k-schedule log --train-batch-size 1 --steps "$STEPS" \
        --lr 0.05 --eval-examples 100)

# task:model:dataset -- ioi is qwen2.5, everything else llama3 (matches the sufficiency sweep,
# and is why any task-average here spans two models). MIB tasks are node-substrate only.
TASKS=(nounpp:llama3:sva rc:llama3:sva simple:llama3:sva within_rc:llama3:sva
       arc_easy:llama3:mib ioi:qwen2.5:mib)

# Reproduce eval_sva.py's output tag (line 677: f"{mode}_{variant}_{optimizer}") so finished
# configs are skipped. k-schedule log adds no suffix; bs1 always.
mattr_tag() {   # $1=variant $2=optimizer $3=loss
  local t="necessary_$1_$2"
  [[ "$3" != "logit_diff" ]] && t="${t}_$3"
  echo "${t}_bs1"
}

n=0; skip=0
for spec in "${TASKS[@]}"; do
  IFS=: read -r task model dataset <<< "$spec"
  for cfg in "${VARIANTS[@]}"; do
    variant=${cfg%:*}; opt=${cfg#*:}
    # job-name abbreviation only (the data-bearing tag is mattr_tag). Deliberately NOT the
    # legacy submit_sva_sweep.sh mapping, which calls hard_topk "soft" (after its sigmoid STE)
    # -- that reads as the soft-forward headline in squeue and is a trap.
    vabbr=$(case "$variant" in
              topk) echo stopk;;                 # soft top-k forward = headline
              hard_topk) echo hste;;             # hard forward, sigmoid STE = "+hard"
              hard_topk_identity) echo idste;;   # hard forward, identity STE
              *) echo "$variant";; esac)
    for loss in "${LOSSES[@]}"; do
      tag=$(mattr_tag "$variant" "$opt" "$loss")
      f="$OUT/${task}_${model}_node_${tag}.json"
      if [[ "${FORCE:-0}" != "1" && -f "$f" ]]; then skip=$((skip+1)); continue; fi
      args=(--model "$model" --task "$task" --dataset "$dataset" --nodes node
            --method mattr --loss "$loss" --variant "$variant" --optimizer "$opt"
            "${COMMON[@]}" --output "$OUT")
      name="cause_${task}_${vabbr}_${loss}"
      if [[ "${DRY:-0}" == "1" ]]; then echo "sbatch -J $name sva_sweep.sbatch ${args[*]}"
      else sbatch -J "$name" sva_sweep.sbatch "${args[@]}"; fi
      n=$((n+1))
    done
  done
done
echo "submitted $n, skipped $skip (already present) -> $OUT"
