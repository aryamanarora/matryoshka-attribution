#!/bin/bash
# Random-ranking baseline for every panel of fig:acc-faith (plots/plot_accauc_vs_faithauc.py).
#
# WHY THIS EXISTS. Random runs were only ever submitted for the two MIB tasks at the node
# substrate, and only in the two PATCHED dirs -- 12 runs total (2 tasks x 3 seeds x 2 dirs), the
# `random` arm of submit_input_replication.sh. The figure macro-averages over the panel's full
# required task-group set and drops any (method, loss, substrate) that is missing even one
# subtask, so 2-of-4 groups at node and 0-of-2 at mlp meant Random drew NOTHING in all 8 panels.
# This fills the other 68 cells so the baseline can be drawn as a floor in each.
#
#   dir                 substrates                  tasks  missing cells
#   sva_sweep           node, mlp, mlp+attn_head    10/8/8   8 + 8 + 8 = 24
#   sva_sweep_input     node                        10                  8
#   sva_zeroabl         node, mlp, mlp+attn_head    10/8/8  10 + 8 + 8 = 26
#   sva_zeroabl_input   node                        10                 10
#                                                              total   68  x3 seeds = 204 jobs
#
# Only the node substrate carries the two MIB tasks and the +input variant -- mlp and
# mlp+attn_head are per-POSITION layouts that filter to the modal prompt length (3.8% of ARC-E
# survives), which is why those cells are not run. Same structural fact the figure's REQUIRED
# dict encodes.
#
# CHEAP. --method random builds a random score vector and goes straight to the eval sweep; there
# is no training loop, so these cost a fraction of the 2000-step MAttr jobs that filled the same
# dirs. That is why 3 seeds per cell is affordable and worth it: the figure averages 4 groups,
# but SVA and Arith are themselves 4-task means, so per-task seed noise propagates.
#
# PROTOCOL matches whichever dir the cell lands in: --eval-examples 100 everywhere (the shared
# value in both submit_sva_sweep.sh and submit_input_replication.sh), --include-input for the
# _input dirs, --ablation zero for the zeroabl dirs. No --grad-examples: that flag only caps the
# all-layer gradient capture, and random takes no gradients.
#
# TAG. eval_sva.run_tag gives `random_s<seed>` plus the `_zeroabl` fragment when --ablation zero
# (it is appended after the loss fragment, and random has no loss fragment since it defaults to
# logit_diff). So the predicted filenames are random_s42.json and random_s42_zeroabl.json --
# which is what the skip-if-exists check below builds.
#
#   bash scripts/submit_random_baseline.sh          # submit missing
#   DRY=1 bash scripts/submit_random_baseline.sh    # print only
#   SEEDS="42" bash scripts/submit_random_baseline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
SEEDS=${SEEDS:-"42 43 44"}

# task -> "model dataset"
declare -A CFG=(
  [nounpp]="llama3 sva" [rc]="llama3 sva" [simple]="llama3 sva" [within_rc]="llama3 sva"
  [addition]="llama3 arith" [months]="llama3 arith"
  [weekdays]="llama3 arith" [hours]="llama3 arith"
  [arc_easy]="llama3 mib" [ioi]="qwen2.5 mib"
)
SVA_ARITH=(nounpp rc simple within_rc addition months weekdays hours)
NODE_TASKS=("${SVA_ARITH[@]}" arc_easy ioi)

# dir|ablation|include-input|job-prefix. Only the node substrate has a +input variant.
SOURCES=(
  "results/sva_sweep|patch|0|rnd"
  "results/sva_sweep_input|patch|1|rndi"
  "results/sva_zeroabl|zero|0|rnd0"
  "results/sva_zeroabl_input|zero|1|rnd0i"
)

# Job names already queued. The skip check below only sees FINISHED runs, so a partial
# submission would resubmit everything still pending and race two jobs onto the same json.
# Names are unique per (dir, substrate, task, seed), so matching on name is exact.
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

n=0; skip=0; qskip=0
sub() {  # $1=jobname $2=out-dir $3=predicted-filename-tag ; rest = eval_sva args
  local name=$1 out=$2 tag=$3; shift 3
  local task model nodes
  task=$(grep -oP '(?<=--task )\S+' <<<"$*")
  model=$(grep -oP '(?<=--model )\S+' <<<"$*")
  nodes=$(grep -oP '(?<=--nodes )\S+' <<<"$*" | tr '+' '-')
  local f="$out/${task}_${model}_${nodes}_${tag}.json"
  if [[ "${FORCE:-0}" != 1 && -f "$f" ]]; then skip=$((skip+1)); return; fi
  if [[ "${FORCE:-0}" != 1 ]] && grep -qxF "$name" <<<"$QUEUED"; then qskip=$((qskip+1)); return; fi
  if [[ "${DRY:-0}" == 1 ]]; then echo "sbatch -J $name ... $out/${task}_${model}_${nodes}_${tag}.json"
  else sbatch -J "$name" sva_sweep.sbatch "$@" >/dev/null; fi
  n=$((n+1))
}

for src in "${SOURCES[@]}"; do
  IFS='|' read -r OUT ABL INP PFX <<< "$src"
  mkdir -p "$OUT"
  ab=(); tagsuf=""
  [[ "$ABL" != patch ]] && { ab=(--ablation "$ABL"); tagsuf="_${ABL}abl"; }
  ii=(); [[ "$INP" == 1 ]] && ii=(--include-input)
  # +input exists for the node substrate only (see header)
  subs=(node); [[ "$INP" == 1 ]] || subs=(node mlp "mlp+attn_head")
  for nodes in "${subs[@]}"; do
    nabbr=${nodes//+/-}
    tasks=("${SVA_ARITH[@]}"); [[ "$nodes" == node ]] && tasks=("${NODE_TASKS[@]}")
    for task in "${tasks[@]}"; do
      read -r model ds <<< "${CFG[$task]}"
      for seed in $SEEDS; do
        sub "${PFX}_${task}_${nabbr}_s${seed}" "$OUT" "random_s${seed}${tagsuf}" \
          --model "$model" --task "$task" --dataset "$ds" --nodes "$nodes" \
          --method random --seed "$seed" --eval-examples 100 \
          "${ab[@]}" "${ii[@]}" --output "$OUT"
      done
    done
  done
done
echo "submitted $n, skipped $skip (done) + $qskip (already queued)"
