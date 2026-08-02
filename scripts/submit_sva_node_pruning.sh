#!/bin/bash
# Node Pruning (Bhaskar et al., 2024) on the SVA-sweep harness, so it appears as a series in
# plots/accauc_vs_faithauc.pdf alongside IG, I×G and MAttr.
#
# This is NOT the MIB Node Pruning baseline re-scored. It is the same hard-concrete + L0
# recipe run through eval_sva.py's own loss_fn, which means it shares MAttr's objective and
# substrate exactly and differs ONLY in how the mask is parameterized (annealed L0 budget vs
# top-k). That is the comparison the figure is about. Consequence: `s` here is a fraction of
# the SVA substrate (thousands of MLP neurons, or +attn heads), not of MIB's ~156 nodes, so
# it is not the same absolute circuit size as the results/eprun_node_s0.9 rows in the tables.
#
# The grid is DERIVED from the MAttr headline runs already on disk (sufficient_topk_adam*_bs1,
# excluding uniformk/ig variants) rather than restated here, so Node Pruning lands on exactly
# the cells the figure averages over and cannot drift out of sync with them. 60 cells:
#   -input : mlp x 4 SVA, mlp+attn_head x 4 SVA, node x (4 SVA + arc_easy + ioi)
#   +input : node x (4 SVA + arc_easy + ioi)
#   x 3 losses (ce, acc, logit_diff) -- the loss axis is the shape axis in that figure.
# Note the model varies by task (ioi is qwen2.5, everything else llama3); that too is read
# off the MAttr json rather than assumed.
#
#   bash scripts/submit_sva_node_pruning.sh            # submit missing cells, s=0.9
#   S=0.8 bash scripts/submit_sva_node_pruning.sh      # a different budget
#   DRY=1 bash scripts/submit_sva_node_pruning.sh      # print, submit nothing
#   FORCE=1 ...                                        # resubmit even if the json exists
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
S=${S:-0.9}
STEPS=${STEPS:-2000}          # matches MATTR_COMMON in submit_sva_sweep.sh -- keep it matched
# Known caveat, do NOT "fix" it by raising STEPS here: the L0 Lagrangian anneals its target
# over the run, and on this substrate (~10^5-10^6 units, vs MIB's ~157 nodes) the gates do not
# fully track a 2000-step anneal. Measured on qwen2.5/nounpp/mlp (583,680 units), target 0.9:
#   steps=600  -> achieved 0.709,  faith AUC 0.527, acc-AUC 0.154
#   steps=2000 -> achieved 0.842,  faith AUC 0.999, acc-AUC 0.209
#   steps=4000 -> achieved 0.892,  faith AUC 1.302, acc-AUC 0.263
# So s=0.9 here means "annealed toward 0.9", and the achieved value is what the run should be
# reported as. Every method in this comparison gets the same step budget; handing the baseline
# 2x the optimization steps would break the comparison in the other direction, which matters
# more than the baseline hitting its nominal target exactly.
TAG="eprun_s$(printf '%03d' "$(python3 -c "print(round($S*100))")")"

# (sweep_dir, nodes, task, model, dataset, loss) for every headline-MAttr cell on disk.
GRID=$(.venv/bin/python - <<'EOF'
import glob, json, os
seen = set()
for res in ("results/sva_sweep", "results/sva_sweep_input"):
    for f in glob.glob(res + "/*_sufficient_topk_adam*_bs1.json"):
        mid = os.path.basename(f).split("_bs1")[0].split("adam", 1)[1]
        if "uniformk" in mid or "ig" in mid:      # ablations, not the headline series
            continue
        d = json.load(open(f))
        seen.add((res, d["nodes"], d["task"], d["model"], d["loss"]))
for row in sorted(seen):
    print(" ".join(row))
EOF
)

n=0; skip=0
while read -r res nodes task model loss; do
  [ -z "$res" ] && continue
  # eval_sva.py appends _<loss> to the tag for anything but logit_diff (its default).
  fulltag=$TAG; [ "$loss" != "logit_diff" ] && fulltag="${TAG}_${loss}"
  out="$res/${task}_${model}_${nodes//+/-}_${fulltag}.json"
  if [ "${FORCE:-0}" != "1" ] && [ -f "$out" ]; then skip=$((skip+1)); continue; fi
  # arc_easy/ioi come from MIB, the four SVA tasks from the SVA dataset; +input is the
  # sva_sweep_input dir and is the only place --include-input is passed.
  case "$task" in arc_easy|ioi) ds=mib ;; *) ds=sva ;; esac
  extra=(); [ "$res" = "results/sva_sweep_input" ] && extra=(--include-input)
  name="npsva_${task}_${nodes//+/-}_${loss}"
  if [ "${DRY:-0}" = "1" ]; then
    # Keep this in sync with the real sbatch below -- a preview that hides --steps is how a
    # step-count change ships unnoticed.
    echo "sbatch -J $name sva_sweep.sbatch --model $model --task $task --dataset $ds --nodes $nodes --method edge_pruning --loss $loss --target-sparsity $S --mode sufficient --train-batch-size 1 --steps $STEPS --eval-examples 100 ${extra[*]-} --output $res"
  else
    sbatch -J "$name" sva_sweep.sbatch \
      --model "$model" --task "$task" --dataset "$ds" --nodes "$nodes" \
      --method edge_pruning --loss "$loss" --target-sparsity "$S" \
      --mode sufficient --train-batch-size 1 --steps "$STEPS" --eval-examples 100 \
      "${extra[@]+"${extra[@]}"}" --output "$res" >/dev/null
  fi
  n=$((n+1))
done <<< "$GRID"
echo "== ${DRY:+DRY }$n Node Pruning SVA jobs at s=$S ($skip already done) =="
