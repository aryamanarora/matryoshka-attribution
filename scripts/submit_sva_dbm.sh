#!/bin/bash
# The pyvene sigmoid-mask baseline (DBM in the paper) on the SVA-sweep harness, so it appears
# as a series wherever Node Pruning already does -- accauc_vs_faithauc, the fingerprint tables
# and the top-neuron table.
#
# This is NOT the MIB DBM baseline re-scored. It is the same SigmoidMaskIntervention recipe run
# through eval_sva.py's own loss_fn, so it shares MAttr's objective, substrate and step budget
# and differs ONLY in how the mask is parameterized (deterministic sigmoid(mask/temp) with the
# temperature annealed 50 -> 0.1, vs top-k). That is the comparison the figure is about.
#
# RECIPE: lr 0.3, L1 6.0 on the gate -- the argmax of the MIB validation sweep
# (results/eprun_eval_ld_sig_lr0.3_l16.0, avg CPR 1.50 vs 1.31 unpenalised), taken across
# as-is rather than re-tuned here. Two things to know about that transfer:
#   - the L1 term is coeff * z.mean(), i.e. normalised by substrate size, so 6.0 means the same
#     pressure per gate on 10^6 SVA units as on MIB's 156--1056 nodes. That normalisation is the
#     reason the coefficient is portable at all.
#   - lr is NOT scale-free in the same way. 0.3 was the argmax over {0.1, 0.3, 1.0} on three
#     cheap MIB cells, a plateau within noise; it is a defensible default here, not a tuned one.
# Re-sweeping lr on this substrate would be the honest upgrade if these rows end up load-bearing.
#
# The grid is DERIVED from the MAttr headline runs already on disk, exactly as
# submit_sva_node_pruning.sh derives its own, so DBM lands on precisely the cells the figures
# average over and cannot drift out of sync with the other two mask learners. 60 cells:
#   -input : mlp x 4 SVA, mlp+attn_head x 4 SVA, node x (4 SVA + arc_easy + ioi)
#   +input : node x (4 SVA + arc_easy + ioi)
#   x 3 losses (ce, acc, logit_diff) -- the loss axis is the shape axis in that figure.
# Model varies by task (ioi is qwen2.5, everything else llama3), read off the MAttr json.
#
#   bash scripts/submit_sva_dbm.sh                 # submit missing cells at the recipe above
#   LR=0.1 L1=6.0 bash scripts/submit_sva_dbm.sh   # a different point (its own tag + files)
#   L1=0 bash scripts/submit_sva_dbm.sh            # the unpenalised pyvene-library recipe
#   DRY=1 bash scripts/submit_sva_dbm.sh           # print, submit nothing
#   FORCE=1 ...                                    # resubmit even if the json exists
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LR=${LR:-0.3}
L1=${L1:-6.0}
STEPS=${STEPS:-2000}          # matches MATTR_COMMON in submit_sva_sweep.sh and the Node Pruning
                              # script -- every method in this comparison gets the same budget.
# Mirrors eval_sva.py's own tag construction (--method sigmoid_mask), which spells the knobs
# with str(float): L1=6 would give "_l16" here but "_l16.0" there, the skip check below would
# never match, and every run would resubmit forever. Normalising both through Python's own
# float repr makes the two spellings agree by construction rather than by discipline.
read -r LR L1 <<< "$(.venv/bin/python -c "print(float('$LR'), float('$L1'))")"
TAG="sig_lr${LR}"
[ "$L1" != "0.0" ] && TAG="${TAG}_l1${L1}"

# (sweep_dir, nodes, task, model, loss) for every headline-MAttr cell on disk.
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
  name="dbmsva_${task}_${nodes//+/-}_${loss}"
  if [ "${DRY:-0}" = "1" ]; then
    # Keep this in sync with the real sbatch below -- a preview that hides --steps or --lr is
    # how a recipe change ships unnoticed.
    echo "sbatch -J $name sva_sweep.sbatch --model $model --task $task --dataset $ds --nodes $nodes --method sigmoid_mask --loss $loss --lr $LR --l1-coeff $L1 --mode sufficient --train-batch-size 1 --steps $STEPS --eval-examples 100 ${extra[*]-} --output $res"
  else
    sbatch -J "$name" sva_sweep.sbatch \
      --model "$model" --task "$task" --dataset "$ds" --nodes "$nodes" \
      --method sigmoid_mask --loss "$loss" --lr "$LR" --l1-coeff "$L1" \
      --mode sufficient --train-batch-size 1 --steps "$STEPS" --eval-examples 100 \
      "${extra[@]+"${extra[@]}"}" --output "$res" >/dev/null
  fi
  n=$((n+1))
done <<< "$GRID"
echo "== ${DRY:+DRY }$n DBM SVA jobs at lr=$LR l1=$L1 ($skip already done) =="
