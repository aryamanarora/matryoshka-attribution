#!/bin/bash
# Stage 1a of the interference-weights replication, as ONE slurm job: the cluster gives this
# account 2 concurrent nodes and every GPU is busy, so eight 40-second jobs cost eight waits
# in the queue while one job costs one.
#   sbatch -J vw_s1 scripts/vw/launch/vw.sbatch scripts/vw_stage1.sh
#
# 1. sweep the ONE knob the note does not state (learning rate, and whether the schedule it
#    states for its transcoder also applies to the transformer), 8 cells x ~40s
# 2. pick the cell whose held-out loss is closest to the note's published 3.38, link it as
#    results/vw/base, and record the whole sweep
set -euo pipefail
cd ~/learning-to-attribute
PY=.venv/bin/python

for lr in 1e-3 2e-3 3e-3; do
  $PY scripts/vw/vw_train.py --lr $lr                                --out results/vw/lr_$lr
  $PY scripts/vw/vw_train.py --lr $lr --decay-frac 0.2               --out results/vw/lrd_$lr
done
$PY scripts/vw/vw_train.py --lr 3e-3 --warmup 500 --decay-frac 0.2   --out results/vw/lrwd_3e-3
$PY scripts/vw/vw_train.py --lr 6e-3 --warmup 500 --decay-frac 0.2   --out results/vw/lrwd_6e-3

$PY - <<'PYEOF'
import json, pathlib, shutil
runs = sorted(pathlib.Path("results/vw").glob("lr*/log.json"))
rows = [(json.loads(p.read_text()), p.parent) for p in runs]
rows.sort(key=lambda r: r[0]["val_loss_full"])
print("\n=== LR sweep (the note: train 3.33, test 3.38) ===")
for d, p in rows:
    print(f"  {p.name:<14} lr {d['args']['lr']:<7} decay {d['args']['decay_frac']:<5} "
          f"warmup {d['args']['warmup']:<5} train {d['train_loss_tail20']:.4f} "
          f"val {d['val_loss_full']:.4f}")
best = rows[0][1]
base = pathlib.Path("results/vw/base")
if base.is_symlink() or base.exists():
    base.unlink() if base.is_symlink() else shutil.rmtree(base)
base.symlink_to(best.name)
print(f"\nbase -> {best.name}")
PYEOF

