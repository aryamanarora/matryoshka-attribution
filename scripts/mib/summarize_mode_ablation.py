"""Objective ablation of the headline MAttr (soft top-k, uniform k, Adam) on MIB node level.

    uv run python scripts/mib/summarize_mode_ablation.py [--splits validation test]

Prints, per split, CPR AUC (`area_under`) and acc-AUC (`acc_auc`) for the headline dir (iso =
denoising objective, what CPR measures), the `--mode cause` run (noising /
noising) and the `--mode joint` run (per-step coin flip, losses.resolve_direction), cell by cell
over make_mib_table.COLUMNS, with the average over the cells every row has. Dirs are the ones
scripts/mib/launch/submit_mib_node_mode_ablation_sc.sh writes; the headline dirs come from
mattr_variants via the test table / validation table constants.
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mib_table as M  # noqa: E402

DIRS = {
    "validation": [("iso (headline)", "mib_node_topk_uniform_lr05"),
                   ("cause", "mib_node_cause_topk_uniform_lr05"),
                   ("joint", "mib_node_joint_topk_uniform_lr05")],
    "test": [("iso (headline)", "test_node_topk_uniform_lr05"),
             ("cause", "test_node_cause_topk_uniform_lr05"),
             ("joint", "test_node_joint_topk_uniform_lr05")],
}


def read(d, task, model, split):
    p = M.RESULTS_BASE / d / f"{task}_{model}_{split}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        r = pickle.load(f)
    return r.get("area_under"), r.get("acc_auc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["validation", "test"])
    a = ap.parse_args()
    cols = [(t, m) for t, m, _ in M.COLUMNS]
    for split in a.splits:
        rows = {name: {c: read(d, *c, split) for c in cols} for name, d in DIRS[split]}
        shared = [c for c in cols if all(rows[n][c] is not None for n in rows)]
        print(f"\n=== {split}: {len(shared)}/{len(cols)} cells present in every row ===")
        for key, label in ((0, "CPR AUC"), (1, "acc-AUC")):
            print(f"\n{label:<16}" + "".join(f"{t[:6]}/{m[:5]:<7}" for t, m in cols) + "  avg(shared)")
            for name in rows:
                cells = []
                for c in cols:
                    v = rows[name][c]
                    cells.append("   ---       " if v is None or v[key] is None else f"{v[key]:8.2f}     ")
                vals = [rows[name][c][key] for c in shared if rows[name][c][key] is not None]
                avg = sum(vals) / len(vals) if vals else float("nan")
                print(f"{name:<16}" + "".join(c[:13] for c in cells) + f"  {avg:.3f}")


if __name__ == "__main__":
    main()
