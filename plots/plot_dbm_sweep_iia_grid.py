"""Why the DBM ladder's `iia` is not an acc-AUC: accuracy vs sparsity for MAttr (measured at
MIB's ten fixed proportions) against the ladder's frontier (measured only at its own rungs'
achieved L0 and linearly interpolated between them by the log-trapezoid).

    uv run python plots/plot_dbm_sweep_iia_grid.py [--tasks arc_easy arc_challenge] [--model llama3]
    -> plots/dbm_sweep_iia_grid.png / .pdf

Per panel: MAttr's `accuracies` from results/test_node_topk_uniform_lr05 (headline, test split),
the ladder's rung points from results/dbm_multisparsity/<task>_<model>_test.json with the
interpolation its IIA integrates, and the single lambda=6 rung scored on MIB's grid
(results/eprun_eval_ld_sig_lr0.3_l16.0). The shaded strip is the region below the sparsest rung,
where the ladder measures nothing and the interpolation credits accuracy anyway.
"""
import argparse
import json
import math
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
GRID = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)


def log_auc(xs, ys):
    lx = [math.log(x) for x in xs]
    return sum((lx[i + 1] - lx[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)) / (lx[-1] - lx[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["arc_easy", "arc_challenge"])
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--out", default=str(ROOT / "plots" / "dbm_sweep_iia_grid"))
    a = ap.parse_args()

    fig, axes = plt.subplots(1, len(a.tasks), figsize=(4.6 * len(a.tasks), 3.6), sharey=True)
    axes = [axes] if len(a.tasks) == 1 else list(axes)
    for ax, task in zip(axes, a.tasks):
        m = pickle.load(open(R / "test_node_topk_uniform_lr05" / f"{task}_{a.model}_test.pkl", "rb"))
        j = json.load(open(R / "dbm_multisparsity" / f"{task}_{a.model}_test.json"))
        p6 = (R / "eprun_eval_ld_sig_lr0.3_l16.0" / "EdgePruning_patching_node"
              / f"{task.replace('_', '-')}_{a.model}_test_abs-False.pkl")
        d6 = pickle.load(open(p6, "rb")) if p6.exists() else None

        xs, ys = list(j["percentages"]), list(j["acc_curve"])
        ax.axvspan(xs[0], xs[1], color="0.9", zorder=0,
                   label="unmeasured by the ladder" if ax is axes[0] else None)
        ax.plot(GRID, m["accuracies"], "o-", color="#1f77b4", ms=5,
                label=f"MAttr, MIB grid  (acc-AUC {m['acc_auc']:.2f})")
        if d6 is not None and d6.get("acc_auc") is not None:
            ax.plot(GRID, d6["accuracies"], "s-", color="#d62728", ms=4, alpha=.8,
                    label=f"DBM $\\lambda$=6, MIB grid  (acc-AUC {d6['acc_auc']:.2f})")
        ax.plot(xs, ys, "--", color="#ff7f0e",
                label=f"DBM ladder, interpolated  (IIA {j['iia']:.2f})")
        ax.plot(xs[1:-1], ys[1:-1], "D", color="#ff7f0e", ms=6, label="ladder rungs (own L0)")
        ax.plot([xs[0], xs[-1]], [ys[0], ys[-1]], "D", mfc="none", color="#ff7f0e", ms=6)
        ax.set_xscale("log"); ax.set_xlim(8e-4, 1.2); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("fraction of nodes kept (log)"); ax.set_title(f"{task} / {a.model} (test)")
        ax.grid(alpha=.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("acc-AUC integrates measurements on MIB's grid; the ladder's IIA integrates its own rungs", fontsize=9)
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=170); fig.savefig(a.out + ".pdf")
    print("wrote", a.out + ".png")


if __name__ == "__main__":
    main()
