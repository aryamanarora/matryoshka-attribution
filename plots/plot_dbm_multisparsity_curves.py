"""Faithfulness and accuracy against circuit SIZE, for DBM's own-L0 frontier and the MAttr arms.

The scalar comparison (plot_dbm_multisparsity_compare.py) says the multi-sparsity row wins on
IIA and ties on CPR. This is the curve those scalars are areas under, which is where the reason
lives: MAttr's points are ONE ranking read at MIB's ten fixed proportions, while DBM's are SEVEN
masks each read only at the size it converged to. Where the two disagree is the sparse end.

X IS THE NODE COUNT, not the proportion, because it is the quantity the masks were trained
against and because it makes the cells comparable in kind (they run 157 to 1057 nodes, so the
axis is log). For the grid-based rows k = int(p * n_scored), which is exactly the k MIB applied.

*** EXPLORATORY: draws whichever test cells have landed. *** The panel headers carry the count.
The MAttr arms are four settings of one method, so they share MAttr's hue and separate by
LINETYPE (plots/palette.py's rule); they overlap almost perfectly here, and that is a result
rather than a plotting failure.

Run:  uv run python plots/plot_dbm_multisparsity_curves.py
Out:  plots/dbm_multisparsity_curves.pdf (+ .png)
"""
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette as P                                              # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "mib"))
import dbm_multisparsity as _DBMMS                               # noqa: E402

GRID_ROWS = [
    ("MAttr Adam, log $k$",  "test_node_topk_log_lr05",      "mattr", (0, ())),
    ("MAttr Adam, unif $k$", "test_node_topk_uniform_lr05",  "mattr", (0, (4, 1.5))),
    ("MAttr SGD, log $k$",   "test_node_softlog_sgd_lr_1.0", "mattr", (0, (1, 1))),
    ("MAttr SGD, unif $k$",  "test_node_softuni_sgd_lr_3.0", "mattr", (0, (3, 1, 1, 1))),
    ("DBM ($\\lambda$=6)",   "eprun_eval_ld_sig_lr0.3_l16.0", "eprun", (0, (5, 2))),
]
FIG_W, FIG_H = 5.5, 2.9
FS_AXIS, FS_TICK, FS_ANNOT = 7, 6, 5.5


def grid_curve(dirn, layout, task, model):
    f = (f"results/{dirn}/{task}_{model}_test.pkl" if layout == "mattr" else
         f"results/{dirn}/EdgePruning_patching_node/"
         f"{task.replace('_', '-')}_{model}_test_abs-False.pkl")
    if not os.path.exists(f):
        return None
    r = pickle.load(open(f, "rb"))
    return r["faithfulnesses"], r["accuracies"]


def main():
    cells = []
    for f in sorted(glob.glob("results/dbm_multisparsity/*_test.json")):
        cells.append(json.load(open(f)))
    if not cells:
        raise SystemExit("no results/dbm_multisparsity/*_test.json yet")
    print(f"{len(cells)}/11 cells")

    PCT = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)
    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(2, len(cells), figsize=(FIG_W, FIG_H), squeeze=False)

    for ci, d in enumerate(cells):
        n = d["n_nodes"]
        task = f"{d['task']}/{d['model']}"
        for mi, (key, ylab) in enumerate(((0, "Faithfulness"), (1, "Accuracy"))):
            ax = axes[mi][ci]
            for lab, dirn, layout, ls in GRID_ROWS:
                got = grid_curve(dirn, layout, d["task"], d["model"])
                if got is None:
                    continue
                ks = [max(int(p * n), 1) for p in PCT]     # k=0 has no place on a log axis
                col = P.METHOD["DBM"] if lab.startswith("DBM") else P.METHOD["MAttr"]
                ax.plot(ks, got[key], lw=0.8, ls=ls, color=col, zorder=2,
                        label=lab if (mi == 0 and ci == 0) else None)
            # The multi-sparsity row exactly as it is scored: MIB's grid, and at each point the
            # measured value of the densest trained mask that fits the budget, held until a
            # denser one becomes affordable (full circuit at p=1). A step by construction, so
            # DRAWN AS LINES BETWEEN THE GRID POINTS, not as a staircase: the metric is MIB's
            # trapezoid over (grid x, assigned y), so this is literally the integrand. A
            # staircase would depict a different area from the one reported, and it would also
            # look unlike every other row here, which is the same trapezoid over the same x.
            # Filled markers mark the grid points themselves; the open circles are the
            # underlying rung measurements at their masks' own converged sizes.
            ks_grid = [max(int(p * n), 1) for p in _DBMMS.PCT]
            fs_, accs_ = _DBMMS.curve(d)
            ax.plot(ks_grid, fs_ if key == 0 else accs_, lw=1.1, color=P.METHOD["DBM"],
                    marker="o", ms=1.8, markeredgewidth=0, zorder=4,
                    label="DBM (multi-sparsity)" if (mi == 0 and ci == 0) else None)
            ax.scatter([r["k"] for r in d["points"]],
                       [r["faithfulness"] if key == 0 else r["accuracy"] for r in d["points"]],
                       s=9, facecolor="#ffffff", edgecolor=P.METHOD["DBM"], linewidth=0.7,
                       zorder=5, label="DBM rungs (own $L_0$)" if (mi == 0 and ci == 0) else None)
            ax.set_xscale("log")
            if mi == 0:
                ax.set_title(task, fontsize=FS_AXIS, pad=2)
            else:
                ax.set_xlabel("Nodes kept", fontsize=FS_AXIS)
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=FS_AXIS)
            P.furnish(ax)
            ax.tick_params(labelsize=FS_TICK)
    axes[0][0].legend(fontsize=FS_ANNOT - 0.5, frameon=False, loc="upper left",
                      handlelength=1.6, handletextpad=0.4, labelspacing=0.15,
                      borderaxespad=0.2)
    fig.tight_layout(pad=0.4, w_pad=0.7, h_pad=0.6)
    fig.savefig("plots/dbm_multisparsity_curves.pdf")
    fig.savefig("plots/dbm_multisparsity_curves.png", dpi=200)
    print("wrote plots/dbm_multisparsity_curves.pdf")


if __name__ == "__main__":
    main()
