"""DBM's own-L0 frontier against the MAttr arms, per cell, on both MIB test metrics.

*** THIS IS AN EXPLORATORY CUT, NOT A PAPER FIGURE. *** It draws whichever cells of
results/dbm_multisparsity/*_test.json have landed. The wave was launched with 11 and the small
models finish first, so an early render shows gpt2/qwen2.5 only -- exactly the cells where
MAttr's published margins are SMALLEST. Read nothing into the ordering until the header says
11/11; the count is printed and drawn into the axis label for that reason.

*** THE ROWS ARE NOT THE SAME KIND OF OBJECT, AND THE FIGURE CANNOT SHOW THAT. *** Every MAttr
row and the shipped DBM row sweep ONE score vector over MIB's ten fixed proportions. The
multi-sparsity row reads SEVEN separately trained masks, each only at the sparsity it converged
to -- seven training runs against one, which the table's cost column carries as 21k against 3k.
Comparing the bars without that caveat overstates the result.

SCORED ON MIB'S GRID via scripts/mib/dbm_multisparsity.py: at each proportion the densest
trained mask that fits the budget, full circuit at p=1. An earlier version of this figure
integrated over the rungs' own sparsities and reported a large DBM win on IIA that was entirely
a grid artefact -- see that module's note.

Both metrics are drawn because they disagree in size: the CPR spread across these methods is a
few hundredths while the IIA spread is a few tenths, and a single panel would hide the second.

Run:  uv run python plots/plot_dbm_multisparsity_compare.py
Out:  plots/dbm_multisparsity_compare.pdf (+ .png)
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

MS = "results/dbm_multisparsity/{t}_{m}_test.json"
# (label, results dir, layout) -- "mattr" is the flat {task}_{model}_test.pkl layout, "eprun"
# the run_evaluation.py subfolder one. Order is the y axis, bottom to top.
ROWS = [
    ("MAttr SGD, unif $k$",  "test_node_softuni_sgd_lr_3.0", "mattr"),
    ("MAttr SGD, log $k$",   "test_node_softlog_sgd_lr_1.0", "mattr"),
    ("MAttr Adam, unif $k$", "test_node_topk_uniform_lr05",  "mattr"),
    ("MAttr Adam, log $k$",  "test_node_topk_log_lr05",      "mattr"),
    ("DBM ($\\lambda$=6)",   "eprun_eval_ld_sig_lr0.3_l16.0", "eprun"),
    ("DBM (multi-sparsity)", None,                            "multi"),
]
# Task marker, not task colour: colour is reserved for the METHOD FAMILY (palette.py's rule),
# and these two cells are two draws of the same comparison rather than two series.
TASK_MARK = {"ioi/gpt2": "o", "mcqa/qwen2.5": "s"}
FIG_W, FIG_H = 5.5, 1.9
FS_AXIS, FS_TICK, FS_ANNOT = 7, 6, 5.5


def read(dirn, layout, task, model):
    if layout == "multi":
        # Via the shared reader, which applies the MIB-grid protocol. The `cpr`/`iia` stored in
        # the json are the OLD rung-grid integration and must not be read directly -- see the
        # note in scripts/mib/dbm_multisparsity.py about the grid inflating IIA.
        got = _DBMMS.cell(task, model, "test")
        return (got[0], got[1]) if got else None
    if layout == "mattr":
        f = f"results/{dirn}/{task}_{model}_test.pkl"
    else:
        f = (f"results/{dirn}/EdgePruning_patching_node/"
             f"{task.replace('_', '-')}_{model}_test_abs-False.pkl")
    if not os.path.exists(f):
        return None
    r = pickle.load(open(f, "rb"))
    return r["area_under"], r.get("acc_auc")


def main():
    cells = []
    for f in sorted(glob.glob("results/dbm_multisparsity/*_test.json")):
        d = json.load(open(f))
        cells.append((d["task"], d["model"]))
    if not cells:
        raise SystemExit("no results/dbm_multisparsity/*_test.json yet")
    print(f"{len(cells)}/11 cells: {[t + '/' + m for t, m in cells]}")

    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    ys = np.arange(len(ROWS))
    for ax, mi, xlab in ((axes[0], 0, "CPR (↑)"), (axes[1], 1, "Compactness (↑)")):
        for yi, (lab, dirn, layout) in enumerate(ROWS):
            col = P.METHOD["DBM"] if lab.startswith("DBM") else P.METHOD["MAttr"]
            for task, model in cells:
                got = read(dirn, layout, task, model)
                if got is None or got[mi] is None:
                    continue
                ax.scatter([got[mi]], [yi], s=20, marker=TASK_MARK.get(f"{task}/{model}", "^"),
                           c=col, edgecolor="#000000", linewidth=0.4, zorder=3)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in ROWS], fontsize=FS_TICK)
        ax.set_ylim(-0.6, len(ROWS) - 0.4)
        ax.set_xlabel(xlab, fontsize=FS_AXIS)
        P.furnish(ax)
        ax.grid(False, axis="y")     # horizontal rules between categories are noise here
        ax.tick_params(labelsize=FS_TICK)
    axes[1].set_yticklabels([])
    # Task key as marker-only entries; the method identity is already on the y axis.
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([], [], marker=TASK_MARK.get(f"{t}/{m}", "^"), ls="none",
                                   mfc="#ffffff", mec="#000000", mew=0.5, ms=3.4,
                                   label=f"{t}/{m}") for t, m in cells],
                   fontsize=FS_ANNOT, frameon=False, loc="lower right",
                   handletextpad=0.3, labelspacing=0.2, borderaxespad=0.3)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    fig.savefig("plots/dbm_multisparsity_compare.pdf")
    fig.savefig("plots/dbm_multisparsity_compare.png", dpi=200)
    print("wrote plots/dbm_multisparsity_compare.pdf")
    for lab, dirn, layout in ROWS:
        vals = [read(dirn, layout, t, m) for t, m in cells]
        s = "  ".join("     --" if v is None else f"{v[0]:.3f}/{v[1]:.3f}" for v in vals)
        print(f"  {lab:24s} {s}")


if __name__ == "__main__":
    main()
