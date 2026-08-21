"""Adam vs SGD for \\ourmethod{} on MIB, averaged over the 11 task/model cells.

The figure for sections/optimiser.tex. Two views of the same four arms (optimizer x k-schedule),
each point one (arm, lr) averaged over M.COLUMNS:

  (default)  SCATTER   acc-AUC (x) vs CPR AUC (y), points joined into per-arm LR paths
  --lrx      CURVES    lr on a log x-axis, two stacked panels (CPR, acc-AUC)

The scatter is the honest summary -- it shows both metrics at once and makes the uniform-k
trade-off (high CPR, low acc-AUC) visible -- but the CURVES are the view that carries this
section's actual claim, because the claim is about WHERE the optimum sits, and lr is a
coordinate there rather than a label. Both are emitted; pick one for the paper.

WHAT THE NUMBERS SAY (validation, 11/11 cells, 2026-08-21):
    Adam log-k peaks at lr=0.1   CPR 1.881 / acc 0.5035
    SGD  log-k peaks at lr=1.0   CPR 1.886 / acc 0.5036
i.e. the peaks are level to within 0.005 CPR and 0.0001 acc-AUC, at optima 10-20x apart. The
matched-lr comparison that reads "SGD costs 0.47 CPR" (0.05: 1.879 vs 1.413) is measuring SGD
being 20x below its optimum, not the optimizer. Same story on uniform k: Adam 2.092 at 0.05,
SGD 2.031 at 3.0. See make_mib_table.OUR_METHODS for why the table rows are pinned to own-best
lr rather than to a shared one.

COMPLETENESS BAR: only dirs with 11/11 cells on BOTH metrics are plotted, because a task-average
over a different subset of cells is not comparable to one over all 11 -- the cells differ by far
more than the arms do. Every exclusion is printed, never silently dropped. Currently dropped:
    topklog_lr_0.01                     1/11  (a one-cell probe, not a sweep point)
    softlog_sgd_lr_{0.005,0.01}         5/11  (cheap cells only)
    final_node  (Adam, unif k, lr=0.01) 11/11 CPR but 3/11 acc -- never re-eval'd for acc
That last one leaves the ADAM UNIF-K ARM WITH A SINGLE POINT (lr=0.05). It is drawn as an
unconnected marker rather than a path, so the figure cannot be read as "Adam's uniform-k curve
is flat". If final_node ever gets an acc re-eval, it becomes a 2-point path and still is not a
sweep; the arm needs lr>=0.3 runs before it can be compared to SGD's six.

Not a duplicate of mib_accauc_cpr_scatter_lr.pdf: that figure is every LR sweep in the paper on
one frame (~34 points, 8 series incl. DBM / Node Pruning / +hard / bern), where the optimizer
contrast is two of eight paths. This is the same projection restricted to the four MAttr
optimizer arms, which is what the optimiser section needs to be readable at column width.

Run:  uv run python plots/plot_optimizer_lr.py [--lrx]
Out:  plots/optimizer_lr.pdf, plots/optimizer_lr_curves.pdf   (plots/*.pdf is gitignored)
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import palette as P

sys.path.insert(0, "scripts")
import make_mib_table as M            # noqa: E402  COLUMNS (the 11 task/model cells)

RB = Path("results")

# (arm label, optimizer, k-schedule, [(lr, results dir), ...])
# Adam's log-k grid stops at 0.3 and SGD's starts at 0.05: that asymmetry is the sweeps', not a
# plotting choice, and it is also the finding -- the two arms were swept over the range where
# each was expected to live, and the ranges barely overlap.
ARMS = [
    ("MAttr (Adam)", "Adam", "log", [
        ("0.005", "topklog_lr_0.005"), ("0.01", "topklog_lr_0.01"),
        ("0.05", "topklog_lr_0.05"), ("0.1", "topklog_lr_0.1"), ("0.3", "topklog_lr_0.3")]),
    ("MAttr (SGD)", "SGD", "log", [
        ("0.005", "softlog_sgd_lr_0.005"), ("0.01", "softlog_sgd_lr_0.01"),
        ("0.05", "softlog_sgd_lr_0.05"), ("0.1", "softlog_sgd_lr_0.1"),
        ("0.3", "softlog_sgd_lr_0.3"), ("1.0", "softlog_sgd_lr_1.0"),
        ("3.0", "softlog_sgd_lr_3.0"), ("10.0", "softlog_sgd_lr_10.0")]),
    ("+ unif k (Adam)", "Adam", "uniform", [
        ("0.01", "final_node"), ("0.05", "mib_node_topk_uniform_lr05")]),
    ("+ unif k (SGD)", "SGD", "uniform", [
        ("0.05", "softuni_sgd_lr_0.05"), ("0.1", "softuni_sgd_lr_0.1"),
        ("0.3", "softuni_sgd_lr_0.3"), ("1.0", "softuni_sgd_lr_1.0"),
        ("3.0", "softuni_sgd_lr_3.0"), ("10.0", "softuni_sgd_lr_10.0")]),
]

# Optimizer -> colour, k-schedule -> linetype. The encoding is deliberate: the CONTRAST this
# figure is about is the optimizer, so it gets the strong channel (hue), and the k-schedule --
# a knob the paper ablates elsewhere -- gets linetype. Hexes from plots/palette.py; SGD's black
# is a measured CVD choice documented there, not a stylistic one.
COLOR = {"Adam": P.METHOD["MAttr"], "SGD": P.METHOD["MAttr (SGD)"]}
DASH = {"log": "-", "uniform": "--"}
MARK = {"log": "o", "uniform": "s"}

RC = {
    "font.family": "Inter", "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42, "text.color": "#000000",
    "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
}


def cell_means(dirn):
    """(n_cpr, n_acc, mean CPR AUC, mean acc-AUC) over M.COLUMNS for one results dir.

    Both metrics come from the SAME eval_mib pkl, so a cell can never contribute to one mean and
    not the other -- which is what lets a single n==11 check gate both axes. (The MAttr acc-AUC
    read in make_mib_accauc_table goes through run_evaluation dirs for older runs; every dir here
    is an eval_mib dir, so the direct read is the right one and matches EVALMIB_ACC's branch.)
    Read raw, NOT through A._acc, which rounds to 2dp -- rounding before averaging 11 cells would
    quantise the arm means onto a grid coarser than the differences being plotted.
    """
    cpr, acc = [], []
    for t, m, _ in M.COLUMNS:
        p = RB / dirn / f"{t}_{m}_validation.pkl"
        if not p.exists():
            continue
        d = pickle.load(open(p, "rb"))
        if d.get("area_under") is not None:
            cpr.append(d["area_under"])
        if d.get("acc_auc") is not None:
            acc.append(d["acc_auc"])
    return (len(cpr), len(acc),
            float(np.mean(cpr)) if cpr else None,
            float(np.mean(acc)) if acc else None)


def build():
    """[(label, opt, sched, [(lr, cpr, acc), ...])], complete cells only, exclusions printed."""
    n_cells = len(M.COLUMNS)
    out = []
    for label, opt, sched, grid in ARMS:
        pts = []
        for lr, dirn in grid:
            ncpr, nacc, cpr, acc = cell_means(dirn)
            if ncpr < n_cells or nacc < n_cells:
                print(f"  DROP {label:16s} lr={lr:<6s} {dirn:28s} "
                      f"cpr {ncpr}/{n_cells} acc {nacc}/{n_cells}")
                continue
            pts.append((float(lr), cpr, acc))
        pts.sort()
        if pts:
            out.append((label, opt, sched, pts))
        else:
            print(f"  DROP ARM {label} -- no complete lr")
    return out


def summarise(arms):
    for label, _, _, pts in arms:
        bc = max(pts, key=lambda p: p[1])
        ba = max(pts, key=lambda p: p[2])
        print(f"  {label:16s} n={len(pts)}  best CPR {bc[1]:.3f} @ lr={bc[0]:g}   "
              f"best acc {ba[2]:.4f} @ lr={ba[0]:g}")


def scatter(arms):
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for label, opt, sched, pts in arms:
        x = [p[2] for p in pts]
        y = [p[1] for p in pts]
        c = COLOR[opt]
        # A one-point arm gets a marker and no line: joining nothing would still draw a legend
        # handle with a linetype, which reads as a swept path that happens to be short.
        if len(pts) > 1:
            ax.plot(x, y, DASH[sched], color=c, lw=1.0, alpha=0.55, zorder=1)
        ax.plot(x, y, MARK[sched], color=c, ms=4.2, mec="white", mew=0.6,
                ls="none", label=label, zorder=3)
        for lr, cpr, acc in pts:
            ax.annotate(f"{lr:g}", (acc, cpr), textcoords="offset points",
                        xytext=(4, 3), fontsize=5.6, color=c, zorder=4)
    ax.set_xlabel("acc-AUC (avg over 11 MIB cells)")
    ax.set_ylabel("CPR AUC (avg over 11 MIB cells)")
    ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax.grid(alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = "plots/optimizer_lr.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def curves(arms):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(2, 1, figsize=(4.4, 4.4), sharex=True)
    for j, (key, ylab) in enumerate(((1, "CPR AUC"), (2, "acc-AUC"))):
        ax = axes[j]
        for label, opt, sched, pts in arms:
            x = [p[0] for p in pts]
            y = [p[key] for p in pts]
            c = COLOR[opt]
            if len(pts) > 1:
                ax.plot(x, y, DASH[sched], color=c, lw=1.2, alpha=0.85, zorder=2)
            ax.plot(x, y, MARK[sched], color=c, ms=4.0, mec="white", mew=0.6,
                    ls="none", label=label if j == 0 else None, zorder=3)
            # Mark each arm's own optimum: the whole point is that these sit at different x.
            if len(pts) > 1:
                bx, by = max(zip(x, y), key=lambda p: p[1])
                ax.plot([bx], [by], MARK[sched], color=c, ms=8.5, mfc="none", mew=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_ylabel(f"{ylab} (avg, 11 cells)")
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    # Legend BELOW the frame, not in it: both panels are single-humped curves that fill their
    # own box, so every in-axes corner is occupied at some lr, and an inset legend sat on the
    # SGD log-k rise.
    axes[1].set_xlabel("learning rate")
    fig.tight_layout()
    axes[0].legend(fontsize=6.5, frameon=False, ncol=4, loc="upper center",
                   bbox_to_anchor=(0.5, 1.20), columnspacing=1.1, handletextpad=0.4)
    out = "plots/optimizer_lr_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    print("exclusions (need 11/11 on both metrics):")
    a = build()
    print("arm optima:")
    summarise(a)
    scatter(a)
    curves(a)
