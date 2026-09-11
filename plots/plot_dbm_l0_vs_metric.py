"""Pooled view of DBM's sparsity sweep: does the mask's own L0 explain the metric?

Companion to plot_dbm_sparsity_stitch.py, which reads ONE cell. This one puts all 11 MIB
node-level cells x 7 L1 coefficients on a single axis of MASK L0 -- the absolute number of gates
the run converged with, from k_log[-1] -- and asks whether that axis orders the runs. L0 rather
than the density fraction because the question is about circuit SIZE; density is still carried in
the frame and its correlation is printed beside L0's, and the two differ because the cells span
156 to 1056 nodes.

THE ANSWER IS DIFFERENT FOR THE TWO METRICS, which is the point of drawing it:
  IIA   density is essentially the whole story -- Spearman(density, IIA) = -0.67 pooled over all
        75 runs, monotone and cell-independent. Sparser mask, earlier switch-on, higher IIA.
  CPR   density explains nothing pooled -- Spearman(density, CPR) = +0.09. Every cell has an
        interior optimum but at its OWN density (0.069 to 0.503 across the 11), and the cells sit
        at different levels, so the within-cell curves are hidden by the between-cell spread.
        The per-cell paths are drawn for exactly that reason: the structure is within, not across.

L1 DOES NOT PIN L0, which is why this axis is worth having at all. The same coefficient lands at
wildly different densities per cell -- at L1=20 the range is 0.009 to 0.256, a 27x spread, and by
L1=40 two cells have collapsed to zero gates. Reading a sparsity sweep by its coefficient is
therefore reading a knob, not a sparsity.

ZERO-DENSITY RUNS ARE DROPPED, not clipped: the x axis is log (densities span three decades) and
a run with no gates on has no operating point to place. They are counted in the printout.

Run:  uv run python plots/plot_dbm_l0_vs_metric.py
Out:  plots/dbm_l0_vs_metric.pdf (+ .png)
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "mib"))
import palette as P                                            # noqa: E402
import make_mib_table as M                                     # noqa: E402

L1S = ["0.2", "0.6", "2.0", "6.0", "20.0", "40.0", "60.0"]
EVAL = ("results/eprun_eval_ld_sig_lr0.3_l1{c}/EdgePruning_patching_node/"
        "{t}_{m}_validation_abs-False.pkl")
TRAIN = "results/eprun_node_ld_sig_lr0.3_l1{c}/{t}_{m}_scores.pt"
FIG_W, FIG_H = 5.5, 2.1
FS_AXIS, FS_TICK, FS_ANNOT = 7, 6, 5.5


def load():
    rows, zeros = [], 0
    for task, model, _ in M.COLUMNS:
        for c in L1S:
            f = EVAL.format(c=c, t=task.replace("_", "-"), m=model)
            p = TRAIN.format(c=c, t=task, m=model)
            if not (os.path.exists(f) and os.path.exists(p)):
                continue
            d = pickle.load(open(f, "rb"))
            s = torch.load(p, map_location="cpu")
            L0 = float(s["k_log"][-1])
            dens = L0 / len(s["scores"])
            # DEGENERATE RUNS ARE DROPPED AT L0 < 0.5, not at 0. Two arithmetic/llama3 runs
            # converge to an expected gate count of ~1e-24 -- not literally zero, so a `<= 0`
            # test misses them, and on a log axis they stretch the x range over thirty decades
            # and squash all 73 real runs into one column. Below half a node there is no
            # operating point to plot, whatever the float says.
            if L0 < 0.5:
                zeros += 1
                continue
            rows.append(dict(cell=f"{task}/{model}", l1=float(c), L0=L0, dens=dens,
                             N=len(s["scores"]), cpr=d["area_under"], iia=d["acc_auc"]))
    return rows, zeros


def main():
    rows, zeros = load()
    # Lightness ramp of DBM's hue over the coefficient ladder -- one method, one ordered knob,
    # so palette.py's "colour means method" rule says shade rather than new hues.
    base = np.array([int(P.METHOD["DBM"][i:i + 2], 16) for i in (1, 3, 5)], float)
    shade = lambda i: tuple((base + 0.62 * (1 - i / (len(L1S) - 1)) * (255 - base)) / 255)

    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))
    for ax, key, ylab in ((axes[0], "cpr", "CPR (↑)"), (axes[1], "iia", "Compactness (↑)")):
        # Per-cell paths first, under the points: the within-cell trajectory is the structure
        # the pooled cloud hides, and a faint line is the cheapest way to show 11 of them.
        for cell in sorted({r["cell"] for r in rows}):
            rr = sorted([r for r in rows if r["cell"] == cell], key=lambda r: r["L0"])
            ax.plot([r["L0"] for r in rr], [r[key] for r in rr],
                    lw=0.5, color="#999999", alpha=0.55, zorder=1)
        for i, c in enumerate(L1S):
            rr = [r for r in rows if r["l1"] == float(c)]
            ax.scatter([r["L0"] for r in rr], [r[key] for r in rr], s=12, color=shade(i),
                       edgecolor="#000000", linewidth=0.3, zorder=3,
                       label=f"$\\lambda$={c}")
        rho = spearmanr([r["L0"] for r in rows], [r[key] for r in rows])[0]
        ax.annotate(f"$\\rho$ = {rho:+.2f}", (0.03, 0.97), xycoords="axes fraction",
                    ha="left", va="top", fontsize=FS_ANNOT + 0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Mask $L_0$ (nodes kept)", fontsize=FS_AXIS)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        P.furnish(ax)
        ax.tick_params(labelsize=FS_TICK)
    axes[1].legend(fontsize=FS_ANNOT, frameon=False, loc="lower left", ncol=2,
                   handlelength=0.8, handletextpad=0.3, labelspacing=0.15,
                   columnspacing=0.6, borderaxespad=0.3)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig("plots/dbm_l0_vs_metric.pdf")
    fig.savefig("plots/dbm_l0_vs_metric.png", dpi=200)
    print(f"wrote plots/dbm_l0_vs_metric.pdf  ({len(rows)} runs, {zeros} dropped at L0<0.5)")
    for key, lab in (("cpr", "CPR"), ("iia", "IIA")):
        sp = lambda k: spearmanr([r[k] for r in rows], [r[key] for r in rows])[0]
        print(f"  pooled Spearman({lab:3s}):  L0 = {sp('L0'):+.3f}   "
              f"density = {sp('dens'):+.3f}   lambda = {sp('l1'):+.3f}")


if __name__ == "__main__":
    main()
