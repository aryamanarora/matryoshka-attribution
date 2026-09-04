"""Faithfulness vs. the ABSOLUTE number of units kept, for three methods at five substrates.

THE QUESTION this answers that no other figure in the repo does: every faithfulness/CPR curve we
plot plots against a SPARSITY FRACTION, which silently rescales the x axis per substrate -- 1% of
`node` is 11 units and 1% of an SAE span is 52,000. That makes "the SAE circuit is sparser" an
untestable statement, because each substrate is measured against its own denominator. Here the x
axis is the raw unit count, common to all fifteen curves, so the substrates can be read against
each other: how many things do you have to keep, in absolute terms, to recover the behaviour.

*** THE AVERAGE IS OVER TASKS AT A FIXED UNIT COUNT, WHICH IS NOT A FIXED SPARSITY. *** The eight
llama3 tasks have different prompt lengths (seq_len 3 for `simple`, 38 for `hours`) and every
substrate but `node` is laid out per-position, so `total` ranges 1.4M-17M within a single
substrate. k=10,000 is therefore 0.7% of `simple`'s MLP units and 0.06% of `hours`'s. That is the
consequence of the requested x axis and it is the honest one for "how many units", but it means a
substrate's curve is NOT the curve of a typical task at a typical sparsity.

EACH CURVE STOPS AT ITS SUBSTRATE'S SMALLEST `total` (node 1,056; MLP 1.4M; SAE 3.1M) and not one
point further. Past that, the shortest task has no measurement, and the two ways of continuing are
both worse than stopping: dropping the task mid-curve changes WHICH tasks are averaged from one x
to the next, and clamping it to its endpoint injects a synthetic 1.0 (faithfulness is normalised
so F(total)=1 by construction) that would drag every mean upward exactly where the curves are
being compared. The termination x is itself informative -- `node` runs out of units at 1,056.

*** FAITHFULNESS OVERSHOOTS 1 AND THE AXIS HAS TO SHOW IT. *** A partly-restored model can carry a
LARGER logit difference than the clean one, so these means peak as high as 3.4 (MLP+Attn, MAttr);
the per-task peak reaches 5.96. The first draft capped y at 1.12 and the excess simply vanished off
the top, which read as five methods saturating together when they do nothing of the sort. The
panels therefore share a y axis wide enough for the largest mean, and the y=1 line is drawn so
"recovered the behaviour" stays locatable.

ONE PANEL PER SUBSTRATE, on a SHARED x. The requested single-axes version exists behind
--layout single, and is genuinely hard to read: fifteen curves need colour AND dash to be
separated, and the two SAE substrates then differ from MLP only by dash pattern in the same
crowded decade. Faceting spends the redundant dash encoding on horizontal space instead. Because x
is shared and absolute, dropping a vertical at any k and reading across the row is still exactly
the cross-substrate comparison the figure is for.

INTERPOLATION is linear in log k, per task, onto a common grid. The measured grids are 24
log-spaced points that differ per task, so some interpolation is unavoidable; log is the space the
points are spaced in and the space the figure is drawn in.

TASKS are the 8 llama3 cells present at every substrate. `arc_easy` and `ioi` are node-only and
are excluded, since a substrate average over a different task set per substrate would confound the
two axes of the comparison.

METHODS: IG, MAttr+Adam at eps=1e-2 (the configuration we ship -- torch's 1e-8 default degenerates
to ~sign(g) at 2.29M mask logits) and MAttr+SGD. Colour is the METHOD, per the paper-wide rule in
plots/palette.py (SGD black, Adam blue, IG orange).

Run:  uv run python plots/plot_faith_curves_substrates.py [--layout facet|single]
Out:  plots/faith_curves_substrates.pdf  (+ .png)
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                            # noqa: E402

RES = "results/sva_sweep"
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
# (disk key, label, linestyle). Ordered by size. The dash patterns are used ONLY by --layout
# single; the faceted default separates substrates by panel and draws every curve solid.
SUBS = [("node", "Node", "solid"),
        ("mlp", "MLP neuron", (0, (3.5, 1.2))),
        ("mlp-attn_head", "MLP+Attn", (0, (1.1, 1.1))),
        ("mlp_sae_span", "SAE (MLP out)", (0, (5, 1.2, 1, 1.2))),
        ("resid_sae_span", "SAE (resid)", (0, (1.1, 1.1, 4, 1.1)))]
# (tag, palette key, legend label). eps=1e-2 is the shipped Adam; see the module docstring.
METHODS = [("ig", "IG", "IG"),
           ("sufficient_topk_adam_eps1e-2_bs1", "MAttr", "MAttr (Adam)"),
           ("sufficient_topk_sgd_bs1", "MAttr (SGD)", "MAttr (SGD)")]
LW = 1.1
XMAX = 4e6
FS_LAB, FS_TICK, FS_LEG = 7, 6, 6.5
# Decade ticks only, and only every OTHER decade: at ~1in per panel, 10^0..10^6 labelled in full
# collides with itself. The unlabelled decades keep their minor gridline.
XTICKS = [1e0, 1e2, 1e4, 1e6]
XLAB = ["10⁰", "10²", "10⁴", "10⁶"]


def curve(task, sub, tag):
    """(n_nodes, faithfulness) for one cell, or None if the run is missing."""
    p = f"{RES}/{task}_llama3_{sub}_{tag}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return np.asarray(d["n_nodes"], float), np.asarray(d["faithfulness"], float)


def mean_curve(sub, tag, grid):
    """Task-averaged faithfulness on `grid`, truncated to where EVERY task has data."""
    ys, hi = [], np.inf
    for t in TASKS:
        c = curve(t, sub, tag)
        if c is None:
            return None, None
        k, f = c
        hi = min(hi, k[-1])
        ys.append(np.interp(np.log(grid), np.log(k), f))
    g = grid <= hi
    return grid[g], np.mean(ys, axis=0)[g]


def furnish_x(ax):
    ax.set_xscale("log")
    ax.set_xlim(1, XMAX)
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.set_xticklabels(XLAB)
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", labelsize=FS_TICK, length=2, pad=1.5)
    ax.tick_params(axis="x", which="minor", length=1.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="facet", choices=["facet", "single"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or ("plots/faith_curves_substrates.pdf" if a.layout == "facet"
                    else "plots/faith_curves_substrates_single.pdf")

    grid = np.unique(np.round(np.logspace(0, np.log10(XMAX), 400)).astype(float))
    data, missing = {}, []
    for sub, slab, _ in SUBS:
        for tag, ckey, mlab in METHODS:
            x, y = mean_curve(sub, tag, grid)
            if x is None:
                missing.append(f"{slab}/{mlab}")
                continue
            data[(slab, mlab)] = (x, y)
    if not data:
        raise SystemExit(f"no runs found under {RES}")
    # Shared y across panels -- the comparison is between substrates, so a per-panel y would let
    # a curve that peaks at 0.9 look like one that peaks at 3.4. Top from the data, not a constant.
    ytop = max(y.max() for _, y in data.values())

    plt.rcParams.update(P.RC)
    if a.layout == "facet":
        fig, axes = plt.subplots(1, len(SUBS), figsize=(5.5, 1.75), sharex=True, sharey=True)
        for ax, (sub, slab, _) in zip(axes, SUBS):
            for tag, ckey, mlab in METHODS:
                if (slab, mlab) not in data:
                    continue
                x, y = data[(slab, mlab)]
                ax.plot(x, y, color=P.color(ckey), lw=LW, zorder=3, solid_capstyle="round")
            ax.axhline(1.0, color="#999999", lw=0.4, ls=(0, (1, 2)), zorder=1)
            ax.set_title(slab, fontsize=FS_LAB, pad=2.5)
            furnish_x(ax)
            P.furnish(ax)
        axes[0].set_ylim(-0.05 * ytop, 1.06 * ytop)
        axes[0].set_ylabel("Faithfulness", fontsize=FS_LAB)
        # xlabel on the CENTRE panel rather than fig.supxlabel: supxlabel is positioned in figure
        # coordinates and tight_layout does not reserve space for it, which left a ~0.3in band of
        # dead white between the tick labels and the label.
        axes[len(SUBS) // 2].set_xlabel("Units kept", fontsize=FS_LAB, labelpad=1.5)
        h = [Line2D([], [], color=P.color(c), lw=LW, label=lab) for _, c, lab in METHODS]
        fig.legend(handles=h, fontsize=FS_LEG, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, 1.03), frameon=False, handlelength=1.9,
                   handletextpad=0.5, columnspacing=1.4)
        fig.tight_layout(pad=0.3, w_pad=0.4, rect=(0, 0, 1, 0.94))
    else:
        fig, ax = plt.subplots(figsize=(5.5, 2.9))
        for sub, slab, ls in SUBS:
            for tag, ckey, mlab in METHODS:
                if (slab, mlab) not in data:
                    continue
                x, y = data[(slab, mlab)]
                ax.plot(x, y, color=P.color(ckey), ls=ls, lw=LW, zorder=3,
                        solid_capstyle="round", dash_capstyle="round")
        ax.axhline(1.0, color="#999999", lw=0.4, ls=(0, (1, 2)), zorder=1)
        ax.set_ylim(-0.05 * ytop, 1.06 * ytop)
        ax.set_xlabel("Units kept", fontsize=FS_LAB)
        ax.set_ylabel("Faithfulness", fontsize=FS_LAB)
        furnish_x(ax)
        ax.set_xticklabels(XLAB)
        P.furnish(ax)
        h_m = [Line2D([], [], color=P.color(c), lw=LW, label=lab) for _, c, lab in METHODS]
        h_s = [Line2D([], [], color="#555555", ls=ls, lw=LW, label=lab) for _, lab, ls in SUBS]
        l1 = ax.legend(handles=h_m, fontsize=FS_LEG, ncol=3, loc="lower left",
                       bbox_to_anchor=(-0.01, 1.0), frameon=False, handlelength=1.9,
                       handletextpad=0.5, columnspacing=1.1)
        ax.add_artist(l1)
        ax.legend(handles=h_s, fontsize=FS_LEG, ncol=1, loc="upper left", frameon=False,
                  handlelength=2.6, handletextpad=0.5, labelspacing=0.25, borderpad=0.2)
        fig.tight_layout(pad=0.3)

    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} (+ .png)   {len(TASKS)} tasks x {len(SUBS)} substrates x "
          f"{len(METHODS)} methods, layout={a.layout}")
    if missing:
        print(f"missing: {', '.join(missing)}")

    # Two numbers per curve, because they answer different questions: k* is "how many units do you
    # need", the log-AUC is "how good is the whole curve". AUC is normalised by the width of the
    # plotted decade range and is NOT the `faith_auc` in the json, which spans each cell's own
    # full 1..total.
    print(f"\n{'substrate':<15}{'method':<14}{'k* (faith>=0.5)':>17}{'peak':>7}{'log-AUC':>9}")
    for sub, slab, _ in SUBS:
        for tag, ckey, mlab in METHODS:
            if (slab, mlab) not in data:
                continue
            x, y = data[(slab, mlab)]
            i = np.argmax(y >= 0.5)
            ks = f"{x[i]:.0f}" if (y >= 0.5).any() else f">{x[-1]:.0f}"
            lx = np.log10(x)
            print(f"{slab:<15}{mlab:<14}{ks:>17}{y.max():>7.2f}"
                  f"{np.trapezoid(y, lx) / (lx[-1] - lx[0]):>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
