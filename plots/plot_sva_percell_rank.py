"""Per-CELL acc-vs-faith scatter: one panel per (task x substrate x ablation), 5 methods.

plots/plot_accauc_vs_faithauc.py macro-averages each method to ONE point per panel over four
task-groups. That is the right summary for a paper figure and it hides the thing this figure is
for: whether an ordering holds cell by cell, or is carried by a few cells and reversed in the
rest. Here nothing is averaged -- every point is a single run.

FIVE METHODS, ONE LOSS. Adam (the sweep's default eps=1e-8), Adam at eps=1e-2, SGD, IG, I x G,
all at logit_diff. Fixing the loss is what makes a cell a clean 5-way comparison: the parent
figure spends marker SHAPE on the loss and draws three points per method, which at one panel per
cell would be 15 markers in a 0.9in box. All 52 x 5 = 260 runs are on disk (asserted in main()).

LAYOUT IS A GRID WITH HOLES, which is why this is raw matplotlib and not plotnine. Rows are the
10 tasks, columns the 6 (ablation x substrate) settings, so a ROW reads "does this task rank the
methods the same way everywhere" and a COLUMN reads "does this setting rank them the same way on
every task". The two MIB tasks (arc_easy, ioi) exist only at the `node` substrate -- the
per-position substrates filter to the modal prompt length and would keep 3.8% of ARC-E -- so 8 of
the 60 slots are structurally empty and are drawn blank rather than filled with something. Neither
facet_wrap (no holes, no row/col headers) nor facet_grid (frees y per ROW, not per panel) can do
that; see the parent figure's docstring for the same limitation hit one figure earlier.

EVERY PANEL HAS ITS OWN AXES AND THEY DO NOT INCLUDE ZERO. That is deliberate and it is the one
thing that can be misread here. Faith AUC spans 0.04 (addition/mlp, zeroed) to 8.4 (nounpp/mlp,
zeroed) across panels, so a shared scale would flatten most panels to a single dot; and anchoring
at 0 would push the five points of a typical panel into a corner, which destroys exactly the
separation this figure exists to show. The cost is that DISTANCE IS NOT COMPARABLE ACROSS PANELS
-- only the ORDER of the five points within a panel is. Read this figure for rank, and the parent
figure for magnitude.

FAITH AUC (y) IS THE NOISY AXIS. Six cells re-run at identical config reproduce acc AUC to
+-0.002 but faith AUC only to ~10%, because faith normalises by (F_clean - F_patch), a
denominator that collapses where the ablated model is badly damaged. A vertical ordering inside
one panel that is not repeated down its column is noise. The x axis carries the weight.

Run:  uv run python plots/plot_sva_percell_rank.py
Out:  plots/sva_percell_rank.pdf  (+ .png sibling; plots/*.pdf is gitignored)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402
import plot_accauc_vs_faithauc as V                    # loaders, task lists, the model pin  # noqa: E402

LOSS = "logit_diff"
# (parse_method key, legend label). Order is the legend order and the z-order: later entries are
# drawn on top, so the three MAttr arms sit above the two gradient baselines where they collide.
METHODS = [
    ("IxG",               "I×G",                 P.color("I×G")),
    ("IG",                "IG",                  P.color("IG")),
    ("stopk-log",         "MAttr (Adam)",        P.color("MAttr")),
    ("softsgd-log",       "MAttr (SGD)",         P.color("MAttr (SGD)")),
    ("stopk-log-eps1e-2", "MAttr (Adam, ε=10⁻²)", P.color("MAttr (Adam eps=1e-2)")),
]
# (results dir, column-header line 1). -input only: `--include-input` is node-only in the hooker,
# so an +input column would exist for 1 of the 3 substrates and the grid would grow a second,
# raggeder set of holes for a variable this figure is not about.
SOURCES = [("results/sva_sweep", "Patched"), ("results/sva_zeroabl", "Zero-abl.")]
SUBS = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]
# Row order: the two 4-task groups first, then the two node-only MIB tasks last so every hole in
# the grid sits in one block at the bottom right rather than being scattered through it.
ROWS = V.SVA + V.ARITH + ["arc_easy", "ioi"]
NODE_ONLY = {"arc_easy", "ioi"}

FIG_W, FIG_H = 5.5, 7.6          # full-page portrait; 60 panels need the height
FS_HEAD, FS_ROW, FS_TICK, FS_LEG = 6.5, 6.5, 5, 6
MARGIN = 0.16                    # fraction of each panel's data range left as padding
MS = 13                          # marker area (pt^2); small enough that 5 fit a 0.8in panel


def cells():
    """{(dir, substrate, task): {method key: (acc_auc, faith_auc)}} for the 5 methods at LOSS."""
    out = {}
    for res, _ in SOURCES:
        raw = V.load(res)          # already applies the task->model pin and averages Random seeds
        for sub, _ in SUBS:
            for task in (ROWS if sub == "node" else [t for t in ROWS if t not in NODE_ONLY]):
                got = {m: raw[(m, LOSS, sub, task)]
                       for m, _, _ in METHODS if (m, LOSS, sub, task) in raw}
                out[(res, sub, task)] = got
    return out


# Below these differences two runs are not distinguishable, so ranking them is ranking noise.
# MEASURED, not guessed: six SVA+ cells re-run at identical config (2026-08-28) reproduced
# acc AUC to |Δ| ≤ 0.0074 with a median near 0.002, and faith AUC only to ~10% relative. The
# acc tolerance is set at the loose end of that spread rather than the median, so a "tie" here
# means "this figure cannot tell them apart", not "they are equal".
#
# Faith's tolerance is RELATIVE because its scale varies 200x across cells (0.04 on
# addition/mlp zeroed, 8.4 on nounpp/mlp zeroed) -- an absolute one would call everything a tie
# in the small-faith panels and nothing a tie in the large ones. Compared against the larger of
# the two values so the test is symmetric.
TOL_ACC, TOL_FAITH_REL = 0.0075, 0.10


def close(a, b, idx):
    """True if the two values are within the measured reproducibility floor on axis `idx`."""
    return (abs(a - b) <= TOL_ACC if idx == 0
            else abs(a - b) <= TOL_FAITH_REL * max(abs(a), abs(b)))


def ranks(data, idx, tol=True):
    """Per-cell competition ranks (1 = best) on axis `idx`, as {cell: {method: rank}}.

    Ties share the smaller rank, matching scipy's 'min' method: two methods that tie are both
    "joint first", and calling one of them second would invent a distinction the numbers do not
    support. So win counts can exceed the cell count -- a 3-way tie for first is three wins.

    With `tol`, "tie" means within the measured run-to-run floor (see TOL_ACC /
    TOL_FAITH_REL) rather than bitwise equal. That matters most at the `node` substrate, where
    the three MAttr arms routinely land inside 0.002 of each other and a strict ranking would
    report a confident ordering of three indistinguishable runs.

    The tolerance is applied to CONSECUTIVE pairs in sorted order, which is the standard
    (and only cheap) reading; it is not transitive, so a chain of near-neighbours spanning more
    than one tolerance can end up joint-ranked. With five points inside one panel that is the
    intended behaviour -- the chain is exactly the case where no ordering is defensible.
    Only complete cells are ranked, so a rank is always out of 5.
    """
    out = {}
    for cell, got in data.items():
        if len(got) < len(METHODS):
            continue
        order = sorted(got, key=lambda m: -got[m][idx])
        r, prev = {}, None
        for i, m in enumerate(order):
            same = prev is not None and (close(got[m][idx], got[prev][idx], idx) if tol
                                         else got[m][idx] == got[prev][idx])
            r[m] = r[prev] if same else i + 1
            prev = m
        out[cell] = r
    return out


def draw(ax, got):
    """One cell: five points, own scale, ~2 ticks per axis."""
    for m, _, c in METHODS:
        if m not in got:
            continue
        ax.scatter(*got[m], s=MS, facecolor=c, edgecolor="#000000", linewidth=0.3, zorder=3)
    xs = [got[m][0] for m, _, _ in METHODS if m in got]
    ys = [got[m][1] for m, _, _ in METHODS if m in got]
    for lim, vs in ((ax.set_xlim, xs), (ax.set_ylim, ys)):
        lo, hi = min(vs), max(vs)
        pad = (hi - lo) * MARGIN or (abs(hi) * MARGIN or 0.01)
        lim(lo - pad, hi + pad)
    # Two ticks per axis, at the data's own extremes rather than at round numbers: a 0.9in panel
    # fits two labels, and the pair that says most is the range the five points actually span.
    for axis, vs, fmt in ((ax.xaxis, xs, "{:.2f}"), (ax.yaxis, ys, "{:.2f}")):
        axis.set_ticks([min(vs), max(vs)])
        axis.set_major_formatter(lambda v, _p, f=fmt: f.format(v))
    P.furnish(ax)
    ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/sva_percell_rank.pdf")
    a = ap.parse_args()

    data = cells()
    short = {k: sorted({m for m, _, _ in METHODS} - set(v)) for k, v in data.items() if
             len(v) < len(METHODS)}
    if short:
        raise SystemExit(f"{len(short)} incomplete cells, e.g. "
                         f"{list(short.items())[:3]} -- a 5-way panel cannot be drawn from 4 "
                         "runs. Wait for the sweep, or drop the method.")

    plt.rcParams.update(P.RC)
    cols = [(res, rlab, sub, slab) for res, rlab in SOURCES for sub, slab in SUBS]
    fig, axes = plt.subplots(len(ROWS), len(cols), figsize=(FIG_W, FIG_H))
    for i, task in enumerate(ROWS):
        for j, (res, rlab, sub, slab) in enumerate(cols):
            ax = axes[i][j]
            if task in NODE_ONLY and sub != "node":
                ax.set_visible(False)          # structurally absent, not a pending run
                continue
            draw(ax, data[(res, sub, task)])
            if i == 0:
                ax.set_title(f"{rlab}\n{slab}", fontsize=FS_HEAD, pad=3)
            if j == 0:
                ax.set_ylabel(task.replace("_", "\n"), fontsize=FS_ROW, rotation=0,
                              ha="right", va="center", labelpad=12)

    fig.supxlabel("IIA AUC (↑)  —  per-panel scale, order is comparable and distance is not",
                  fontsize=FS_HEAD, y=0.012)
    fig.supylabel("Faith AUC (↑)", fontsize=FS_HEAD, x=0.008)
    fig.tight_layout(pad=0.25, w_pad=0.5, h_pad=0.55, rect=(0.012, 0.022, 1, 0.955))
    fig.legend(handles=[Line2D([], [], marker="o", ls="", markerfacecolor=c,
                               markeredgecolor="#000000", markeredgewidth=0.3, markersize=3.4,
                               label=lab) for _, lab, c in METHODS],
               fontsize=FS_LEG, ncol=len(METHODS), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=1.1)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out, f"({sum(len(v) for v in data.values())} points, {len(data)} cells)")

    # --- the ranking, as numbers. The figure shows order per panel; this aggregates it.
    for idx, name in ((0, "IIA AUC"), (1, "Faith AUC")):
        R = ranks(data, idx)
        print(f"\nmean rank on {name} (1 = best of 5; ties within the measured noise floor "
              f"{'±%.4f abs' % TOL_ACC if idx == 0 else '±%d%% rel' % (TOL_FAITH_REL * 100)} "
              "share a rank, so wins can exceed n):")
        hdr = "  ".join(f"{lab:>21}" for _, lab, _ in METHODS)
        print(f"  {'setting':<26} n   {hdr}")
        for res, rlab in SOURCES:
            for sub, slab in SUBS:
                ks = [k for k in R if k[0] == res and k[1] == sub]
                if not ks:
                    continue
                mr = [np.mean([R[k][m] for k in ks]) for m, _, _ in METHODS]
                w = [sum(R[k][m] == 1 for k in ks) for m, _, _ in METHODS]
                print(f"  {rlab + ' / ' + slab:<26} {len(ks):<3} "
                      + "  ".join(f"{v:>10.2f} ({n:>2} win)" for v, n in zip(mr, w)))
        ks = list(R)
        mr = [np.mean([R[k][m] for k in ks]) for m, _, _ in METHODS]
        strict = ranks(data, idx, tol=False)
        sr = [np.mean([strict[k][m] for k in ks]) for m, _, _ in METHODS]
        print(f"  {'ALL':<26} {len(ks):<3} " + "  ".join(f"{v:>21.2f}" for v in mr))
        # Strict ranks alongside, so the effect of the tolerance is visible rather than baked
        # in silently -- where the two rows disagree, the ordering is inside run-to-run noise.
        print(f"  {'ALL (strict, no tolerance)':<26} {len(ks):<3} "
              + "  ".join(f"{v:>21.2f}" for v in sr))


if __name__ == "__main__":
    sys.exit(main())
