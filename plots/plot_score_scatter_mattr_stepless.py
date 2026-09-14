"""\\ourmethod{}+Adam vs Expected Gradients, unit by unit, on addition / llama3 / MLP neurons.

WHY A DENSITY PLOT AND NOT A SCATTER. The substrate is 32 x 5 x 14,336 = 2,293,760 units. At
that count a scatter is a solid block wherever the bulk sits and tells you nothing about it;
every visible feature would be the few hundred points in the tails. hexbin with a log count
scale keeps the bulk readable and the tails visible at the same time.

*** BOTH AXES ARE asinh-SCALED, AND WITHOUT IT THERE IS NO FIGURE. *** The two methods live on
scales three orders of magnitude apart -- MAttr spans -4.7 to 12.4 with a median of -0.033,
Expected Gradients spans -0.135 to 0.282 with a median of exactly 0 and 21.9% of units exactly zero.
On linear axes the Expected Gradients axis collapses to a line. asinh is used rather than symlog
because it is smooth through zero, which matters when a fifth of one axis IS zero: symlog's
linear-to-log seam would fall inside the densest part of the data. `LINW` per axis is the width
below which the transform is effectively linear, set from each method's own 75th percentile of
|score| so neither axis is chosen to flatter the other.

*** PEARSON AND SPEARMAN DISAGREE BY 4x HERE AND THE FIGURE IS WHY. *** r = +0.304 against
rho = +0.081. The agreement is almost entirely in the HEAD: top-1000 overlap is 73.6% while
top-10,000 overlap is 32.8% and the full-vector rank correlation is near zero. So the ridge in
the upper right is real and the rest of the plane is not, and quoting r alone would describe a
relationship that holds for ~0.04% of the units. Both statistics are printed on the panel for
that reason.

RUNS ARE THE PUBLISHED ONES: the MAttr arm is eps=1e-2 (the configuration the paper ships;
torch's 1e-8 default degenerates to sign(g) at this scale) at 5,000 steps, from the tree
plot_accauc_vs_faithauc routes the MLP column to. Expected Gradients is alpha ~ U(0,1) drawn per
example at m=1, seed 42 -- one backward per example, the same cost as I x G.

Run:  uv run python plots/plot_score_scatter_mattr_stepless.py
Out:  plots/score_scatter_mattr_stepless.pdf  (+ .png)
"""
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402

RES = "results/sva_sweep_5k"
X = ("sufficient_topk_adam_eps1e-2_bs1_s5000", "\\ourmethod{}$+$Adam score")
Y = ("mc_ig_m1_s42", "Expected Gradients score")
CELL = "addition_llama3_mlp"
FIG_W, FIG_H = 3.4, 2.9
FS_LAB, FS_TICK, FS_NOTE = 7, 6, 5.5
GRID = 74
# Candidate tick values in ORIGINAL units; only those inside each axis's range are drawn. Chosen
# per decade rather than by a locator because the asinh transform makes an even spacing in
# transformed space land on unreadable numbers.
TICKS = [-10, -1, -0.1, -0.01, -0.001, 0, 0.001, 0.01, 0.1, 1, 10]


def load(tag):
    p = os.path.join(RES, f"{CELL}_{tag}.scores.pt")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    return torch.load(p, map_location="cpu").float().numpy()


def fmt(v):
    if v == 0:
        return "0"
    return f"{v:g}".replace("0.001", "10⁻³").replace("0.01", "10⁻²").replace("0.1", "10⁻¹")


def main():
    from scipy.stats import rankdata
    x, y = load(X[0]), load(Y[0])
    assert x.shape == y.shape, (x.shape, y.shape)
    # Linear width per axis: the 75th percentile of |score|, so the transform is linear through
    # the bulk of each distribution and compresses only its tail.
    wx, wy = np.percentile(np.abs(x), 75), np.percentile(np.abs(y), 75)
    tx, ty = np.arcsinh(x / wx), np.arcsinh(y / wy)

    r = float(np.corrcoef(x, y)[0, 1])
    rho = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    ov = {k: len(set(np.argpartition(x, -k)[-k:]) & set(np.argpartition(y, -k)[-k:])) / k
          for k in (100, 1000, 10000)}

    plt.rcParams.update(P.RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    hb = ax.hexbin(tx, ty, gridsize=GRID, cmap="viridis", norm=LogNorm(), linewidths=0,
                   mincnt=1)
    ax.axhline(0, color="#999999", lw=0.4, ls=(0, (1, 2)), zorder=1)
    ax.axvline(0, color="#999999", lw=0.4, ls=(0, (1, 2)), zorder=1)
    # TICKS ARE THINNED PER AXIS, and they have to be: the two linear widths differ by ~2,300x
    # (x 0.149, y 6.4e-5), so the same candidate list that is well spread on the Expected Gradients axis
    # collapses to an unreadable pile around zero on the MAttr one -- 10^-3, 10^-2 and 10^-1 all
    # land inside 0.7 of a unit there. Greedy left-to-right with a minimum separation of 6% of
    # the axis span keeps whichever candidates are actually distinguishable.
    for w, setter, lo, hi in ((wx, "x", tx.min(), tx.max()), (wy, "y", ty.min(), ty.max())):
        # ZERO IS KEPT FIRST, then candidates are accepted outward only if they clear EVERY tick
        # already kept by `gap`. A left-to-right sweep was tried and drops zero on the MAttr axis
        # (its neighbours at +-10^-1 sit 0.63 units away and win the sweep), which is the one
        # tick this plot cannot lose -- both distributions are centred there. 12% of the span is
        # set from the label width, not the data: "-10^-1" at 6pt is ~0.25in against a ~2.4in
        # axis, and 6% left three labels overlapping in a 0.16in stretch.
        gap = 0.12 * (hi - lo)
        kept = [0.0] if lo <= 0 <= hi else []
        for v in sorted(TICKS, key=abs):
            t = np.arcsinh(v / w)
            if lo <= t <= hi and all(abs(t - np.arcsinh(u / w)) >= gap for u in kept):
                kept.append(float(v))
        kept.sort()
        (ax.set_xticks if setter == "x" else ax.set_yticks)([np.arcsinh(v / w) for v in kept])
        (ax.set_xticklabels if setter == "x" else ax.set_yticklabels)([fmt(v) for v in kept])
    ax.set_xlabel(X[1].replace("\\ourmethod{}", "MAttr"), fontsize=FS_LAB)
    ax.set_ylabel(Y[1], fontsize=FS_LAB)
    ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1.5)
    P.furnish(ax)
    ax.set_axisbelow(False)          # the hairline grid must not sit over the density
    note = (f"Pearson $r$ = {r:+.3f}\nSpearman $\\rho$ = {rho:+.3f}\n"
            f"top-1k overlap = {ov[1000]:.0%}\ntop-10k overlap = {ov[10000]:.0%}")
    ax.text(0.03, 0.97, note, transform=ax.transAxes, fontsize=FS_NOTE, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#cccccc", lw=0.4))
    cb = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("units per bin", fontsize=FS_LAB, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=1.2, pad=1.0)
    cb.outline.set_linewidth(0.5)

    fig.tight_layout(pad=0.3)
    out = "plots/score_scatter_mattr_stepless.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {out}   n={x.size:,}")
    print(f"  Pearson r {r:+.4f}   Spearman rho {rho:+.4f}")
    for k, v in ov.items():
        print(f"  top-{k:<6} overlap {v:.1%}")
    print(f"  asinh linear width: x {wx:.4g}, y {wy:.4g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
