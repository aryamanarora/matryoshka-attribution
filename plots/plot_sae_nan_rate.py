"""Where the SAE residual intervention diverges: non-finite rate per method x task.

WHAT THIS IS. With the numerical guard removed from `_sae_interchange` (2026-08-29), the
Llama-Scope residual-SAE intervention diverges on some cells: the base branch compounds ~1000x
per layer from layer ~19, overflows fp32 by layer 24, and the run finishes with a non-finite
AUC. This figure is the map of which cells. It is a DIAGNOSTIC, not a result -- the numbers it
plots are failure rates, not scores.

The guard used to hide all of this by rescaling the diverged activation to 8x the source norm,
which silently substituted a fabricated activation at exactly the sparsities where the
intervention had already blown up. Measured on 10 matched pairs, that inflated acc-AUC by up to
0.087 (mean +0.013, never negative). So every SAE number produced before that date is biased
upward by an unknown amount on an unknown subset of cells -- which is why they were quarantined
rather than corrected.

THE TWO THINGS THE MAP SHOWS, both of which a table of totals hides:

  * IT IS A TASK-FAMILY EFFECT, not a method effect. All four arithmetic tasks fail heavily
    (54-69%); three of the four SVA tasks never fail at all. That matches the pre-removal
    finding that arith is where the SAE reconstruction is worst -- where the error node ranked
    #0 of 6.29M units and IIA sat at the 0.5 coin-flip plateau.

  * RANDOM NEVER DIVERGES, 0 of 24. A random ranking spreads its top-k uniformly over 6.29M
    units, so it touches few error nodes and no coherent set of latents. Every method that
    actually SELECTS -- gradient or learned, it makes no difference -- concentrates the budget
    on high-leverage units, and that concentration is what triggers the loop. Divergence is a
    property of selecting well, not of any one estimator. The Random row is therefore the
    control that makes the rest of the figure interpretable, and must not be dropped as "the
    boring row".

Cell text is n_nonfinite / n_runs, so a cell can never be read as a rate without its
denominator (methods differ: 3 losses for the gradient arms and MAttr+SGD, 3 seeds for Random,
1 for the eps arm, which is logit-diff only).

Run:  uv run python plots/plot_sae_nan_rate.py
Out:  plots/sae_nan_rate.pdf  (+ .png)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402
import plot_accauc_vs_faithauc as V                    # parse_method, task lists  # noqa: E402

RES = "results/sva_sweep"
ORDER = ["Random", "IxG", "IG", "softsgd-log", "stopk-log-eps1e-2"]
TASKS = V.SVA + V.ARITH
FIG_W, FIG_H = 5.5, 1.75
FS_TICK, FS_CELL, FS_LAB = 6.5, 5.5, 6.5


def tally():
    """{(method, task): [n_runs, n_nonfinite]} over the post-fix SAE residual runs."""
    out = {}
    for f in glob.glob(f"{RES}/*_resid_sae_span_*.json"):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = V.parse_method(os.path.basename(f), d)
        if m is None:
            continue
        ok = np.isfinite(d.get("acc_auc", np.nan)) and np.isfinite(d.get("faith_auc", np.nan))
        c = out.setdefault((m, d["task"]), [0, 0])
        c[0] += 1
        c[1] += 0 if ok else 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/sae_nan_rate.pdf")
    a = ap.parse_args()
    T = tally()
    rows = [m for m in ORDER if any(k[0] == m for k in T)]
    if not rows:
        raise SystemExit(f"no SAE residual runs in {RES}")

    rate = np.full((len(rows), len(TASKS)), np.nan)
    txt = [["" for _ in TASKS] for _ in rows]
    for i, m in enumerate(rows):
        for j, t in enumerate(TASKS):
            n, b = T.get((m, t), [0, 0])
            if n:
                rate[i, j] = b / n
                txt[i][j] = f"{b}/{n}"

    plt.rcParams.update(P.RC)
    # White -> the palette's I x G wine. A sequential ramp anchored at white so 0 reads as
    # "nothing wrong" rather than as a colour; the dark end borrows an existing hue instead of
    # inventing one, since this figure shares a page with the method figures.
    cmap = LinearSegmentedColormap.from_list("nan", ["#ffffff", P.color("I×G")])
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.imshow(rate, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(TASKS)):
            if not txt[i][j]:
                continue
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=FS_CELL,
                    color="#ffffff" if rate[i, j] > 0.55 else "#000000")
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([t.replace("_", "\n") for t in TASKS], fontsize=FS_TICK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([V.METHODS[m][0] for m in rows], fontsize=FS_TICK)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(TASKS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    # GREY, not white: a 0/3 cell is white, so white separators made the clean half of the map
    # dissolve into the page and the row read as floating text with no grid.
    ax.grid(which="minor", color="#bbbbbb", lw=0.6)
    ax.tick_params(which="minor", length=0)
    # The SVA / Arith boundary is the whole finding; mark it.
    ax.axvline(len(V.SVA) - 0.5, color="#000000", lw=0.8)
    ax.text(len(V.SVA) / 2 - 0.5, -0.85, "SVA", ha="center", fontsize=FS_LAB)
    ax.text(len(V.SVA) + len(V.ARITH) / 2 - 0.5, -0.85, "Arithmetic", ha="center", fontsize=FS_LAB)
    ax.set_ylim(len(rows) - 0.5, -1.15)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("non-finite AUC", fontsize=FS_LAB)
    cb.ax.tick_params(labelsize=FS_CELL, length=2)
    cb.outline.set_linewidth(0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)
    tot = np.nansum([T[k][1] for k in T]), sum(T[k][0] for k in T)
    print(f"\noverall: {tot[0]:.0f}/{tot[1]} runs non-finite")
    for i, m in enumerate(rows):
        n = sum(T.get((m, t), [0, 0])[0] for t in TASKS)
        b = sum(T.get((m, t), [0, 0])[1] for t in TASKS)
        print(f"  {V.METHODS[m][0]:<24}{b:>3}/{n:<3}")


if __name__ == "__main__":
    sys.exit(main())
