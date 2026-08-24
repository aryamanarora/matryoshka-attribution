"""Every LR sweep in tabs/lr_sweep.tex as one full-width figure: metric vs learning rate.

This is plot_optimizer_lr.py's chart (mean over the 11 MIB validation cells against lr, each
arm's own optimum ringed) widened from the 4 MAttr optimizer arms to ALL NINE blocks of the LR
table, so the appendix table's ~50 rows have a summary a reader can take in at once. The thing
the table makes hard to see and this makes obvious is that the optima sit at DIFFERENT
learning rates -- MAttr+Adam at 0.05-0.1, MAttr+SGD at 1.0, DBM at 0.3, Node Pruning at 0.1 --
so any comparison at one shared LR is reading distance-from-own-optimum, not method quality.

THE SERIES ARE IMPORTED FROM make_lr_table.LR_METHODS, not restated. That module is what
generates lr_sweep.tex, so "the blocks of the table" and "the series of this figure" are the
same list by construction and cannot drift apart. Adding a block to the table therefore lands
here automatically -- and if it has no entry in STYLE below, this script RAISES rather than
silently dropping it, because a block missing from a figure captioned "all of them" is the one
failure mode that looks like a result.

FACETS ARE OURS vs THE MASK-LEARNING BASELINES (columns), METRIC (rows). Note this is NOT
plot_mib_accauc_cpr_scatter's GRADIENT/MASK split: MAttr is itself a mask-learning method, so
by that taxonomy all nine blocks are one family. The split here is the one the table's own
ordering implies -- the six MAttr variants, then the three published baselines we swept.

COMPLETENESS IS PER-METRIC, which deliberately DIFFERS from build_lr_rows' rule. That function
demands 11/11 cells on acc AND CPR because it feeds a scatter of one against the other, where a
point needs both coordinates. These are separate panels, so a CPR mean over 11/11 cells is
comparable to every other CPR mean whatever the acc coverage of the same dir. Applying the
stricter rule here would drop 3 of the 5 "+ unif k, + hard" LRs -- all 11/11 on CPR -- out of
the CPR panel over a missing acc cell, i.e. hide measured data to satisfy a constraint the
panel does not have. What is NOT relaxed is the 11/11 bar itself: a mean over 10 cells is not
comparable to a mean over 11, and every exclusion is printed with its counts.

NON-LR ROWS ARE DROPPED, with a printed reason. The REINFORCE block carries a "0.1, 2k" row,
which varies the STEP COUNT at an lr already in the block; on an lr axis it would plot a second,
better point at x=0.1 and read as LR sensitivity. (plot_mib_accauc_cpr_scatter excludes it for
the same reason.) The DBM "0.001 (pyvene)" and Node Pruning "0.8 (default)" rows ARE lr rows --
only their parenthetical is stripped.

ENCODING: colour is the FORWARD/optimizer family and linetype is the k-schedule, the same rule
plot_optimizer_lr.py and plot_train_curves.py use, so a reader moving between the three figures
does not relearn it. Blue = MAttr as published (Adam), black = the SGD arm, green = the hard-STE
family (with REINFORCE dash-dotted as a third member -- it is a hard BACKWARD, so it belongs to
that family), pink = DBM, indigo = Node Pruning. All hexes come from palette.py; none is local.

RINGS MARK EACH ARM'S ARGMAX, and an arm whose argmax is at an ENDPOINT of its own grid gets no
ring -- an endpoint maximum is "the best lr we swept", not a measured optimum, and ringing it
would claim a turnover that was never observed. plot_optimizer_lr hardcodes that exclusion as
NO_RING; here it is computed, since nine grids are too many to track by hand.

Run:  uv run python plots/plot_lr_sweep_summary.py
Out:  plots/lr_sweep_summary.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import palette as P                                     # noqa: E402
import plot_mib_accauc_cpr_scatter as S                 # _pair, COLS, RC   # noqa: E402
import make_lr_table as M                               # LR_METHODS -- the table's own list  # noqa: E402

OURS, BASE = "\\ourmethod{} and ablations", "Mask-learning baselines"
# block name (verbatim from make_lr_table.LR_METHODS) -> (facet, legend label, colour, linestyle).
# Legend labels are the TABLE's row labels, flattened -- the figure is a view of lr_sweep.tex and
# a reader holding both should not have to translate. That is also why the first block reads
# "MAttr" and not "MAttr (Adam)" even though the topklog_* dirs are Adam: the table names its
# ablations RELATIVE to that row ("$+$ SGD"), so renaming it here would break the correspondence
# in the one direction that matters.
# Dashes are given as explicit on/off point tuples, not the "dashed"/"dashdot" names. Matplotlib
# scales the NAMED patterns by linewidth, so at lw=0.9 a named dashdot is ~3pt per cycle and its
# dot merges with its dash in a 1.9x legend handle -- "$+$ hard bwd" printed as a solid green line
# identical to "$+$ hard". These patterns are fixed in points and survive both the thin line and
# the short handle; check the LEGEND, not the axes, when changing them.
DASH, DOTDASH = (0, (3.2, 1.4)), (0, (4.5, 1.2, 0.9, 1.2))
STYLE = {
    "\\ourmethod{}":            (OURS, "MAttr",              P.METHOD["MAttr"],        "solid"),
    "$+$ SGD":                  (OURS, "$+$ SGD",            P.METHOD["MAttr (SGD)"],  "solid"),
    "$+$ SGD, $+$ unif $k$":    (OURS, "$+$ SGD, unif. $k$", P.METHOD["MAttr (SGD)"],  DASH),
    "$+$ hard":                 (OURS, "$+$ hard",           P.METHOD["+hard"],        "solid"),
    "$+$ unif $k$, $+$ hard":   (OURS, "$+$ hard, unif. $k$", P.METHOD["+hard"],       DASH),
    "$+$ hard bwd (REINFORCE)": (OURS, "$+$ hard bwd",       P.METHOD["+hard"],        DOTDASH),
    "DBM":                                 (BASE, "DBM",              P.METHOD["DBM"], "solid"),
    "Node Pruning ($s{=}0.5$, logit-diff)": (BASE, "Node Pruning, $s{=}0.5$",
                                            P.METHOD["Node Pruning"], "solid"),
    "Node Pruning ($s{=}0.8$, logit-diff)": (BASE, "Node Pruning, $s{=}0.8$",
                                            P.METHOD["Node Pruning"], DASH),
}
FACETS = [OURS, BASE]
METRICS = [("cpr", "CPR AUC (↑)"), ("acc", "IIA log-AUC (↑)")]
FIG_W, ROW_H = 5.4, 1.55
FS_LABEL, FS_TICK, FS_LEGEND = 7.5, 7, 6
# Header strip, split as in plot_train_curves: handles plus the column titles, which set_title
# draws ABOVE the axes rectangle and so into the same band.
LEG_H, TITLE_H = 0.50, 0.20


def lr_of(label):
    """The numeric lr in a table row label, or None if the row does not vary lr.

    Accepts a bare number or a number with a trailing parenthetical ("0.001 (pyvene)",
    "0.8 (default)"). Rejects "0.1, 2k" -- see the module docstring.
    """
    m = re.fullmatch(r"\s*([\d.]+)\s*(\([^)]*\))?\s*", label)
    return float(m.group(1)) if m else None


def load():
    """{block: [(lr, {metric: mean or None})]}, plus the printed exclusion log.

    Per-metric completeness: a metric's mean is kept only if all len(S.COLS) cells have it.
    """
    out, skipped = {}, []
    for name, vals, *_ in M.LR_METHODS:
        if name not in STYLE:
            raise SystemExit(f"block {name!r} is in lr_sweep.tex but has no STYLE entry -- "
                             "add one (see the module docstring) rather than letting it vanish")
        pts = []
        for lab, dirn in vals:
            lr = lr_of(lab)
            if lr is None:
                skipped.append(f"  {name} row {lab!r} ({dirn}): not an lr row -- dropped")
                continue
            pairs = [S._pair(dirn, t, m) for t, m, _ in S.COLS]
            vals_by = {"acc": [a for a, _ in pairs if a is not None],
                       "cpr": [c for _, c in pairs if c is not None]}
            rec, miss = {}, []
            for key in ("cpr", "acc"):
                got = vals_by[key]
                if len(got) == len(S.COLS):
                    rec[key] = float(np.mean(got))
                else:
                    rec[key] = None
                    miss.append(f"{key} {len(got)}/{len(S.COLS)}")
            if miss:
                skipped.append(f"  {name} lr={lab} ({dirn}): {', '.join(miss)} -- "
                               f"dropped from {'those panels' if rec else 'both panels'}")
            if any(v is not None for v in rec.values()):
                pts.append((lr, rec))
        out[name] = sorted(pts)
    return out, skipped


def draw(ax, data, facet, metric):
    """One panel. Returns the legend handles for this facet, in STYLE order."""
    handles = []
    for name, (fct, leg, colr, ls) in STYLE.items():
        if fct != facet:
            continue
        pts = [(lr, r[metric]) for lr, r in data[name] if r[metric] is not None]
        handles.append(Line2D([0], [0], color=colr, ls=ls, lw=0.9, marker="s", ms=2.6,
                              mec="#000000", mew=0.35, label=leg))
        if not pts:
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        if len(pts) > 1:
            ax.plot(x, y, ls=ls, lw=0.9, color=colr, zorder=2)
            # Ring the argmax only when it is INTERIOR to the swept grid -- see the docstring.
            bi = int(np.argmax(y))
            if 0 < bi < len(y) - 1:
                ax.plot([x[bi]], [y[bi]], "o", ms=6.5, mfc="none", mec=colr, mew=0.8, zorder=4)
        ax.plot(x, y, "s", ms=2.6, color=colr, mec="#000000", mew=0.35, ls="none", zorder=3)
    ax.set_xscale("log")
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK)
    return handles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/lr_sweep_summary.pdf")
    a = ap.parse_args()

    data, skipped = load()
    plt.rcParams.update(S.RC)
    nr, nc = len(METRICS), len(FACETS)
    head = LEG_H + TITLE_H
    fh = ROW_H * nr + head
    # sharex across BOTH columns on purpose: the whole point is that the optima sit at different
    # learning rates, which is only legible if 0.3 is the same place in every panel.
    fig, axes = plt.subplots(nr, nc, figsize=(FIG_W, fh), sharex=True, sharey="row",
                             squeeze=False)
    legs = {}
    for r, (metric, ylab) in enumerate(METRICS):
        for c, facet in enumerate(FACETS):
            ax = axes[r][c]
            legs[facet] = draw(ax, data, facet, metric)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=FS_LABEL)
            if r == 0:
                ax.set_title(facet.replace("\\ourmethod{}", "MAttr"), fontsize=FS_LABEL, pad=3)
            if r == nr - 1:
                ax.set_xlabel("learning rate", fontsize=FS_LABEL)

    fig.tight_layout()
    top = 1.0 - head / fh
    fig.subplots_adjust(top=top)
    # One legend per COLUMN, over that column, because the two facets share no series. Anchored
    # a title's height above `top` -- the titles live above the axes box, so an anchor at top+0
    # lands on them even though subplots_adjust reserved the strip.
    for c, facet in enumerate(FACETS):
        box = axes[0][c].get_position()
        fig.legend(handles=legs[facet], fontsize=FS_LEGEND, ncol=2 if c == 0 else 1,
                   loc="lower center", bbox_to_anchor=(box.x0 + box.width / 2,
                                                       top + TITLE_H / fh),
                   frameon=False, handletextpad=0.4, handlelength=2.6, columnspacing=1.0,
                   labelspacing=0.25)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    if skipped:
        print(f"\nexcluded ({len(S.COLS)} MIB validation cells required per metric):")
        print("\n".join(skipped))
    print("\nbest lr per block (n = lr values plotted):")
    for name, (fct, leg, _, _) in STYLE.items():
        for metric, _ in METRICS:
            pts = [(lr, r[metric]) for lr, r in data[name] if r[metric] is not None]
            if not pts:
                print(f"  {leg:<24} {metric:<4} -- no complete lr")
                continue
            bi = int(np.argmax([p[1] for p in pts]))
            edge = "" if 0 < bi < len(pts) - 1 else "  (grid endpoint, not an optimum)"
            print(f"  {leg:<24} {metric:<4} n={len(pts)}  best {pts[bi][1]:.3f} "
                  f"@ lr={pts[bi][0]:g}{edge}")


if __name__ == "__main__":
    sys.exit(main())
