"""tabs/sparsity_sweep.tex as one figure: both metrics against each baseline's sparsity knob.

The sibling of plot_lr_sweep_summary.py, over the OTHER sweep table -- rows are the two metrics,
columns are the two blocks of make_lr_table.SPARSITY_METHODS (Node Pruning's target sparsity s,
DBM's L1 coefficient). Same import-the-table discipline: the blocks come from that module, so
"the blocks of the table" and "the panels of this figure" are one list and an unstyled block
RAISES rather than quietly vanishing.

WHAT IT SHOWS, and it is not what the CPR table alone suggests: THE TWO METRICS DISAGREE ABOUT
THE KNOB, in the same direction for both methods.

    Node Pruning   CPR AUC   peaks at s=0.5 (1.67), falls to 1.24 by s=0.99
                   acc-AUC   rises MONOTONICALLY across the whole grid, 0.09 -> 0.38
    DBM            CPR AUC   peaks at lambda=6 (1.50), falls at 20
                   acc-AUC   rises MONOTONICALLY, 0.19 -> 0.35

That is the cpr-auc-is-dense-end-dominated fact made visible: MIB's `area_under` is a linear
trapezoid over 0.001--1.0 sparsity, so ~90% of it comes from k>=20% where a denser circuit simply
scores better, while acc-AUC is log-normalised and reads the sparse end. Turning either knob up
trades one for the other. So "the best sparsity setting" is metric-dependent, and
make_mib_table.EPRUN_BEST_SPARSITY (s=0.5, picked by CPR) is SIXTH OF SEVEN on acc-AUC. That
caveat is already in that constant's comment; this figure is where it is legible.

THE RING RULE CARRIES THE POINT BY ITSELF. As in the LR figure, an argmax is ringed only when it
is INTERIOR to the swept grid. Both CPR panels ring; NEITHER acc-AUC panel does, because both
acc-AUC curves are still climbing at the sparsest setting we ran. An unringed maximum here is a
statement that the grid does not bracket the optimum -- these two sweeps do not establish a best
acc-AUC operating point, they establish that it is sparser than anything swept.

NO SHARED X AXIS, which is the deliberate difference from plot_lr_sweep_summary.py. There both
columns were learning rates and sharex was the whole point (the optima sit at different LRs on
one scale). Here the columns sweep different quantities in different units -- a target sparsity
in [0,1] against an unbounded L1 coefficient -- and putting them on one axis would invite reading
"s=0.5" against "lambda=0.5" as the same setting. The y axes ARE shared per row, so the vertical
comparison across columns stays honest: Node Pruning's knob moves CPR over a ~1.3 range and DBM's
over ~0.19, and that difference in leverage is real, not a scaling choice.

X SCALES DIFFER PER COLUMN for the same reason. s is plotted LINEARLY: it is a fraction, its
grid is not geometric, and the CPR turnover at 0.5 sits mid-axis where it reads. lambda is
plotted on a SYMLOG axis with linthresh below the smallest nonzero point, because its grid IS
geometric (0.2/0.6/2/6/20, x3 apart) and because lambda=0 is a real swept point -- the
unpenalised control -- that a log axis cannot place at all.

THE X AXIS IS A REQUEST, NOT AN OUTCOME, for the Node Pruning column. s is the L0 anneal's
TARGET and the anneal does not reach it: logs/eprun_126314.out ends at sparsity 0.8333 against
target 0.9000. Achieved sparsity is NOT in the artifacts this script reads -- the graph JSONs
store unthresholded logits for MIB's own top-k sweep to re-threshold, and `in_graph` is False on
every maskable node -- so it cannot be plotted here without mining the training logs. Do not
caption the s axis as achieved sparsity. (See node-pruning-l0-anneal-steps.)

COLOURS MATCH plot_lr_sweep_summary.py cell-for-cell (indigo Node Pruning, pink DBM, both from
palette.py), so the two sweep figures read as one pair. There is no legend: one series per panel,
already named by the column title.

Run:  uv run python plots/plot_sparsity_sweep_summary.py
Out:  plots/sparsity_sweep_summary.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import palette as P                                     # noqa: E402
import plot_mib_accauc_cpr_scatter as S                 # RC only  # noqa: E402
import make_lr_table as M                               # SPARSITY_METHODS + cpr()  # noqa: E402
# The row-label parser is IMPORTED, not restated. Both sweep tables write their knob values as
# "<number>" or "<number> (<note>)" and both figures have to strip the note the same way; two
# copies of that regex is how the two figures would come to disagree about which rows are
# plottable. This is a pure function -- importing it cannot reshape the other figure the way
# importing a layout CONSTANT can (see plot_train_curves.py's ROW_H warning).
from plot_lr_sweep_summary import lr_of as knob_of      # noqa: E402

# block name (verbatim from make_lr_table.SPARSITY_METHODS) -> (column title, x label, colour,
# x-scale). Titles are shortened from the table's block names, which carry the fixed hyperparameter
# inline ("Node Pruning (logit-diff, LR $=$ 0.8)"); that belongs in a caption, not on an axis.
STYLE = {
    "Node Pruning (logit-diff, LR $=$ 0.8)":
        ("Node Pruning (logit-diff, lr 0.8)", "target sparsity $s$",
         P.METHOD["Node Pruning"], "linear"),
    "DBM $+$ L1 (lr $=$ 0.3)":
        ("DBM (logit-diff, lr 0.3)", r"L1 coefficient $\lambda$",
         P.METHOD["DBM"], "symlog"),
}
METRICS = [("area_under", "CPR AUC (↑)"), ("acc_auc", "IIA log-AUC (↑)")]
FIG_W, ROW_H = 5.4, 1.55
FS_LABEL, FS_TICK, FS_ANNOT = 7.5, 7, 6
TITLE_H = 0.20          # set_title draws ABOVE the axes rect, so the header strip must hold it
SYMLOG_LINTHRESH = 0.15  # below the smallest nonzero lambda (0.2), so only 0 sits in the linear leg


def load():
    """{block: [(knob, {metric: mean over the 11 cells or None})]}, plus the exclusion log.

    Per-metric completeness over make_lr_table.COLUMNS, the same rule and the same 11 MIB
    validation cells as plot_lr_sweep_summary.load(). As of this writing NOTHING is excluded --
    all 13 rows are 11/11 on both metrics -- but the check stays: a partially-landed row would
    otherwise contribute a mean over a different cell set than the point next to it.
    """
    out, skipped = {}, []
    for name, vals, *_ in M.SPARSITY_METHODS:
        if name not in STYLE:
            raise SystemExit(f"block {name!r} is in sparsity_sweep.tex but has no STYLE entry -- "
                             "add one (see the module docstring) rather than letting it vanish")
        pts = []
        for lab, dirn in vals:
            knob = knob_of(lab)
            if knob is None:
                skipped.append(f"  {name} row {lab!r} ({dirn}): not a knob row -- dropped")
                continue
            rec, miss = {}, []
            for key, _ in METRICS:
                got = [v for v in (M.cpr(dirn, t, m, key) for t, m, _ in M.COLUMNS)
                       if v is not None]
                rec[key] = float(np.mean(got)) if len(got) == len(M.COLUMNS) else None
                if rec[key] is None:
                    miss.append(f"{key} {len(got)}/{len(M.COLUMNS)}")
            if miss:
                skipped.append(f"  {name} {lab} ({dirn}): {', '.join(miss)} -- dropped from "
                               f"{'those panels' if any(rec.values()) else 'both panels'}")
            if any(v is not None for v in rec.values()):
                pts.append((knob, rec))
        out[name] = sorted(pts)
    return out, skipped


def draw(ax, pts, colr, xscale, metric):
    """One panel: the block's curve for one metric, argmax ringed only when interior."""
    xy = [(k, r[metric]) for k, r in pts if r[metric] is not None]
    if xy:
        x, y = [p[0] for p in xy], [p[1] for p in xy]
        ax.plot(x, y, "-", lw=0.9, color=colr, zorder=2)
        ax.plot(x, y, "s", ms=2.6, color=colr, mec="#000000", mew=0.35, ls="none", zorder=3)
        bi = int(np.argmax(y))
        if 0 < bi < len(y) - 1:
            ax.plot([x[bi]], [y[bi]], "o", ms=6.5, mfc="none", mec=colr, mew=0.8, zorder=4)
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.4)
        # Tick the SWEPT VALUES, not the decades. Symlog's default locator labels 1e-1/1e0/1e1,
        # and not one of those three is a setting we ran -- a reader would take the axis for a
        # continuum that was sampled at those points. These ticks are exactly the six rows of the
        # table's DBM block, so every label has a square sitting on it and gaps in the grid are
        # gaps in the sweep. Minor ticks off for the same reason.
        ax.set_xticks([k for k, _ in pts])
        ax.set_xticklabels([f"{k:g}" for k, _ in pts])
        ax.set_xticks([], minor=True)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/sparsity_sweep_summary.pdf")
    a = ap.parse_args()

    data, skipped = load()
    blocks = [n for n, *_ in M.SPARSITY_METHODS]
    plt.rcParams.update(S.RC)
    nr, nc = len(METRICS), len(blocks)
    fh = ROW_H * nr + TITLE_H
    # sharey per ROW only. sharex is deliberately OFF -- see the docstring; the two columns are
    # different quantities and squeeze=False keeps the indexing uniform either way.
    fig, axes = plt.subplots(nr, nc, figsize=(FIG_W, fh), sharey="row", squeeze=False)
    for r, (metric, ylab) in enumerate(METRICS):
        for c, name in enumerate(blocks):
            title, xlab, colr, xscale = STYLE[name]
            ax = axes[r][c]
            draw(ax, data[name], colr, xscale, metric)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=FS_LABEL)
            if r == 0:
                ax.set_title(title, fontsize=FS_LABEL, pad=3)
            if r == nr - 1:
                ax.set_xlabel(xlab, fontsize=FS_LABEL)

    fig.tight_layout()
    fig.subplots_adjust(top=1.0 - TITLE_H / fh)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    if skipped:
        print(f"\nexcluded ({len(M.COLUMNS)} MIB validation cells required per metric):")
        print("\n".join(skipped))
    else:
        print(f"\nno exclusions: every row is {len(M.COLUMNS)}/{len(M.COLUMNS)} on both metrics")
    print("\nbest knob per block:")
    for name in blocks:
        for metric, _ in METRICS:
            xy = [(k, r[metric]) for k, r in data[name] if r[metric] is not None]
            if not xy:
                print(f"  {STYLE[name][0]:<30} {metric:<10} -- no complete row")
                continue
            bi = int(np.argmax([p[1] for p in xy]))
            edge = "" if 0 < bi < len(xy) - 1 else "  (grid endpoint -- optimum NOT bracketed)"
            print(f"  {STYLE[name][0]:<30} {metric:<10} n={len(xy)}  best {xy[bi][1]:.3f} "
                  f"@ {xy[bi][0]:g}{edge}")


if __name__ == "__main__":
    sys.exit(main())
