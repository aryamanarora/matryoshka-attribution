"""tabs/sparsity_sweep.tex as one figure: both metrics against each baseline's sparsity knob.

The sibling of plot_lr_sweep_summary.py, over the OTHER sweep table -- rows are the two metrics,
columns are the two blocks of make_lr_table.SPARSITY_METHODS (Node Pruning's target sparsity s,
DBM's L1 coefficient). Same import-the-table discipline: the blocks come from that module, so
"the blocks of the table" and "the columns of this figure" are one list and an unstyled block
RAISES rather than quietly vanishing.

WHAT IT SHOWS, and it is not what the CPR table alone suggests: THE TWO METRICS DISAGREE ABOUT
THE KNOB, in the same direction for every series.

    Node Pruning (logit-diff)  CPR AUC  peaks at s=0.5 (1.67), falls to 1.24 by s=0.99
                               acc-AUC  rises MONOTONICALLY across the whole grid, 0.09 -> 0.38
    Node Pruning (KL)          CPR AUC  strictly lower (~0.91--1.00) at every shared setting
                               acc-AUC  strictly HIGHER (~0.40--0.46), above logit-diff's best
    DBM                        CPR AUC  peaks at lambda=6 (1.50), falls at 20
                               acc-AUC  rises MONOTONICALLY, 0.19 -> 0.35

That is the cpr-auc-is-dense-end-dominated fact made visible: MIB's `area_under` is a linear
trapezoid over 0.001--1.0 sparsity, so ~90% of it comes from k>=20% where a denser circuit simply
scores better, while acc-AUC is log-normalised and reads the sparse end. Anything that makes the
circuit sparser trades one metric for the other. So "the best sparsity setting" is
metric-dependent, and make_mib_table.EPRUN_BEST_SPARSITY (s=0.5, picked by CPR) is SIXTH OF SEVEN
on acc-AUC. That caveat is already in that constant's comment; this figure is where it is legible.

BOTH NODE PRUNING LOSSES ARE DRAWN, and the pair is the cleanest evidence for that reading. KL is
Edge Pruning's own objective and logit-diff is MAttr's; the KL runs reach a much sparser circuit
at the same target (see plot_achieved_sparsity.py) and their two metrics move in exactly the
opposite directions -- worse CPR, better acc-AUC. The objective is not what separates them on
these two metrics; DENSITY is, and the objective matters here only because it changes how close
the Lagrangian gets to its target.

THE X AXIS IS A REQUEST, NOT AN OUTCOME. s is the L0 anneal's TARGET and the anneal does not
reach it -- it saturates around 0.86 on the logit-diff series, so the three sparsest settings in
this grid are very nearly the same circuit. That is a separate figure
(plots/plot_achieved_sparsity.py, which is also where the log-mining lives); read it before
reading anything into the right-hand end of these panels, and do not caption this axis as
achieved sparsity. See node-pruning-l0-anneal-steps.

THE RING RULE CARRIES THE POINT BY ITSELF. As in the LR figure, an argmax is ringed only when it
is INTERIOR to the swept grid -- and, added here, only when it is the STRICT unique maximum. A
tie between two settings does not identify an operating point either, and KL's acc-AUC ties its
top two to 3dp, which a bare argmax would have silently ringed as the first of them. Both CPR
panels ring the logit-diff series; NO acc-AUC panel rings anything, because every acc-AUC curve
is still climbing at the sparsest setting we ran. An unringed maximum is a statement that the
grid does not bracket the optimum.

NO SHARED X AXIS, which is the deliberate difference from plot_lr_sweep_summary.py. There both
columns were learning rates and sharex was the whole point (the optima sit at different LRs on
one scale). Here the columns sweep different quantities in different units -- a target sparsity
against an unbounded L1 coefficient -- and putting them on one axis would invite reading "s=0.5"
against "lambda=0.5" as the same setting. The y axes ARE shared per row, so the vertical
comparison across columns stays honest: Node Pruning's knob moves CPR over a ~1.3 range and DBM's
over ~0.19, and that difference in leverage is real, not a scaling choice.

X SCALES DIFFER PER COLUMN for the same reason. s is plotted LINEARLY: it is a fraction, its
grid is not geometric, and the CPR turnover at 0.5 sits mid-axis where it reads. lambda is
plotted on a SYMLOG axis with linthresh below the smallest nonzero point, because its grid IS
geometric (0.2/0.6/2/6/20, x3 apart) and because lambda=0 is a real swept point -- the
unpenalised control -- that a log axis cannot place at all.

COLOURS MATCH plot_lr_sweep_summary.py cell-for-cell (indigo Node Pruning, pink DBM, both from
palette.py), so the two sweep figures read as one pair, and linetype separates the two Node
Pruning objectives the way it separates k-schedules there.

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
# The KL series is shared with the achieved-sparsity figure and defined there, next to the
# comment explaining why its s=0.9 dir is the unsuffixed one.
from plot_achieved_sparsity import KL_ROWS              # noqa: E402

# Explicit point tuples, not the named "dashed" -- matplotlib scales the named patterns by
# linewidth, and at lw=0.9 they collapse toward solid inside a short legend handle. See the
# comment in plot_lr_sweep_summary.py, which this figure is styled to match.
DASH = (0, (3.2, 1.4))

# block name (verbatim from make_lr_table.SPARSITY_METHODS) -> column style. `extra` lists series
# that are NOT rows of that table: sparsity_sweep.tex reports the logit-diff block only, and the
# KL block is drawn here as a contrast without being claimed as part of it.
STYLE = {
    "Node Pruning (logit-diff, LR $=$ 0.8)": dict(
        title="Node Pruning (lr 0.8)", xlabel="target sparsity $s$",
        colour=P.METHOD["Node Pruning"], xscale="linear",
        label="logit-diff loss", dash="solid",
        extra=[dict(label="KL loss", rows=KL_ROWS, dash=DASH)]),
    "DBM $+$ L1 (lr $=$ 0.3)": dict(
        title="DBM (logit-diff, lr 0.3)", xlabel=r"L1 coefficient $\lambda$",
        colour=P.METHOD["DBM"], xscale="symlog", label="logit-diff loss", dash="solid", extra=[]),
}
METRICS = [("area_under", "CPR AUC (↑)"), ("acc_auc", "IIA log-AUC (↑)")]
FIG_W, ROW_H = 5.4, 1.55
FS_LABEL, FS_TICK, FS_ANNOT = 7.5, 7, 6
TITLE_H = 0.20          # set_title draws ABOVE the axes rect, so the header strip must hold it
SYMLOG_LINTHRESH = 0.15  # below the smallest nonzero lambda (0.2), so only 0 sits in the linear leg


def series_of(block):
    """[(label, rows, dash)] for a block: the table's own series first, then the extras."""
    st = STYLE[block]
    rows = dict((n, v) for n, v, *_ in M.SPARSITY_METHODS)[block]
    return ([(st["label"], rows, st["dash"])]
            + [(e["label"], e["rows"], e["dash"]) for e in st["extra"]])


def load():
    """{block: [(label, dash, [(knob, {metric: mean or None})])]}, plus the exclusion log.

    Per-metric completeness over make_lr_table.COLUMNS, the same rule and the same 11 MIB
    validation cells as plot_lr_sweep_summary.load(). The table's own rows are all 11/11; the KL
    series is where this bites, since its sparsest budgets were trained before the others.
    """
    out, skipped = {}, []
    for name, *_ in M.SPARSITY_METHODS:
        if name not in STYLE:
            raise SystemExit(f"block {name!r} is in sparsity_sweep.tex but has no STYLE entry -- "
                             "add one (see the module docstring) rather than letting it vanish")
        built = []
        for label, vals, dash in series_of(name):
            pts = []
            for lab, dirn in vals:
                knob = knob_of(lab)
                if knob is None:
                    skipped.append(f"  {name} / {label} row {lab!r}: not a knob row -- dropped")
                    continue
                rec, miss = {}, []
                for key, _ in METRICS:
                    got = [v for v in (M.cpr(dirn, t, m, key) for t, m, _ in M.COLUMNS)
                           if v is not None]
                    rec[key] = float(np.mean(got)) if len(got) == len(M.COLUMNS) else None
                    if rec[key] is None:
                        miss.append(f"{key} {len(got)}/{len(M.COLUMNS)}")
                if miss:
                    skipped.append(f"  {name} / {label} {lab} ({dirn}): {', '.join(miss)} -- "
                                   f"dropped from "
                                   f"{'those panels' if any(rec.values()) else 'both panels'}")
                if any(v is not None for v in rec.values()):
                    pts.append((knob, rec))
            built.append((label, dash, sorted(pts)))
        out[name] = built
    return out, skipped


def ringed(y):
    """Index of the argmax, or None when it does not identify an operating point.

    None when the max sits at either end of the swept grid (then it is "the best we ran", not a
    measured optimum) and None when it is tied (then the grid does not pick between the tied
    settings either -- KL's acc-AUC ties its top two, and a bare argmax would ring the first).
    """
    if len(y) < 3:
        return None
    bi = int(np.argmax(y))
    if not 0 < bi < len(y) - 1:
        return None
    return bi if sum(v == y[bi] for v in y) == 1 else None


def style_axis(ax, xscale, ticks):
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.4)
        # Tick the SWEPT VALUES, not the decades. Symlog's default locator labels 1e-1/1e0/1e1,
        # and not one of those three is a setting we ran -- a reader would take the axis for a
        # continuum that was sampled at those points. These ticks are exactly the rows of the
        # table's DBM block, so every label has a square sitting on it and gaps in the grid are
        # gaps in the sweep. Minor ticks off for the same reason.
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{k:g}" for k in ticks])
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
    for c, name in enumerate(blocks):
        st = STYLE[name]
        ser = data[name]
        ticks = sorted({k for _, _, pts in ser for k, _ in pts})
        for r, (metric, ylab) in enumerate(METRICS):
            ax = axes[r][c]
            for label, dash, pts in ser:
                xy = [(k, rec[metric]) for k, rec in pts if rec[metric] is not None]
                if not xy:
                    continue
                x, y = [p[0] for p in xy], [p[1] for p in xy]
                ax.plot(x, y, ls=dash, lw=0.9, color=st["colour"], zorder=2, label=label)
                ax.plot(x, y, "s", ms=2.6, color=st["colour"], mec="#000000", mew=0.35,
                        ls="none", zorder=3)
                bi = ringed(y)
                if bi is not None:
                    ax.plot([x[bi]], [y[bi]], "o", ms=6.5, mfc="none", mec=st["colour"],
                            mew=0.8, zorder=4)
            style_axis(ax, st["xscale"], ticks)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=FS_LABEL)
            if r == 0:
                ax.set_title(st["title"], fontsize=FS_LABEL, pad=3)
                if len(ser) > 1:
                    ax.legend(fontsize=FS_ANNOT, loc="lower right", frameon=True,
                              framealpha=0.9, borderpad=0.3, handlelength=2.4,
                              handletextpad=0.5, labelspacing=0.25).get_frame().set_linewidth(0.4)
            if r == nr - 1:
                ax.set_xlabel(st["xlabel"], fontsize=FS_LABEL)

    fig.tight_layout()
    fig.subplots_adjust(top=1.0 - TITLE_H / fh)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    if skipped:
        print(f"\nexcluded ({len(M.COLUMNS)} MIB validation cells required per metric):")
        print("\n".join(skipped))
    else:
        print(f"\nno exclusions: every series is {len(M.COLUMNS)}/{len(M.COLUMNS)} everywhere")
    print("\nbest knob per series:")
    for name in blocks:
        for label, _, pts in data[name]:
            for metric, _ in METRICS:
                xy = [(k, r[metric]) for k, r in pts if r[metric] is not None]
                tag = f"{STYLE[name]['title']} / {label}"
                if not xy:
                    print(f"  {tag:<38} {metric:<10} -- no complete row")
                    continue
                y = [p[1] for p in xy]
                best = int(np.argmax(y))
                why = "" if ringed(y) is not None else "  (not ringed: endpoint or tie)"
                print(f"  {tag:<38} {metric:<10} n={len(xy)}  best {y[best]:.3f} "
                      f"@ {xy[best][0]:g}{why}")


if __name__ == "__main__":
    sys.exit(main())
