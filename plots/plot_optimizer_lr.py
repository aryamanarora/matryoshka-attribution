"""Adam vs SGD for \\ourmethod{} on MIB: performance against learning rate, averaged over cells.

Two SEPARATE single-panel PDFs, sized for side-by-side \\subfigure at 0.48\\linewidth:

    plots/optimizer_lr_cpr.pdf   CPR AUC      vs lr   (carries the legend)
    plots/optimizer_lr_acc.pdf   IIA log-AUC  vs lr   (no legend -- see below)

Each point is one (arm, lr) averaged over the 11 MIB validation cells in M.COLUMNS; each arm's
own optimum is ringed, because that the rings sit at different x IS the result.

THE LEGEND IS ONLY IN THE CPR PANEL. The two figures are meant to sit side by side in one float,
where a second copy is redundant and eats a quarter of a 2.6in panel. If either is ever used
alone, pass --legend-both.

WHAT THE NUMBERS SAY (validation, 11/11 cells, 2026-08-21):
    Adam log-k peaks at lr=0.1   CPR 1.881 / IIA 0.5035
    SGD  log-k peaks at lr=1.0   CPR 1.886 / IIA 0.5036
i.e. the peaks are level to within 0.005 CPR and 0.0001 IIA, at optima 10-20x apart. The
matched-lr comparison that reads "SGD costs 0.47 CPR" (0.05: 1.879 vs 1.413) is measuring SGD
being 20x below its optimum, not the optimizer. Same story on uniform k: Adam 2.092 at 0.05,
SGD 2.031 at 3.0. See make_mib_table.OUR_METHODS for why the table rows are pinned to own-best
lr rather than to a shared one.

Data loading, the completeness bar and the axis furniture are IMPORTED from
plot_mib_accauc_cpr_scatter (build_lr_rows/RC/LAB sizes) so these panels stay in the same visual
language as the paper's other MIB figures and inherit the same "11/11 cells on BOTH metrics or
it is not plotted" rule, with every exclusion printed. What is NOT reused is that module's
direct-labelling machinery: lr is a coordinate here, not a per-point label, so there is nothing
to place. (Aside, if the acc-vs-CPR projection is ever wanted for these four arms: embedding
draw_points/place_labels in a gridspec panel mis-measured its label boxes ~7x small and reported
"0 overlaps" over visibly colliding labels. Whatever that is, it does not bite the standalone
figures in that module, and it is not worked around here.)

COLOUR IS THE OPTIMIZER, which inverts plot_mib_accauc_cpr_scatter's --lr encoding (there all
four MAttr paths take MAttr's blue and separate by dash, since colour = method). This figure
exists to contrast two optimizers, so the contrast gets the strong channel and the k-schedule
takes the linetype. palette.py gives "MAttr (SGD)" its own hex for exactly this case -- see its
comment, the black is a measured CVD choice rather than a stylistic one.

ADAM'S UNIFORM-K ARM IS TWO POINTS, lr=0.01 (final_node) and 0.05, against SGD's six, and its
right end is a cliff edge rather than a peak -- nothing above 0.05 was ever swept, so its
maximum is just the largest lr it has. It is therefore drawn WITHOUT the optimum ring the other
three carry. final_node's own pkl has no acc_auc; it clears the completeness bar because
build_lr_rows' _pair() falls back to A.acc_mattr, which finds the run_evaluation copy under
MIB-circuit-track/results/mattr_accauc. Two lr>=0.3 runs would make this arm comparable.

Run:  uv run python plots/plot_optimizer_lr.py [--legend-both]
Out:  plots/optimizer_lr_{cpr,acc}.pdf   (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import palette as P

import plot_mib_accauc_cpr_scatter as S   # build_lr_rows, RC

# (legend name, short label build_lr_rows keys the path on, colour, linestyle, [(lr, dir), ...])
# Adam's log-k grid stops at 0.3 and SGD's starts at 0.05: that asymmetry is the sweeps', not a
# plotting choice, and it is also the finding -- each arm was swept over the range it was
# expected to live in, and the two ranges barely overlap. It does mean Adam's right shoulder
# rests on a single point at 0.3, so "Adam degrades above 0.3" is drawn but not measured.
OPT_SERIES = [
    ("MAttr, Adam (log $k$)", "Adam", P.METHOD["MAttr"], "solid",
     [("0.005", "topklog_lr_0.005"), ("0.01", "topklog_lr_0.01"),
      ("0.05", "topklog_lr_0.05"), ("0.1", "topklog_lr_0.1"), ("0.3", "topklog_lr_0.3")]),
    ("MAttr, SGD (log $k$)", "SGD", P.METHOD["MAttr (SGD)"], "solid",
     [("0.005", "softlog_sgd_lr_0.005"), ("0.01", "softlog_sgd_lr_0.01"),
      ("0.05", "softlog_sgd_lr_0.05"), ("0.1", "softlog_sgd_lr_0.1"),
      ("0.3", "softlog_sgd_lr_0.3"), ("1.0", "softlog_sgd_lr_1.0"),
      ("3.0", "softlog_sgd_lr_3.0"), ("10.0", "softlog_sgd_lr_10.0")]),
    ("$+$ unif. $k$, Adam", "Adam-u", P.METHOD["MAttr"], "dashed",
     [("0.01", "final_node"), ("0.05", "mib_node_topk_uniform_lr05")]),
    ("$+$ unif. $k$, SGD", "SGD-u", P.METHOD["MAttr (SGD)"], "dashed",
     [("0.05", "softuni_sgd_lr_0.05"), ("0.1", "softuni_sgd_lr_0.1"),
      ("0.3", "softuni_sgd_lr_0.3"), ("1.0", "softuni_sgd_lr_1.0"),
      ("3.0", "softuni_sgd_lr_3.0"), ("10.0", "softuni_sgd_lr_10.0")]),
]

# Arms whose maximum is NOT an optimum, so they get no ring: see the Adam-unif note above.
NO_RING = {"Adam-u"}

# Sized for two \subfigure[0.48\linewidth] panels in one float: ICLR's \linewidth is 5.5in, so
# each renders at ~2.64in and LaTeX scales 2.7in down by 0.98 -- i.e. these are drawn very close
# to final size, which is why the point sizes below are ~2/3 of the full-page figures' (a 4.5pt
# marker on a 5.4in page is a 2.2pt marker once that page is a 2.6in panel).
FIG_W, FIG_H = 2.7, 2.15
FS_LABEL, FS_TICK, FS_LEGEND = 8, 7, 5.8


def series_points(rows, short, metric):
    """[(lr, value)] for one arm, sorted by lr. Values are the cell means build_lr_rows made."""
    return sorted((v, r[metric]) for r in rows for k, v in r["paths"] if k == f"lr:{short}")


def panel(rows, metric, ylabel, out, legend):
    plt.rcParams.update(S.RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    drawn = []
    for leg, short, colr, ls, _ in OPT_SERIES:
        pts = series_points(rows, short, metric)
        if not pts:
            print(f"  arm {leg} has no complete lr -- not drawn", file=sys.stderr)
            continue
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        if len(pts) > 1:
            ax.plot(x, y, ls=ls, lw=0.9, color=colr, zorder=2)
            if short not in NO_RING:
                bx, by = max(zip(x, y), key=lambda p: p[1])
                ax.plot([bx], [by], "o", ms=7.5, mfc="none", mec=colr, mew=0.9, zorder=4)
        ax.plot(x, y, "s", ms=3.0, color=colr, mec="#000000", mew=0.4, ls="none", zorder=3)
        drawn.append((leg, colr, ls))
    ax.set_xscale("log")
    ax.set_xlabel("learning rate", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    # Same furniture as draw_points in plot_mib_accauc_cpr_scatter: hairline grid behind the
    # data, half-weight spines. Copied rather than factored out of it -- that function is
    # specifically an acc-vs-CPR scatter (sets both axis labels, pads x for point labels, takes a
    # group encoding), and these four lines are all this panel shares with it.
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if legend:
        # RESERVE the band the legend sits in rather than letting it float over the data. Four
        # rows at 5.8pt need ~35% of a 2.15in panel, and every corner of this panel is occupied
        # at some lr -- the two log-k arms rise from the bottom-left and the two uniform-k arms
        # own the top-right, so "lower right" put the frame straight over SGD's 0.05-0.3 rise.
        # Extending the y range downward costs vertical resolution on the curves; hiding a
        # quarter of a series costs the reader the shape of it, which is the whole figure.
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0 - 0.55 * (y1 - y0), y1)
        # Line2D handles, not the scatter's: two of the four series share each colour and
        # separate only by dash, so marker-only handles would show two identical squares twice.
        ax.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=0.9, marker="s", ms=3.0,
                                  mec="#000000", mew=0.4, label=leg) for leg, c, ls in drawn],
                  fontsize=FS_LEGEND, loc="lower right", frameon=True, framealpha=0.95,
                  borderpad=0.35, handletextpad=0.4, handlelength=2.0, labelspacing=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(drawn)} arms)")


def main():
    series = [(leg, short, vals) for leg, short, _, _, vals in OPT_SERIES]
    rows = S.build_lr_rows(series)
    if not rows:
        raise SystemExit("no complete optimizer series on disk")
    both = "--legend-both" in sys.argv
    panel(rows, "cpr", "CPR AUC (↑)", "plots/optimizer_lr_cpr.pdf", legend=True)
    panel(rows, "acc", "IIA log-AUC (↑)", "plots/optimizer_lr_acc.pdf", legend=both)
    for leg, short, _, _, _ in OPT_SERIES:
        c, a = series_points(rows, short, "cpr"), series_points(rows, short, "acc")
        if not c:
            continue
        bc, ba = max(c, key=lambda p: p[1]), max(a, key=lambda p: p[1])
        note = "  (largest lr swept, not an optimum)" if short in NO_RING else ""
        print(f"  {leg:<24} n={len(c)}  best CPR {bc[1]:.3f} @ lr={bc[0]:g}   "
              f"best IIA {ba[1]:.4f} @ lr={ba[0]:g}{note}")


if __name__ == "__main__":
    sys.exit(main())
