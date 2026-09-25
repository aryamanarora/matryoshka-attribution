"""Adam vs SGD for \\ourmethod{} on MIB: performance against learning rate, averaged over cells.

THREE outputs. The merged one is what the paper places:

    plots/optimizer_lr.pdf       both metrics, full width, ONE shared legend above the panels

and the two single-panel PDFs it supersedes, sized for side-by-side \\subfigure at 0.48\\linewidth:

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


def draw(ax, rows, metric, ylabel, fs=None):
    """One metric-vs-lr panel into a provided Axes. Returns [(legend name, colour, linestyle)].

    Split out of panel() so the merged full-width figure draws the SAME panels rather than a
    second implementation of them -- the ring rule, the completeness handling and the axis
    furniture all have to stay identical between the two outputs.
    """
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
    lab, tick = fs or (FS_LABEL, FS_TICK)
    ax.set_xlabel("learning rate", fontsize=lab)
    ax.set_ylabel(ylabel, fontsize=lab)
    ax.tick_params(labelsize=tick)
    # Same furniture as draw_points in plot_mib_accauc_cpr_scatter: hairline grid behind the
    # data, half-weight spines. Copied rather than factored out of it -- that function is
    # specifically an acc-vs-CPR scatter (sets both axis labels, pads x for point labels, takes a
    # group encoding), and these four lines are all this panel shares with it.
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    return drawn


def panel(rows, metric, ylabel, out, legend):
    plt.rcParams.update(S.RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    drawn = draw(ax, rows, metric, ylabel)
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


# One full-width float instead of two 0.48\\textwidth subfigures (2026-09-05, requested).
# 5.5in is iclr2026_conference.sty's \\textwidth, so at width=\\linewidth this is placed 1:1.
# The merged figure is placed at FULL width, so it renders 1:1 and can carry smaller type than
# the 0.48-width singles, which LaTeX scales UP. That is most of where the height came from:
# 8/7/5.8pt at 2.25in -> 7/6/5.5pt at 1.75in with the curves no more crowded than before.
MERGED_W, MERGED_H = 5.5, 1.75
MERGED_FS = (7, 6)          # (axis label, tick); legend uses MERGED_FS_LEG
MERGED_FS_LEG = 5.5


def merged(rows, out="plots/optimizer_lr.pdf"):
    """Both metrics in one full-width figure, sharing a legend above the panels.

    *** THE SHARED LEGEND IS WHY THIS IS NOT JUST THE TWO PDFs PLACED TOGETHER. *** In the
    single-panel build the legend sits INSIDE the CPR axes, and every corner of that panel is
    occupied at some lr -- the two log-k arms rise from the bottom-left, the two uniform-k arms
    own the top-right -- so panel() has to expand the y range 55% downward to make room. That
    costs vertical resolution on exactly the curves the figure is about. Moving the legend above
    both panels gives it back: the y limits here are the data's.
    """
    plt.rcParams.update(S.RC)
    fig, axes = plt.subplots(1, 2, figsize=(MERGED_W, MERGED_H))
    drawn = draw(axes[0], rows, "cpr", "CPR AUC (↑)", fs=MERGED_FS)
    draw(axes[1], rows, "acc", "Compactness (↑)", fs=MERGED_FS)
    # Legend band as a LENGTH, not a fraction of the height -- at 1.75in a fixed 0.88 rect top
    # reserves 0.21in where 0.26in is needed, and the legend lands on the panel titles' space.
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 1 - 0.20 / MERGED_H))
    fig.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=0.9, marker="s", ms=3.0,
                               mec="#000000", mew=0.4, label=leg) for leg, c, ls in drawn],
               fontsize=MERGED_FS_LEG, ncol=len(drawn), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.4,
               handlelength=2.0, columnspacing=1.4)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {out} ({len(drawn)} arms, merged)")


# ---------------------------------------------------------------- all-ablation variant
# Every node-level MAttr ablation with a COMPLETE 11/11 validation score on both metrics, which
# as of 2026-09-05 is all 15 rows of make_mib_table.OUR_METHODS at node level. Most were only
# ever run at one learning rate, so most contribute a single point rather than a curve -- that
# is the requested cut, and it is why the swept arms are drawn as lines and the rest as labelled
# markers rather than everything being forced into one visual form.
#
# ENCODING. Colour is the OPTIMIZER (palette's blue/black rule) and marker FILL is the
# k-schedule (filled = log, open = uniform). The ablation itself is named by a direct text label
# rather than a hue, because there are seven of them and inventing seven more hexes would either
# collide with the method colours this palette already fixes (orange = IG everywhere in this
# paper) or drop below its CVD floor. That is the documented fallback in the plotting guide:
# colour the grouping, label the points.
#
# *** THE TWO "hard bwd" DIRS HAVE NO SINGLE LEARNING RATE. *** results/mib_node_bernoulli_
# reinforce{,_log} were run at lr=0.1 on 10 cells and 0.01 on ioi/llama3, which is why
# mib_results.tex prints "0.01/0.1" for those rows. They are placed at the MODAL lr (0.1) and
# drawn with a cross marker so the x is not read as exact; ours_lr's own warning prints on every
# run. Excluding them was the alternative and is worse -- "+ hard bwd" is the ablation that
# collapses, and a figure of ablations that silently drops the collapsing one is misleading.
ABL_MIXED_LR = {"mib_node_bernoulli_reinforce_log", "mib_node_bernoulli_reinforce"}


def ablation_series():
    """[(label, dir, lr, optimizer, k-schedule)] for every complete node-level ablation."""
    import make_mib_table as M
    out = []
    for name, dirn, level, grp in M.OUR_METHODS:
        if level != "node":
            continue
        lr = (M.ours_lr(dirn) or "")
        if not lr:
            continue
        lr = lr.split("/")[-1] if "/" in lr else lr      # mixed-lr dirs -> modal value
        out.append((S.delatex(name), dirn, lr, M.opt_of(dirn), grp))
    return out


def place(ax, fig, pts, gap_pt=5.2, dx_pt=3.5):
    """Label each point to its right, pushed apart vertically so none overlaps.

    A GREEDY 1-D SEPARATION, not a repel: these labels only ever collide in y (every point sits
    at one of five swept learning rates, so x is quantised and the columns are far apart), and a
    sweep from the bottom up enforcing a minimum gap is exact for that case. The general
    repel in plot_mib_accauc_cpr_scatter is deliberately not reused -- its own docstring records
    that embedding it in a gridspec panel mis-measured the label boxes ~7x small and reported
    "0 overlaps" over visibly colliding text, which is worse than no de-overlap at all.

    Offsets are in POINTS, which is what annotate's `textcoords="offset points"` wants and is
    independent of the dpi savefig later uses.
    """
    inv = 72.0 / fig.dpi                      # display px -> points
    order = sorted(range(len(pts)), key=lambda i: pts[i][1])
    ys = [ax.transData.transform((x, y))[1] * inv for x, y, _ in pts]
    placed = {}
    prev = -1e9
    for i in order:
        t = max(ys[i], prev + gap_pt)
        placed[i] = t - ys[i]                 # how far the label sits above its own marker
        prev = t
    for i, (x, y, lab) in enumerate(pts):
        ax.annotate(lab, (x, y), fontsize=4.6, xytext=(dx_pt, placed[i] - 1.6),
                    textcoords="offset points", color="#000000", zorder=5,
                    annotation_clip=False)


def fig_ablations(rows_lr, out="plots/ablation_lr.pdf"):
    ser = ablation_series()
    rows = S.build_lr_rows([(d, d, [(lr, d)]) for _, d, lr, _, _ in ser])
    by = {r["grp"]: r for r in rows}
    meta = {d: (lab, lr, opt, grp) for lab, d, lr, opt, grp in ser}
    have = [(d, by[d]) for _, d, _, _, _ in ser if d in by]
    if not have:
        raise SystemExit("no complete ablation dirs")

    plt.rcParams.update(S.RC)
    fig, axes = plt.subplots(1, 2, figsize=(MERGED_W, 3.2))
    for ax, metric, ylab in ((axes[0], "cpr", "CPR AUC (↑)"), (axes[1], "acc", "Compactness (↑)")):
        # The four swept arms first, as lines, so the single points read against a curve.
        for leg, short, colr, ls, _ in OPT_SERIES:
            pts = series_points(rows_lr, short, metric)
            if len(pts) > 1:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls=ls, lw=0.8,
                        color=colr, zorder=2, alpha=0.55)
        texts = []
        for d, r in have:
            lab, lr, opt, grp = meta[d]
            colr = P.METHOD["MAttr"] if opt == "adam" else P.METHOD["MAttr (SGD)"]
            mk = "X" if d in ABL_MIXED_LR else "o"
            ax.plot([float(lr)], [r[metric]], mk, ms=4.0,
                    mfc=colr if grp == "ours" else "none", mec=colr, mew=0.8, zorder=4)
            texts.append((float(lr), r[metric], lab))
        ax.set_xscale("log")
        ax.set_xlabel("learning rate", fontsize=FS_LABEL)
        ax.set_ylabel(ylab, fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, lw=0.25, color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.margins(x=0.18, y=0.10)
        place(ax, fig, texts)
    h = [Line2D([0], [0], color=P.METHOD["MAttr"], ls="none", marker="o", ms=4.0,
                mec=P.METHOD["MAttr"], label="Adam"),
         Line2D([0], [0], color=P.METHOD["MAttr (SGD)"], ls="none", marker="o", ms=4.0,
                mec=P.METHOD["MAttr (SGD)"], label="SGD"),
         Line2D([0], [0], color="#555555", ls="none", marker="o", ms=4.0, mfc="#555555",
                mec="#555555", label="log $k$"),
         Line2D([0], [0], color="#555555", ls="none", marker="o", ms=4.0, mfc="none",
                mec="#555555", label="unif. $k$"),
         Line2D([0], [0], color="#555555", ls="none", marker="X", ms=4.0, label="mixed lr")]
    fig.tight_layout(pad=0.3, w_pad=1.2, rect=(0, 0, 1, 0.91))
    fig.legend(handles=h, fontsize=FS_LEGEND, ncol=len(h), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=1.3)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {out} ({len(have)} ablations)")


def main():
    series = [(leg, short, vals) for leg, short, _, _, vals in OPT_SERIES]
    rows = S.build_lr_rows(series)
    if not rows:
        raise SystemExit("no complete optimizer series on disk")
    both = "--legend-both" in sys.argv
    panel(rows, "cpr", "CPR AUC (↑)", "plots/optimizer_lr_cpr.pdf", legend=True)
    panel(rows, "acc", "Compactness (↑)", "plots/optimizer_lr_acc.pdf", legend=both)
    # The two single-panel PDFs are still written: they are what --legend-both exists for, and
    # dropping them would break any float still placing them as subfigures.
    merged(rows)
    fig_ablations(rows)
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
