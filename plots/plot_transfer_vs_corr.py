"""Scatter: Pearson r of two tasks' scores vs their average causal transfer.

Third panel of the cross-task family (plots/plot_task_score_corr.py, plot_task_transfer.py):
does score similarity PREDICT causal transfer? One point per unordered task pair (13 choose 2
= 78). x = Pearson r of the two headline-MAttr score vectors (the quantity the left heatmap
shows); y = relative acc-AUC transfer symmetrised over direction, (T[A<-B] + T[B<-A])/2, each
T relative to the target's own diagonal (the quantity the right heatmap shows -- acc because
it is the one metric both harnesses compute identically; relative because raw acc-AUC folds
task difficulty into y). Points are coloured by whether the pair sits within one task family
(MIB / SVA / Arith.) or across two, since the heatmaps' block structure is the obvious
confounder for any trend.

Symmetrising is a presentation choice: transfer is directional (Addition <- Arith.(sub.) 0.99
vs Months <- Arith.(sub.) 0.86) and Pearson is not, so the y averaged over direction is the
part of transfer r could possibly explain. The Spearman rho of the scatter prints to stdout
(and is annotated on the panel) -- quote that, not a fitted line; 78 points with this much
block structure do not need a regression to read.

Out: transfer_vs_corr_half.pdf (0.48\\textwidth slot, default) or _third.pdf (--third,
1.8 x 2.1in -- the box every third-width panel of this figure family shares).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from plotnine import (ggplot, aes, geom_point, annotate, facet_wrap, labs,
                      scale_color_manual, theme, element_blank, element_text)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_task_corr_heatmap import TASKS
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "transfer"))
from collect_transfer import load_row

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("results/transfer_src")   # prep_transfer_sources.py's normalised [1057] vectors

# facet order + display names for --facet; keys are prep_transfer_sources.METHODS entries and
# collect_transfer round suffixes ("" = the original MAttr round's dirs)
# IxG's MIB side is the `ig1` run (EAP at m=1 = attribution patching = I x G; see
# prep_transfer_sources), so all four facets cover the full 78 pairs.
FACET_METHODS = [("mattr", "MAttr", ""), ("adam", "MAttr + Adam", "adam"),
                 ("mc_ig", "Stepless IG", "mc_ig"), ("ixg", "I×G", "ixg")]


def pairs_frame(method_key, round_suffix, raw=False, xstat="pearson"):
    """One row per unordered task pair: Pearson r of the two score vectors vs symmetrised
    acc-AUC transfer -- diagonal-relative by default, raw with ``raw=True`` (then a cell needs
    no diagonal, so partial rounds yield more pairs). Pairs with a missing transfer cell are
    dropped (a per-method round that is still running yields a partial facet, not a crash)."""
    vecs = {t: torch.load(SRC / method_key / f"{t}.pt").tolist()
            for t, _l, _h, _g in TASKS if (SRC / method_key / f"{t}.pt").exists()}
    rel = {}
    for tgt, _l, _h, _g in TASKS:
        row = load_row(tgt, "acc", round_suffix)
        diag = row.get(tgt)
        for src, _l2, _h2, _g2 in TASKS:
            if raw and src in row:
                rel[(tgt, src)] = row[src]
            elif diag and src in row:
                rel[(tgt, src)] = row[src] / diag
    grp = {t: g for t, _l, _h, g in TASKS}
    lab = {t: l for t, l, _h, _g in TASKS}
    ts = [t for t, _l, _h, _g in TASKS if t in vecs]
    rows = []
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            if (a, b) not in rel or (b, a) not in rel:
                continue
            corr = pearsonr if xstat == "pearson" else spearmanr
            rows.append(dict(
                pair=f"{lab[a]} ~ {lab[b]}",
                r=corr(vecs[a], vecs[b])[0],
                xfer=(rel[(a, b)] + rel[(b, a)]) / 2,
                kind="Within" if grp[a] == grp[b] else "Across"))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    sz = ap.add_mutually_exclusive_group()
    sz.add_argument("--third", action="store_true",
                    help="1.8 x 2.1in box for a 0.31\\textwidth three-in-a-row slot, equal to "
                         "the --third heatmaps")
    ap.add_argument("--x", default="pearson", choices=["pearson", "spearman"],
                    help="score-similarity statistic on the x axis")
    ap.add_argument("--raw", action="store_true",
                    help="y = raw acc-AUC averaged over the two directions, instead of "
                         "diagonal-relative -- shows absolute transferred performance, folding "
                         "each task's difficulty (diagonal 0.49-0.60) back in")
    sz.add_argument("--facet", action="store_true",
                    help="full-width: one panel per method (MAttr / +Adam / Stepless IG / "
                         "IxG), each method's OWN scores and OWN transfer round; per-panel "
                         "Spearman rho annotated")
    args = ap.parse_args()
    third = args.third

    if args.facet:
        frames = []
        for key, label, suf in FACET_METHODS:
            d = pairs_frame(key, suf, raw=args.raw, xstat=args.x)
            if d.empty:
                print(f"note: {label}: no transfer cells yet, facet skipped")
                continue
            d["method"] = label
            rho_m = spearmanr(d.r, d.xfer)[0]
            d["ann"] = f"\u03c1 = {rho_m:.2f}"
            print(f"{label}: {len(d)} pairs  Spearman(r, transfer) = {rho_m:.3f}")
            frames.append(d)
        df = pd.concat(frames, ignore_index=True)
        df["method"] = pd.Categorical(df["method"], ordered=True,
                                      categories=[l for _k, l, _s in FACET_METHODS
                                                  if l in set(df.method)])
        ann = df.drop_duplicates("method")[["method", "ann"]].copy()
        ann["ax"], ann["ay"] = float(df.r.min()), float(df.xfer.max()) * 1.02   # top-left
        from plotnine import geom_text
        p = (ggplot(df, aes("r", "xfer", color="kind"))
             + geom_point(size=1.3, alpha=0.85, stroke=0)
             + scale_color_manual(values={"Within": "#0072b2", "Across": "#e69f00"})
             + facet_wrap("~method", nrow=1)
             + geom_text(ann, aes(x="ax", y="ay", label="ann"), size=5.5,
                         ha="left", color="#000000", family="Inter", inherit_aes=False)
             + labs(x=("Spearman ρ of scores" if args.x == "spearman" else "Pearson r of scores"),
                    y=("acc-AUC (sym. avg)" if args.raw else "Rel. transfer (acc-AUC, sym.)"),
                    color="")
             + theme(figure_size=(5.9, 1.85),
                     legend_position="top", legend_direction="horizontal",
                     legend_box_margin=0, legend_text=element_text(size=6),
                     strip_background=element_blank(), strip_text=element_text(size=7),
                     panel_spacing_x=0.02,
                     axis_title=element_text(size=7), axis_text=element_text(size=6)))
        fn = OUT / ("transfer_vs_corr_methods"
                    + ("_rho" if args.x == "spearman" else "")
                    + ("_raw" if args.raw else "") + ".pdf")
        p.save(fn, dpi=300, verbose=False)
        print("wrote", fn)
        return

    df = pairs_frame("mattr", "")
    rho = spearmanr(df.r, df.xfer)[0]
    if third:
        # Rendered with the exact matplotlib style of plot_shared_circuit_scatter.py (same
        # rcParams, marker/edge geometry, legend and grid), so the two scatters in the
        # third-width row read as one figure family. rho prints above for the caption; no
        # stat box on the panel, matching that scatter.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.family": "Inter",
            "mathtext.fontset": "custom", "mathtext.rm": "Inter",
            "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
            "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
            "pdf.fonttype": 42,
            "text.color": "#000000", "axes.labelcolor": "#000000",
            "xtick.color": "#000000", "ytick.color": "#000000",
        })
        # Three-panel row of fig:task-transfer: square plot rectangle, authored at final page
        # size. The full solve and the LaTeX subfigure widths live in
        # plots/plot_method_corr_heatmap.py above its p1b spec -- one copy, since changing any
        # panel means resolving all three.
        fig, ax = plt.subplots(figsize=(1.518, 1.457))
        ax.set_box_aspect(1)          # square plotting rectangle, matching the two heatmaps
        # ONE SERIES, ONE COLOUR (2026-09-08, requested). The Within/Across split used to be
        # drawn as two hues (Tol indigo / Wong reddish purple) with the words as the key.
        #
        # *** WHAT THE FIGURE NO LONGER SHOWS. *** `kind` was on the panel because the family
        # block structure is the obvious confounder for the trend this figure claims: pairs
        # inside one task family sit high on both axes and pairs across families sit low, so
        # some of the r below is between-family separation rather than a within-family relation.
        # The column is still computed and still in the frame -- put the hues back by restoring
        # the two-series loop -- but a reader of the panel alone cannot check that confound now,
        # so the caption or the prose has to carry it.
        ax.scatter(df.r, df.xfer, s=6, c="#332288", alpha=1.0, linewidths=0.25,
                   edgecolors="#000000", zorder=2)
        ax.set_xlabel("Pearson r of scores", fontsize=5.5)
        # "IIA log-AUC", not "acc-AUC": the paper's prose and every other figure call this
        # metric IIA log-AUC (it is the log-weighted interchange-intervention accuracy of
        # eq. logauc), and the pkl key `acc_auc` is an implementation name, not a display one.
        # 5.5pt, not 6: at 6pt this 26-character string is taller than the 1.258in box and
        # its closing paren clipped off the top of the PDF.
        ax.set_ylabel("Rel. transfer (compactness)", fontsize=5.5)
        ax.tick_params(labelsize=4.5, width=0.5, length=2)
        ax.grid(True, lw=0.25, color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("#000000")          # palette.SPINE_COLOR; this file predates it
        # House-style labels (leader + white bbox, series-coloured text) on three pairs the
        # prose leans on: the cross-benchmark arithmetic pair sitting inside the within-family
        # cluster, the ARC ceiling, and the biggest r-much-greater-than-transfer outlier. Short
        # forms -- the full task names are wider than this panel affords (same rule as
        # POINT_LABEL in plot_accauc_vs_faithauc).
        #
        # These were dropped for one revision when the row was solved at a 0.68in axes box and
        # all three collided; the squared row gives this panel 1.13in and they fit again. If the
        # row is ever re-solved smaller, check this block before anything else -- positions are
        # DATA coordinates tuned against xlim/ylim below, so they do not follow a resize.
        ax.set_xlim(0.10, 1.02); ax.set_ylim(0.0, 1.14)
        # Least-squares fit across the drawn x range, annotated with its PEARSON r.
        #
        # *** THE TWO r's ON THIS PANEL ARE NOT THE SAME QUANTITY. *** The x AXIS is the
        # Pearson r between two tasks' score vectors; this annotation is the Pearson r between
        # that x and the transfer on y, i.e. how well score similarity predicts transfer. The
        # label says "r =" as requested; if a reader ever conflates the two, disambiguate in
        # the caption rather than by renaming the axis.
        #
        # A LINE IS A WEAKER CLAIM HERE THAN IT LOOKS, which is why the module docstring told
        # callers to quote the rank correlation instead: the points carry family block
        # structure (Within pairs sit high, Across low), so a straight fit is partly tracing
        # that split rather than a within-family trend. Drawn because it was asked for; both
        # statistics still print to stdout so the caption can use either.
        import numpy as np
        m, b = np.polyfit(df.r.to_numpy(), df.xfer.to_numpy(), 1)
        xs = np.array(ax.get_xlim())
        ax.plot(xs, m * xs + b, lw=0.7, color="#666666", ls=(0, (4, 2)), zorder=0)
        pear = pearsonr(df.r, df.xfer)[0]
        # Upper-left: the fit runs bottom-left to top-right, so this corner is the one region
        # of the panel that no point occupies.
        ax.annotate(f"r = {pear:.2f}", (0.04, 0.96), xycoords="axes fraction",
                    fontsize=5, ha="left", va="top", color="#000000")
        # Labels in the single series colour now that `kind` is not encoded.
        for pair, short, (tx, ty), ha in (
                # Positions avoid three occupied regions, in this order of priority: the
                # upper-LEFT block (r + the two series words), the point cloud itself, and the
                # right frame. "Arith.~Add." moved twice: at y 1.03 its white bbox covered the
                # "0.90" of the r annotation, and at (0.560, 0.940) it covered the tail of
                # "Within" -- it now starts right of that block at x 0.35. "Arith.~Simple" is
                # right-ALIGNED at the frame rather than left-aligned at its point, which ran
                # ~0.05 of the x range past the edge.
                ("Arith. (sub.) ~ Addition", "Arith.~Add.",   (0.660, 0.900), "right"),
                ("ARC-E ~ ARC-C",            "ARC-E~ARC-C",   (0.820, 1.100), "right"),
                ("Arith. (sub.) ~ Simple",   "Arith.~Simple", (1.010, 0.080), "right")):
            hit = df[df.pair == pair]
            if hit.empty:                       # a pair named here but not drawn is a bug,
                raise SystemExit(f"label pair {pair!r} is not in the plotted set")
            row = hit.iloc[0]
            ax.plot([row.r, tx + (0.01 if ha == "left" else -0.01)], [row.xfer, ty],
                    lw=0.35, color="#888888", zorder=3)
            ax.annotate(short, (tx, ty), fontsize=4.5, va="center", ha=ha,
                        color="#332288", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        # NO LEGEND AND NO KEY OF ANY KIND: with one series there is nothing to key. An
        # earlier revision drew a frameless ax.legend inside the axes, whose swatches sat at
        # data-plausible positions and read as two extra points; that is why any future key
        # here should be series-coloured words rather than markers.
        fig.tight_layout(pad=0.3)
        fig.savefig(OUT / "transfer_vs_corr_third.pdf", dpi=300)
        print("wrote", OUT / "transfer_vs_corr_third.pdf")
        return
    print(f"{len(df)} pairs  Spearman(r, transfer) = {rho:.3f}  "
          f"Pearson = {pearsonr(df.r, df.xfer)[0]:.3f}")
    print("\nextremes:")
    for _, p in df.reindex(df.xfer.sort_values(ascending=False).index).head(4).iterrows():
        print(f"  high: {p['pair']:<28} r={p.r:.2f}  transfer={p.xfer:.2f}")
    hi_r_lo_t = df[(df.r > 0.6) & (df.xfer < 0.5)]
    for _, p in hi_r_lo_t.iterrows():
        print(f"  r>>transfer: {p['pair']:<22} r={p.r:.2f}  transfer={p.xfer:.2f}")

    p = (ggplot(df, aes("r", "xfer", color="kind"))
         + geom_point(size=1.2 if third else 1.6, alpha=0.85, stroke=0)
         # Wong blue / orange: the palette's lightness-separated pair; these are pair TYPES,
         # not methods, so plots/palette.py's method map deliberately does not apply.
         + scale_color_manual(values={"Within": "#0072b2", "Across": "#e69f00"})
         + annotate("text", x=0.18, y=1.0, label=f"ρ = {rho:.2f}",
                    size=6 if third else 6.5, ha="left", family="Inter")
         + labs(x="Pearson r of scores", y="Rel. transfer (acc-AUC, sym.)", color="")
         + theme(figure_size=(1.45, 1.5) if third else (2.7, 2.35),
                 legend_position="top", legend_direction="horizontal",
                 legend_box_margin=0, legend_text=element_text(size=5.5 if third else 6),
                 axis_title=element_text(size=6.5 if third else 7),
                 axis_text=element_text(size=5.5 if third else 6)))
    fn = OUT / f"transfer_vs_corr_{'third' if third else 'half'}.pdf"
    p.save(fn, dpi=300, verbose=False)
    print("wrote", fn)


if __name__ == "__main__":
    main()
