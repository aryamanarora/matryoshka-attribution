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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
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
        fig, ax = plt.subplots(figsize=(1.45, 1.5))
        # NOT the blue/orange of the neighbouring shared-circuit scatter: there those hues
        # mean head/MLP, and reusing them here would read as the same encoding. Tol indigo /
        # Wong reddish purple -- palette.py's cool-purple family, lightness-separated
        # (L* ~24 vs ~60), unclaimed by any per-node meaning in this row.
        for kind, c in (("Across", "#cc79a7"), ("Within", "#332288")):
            m = df.kind == kind
            ax.scatter(df.r[m], df.xfer[m], s=11, c=c, alpha=1.0, linewidths=0.3,
                       edgecolors="#000000", label=kind, zorder=2 if kind == "Within" else 1)
        ax.set_xlabel("Pearson r of scores", fontsize=6)
        ax.set_ylabel("Rel. transfer (acc-AUC)", fontsize=6)
        ax.tick_params(labelsize=5, width=0.5, length=2)
        ax.grid(True, lw=0.25, color="#dddddd")
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        # house-style labels (leader + white bbox, series-coloured text) on three pairs the
        # prose leans on: the cross-benchmark arithmetic pair sitting inside the within-family
        # cluster, the ARC ceiling, and the biggest r>>transfer outlier. Short forms -- the
        # full task names are wider than this panel affords (same rule as POINT_LABEL in
        # plot_accauc_vs_faithauc).
        ax.set_xlim(0.10, 1.02); ax.set_ylim(0.0, 1.14)
        CANN = {"Within": "#332288", "Across": "#cc79a7"}
        for pair, short, (tx, ty), ha in (
                ("Arith. (sub.) ~ Addition", "Arith.~Add.",   (0.585, 1.030), "right"),
                ("ARC-E ~ ARC-C",            "ARC-E~ARC-C",   (0.800, 1.095), "right"),
                ("Arith. (sub.) ~ Simple",   "Arith.~Simple", (0.760, 0.300), "left")):
            row = df[df.pair == pair].iloc[0]
            ax.plot([row.r, tx + (0.01 if ha == "left" else -0.01)], [row.xfer, ty],
                    lw=0.35, color="#888888", zorder=3)
            ax.annotate(short, (tx, ty), fontsize=4.5, va="center", ha=ha,
                        color=CANN[row.kind], zorder=5,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
        ax.legend(fontsize=5, frameon=False, loc="lower right", handletextpad=0.1,
                  borderaxespad=0.2, labelspacing=0.2)
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
