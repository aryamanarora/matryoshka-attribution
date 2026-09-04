"""Task x task score correlation, Spearman rho AND Pearson r side by side, with cell text.

Companion to the transfer heatmap (plots/plot_task_transfer.py): same 13 llama3 cells, same
task order and group separators, same headline MAttr scores (soft fwd, log-k, SGD, node,
+input), but the cells are correlations of the raw score VECTORS rather than causal transfer.
Differs from plots/plot_task_corr_heatmap.py in three ways: it adds Pearson r next to Spearman
rho (rank agreement vs linear agreement -- Pearson is dominated by the few large-score nodes,
so rho >> r means tasks agree on the ranking but not on which shared nodes dominate, and
r >> rho means a handful of shared heavy hitters with disagreeing tails), it collapses the
node-subset facets to All nodes (the subset split lives in that figure), and it prints the
value in every cell to match the transfer figure it will sit next to.

Loaders are imported from plot_task_corr_heatmap -- single source of truth for file locations,
recipe assertions, and the tensor-index -> node-name map (verified there by the arc_easy
cross-harness check).

In: results/softlog_sgd_lr_1.0 (MIB cells) + results/sva_sweep_input (SVA/arith cells)
Out: paper/figs/task_score_corr.pdf
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from plotnine import (ggplot, aes, geom_tile, geom_text, geom_vline, geom_hline, labs,
                      facet_wrap, scale_color_identity, scale_fill_gradient2,
                      theme, element_text, element_blank)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_task_corr_heatmap import LOADERS, TASKS, names, check_namespace, SEP, FILL

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="both", choices=["both", "pearson", "spearman"],
                    help="which correlation(s); a single one drops the facet strip")
    sz = ap.add_mutually_exclusive_group()
    sz.add_argument("--half", action="store_true",
                    help="draw at ~2.7in for a 0.48\\textwidth subfigure slot: legend moves to "
                         "a top colourbar so the 13 columns keep their width, cell text shrinks "
                         "to 3.4pt. Output gets a _half suffix.")
    sz.add_argument("--third", action="store_true",
                    help="draw at ~1.8in for a 0.31\\textwidth slot (three-in-a-row); "
                         "everything one notch smaller again. Output gets _third.")
    args = ap.parse_args()

    scores = {t: LOADERS[h](t) for t, _l, h, _g in TASKS}
    keys = set(names())
    check_namespace(scores, keys)
    common = sorted(keys)
    vecs = {t: [scores[t][n] for n in common] for t, _l, _h, _g in TASKS}
    print(f"{len(common)} nodes x {len(TASKS)} tasks")

    corrs = [("Spearman ρ", lambda a, b: spearmanr(a, b)[0]),
             ("Pearson r", lambda a, b: pearsonr(a, b)[0])]
    if args.only != "both":
        corrs = [c for c in corrs if c[0].lower().startswith(args.only)]
    rows = []
    for name, fn in corrs:
        for (ta, la, _h, _g) in TASKS:
            for (tb, lb, _h2, _g2) in TASKS:
                rows.append(dict(a=la, b=lb, v=fn(vecs[ta], vecs[tb]), corr=name))
    df = pd.DataFrame(rows)
    labels = [lab for _, lab, _h, _g in TASKS]
    df["a"] = pd.Categorical(df["a"], categories=labels, ordered=True)
    df["b"] = pd.Categorical(df["b"], categories=labels[::-1], ordered=True)
    df["corr"] = pd.Categorical(df["corr"], categories=["Spearman ρ", "Pearson r"], ordered=True)

    cuts = [i + 0.5 for i in range(1, len(TASKS)) if TASKS[i][3] != TASKS[i - 1][3]]
    hcuts = [len(TASKS) - c + 1 for c in cuts]

    # Cell text: 2 decimals, leading zero dropped, minus kept ("-.12"). White ink once the
    # diverging fill is dark in EITHER direction, |v| past ~0.6 of the scale.
    df["txt"] = df.v.map(lambda v: f"{v:.2f}".replace("0.", ".", 1))
    df["ink"] = (df.v.abs() > 0.6).map({True: "#ffffff", False: "#000000"})

    fill_lab = {"both": "corr.", "pearson": "r", "spearman": "ρ"}[args.only]
    cell_size = 2.9 if args.third else 3.4 if args.half else 3.8
    p = (ggplot(df, aes("a", "b", fill="v")) + geom_tile(color="white", size=0.2)
         + geom_text(aes(label="txt", color="ink"), size=cell_size, show_legend=False)
         + scale_color_identity()
         + geom_vline(xintercept=cuts, **SEP) + geom_hline(yintercept=hcuts, **SEP)
         # Same diverging scale and fixed [-1, 1] limits as the correlation heatmaps, so a
         # given blue means the same value across the figure family (see that script).
         + scale_fill_gradient2(**FILL) + labs(x="", y="", fill=fill_lab)
         + theme(panel_grid=element_blank(),
                 panel_spacing_x=0.02,
                 strip_background=element_blank(),
                 strip_text=element_text(size=7),
                 axis_text_x=element_text(rotation=45, ha="right", size=6),
                 axis_text_y=element_text(size=6),
                 legend_title=element_text(size=7), legend_text=element_text(size=6)))
    if len(corrs) > 1:
        p = p + facet_wrap("~corr", nrow=1)
    if args.half or args.third:
        # Sub-full slots: legend goes to a top colourbar so the panel keeps the width the 13
        # columns and their in-cell numbers need. THIRD_SIZE is shared across all three
        # third-width panels (this, plot_task_transfer, plot_transfer_vs_corr) so a
        # 3-subfigure row gets equal boxes.
        fs = (1.45, 1.5) if args.third else (2.7, 2.95)
        p = p + theme(figure_size=fs, legend_position="top",
                      legend_direction="horizontal", legend_box_margin=0)
        if args.third:
            # 90-degree x labels: 13 columns at ~7pt pitch cannot host 45-degree text without
            # collisions; vertical text only needs the font size as pitch. NO legend at this
            # size -- the in-cell numbers carry the values, and the freed strip goes to tiles.
            p = p + theme(legend_position="none",
                          axis_text_x=element_text(rotation=90, ha="center", va="top", size=4.5),
                          axis_text_y=element_text(size=4.5))
    else:
        p = p + theme(figure_size=(5.9, 2.9) if len(corrs) > 1 else (3.6, 2.9))

    stem = "task_score_corr" + ("" if args.only == "both" else f"_{args.only}") \
        + ("_half" if args.half else "_third" if args.third else "")
    p.save(OUT / f"{stem}.pdf", dpi=300, verbose=False)
    print("wrote", OUT / f"{stem}.pdf")

    if len(corrs) > 1:
        # rho-vs-r divergence, the number the caption will want: where ranking agreement and
        # linear agreement come apart most (off-diagonal pairs, upper triangle).
        piv = df.pivot_table(index=["a", "b"], columns="corr", values="v", observed=True).reset_index()
        piv = piv[piv.a.astype(str) < piv.b.astype(str)]
        piv["gap"] = piv["Spearman ρ"] - piv["Pearson r"]
        top = piv.reindex(piv.gap.abs().sort_values(ascending=False).index).head(5)
        print("\nlargest |rho - r| pairs:")
        for _, r in top.iterrows():
            print(f"  {r.a:>13} ~ {r.b:<13} rho={r['Spearman ρ']:+.3f}  r={r['Pearson r']:+.3f}")


if __name__ == "__main__":
    main()
