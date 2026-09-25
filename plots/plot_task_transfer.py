"""Task x task TRANSFER heatmap: task A's MAttr ranking, causally evaluated on task B.

The causal twin of plots/plot_task_corr_heatmap.py. That figure asks whether two tasks RANK the
nodes the same way (Spearman rho of the score vectors); this one asks what actually happens when
you run task B's faithfulness eval with task A's ranking (results/transfer_{mib,sva}, written by
scripts/transfer/launch/submit_transfer.sh -- headline MAttr soft-fwd/log-k/SGD node scores, 200 eval examples
everywhere including the recomputed diagonal). Rows are the TARGET task whose eval set is scored;
columns are the SOURCE task whose ranking is evaluated -- so unlike the correlation heatmap this
matrix is directional, and asymmetry is part of the content.

Each cell is the target's metric under the source's ranking DIVIDED by the diagonal (the
target's own ranking, same job, same examples): 1.0 on the diagonal by construction, "fraction
of own-circuit performance a foreign circuit retains" off it. The two harness halves score with
their own headline metric -- MIB targets (top 5 rows): CPR AUC; eval_sva targets: faith AUC --
which the diagonal normalisation is what makes readable on one colour scale; still, compare
cells WITHIN a row, never down a column across the harness boundary.

Same fixed task order, display names and group separators as the correlation heatmap, so the
two figures can be read against each other panel-for-panel. No in-cell numbers for the same
reason as there (13 cells across ~3in leaves no honest font size); block means print to stdout
for the caption.

METRIC AND NORMALISATION are independent knobs. --metric cpr (default) is the harness-native
AUC pair above; --metric acc is acc-AUC, which both harnesses compute IDENTICALLY --
log-sparsity-weighted mean of P(base answer beats counterfactual answer), the MIB side
deliberately built to match eval_sva's (MIB-circuit-track/MIB_circuit_track/evaluation.py:80)
-- making it the cross-comparable choice. Both default to diagonal-relative; --raw (acc only:
raw CPR and faith-AUC share no units) plots raw values instead, which keeps the diagonal's
task-difficulty spread (0.49-0.60) visible rather than folding it into "transfer".

In: results/transfer_mib/{task}_llama3_transfer.json + results/transfer_sva/*_xfer_*.json
Out: paper/figs/task_transfer_heatmap{,_acc,_acc_raw}.pdf
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from plotnine import (ggplot, aes, geom_tile, geom_text, geom_vline, geom_hline, labs,
                      scale_color_identity, scale_x_discrete, scale_y_discrete,
                      scale_fill_gradient, theme_bw, theme_set, theme,
                      element_text, element_blank, element_rect)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "transfer"))
from collect_transfer import TASKS, MIB_TASKS, load_row   # single source for files + metric map

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

# Display names and groups, matching plot_task_corr_heatmap.TASKS order exactly.
# "Arith." for arithmetic_subtraction, not "Arith. (sub.)" (2026-09-08, requested): MIB ships
# an addition task too, but it is not a column of any table here and never appears in these
# figures, so the qualifier distinguished nothing a reader could see. NOTE the string now
# collides with the GROUP name "Arith." (addition/months/weekdays/hours) -- that is only a
# collision in the stdout block-means below, where one is a row label and the other a task;
# the figure never prints group names.
LABEL = {"ioi": "IOI", "arithmetic_subtraction": "Arith.", "mcqa": "MCQA",
         "arc_easy": "ARC-E", "arc_challenge": "ARC-C", "simple": "Simple",
         "nounpp": "Noun PP", "rc": "RC", "within_rc": "Within RC", "addition": "Addition",
         "months": "Months", "weekdays": "Weekdays", "hours": "Hours"}
# Task subset for the THIRD-width panel only (2026-09-08, requested). 13x13 tiles in a 1.45in
# box is ~7pt per tile and forced the in-cell numbers down to 2.9pt; at 7x7 the tile roughly
# doubles and the values are legible at the size the rest of this row uses. Two or three of
# each family are kept (MIB: IOI/Arith./MCQA, SVA: Simple/Noun PP, Arith.: Addition/Months), so
# every within-family block the figure is about still has off-diagonal cells.
# --half, --facet and the full-size variants are UNCHANGED and still draw all 13 tasks.
THIRD_TASKS = ["ioi", "arithmetic_subtraction", "mcqa", "simple", "nounpp",
               "addition", "months"]

GROUP = {**{t: "MIB" for t in MIB_TASKS},
         **{t: "SVA" for t in ("simple", "nounpp", "rc", "within_rc")},
         **{t: "Arith." for t in ("addition", "months", "weekdays", "hours")}}
SEP = dict(color="#555555", size=0.35)

theme_set(
    theme_bw(base_size=8)
    + theme(
        # Black panel frame, matching palette.SPINE_COLOR and figs/baseline_strongreject.
        # theme_bw's own panel_border is grey20 and read lighter than the matplotlib figures
        # beside it on the same page.
        panel_border=element_rect(color="#000000", fill=None, size=0.5),
        text=element_text(color="#000", family="Inter"),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
    )
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="cpr", choices=["cpr", "acc"],
                    help="cpr: harness-native AUCs (CPR / faith-AUC); acc: acc-AUC, the "
                         "identical quantity in both harnesses")
    ap.add_argument("--raw", action="store_true",
                    help="plot raw values instead of dividing by the diagonal. Only meaningful "
                         "with --metric acc (raw CPR and faith-AUC are not in the same units); "
                         "shows the diagonal's task-difficulty spread instead of folding it in")
    sz = ap.add_mutually_exclusive_group()
    sz.add_argument("--half", action="store_true",
                    help="draw at ~2.7in for a 0.48\\textwidth subfigure slot: legend moves to "
                         "a top colourbar, cell text shrinks to 3.4pt. Output gets _half.")
    ap.add_argument("--method", default="",
                    help="per-method transfer round to plot (adam/mc_ig/ixg; results dirs "
                         "transfer_{mib,sva}_<m>). Empty = the headline MAttr(SGD) round. "
                         "Output filename gets the same suffix.")
    sz.add_argument("--facet", action="store_true",
                    help="full-width row: one panel per method (MAttr / +Adam / Expected Gradients / "
                         "IxG), shared fill scale, no cell text (pattern comparison, not "
                         "value lookup). Partial rounds show as grey NA tiles.")
    sz.add_argument("--third", action="store_true",
                    help="draw at ~1.8in for a 0.31\\textwidth slot (three-in-a-row), matching "
                         "the (removed) plot_task_score_corr --third box-for-box. Output gets _third.")
    args = ap.parse_args()
    if args.raw and args.metric == "cpr":
        raise SystemExit("--raw needs --metric acc: raw CPR and faith-AUC share no units")
    relative = not args.raw

    methods = [(args.method, None)]
    if args.facet:
        methods = [("", "MAttr"), ("adam", "MAttr + Adam"),
                   ("mc_ig", "Expected Gradients"), ("ixg", "I×G")]
    rows = []
    for meth, mlabel in methods:
      for tgt in TASKS:
        row = load_row(tgt, args.metric, meth)
        missing = [s for s in TASKS if s not in row]
        if missing and not args.facet:   # partial-tolerant: absent cells become NA tiles
            print(f"note: {tgt} missing {len(missing)} cells ({', '.join(missing)})")
        diag = row.get(tgt)
        for src in TASKS:
            v = row.get(src)
            if relative:
                v = v / diag if v is not None and diag else None
            rows.append(dict(src=LABEL[src], tgt=LABEL[tgt], rel=v, method=mlabel))
    df = pd.DataFrame(rows)
    if args.facet:
        done = df.rel.notna().groupby(df.method).sum()
        print("cells per facet:", dict(done))
        keep = [l for _m, l in methods if done.get(l, 0) > 0]   # a method with no round yet
        df = df[df.method.isin(keep)]                            # gets no (empty) panel
        df["method"] = pd.Categorical(df["method"], ordered=True, categories=keep)
    # The --third panel draws a SUBSET; every other variant keeps all 13. Narrowed here, above
    # the labels/categories/cuts, so the group separators and the caption block-means below are
    # all computed on the tasks actually drawn rather than on the full grid.
    tasks = TASKS
    if args.third:
        missing = [t for t in THIRD_TASKS if t not in TASKS]
        if missing:
            raise SystemExit(f"THIRD_TASKS not in collect_transfer.TASKS: {missing}")
        tasks = [t for t in TASKS if t in set(THIRD_TASKS)]
        keep = {LABEL[t] for t in tasks}
        df = df[df.src.isin(keep) & df.tgt.isin(keep)]
        print(f"third panel: {len(tasks)} tasks, {len(df)} cells")
    TASKS_D = tasks
    labels = [LABEL[t] for t in TASKS_D]
    df["src"] = pd.Categorical(df["src"], categories=labels, ordered=True)
    df["tgt"] = pd.Categorical(df["tgt"], categories=labels[::-1], ordered=True)

    # Group boundaries, same construction as the correlation heatmap's separators().
    cuts = [i + 0.5 for i in range(1, len(TASKS_D))
            if GROUP[TASKS_D[i]] != GROUP[TASKS_D[i - 1]]]
    hcuts = [len(TASKS_D) - c + 1 for c in cuts]

    # Relative: ceiling just past 1 covers the rare foreign-beats-own cells (ARC-C <- ARC-E at
    # 1.10 on CPR, ARC-C <- MCQA at 1.03 on acc). Raw acc-AUC: [0, 0.65] spans the diagonals
    # (0.49-0.60); floor stays 0, not chance, because "chance" is task-dependent here
    # (fully-patched accuracy is ~0 on these counterfactual pairs, not 0.5).
    lim = [0, 1.15] if relative else [0, 0.65]
    over = df.rel.max()
    if over > lim[1]:   # an overshoot should be seen as a warning, not silently clamped
        print(f"WARNING: max value {over:.2f} exceeds the {lim[1]} scale cap")
    # In-cell values: 2 decimals, no leading zero (".47"), so 3 glyphs fit the ~14pt cell a
    # 3.6in-wide 13-column panel affords at 4.5pt. White ink on dark tiles, black elsewhere;
    # the flip point is halfway up the fill scale.
    d2 = df.dropna(subset=["rel"]).copy()
    if args.facet:
        d2 = d2.iloc[0:0]   # pattern comparison at ~1.2in/panel: no honest font size exists
    d2["txt"] = d2.rel.map(lambda v: f"{v:.2f}".replace("0.", ".", 1))
    d2["ink"] = (d2.rel > lim[1] / 2).map({True: "#ffffff", False: "#000000"})
    p = (ggplot(df, aes("src", "tgt", fill="rel")) + geom_tile(color="white", size=0.2)
         + geom_text(d2, aes(label="txt", color="ink"),
                     # 4.4 at --third, not 2.9: the subset halves the column count, so the
                     # tile goes from ~7pt to ~13pt and a 3-glyph ".47" at 4.4 (~7.5pt) clears
                     # its gutters. Retune with the column count if THIRD_TASKS changes.
                     size=4.4 if args.third else 3.4 if args.half else 4.5, show_legend=False)
         + scale_color_identity()
         + geom_vline(xintercept=cuts, **SEP) + geom_hline(yintercept=hcuts, **SEP)
         # Sequential, not the correlation figures' diverging scale: this quantity is a fraction
         # of the diagonal, not a signed rho, and reusing their red-blue would wrongly suggest
         # the two figures share units. High end is the same #2166ac so "dark blue = strong" at
         # least rhymes across the pair.
         + scale_fill_gradient(low="#f7f7f7", high="#2166ac", limits=lim, na_value="#eeeeee")
         # Flush tiles: discrete scales default to 0.6 of a category of padding per side, which
         # on a 7-column panel is ~17% of the width spent on blank strip inside the frame.
         + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
         + labs(x="Source task (whose ranking)", y="Target task (whose eval)",
                fill="Rel. transfer" if relative else "acc-AUC")
         + theme(figure_size=(3.6, 2.9),
                 axis_text_x=element_text(rotation=45, ha="right", size=6),
                 axis_text_y=element_text(size=6)))
    if args.facet:
        from plotnine import facet_wrap
        p = (p + facet_wrap("~method", nrow=1)
               + labs(x="Source task", y="Target task")
               + theme(figure_size=(5.9, 1.95),
                       strip_background=element_blank(), strip_text=element_text(size=7),
                       panel_spacing_x=0.02,
                       axis_text_x=element_text(rotation=90, ha="center", va="top", size=4.5),
                       axis_text_y=element_text(size=4.5)))
    if args.half or args.third:
        # Sub-full slots: top colourbar keeps panel width for the 13 columns. The --third box
        # (1.8 x 2.1) matches the old plot_task_score_corr --third so a 3-subfigure row sits equal;
        # the axis TITLES are dropped there -- at 1.8in they cost a tile-row each and the
        # caption already says which side is source and which is target.
        # Three-panel row of fig:task-transfer: square plot rectangle, authored at final page
        # size. The full solve and the LaTeX subfigure widths live in
        # plots/plot_method_corr_heatmap.py above its p1b spec -- one copy, since changing any
        # panel means resolving all three.
        fs = (1.497, 1.457) if args.third else (2.7, 3.15)
        p = p + theme(figure_size=fs, legend_position="top",
                      legend_direction="horizontal", legend_box_margin=0)
        if args.third:
            p = p + labs(x="", y="") \
                  + theme(legend_position="none", aspect_ratio=1,
                          axis_text_x=element_text(rotation=45, ha="right", size=5.5),
                          axis_text_y=element_text(size=5.5))
    suffix = ("_methods" if args.facet else (f"_{args.method}" if args.method else "")) \
        + {"cpr": "", "acc": "_acc"}[args.metric] + ("_raw" if args.raw else "") \
        + ("_half" if args.half else "_third" if args.third else "")
    fn = OUT / f"task_transfer_heatmap{suffix}.pdf"
    p.save(fn, dpi=300, verbose=False)
    print("wrote", fn)

    # Caption numbers: mean relative transfer per (target-group, source-group) block,
    # diagonal excluded. Directional, so all 9 ordered pairs print (rows = target group).
    # subset=["rel"], not a bare dropna(): df carries columns that are NaN by construction
    # outside --facet (e.g. `method`), so dropna() over every column emptied `d` and printed
    # nine "--" for every variant of this figure. Pre-existing; the figure itself was fine
    # because its text layer already used dropna(subset=["rel"]).
    d = df.dropna(subset=["rel"])
    d = d[d.src.astype(str) != d.tgt.astype(str)].copy()
    g = {LABEL[t]: GROUP[t] for t in TASKS_D}
    # bracket access throughout: `gt` as an attribute is DataFrame.gt (greater-than), not a column
    d["sgrp"], d["tgrp"] = d.src.astype(str).map(g), d.tgt.astype(str).map(g)
    if args.facet:
        return
    qty = "relative transfer" if relative else "raw acc-AUC"
    print(f"\nmean {qty} (target-group <- source-group), diagonal excluded:")
    for tgrp in ("MIB", "SVA", "Arith."):
        cells = []
        for sgrp in ("MIB", "SVA", "Arith."):
            v = d[(d["tgrp"] == tgrp) & (d["sgrp"] == sgrp)].rel
            cells.append(f"{sgrp:>7}: {v.mean():.3f}" if len(v) else f"{sgrp:>7}:     --")
        print(f"  {tgrp:>7} <- " + "  ".join(cells))


if __name__ == "__main__":
    main()
