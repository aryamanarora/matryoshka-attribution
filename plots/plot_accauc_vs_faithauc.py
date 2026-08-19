"""Scatter of accuracy-AUC (x) vs faithfulness-AUC (y), one point per (method, loss).

Each point is averaged over TASK-GROUPS: SVA (mean of its 4 subtasks) + Arith (mean of its 4)
+ the 2 MIB tasks (ARC-E, IOI) when present. Columns are substrate x whether the input node is
included in scoring/ablation; only the `node` substrate has the MIB tasks and the +input
variant, so mlp / mlp+attn_head carry SVA and Arith only, no-input.

ROWS are the ablation SETTING: `Patched` sets every non-top-k unit to its counterfactual source
activation, `Zero-abl.` sets it to 0. Read the ordering WITHIN a row and never a point's
position across rows -- the two rows are different experiments, not two scorings of one. MAttr
and the two mask baselines are retrained through whichever intervention they are scored under,
and the gradient baselines change estimator outright (I×G -> Gradient×Input, IG -> textbook
zero-baseline IG). The settings agree at only Spearman ~0.44 on matched cells, which is the
reason the second row is worth drawing at all. The x axis of the zero row is chance-corrected
(see `load`); its y axis is on a ~1.9x inflated scale because faith-AUC's (F_clean - F_patch)
denominator shrinks when the ablated model is destroyed rather than flipped.

Data: results/sva_sweep (patched, input excluded), results/sva_sweep_input (patched, included),
results/sva_zeroabl (zeroed, input excluded -- the `+input` column is empty there by design).
Run:  uv run python plots/plot_accauc_vs_faithauc.py  ->  plots/accauc_vs_faithauc.pdf
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import palette as P
from plotnine import (
    ggplot, aes, geom_point, geom_path, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_fill_manual, scale_shape_manual,
    scale_color_manual, guides, guide_legend, expand_limits,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.0),
        axis_title=element_text(size=8),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=8,
        legend_position="top",
        legend_direction="horizontal",
        legend_box="horizontal",
        legend_box_margin=0,
        legend_margin=0,
    )
)

SVA = ["nounpp", "rc", "simple", "within_rc"]
# goodfire-ai/arithmetic-wild, same model (llama3) and same three substrates as SVA. Kept as a
# SEPARATE group rather than folded into SVA: these are not agreement tasks, and averaging them
# into SVA would hide that they are where the methods separate most (I×G floors at acc-AUC 0.022
# on all four while MAttr reaches ~0.50). As a fourth group each contributes 1/4 of every point.
ARITH = ["addition", "months", "weekdays", "hours"]
# (results dir, input-included label, ablation-row label). The ablation dimension is the FACET
# ROW: `Patched` ablates non-top-k units to the counterfactual source activation, `Zero-abl.`
# sets them to 0. That is a different SETTING, not a rescoring -- MAttr trains through it, and
# the gradient baselines change estimator (I×G -> Gradient×Input, IG -> zero-baseline IG) -- so
# read the ORDERING within a row, never a point's position across rows. The two settings agree
# at only Spearman ~0.44 on matched cells, which is why the row is worth drawing.
# The zero sweep was run without --include-input, so its `+input` column is empty by design.
SOURCES = [("results/sva_sweep", "−input", "Patched"),
           ("results/sva_sweep_input", "+input", "Patched"),
           ("results/sva_zeroabl", "−input", "Zero-abl.")]
SUBSTRATES = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]

# method key -> (display label, colour); order = legend order.
# Colours come from plots/palette.py -- the single source of truth for every figure. Do not
# write hex codes here; plot_accauc_vs_faithauc_cause.py and plot_faith_vs_acc_k1.py read this
# dict directly, and three more figures read palette.py, so a local override desyncs the paper.
METHODS = {
    "IG":         ("IG",           P.color("IG")),
    "IxG":        ("I×G",          P.color("I×G")),
    # Single-pass like I×G (only the backward RULES change): LN-freeze, gated-MLP secant +
    # half-rule, and the uniform half-rule on the QK/OV matmuls. The HF-side implementation is
    # src/learning_to_attribute/grad_attribution.py, verified against vanilla eager attention
    # by scripts/test_attnlrp_hf.py; on MIB the equivalent TransformerLens path is within
    # Spearman 0.96 of GIM (MIB-circuit-track/gim_attnlrp_decomp.py), so this series stands in
    # for the whole LRP family here.
    "AttnLRP":    ("AttnLRP",      P.color("AttnLRP")),
    "stopk-log":  ("MAttr (log)",  P.color("MAttr")),   # headline = soft top-k fwd, log k
    "soft-log":   ("+hard (log)",  P.color("+hard")),   # sigmoid-STE hard forward ablation
    # Node Pruning trained through the SAME loss_fn as MAttr (eval_sva.py --method edge_pruning),
    # so its points move along the loss axis like every other series here and the ONLY difference
    # from MAttr is how the mask is parameterized: hard-concrete gates under an annealed L0
    # budget vs top-k. The budget is in the key rather than hidden -- s=0.9 of the SUBSTRATE
    # (MLP neurons, or +attn heads), a far larger unit count than MIB's ~156 nodes, so this is
    # NOT the same absolute circuit size as the results/eprun_node_s0.9 rows in the tables.
    "eprun-s090": ("Node Pruning", P.color("Node Pruning")),
    # The pyvene sigmoid-mask baseline, likewise trained through eval_sva.py's own loss_fn, so
    # the same "only the mask parameterization differs" reading applies: deterministic
    # sigmoid(mask/temp) with temp annealed 50 -> 0.1, vs top-k. The key spells the recipe
    # (lr 0.3, L1 6.0) because that pair, not the method name, decides the circuit -- it is the
    # MIB validation argmax carried over, and a re-swept lr would be a DIFFERENT series.
    "sig_lr0.3_l16.0": ("DBM", P.color("DBM")),
}
LOSSES = {"acc": "acc", "ce": "CE", "logit_diff": "logit-diff"}
LOSS_SHAPE = {"acc": "o", "CE": "^", "logit-diff": "s"}   # all fillable: black edge + method fill
# Order the dashed guide visits a method's three points. NOT the legend order (that stays
# LOSSES order) and not sorted by x -- it is the loss's own sharpness ordering, CE (softest
# training signal) -> acc -> logit-diff (hardest), so the line reads as a trajectory rather
# than a shape. geom_path honours row order, which is why the frame is sorted by it.
LOSS_PATH = ["CE", "acc", "logit-diff"]

# Methods drawn in THIS figure. METHODS itself stays the full registry -- it is the shared
# method set/colour map that plot_accauc_vs_faithauc_cause.py and plot_faith_vs_acc_k1.py
# iterate, so deleting a key there would silently drop the series from those figures too.
# "+hard (log)" is omitted here only: it sits nearly on top of the MAttr (log) points in every
# facet, so it costs a legend entry and 12 overlapping markers without separating anything.
# It is still the "$+$ hard" ablation row in the tables, and still drawn in the cause figure.
FIGURE_METHODS = [k for k in METHODS if k != "soft-log"]


def parse_method(fname, d):
    """Method label from filename tag (mirrors make_fingerprint_tables.parse_method)."""
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    tag = tag.replace("_zeroabl", "")   # ablation is a facet ROW, not a method
    if tag.startswith(("random", "conductance")) or "fixedk" in tag:
        return None
    if tag.startswith("eprun_s"):        # eprun_s090[_ce|_acc] -> one key per budget
        return "eprun-s" + tag.split("_")[1][1:]
    if tag.startswith("sig_"):           # sig_lr0.3_l16.0[_ce|_acc] -> one key per recipe
        return re.sub(r"_(ce|acc)$", "", tag)
    if "hard_topk" in tag:
        if re.search(r"_ig\d+", tag):
            return None
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "unif" if "uniformk" in tag else "log"
        return f"{fam}-{ks}"
    if "sufficient_topk_" in tag:   # soft top-k forward (differentiable, no STE)
        if re.search(r"_ig\d+", tag):
            return None
        ks = "unif" if "uniformk" in tag else "log"
        return f"stopk-{ks}"
    # Be STRICT here. This used to fall through to "IG" for anything unrecognised, which meant a
    # cause-trained MAttr run (tag `necessary_topk_adam_bs1`, from --mode necessary) would be
    # silently relabelled "IG" and averaged into the IG points. Unknown tags must drop out, not
    # masquerade as a baseline. All `necessary_*` runs are therefore invisible to these figures
    # by design -- they belong in a cause-trained figure of their own.
    if tag.startswith("ixg"):
        return "IxG"
    if tag.startswith("attnlrp"):
        return "AttnLRP"
    if tag.startswith("ig"):
        return "IG"
    return None


def _auc_of(xs, ya):
    """Trapezoid on a log-x grid, normalised by the log span. Mirrors eval_sva.py:738."""
    lx = np.log10(np.asarray(xs, float))
    ya = np.asarray(ya, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


def load(res, corrected=False):
    """(method, loss, substrate, task) -> (acc_auc, faith_auc).

    `corrected=True` replaces the stored acc-AUC with a CHANCE-CORRECTED one, and is required
    for the zero-ablation sweep. Zeroing every non-top-k unit destroys the model to
    logit_diff ~ 0, so `acc_base` -- a binary base-vs-source preference -- sits at a 0.5 chance
    floor rather than patching's 0.0, and the stored acc-AUC reads ~0.5 for a circuit carrying
    no signal whatsoever. Plotting that raw against faith-AUC would put every zero-row method in
    a fake cluster near x=0.5 and invert the ordering. Renormalising against the measured floor,

        acc' = (acc - acc[0]) / (1 - acc[0])

    restores the spread and is the IDENTITY when acc[0] = 0, i.e. it would not move a single
    patched point -- so the x axis still means the same thing in both rows, up to the extra
    sampling noise in acc[0] (~+-0.05 at 100 examples). See scripts/compare_ablation.py.
    """
    raw = {}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        m = parse_method(os.path.basename(f), d)
        if m is None or m not in METHODS:
            continue
        acc_auc = d["acc_auc"]
        if corrected:
            acc = d["iso_metrics"]["acc_base"]
            a0 = acc[0]
            acc_auc = (np.nan if a0 == 1 else
                       _auc_of(d["n_nodes"], [(x - a0) / (1 - a0) for x in acc]))
        raw[(m, d["loss"], d["nodes"], d["task"])] = (acc_auc, d["faith_auc"])
    return raw


def group_avg(raw, m, loss, sub):
    """Average over task-groups: SVA (mean of 4) + Arith (mean of 4) + ARC-E + IOI when present.

    Macro-average over groups, not over tasks, so the eight subtasks that come in fours do not
    outvote the two single-task MIB cells. Groups with no runs at this substrate drop out (the
    MIB tasks exist at `node` only), which is why the mean is over `gx` rather than len(groups).
    """
    groups = [("SVA", SVA), ("Arith", ARITH), ("ARC-E", ["arc_easy"]), ("IOI", ["ioi"])]
    gx, gy, names = [], [], []
    for gname, tasks in groups:
        xs = [raw[(m, loss, sub, t)] for t in tasks if (m, loss, sub, t) in raw]
        if xs:
            gx.append(np.mean([v[0] for v in xs]))
            gy.append(np.mean([v[1] for v in xs]))
            names.append(gname)
    if not gx:
        return None
    return float(np.mean(gx)), float(np.mean(gy)), tuple(names)


def main():
    rows = []
    for res, inp_label, abl in SOURCES:
        raw = load(res, corrected=abl != "Patched")
        for m in FIGURE_METHODS:
            mlabel = METHODS[m][0]
            for lkey, llabel in LOSSES.items():
                for sub, slabel in SUBSTRATES:
                    r = group_avg(raw, m, lkey, sub)
                    if r is None:
                        continue
                    facet = f"{slabel}, {inp_label}"
                    rows.append(dict(acc_auc=r[0], faith_auc=r[1], method=mlabel,
                                     loss=llabel, facet=facet, ablation=abl,
                                     groups="+".join(r[2])))
    df = pd.DataFrame(rows)

    # ordering for consistent legends / facets (only 4 non-empty substrate x input combos)
    df["method"] = pd.Categorical(df["method"], [METHODS[m][0] for m in FIGURE_METHODS])
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()))
    facet_order = ["Node, −input", "Node, +input",
                   "MLP, −input", "MLP+Attn, −input"]
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in set(df["facet"])])
    df["ablation"] = pd.Categorical(df["ablation"], ["Patched", "Zero-abl."])
    # geom_path connects rows in FRAME order, so the sort below is what defines the line, not
    # a plotnine setting. Sorting by facet/method too keeps each method's three rows contiguous.
    # `ablation` leads the sort so a method's path never runs between the two settings.
    df["_path"] = pd.Categorical(df["loss"], LOSS_PATH).codes
    df = df.sort_values(["ablation", "facet", "method", "_path"])

    colors = {METHODS[m][0]: METHODS[m][1] for m in FIGURE_METHODS}
    p = (
        ggplot(df, aes("acc_auc", "faith_auc", fill="method", shape="loss"))
        # Dashed guide joining a method's three losses, drawn BEFORE the points so markers sit
        # on top. It carries no information the markers do not -- it groups them, so it is thin,
        # dashed and semi-transparent, and adds no legend entry (the colour scale has guide=None;
        # method is already keyed by fill).
        + geom_path(aes(color="method", group="method"), linetype="dashed",
                    size=0.3, alpha=0.55, show_legend=False)
        # Black edge on every marker: method is carried by FILL, not colour, so points stay
        # legible where two methods land on top of each other and against the grid lines.
        # alpha=1 -- a translucent fill under a black edge reads as a different, muddier colour
        # wherever markers overlap, which is exactly where the distinction has to hold.
        + geom_point(size=1.9, color="#000000", stroke=0.3)
        # Rows = ablation setting, cols = substrate x input. `scales="free"` is per-PANEL here,
        # not per-column, which is what we want: the zero row's corrected acc-AUC and inflated
        # faith-AUC live on their own scales and sharing an axis with the patched row would
        # invite exactly the cross-setting comparison the docstring warns against.
        + facet_grid("ablation ~ facet", scales="free")
        + expand_limits(x=0, y=0)  # anchor each free axis at 0 (upper stays per-facet)
        + scale_fill_manual(values=colors, name="Method")
        + scale_color_manual(values=colors, guide=None)   # line colour only; no second legend
        + scale_shape_manual(values=LOSS_SHAPE, name="Loss")
        + labs(x="IIA AUC (↑)", y="Faith AUC (↑)")
        + guides(fill=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    out = "plots/accauc_vs_faithauc.pdf"
    p.save(out, dpi=300, verbose=False)
    # PNG sibling for eyeballing the result without a PDF viewer, as the cause figure and the
    # iso-vs-cause curves already do. Only the PDF is copied into paper/figs.
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print("wrote", out, f"({len(df)} points)")
    # quick sanity: points per facet cell
    print(df.groupby(["ablation", "facet"], observed=True).size().to_string())
    # COVERAGE, not cosmetics. group_avg silently drops a task-group with no runs, so a row
    # whose sweep is still in flight quietly becomes an SVA-only average while the other row
    # macro-averages four groups -- same axis, different populations, no visible sign of it.
    # Print which groups actually went into each panel so an incomplete row is impossible to
    # mistake for a complete one.
    cov = df.groupby(["ablation", "facet"], observed=True)["groups"].agg(
        lambda s: " / ".join(sorted(set(s))))
    print("\ntask-groups averaged per panel:")
    print(cov.to_string())


if __name__ == "__main__":
    main()
