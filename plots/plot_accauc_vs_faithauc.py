"""Scatter of accuracy-AUC (x) vs faithfulness-AUC (y), one point per (method, loss).

Each point is averaged over TASK-GROUPS: SVA (mean of its 4 subtasks) + the 2 MIB tasks
(ARC-E, IOI) when present. Faceted by substrate (rows) x whether the input node is included
in scoring/ablation (cols). Only the `node` substrate has the MIB tasks and the +input
variant; mlp / mlp+attn_head are SVA-only, no-input (those input=Yes cells stay empty).

Data: results/sva_sweep/*.json (input excluded), results/sva_sweep_input/*.json (included).
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
    ggplot, aes, geom_point, geom_path, facet_wrap, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_fill_manual, scale_shape_manual,
    scale_color_manual, guides, guide_legend, expand_limits,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.6),
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
# (results dir, input-included label)
SWEEPS = [("results/sva_sweep", "−input"),
          ("results/sva_sweep_input", "+input")]
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


def load(res):
    """(method, loss, substrate, task) -> {acc_auc, faith_auc}."""
    raw = {}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        m = parse_method(os.path.basename(f), d)
        if m is None or m not in METHODS:
            continue
        raw[(m, d["loss"], d["nodes"], d["task"])] = (d["acc_auc"], d["faith_auc"])
    return raw


def group_avg(raw, m, loss, sub):
    """Average over task-groups: SVA (mean of 4) + ARC-E + IOI when present."""
    groups = [SVA, ["arc_easy"], ["ioi"]]
    gx, gy = [], []
    for tasks in groups:
        xs = [raw[(m, loss, sub, t)] for t in tasks if (m, loss, sub, t) in raw]
        if xs:
            gx.append(np.mean([v[0] for v in xs]))
            gy.append(np.mean([v[1] for v in xs]))
    if not gx:
        return None
    return float(np.mean(gx)), float(np.mean(gy))


def main():
    rows = []
    for res, inp_label in SWEEPS:
        raw = load(res)
        for m in FIGURE_METHODS:
            mlabel = METHODS[m][0]
            for lkey, llabel in LOSSES.items():
                for sub, slabel in SUBSTRATES:
                    r = group_avg(raw, m, lkey, sub)
                    if r is None:
                        continue
                    facet = f"{slabel}, {inp_label}"
                    rows.append(dict(acc_auc=r[0], faith_auc=r[1], method=mlabel,
                                     loss=llabel, facet=facet))
    df = pd.DataFrame(rows)

    # ordering for consistent legends / facets (only 4 non-empty substrate x input combos)
    df["method"] = pd.Categorical(df["method"], [METHODS[m][0] for m in FIGURE_METHODS])
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()))
    facet_order = ["Node, −input", "Node, +input",
                   "MLP, −input", "MLP+Attn, −input"]
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in set(df["facet"])])
    # geom_path connects rows in FRAME order, so the sort below is what defines the line, not
    # a plotnine setting. Sorting by facet/method too keeps each method's three rows contiguous.
    df["_path"] = pd.Categorical(df["loss"], LOSS_PATH).codes
    df = df.sort_values(["facet", "method", "_path"])

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
        + facet_wrap("facet", nrow=1, scales="free")
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
    print(df.groupby("facet", observed=True).size().to_string())


if __name__ == "__main__":
    main()
