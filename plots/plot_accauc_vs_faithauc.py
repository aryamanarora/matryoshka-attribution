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
from plotnine import (
    ggplot, aes, geom_point, facet_wrap, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_color_manual, scale_shape_manual,
    guides, guide_legend, expand_limits,
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
        legend_position="bottom",
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

# method key -> (display label, colour); order = legend order
METHODS = {
    "IG":         ("IG",           "#8c564b"),
    "IxG":        ("I×G",          "#e377c2"),
    "stopk-log":  ("MAttr (log)",  "#1f77b4"),   # headline = soft top-k fwd, log k
    "soft-log":   ("+hard (log)",  "#2ca02c"),   # sigmoid-STE hard forward ablation
}
LOSSES = {"acc": "acc", "ce": "CE", "logit_diff": "logit-diff"}
LOSS_SHAPE = {"acc": "o", "CE": "^", "logit-diff": "s"}


def parse_method(fname, d):
    """Method label from filename tag (mirrors make_fingerprint_tables.parse_method)."""
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    if tag.startswith(("random", "conductance")) or "fixedk" in tag:
        return None
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
    return "IxG" if tag.startswith("ixg") else "IG"


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
        for m, (mlabel, _) in METHODS.items():
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
    df["method"] = pd.Categorical(df["method"], [v[0] for v in METHODS.values()])
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()))
    facet_order = ["Node, −input", "Node, +input",
                   "MLP, −input", "MLP+Attn, −input"]
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in set(df["facet"])])

    p = (
        ggplot(df, aes("acc_auc", "faith_auc", color="method", shape="loss"))
        + geom_point(size=2.6, alpha=0.85, stroke=0.3)
        + facet_wrap("facet", nrow=1, scales="free")
        + expand_limits(x=0, y=0)  # anchor each free axis at 0 (upper stays per-facet)
        + scale_color_manual(values={lab: col for lab, col in METHODS.values()}, name="Method")
        + scale_shape_manual(values=LOSS_SHAPE, name="Loss")
        + labs(x="IIA AUC (↑)", y="Faith AUC (↑)")
        + guides(color=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    out = "plots/accauc_vs_faithauc.pdf"
    p.save(out, dpi=300, verbose=False)
    print("wrote", out, f"({len(df)} points)")
    # quick sanity: points per facet cell
    print(df.groupby("facet", observed=True).size().to_string())


if __name__ == "__main__":
    main()
