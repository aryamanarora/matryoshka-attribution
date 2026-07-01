"""Compare the identity-STE MAttr variant against sigmoid-STE MAttr and IxG on the linear toy.

hard_topk_identity (from eval_mib): hard top-k forward, identity straight-through backward
(dm/ds=1), so the score gradient is purely g*delta_i per node -- the attribution-patching
signal applied per step via Adam, evaluated at the current hard mask. This sits between
sigmoid-STE MAttr (gate-slope gradient) and pure IxG (g*delta averaged at the corrupt run).

Spearman recovery vs training step, selected n (all batch=1, 2000 steps, 5 seeds).
Reads results/toy_linear_{mattr,identity,ixg}.pkl. Plots paper/figs/toy_linear_identity.pdf.
  uv run python plots/plot_toy_linear_identity.py
"""
import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_hline, geom_line, labs, facet_wrap, scale_color_brewer,
    scale_x_log10, scale_y_continuous, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.9),
        axis_title=element_text(size=7), axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(), panel_spacing_x=0.04,
        strip_background=element_blank(), strip_text=element_text(size=7),
        legend_title=element_text(size=7), legend_text=element_text(size=6),
        legend_key_size=6, legend_position="top", legend_direction="horizontal",
        legend_box_margin=0,
    )
)

SOURCES = [
    ("toy_linear_mattr", "MAttr (sigmoid STE)"),
    ("toy_linear_identity", "MAttr (identity STE)"),
    ("toy_linear_ixg", "IxG"),
]
METHODS = [m for _, m in SOURCES]
SEL_N = [16, 64, 256]

rows = []
for fname, method in SOURCES:
    with open(R / f"{fname}.pkl", "rb") as f:
        runs = pickle.load(f)["runs"]
    for r in runs:
        if r["n"] not in SEL_N:
            continue
        for st, sp in zip(r["steps"], r["spearman"]):
            rows.append({"n": r["n"], "method": method, "step": st, "spearman": sp})

df = pd.DataFrame(rows)
agg = df.groupby(["n", "method", "step"])["spearman"].mean().reset_index()
agg["method"] = pd.Categorical(agg["method"], categories=METHODS, ordered=True)
agg["facet"] = pd.Categorical("$n = " + agg["n"].astype(str) + "$",
                              categories=[f"$n = {n}$" for n in SEL_N], ordered=True)

p = (
    ggplot(agg, aes("step", "spearman", color="method"))
    + geom_line(size=0.6)
    + facet_wrap("facet", nrow=1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Training Step", y=r"Spearman($s$, $|a|$)", color="")
)
p.save(OUT / "toy_linear_identity.pdf", verbose=False)
print(f"Saved {OUT / 'toy_linear_identity.pdf'}")
