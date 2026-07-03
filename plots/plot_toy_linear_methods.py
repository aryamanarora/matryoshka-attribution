"""Three-way method comparison on the linear toy: MAttr (uniform k) vs MAttr (log k) vs
IxG (attribution patching). Convergence of rank recovery (Spearman vs |a|) faceted by n.

All three use one counterfactual pair per step/sample (batch=1), so the x-axis is directly
comparable. Reads the three n-sweep pickles in results/. \textwidth single-row figure.

Regenerate with: uv run python plots/plot_toy_linear_methods.py
"""
import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_hline, labs, facet_wrap,
    scale_color_brewer, scale_x_log10, scale_y_continuous,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.04,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

SOURCES = [
    (R / "toy_linear_mattr.pkl", "MAttr (uniform $k$)"),
    (R / "toy_linear_mattr_logk.pkl", "MAttr (log $k$)"),
    (R / "toy_linear_mattr_sumpow2.pkl", r"MAttr ($\sum_k$ pow2)"),
    (R / "toy_linear_ixg.pkl", "IxG"),
]
METHODS = [m for _, m in SOURCES]
SEL_N = [16, 64, 256]

thresh = None
rows = []
for path, method in SOURCES:
    with open(path, "rb") as f:
        d = pickle.load(f)
    thresh = d["args"]["thresh"]
    for r in d["runs"]:
        if r["n"] not in SEL_N:
            continue
        for st, sp in zip(r["steps"], r["spearman"]):
            rows.append({"n": r["n"], "method": method, "pairs": st, "spearman": sp})

df = pd.DataFrame(rows)
agg = df.groupby(["n", "method", "pairs"])["spearman"].mean().reset_index()
agg["method"] = pd.Categorical(agg["method"], categories=METHODS, ordered=True)
agg["facet"] = pd.Categorical(
    "$n = " + agg["n"].astype(str) + "$",
    categories=[f"$n = {n}$" for n in SEL_N], ordered=True)

p = (
    ggplot(agg, aes("pairs", "spearman", color="method"))
    + geom_hline(yintercept=thresh, color="#888888", linetype="dotted", size=0.3)
    + geom_line(size=0.6)
    + facet_wrap("facet", nrow=1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="# Counterfactual Pairs", y=r"Spearman($s$, $|a|$)", color="")
)
p.save(OUT / "toy_linear_methods.pdf", verbose=False)
print(f"Saved {OUT / 'toy_linear_methods.pdf'}")
