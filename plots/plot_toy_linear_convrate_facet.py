"""Convergence rate vs n, faceted by Spearman target rho, for IxG / MAttr-uniform / MAttr-log.

y = steps (= counterfactual pairs, batch=1) to first reach Spearman(s, |a|) >= rho;
x = n (number of terms); one facet per rho target; one Set1 line per method. Mean over
the seeds that reach the target (a (method, n, rho) point is plotted only if a majority of
seeds reach it, to avoid survivorship-biased single-seed points).

Reads the three n-sweep pickles in results/. Full-width figure -> paper/figs/.
Regenerate with: uv run python plots/plot_toy_linear_convrate_facet.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, labs, facet_wrap,
    scale_color_brewer, scale_x_log10, scale_y_log10,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.8),
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

import sys
TASK = sys.argv[1] if len(sys.argv) > 1 else "linear"
if TASK == "quadratic":
    SOURCES = [
        ("toy_quadratic_uniform", "MAttr (uniform $k$)"),
        ("toy_quadratic_log", "MAttr (log $k$)"),
        ("toy_quadratic_ixg", "IxG"),
    ]
    OUTNAME = "toy_quadratic_convrate_facet.pdf"
else:
    SOURCES = [
        ("toy_linear_mattr", "MAttr (uniform $k$)"),
        ("toy_linear_mattr_logk", "MAttr (log $k$)"),
        ("toy_linear_ixg", "IxG"),
    ]
    OUTNAME = "toy_linear_convrate_facet.pdf"
METHODS = [m for _, m in SOURCES]
RHOS = [0.7, 0.8, 0.9, 0.95]


def steps_to(steps, trace, rho):
    for st, v in zip(steps, trace):
        if v >= rho:
            return st
    return None


rows = []
for fname, method in SOURCES:
    with open(R / f"{fname}.pkl", "rb") as f:
        runs = pickle.load(f)["runs"]
    by_n = {}
    for r in runs:
        by_n.setdefault(r["n"], []).append(r)
    for n, rs in by_n.items():
        for rho in RHOS:
            tts = [steps_to(r["steps"], r["spearman"], rho) for r in rs]
            hit = [t for t in tts if t is not None]
            if len(hit) > len(rs) / 2:                  # majority of seeds reach it
                rows.append({"method": method, "n": n, "rho": rho,
                             "steps": float(np.mean(hit))})

df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
df["facet"] = pd.Categorical(
    [fr"$\rho \geq {r:.2f}$" for r in df["rho"]],
    categories=[fr"$\rho \geq {r:.2f}$" for r in RHOS], ordered=True)
ns = sorted(df["n"].unique())

p = (
    ggplot(df, aes("n", "steps", color="method"))
    + geom_line(size=0.5)
    + geom_point(size=1.1)
    + facet_wrap("facet", nrow=1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + scale_y_log10(labels=label_log(base=10))
    + labs(x="Number of Terms $n$",
           y=r"Steps to Reach $\rho$", color="")
)
out = OUT / OUTNAME
p.save(out, verbose=False)
print(f"Saved {out}")
