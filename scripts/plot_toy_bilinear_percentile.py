"""Percentile rank of the bilinear (product) nodes over training, faceted by n, one Set1
line per method. Product nodes carry as much true importance as the linear anchors (same
coeff distribution), so an unbiased attributor keeps them at the ~50th percentile (dashed
reference). IxG pins them near the bottom (its zero-baseline gradient can't see products);
MAttr lifts them toward 50; id-STE does so only at small n. Full-width -> paper/figs/.
Regenerate: uv run python scripts/plot_toy_bilinear_percentile.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, geom_hline, labs, facet_wrap,
    scale_color_manual, scale_fill_manual,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    "MAttr (uniform $k$)": "#e41a1c", "MAttr (log $k$)": "#377eb8", "IxG": "#4daf4a",
    "id-STE+SGD (uniform $k$)": "#984ea3", "id-STE+SGD (log $k$)": "#ff7f00",
    "id-STE+Adam (uniform $k$)": "#a65628", "id-STE+Adam (log $k$)": "#f781bf",
}

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.6),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.04,
        panel_spacing_y=0.04,
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
    ("toy_bilinear_mattr", "MAttr (uniform $k$)"),
    ("toy_bilinear_mattr_logk", "MAttr (log $k$)"),
    ("toy_bilinear_ixg", "IxG"),
    ("toy_bilinear_identity_sgd", "id-STE+SGD (uniform $k$)"),
    ("toy_bilinear_identity_sgd_logk", "id-STE+SGD (log $k$)"),
    ("toy_bilinear_identity_adam", "id-STE+Adam (uniform $k$)"),
    ("toy_bilinear_identity_adam_logk", "id-STE+Adam (log $k$)"),
]
METHODS = [m for _, m in SOURCES]

rows = []
for fname, method in SOURCES:
    runs = pickle.load(open(R / f"{fname}.pkl", "rb"))["runs"]
    by = {}
    for r in runs:
        for st, pc in zip(r["steps"], r["prod_pct"]):
            by.setdefault((r["n"], st), []).append(pc)
    for (n, st), vals in by.items():
        vals = np.array(vals)
        rows.append({"method": method, "n": n, "step": st,
                     "pct": vals.mean(), "sd": vals.std()})

df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
df["lo"] = df["pct"] - df["sd"]
df["hi"] = df["pct"] + df["sd"]
ns = sorted(df["n"].unique())
df["facet"] = pd.Categorical([f"n = {n}" for n in df["n"]],
                             categories=[f"n = {n}" for n in ns], ordered=True)

p = (
    ggplot(df, aes("step", "pct", color="method", fill="method"))
    + geom_hline(yintercept=50, color="#999999", linetype="dashed", size=0.3)
    + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.12, color="none")
    + geom_line(size=0.5)
    + facet_wrap("facet", ncol=4)
    + scale_color_manual(values=[COLOR_MAP[m] for m in METHODS])
    + scale_fill_manual(values=[COLOR_MAP[m] for m in METHODS])
    + labs(x="Steps (counterfactual pairs seen)",
           y="Percentile rank of product nodes", color="", fill="")
)
p.save(OUT / "toy_bilinear_percentile.pdf", verbose=False)
print(f"Saved {OUT / 'toy_bilinear_percentile.pdf'}")
