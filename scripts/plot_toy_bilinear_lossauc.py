"""Loss AUC over training, faceted by n, one line per method, for the bilinear toy.

Loss AUC = area under the (expected denoising loss vs sparsity) curve for the ranking the
learned scores induce -- keep the top-k nodes clean, measure loss, integrate over k/n. This
needs NO gold ranking (it uses the objective itself, the toy analog of MIB's faithfulness
AUC); lower = the ranking drives loss down faster = better attribution. Under the zero CF
the loss of any hard mask S is analytic (Σ a_i^2 over unselected linear + Σ c_j^2 over
incomplete pairs), so the curve is exact. Full-width -> paper/figs/.
Regenerate: uv run python scripts/plot_toy_bilinear_lossauc.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_ribbon, labs, facet_wrap,
    scale_color_manual, scale_fill_manual,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

COLOR_MAP = {
    "MAttr (uniform $k$)": "#e41a1c", "MAttr (log $k$)": "#377eb8", "IxG": "#4daf4a",
    "id-STE+SGD (uniform $k$)": "#984ea3", "id-STE+SGD (log $k$)": "#ff7f00",
    "id-STE+Adam (uniform $k$)": "#a65628", "id-STE+Adam (log $k$)": "#f781bf",
    "IG (5 steps)": "#17becf",
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

# argv[1] = "linonly" -> control task (all linear, no product circuit)
TASK = sys.argv[1] if len(sys.argv) > 1 else "bilinear"
PREFIX = "toy_linonly" if TASK == "linonly" else "toy_bilinear"
OUTNAME = "toy_bilinear_lossauc_linonly.pdf" if TASK == "linonly" else "toy_bilinear_lossauc.pdf"
SOURCES = [
    (f"{PREFIX}_mattr", "MAttr (uniform $k$)"),
    (f"{PREFIX}_mattr_logk", "MAttr (log $k$)"),
    (f"{PREFIX}_ixg", "IxG"),
    (f"{PREFIX}_ig", "IG (5 steps)"),
    (f"{PREFIX}_identity_sgd", "id-STE+SGD (uniform $k$)"),
    (f"{PREFIX}_identity_sgd_logk", "id-STE+SGD (log $k$)"),
    (f"{PREFIX}_identity_adam", "id-STE+Adam (uniform $k$)"),
    (f"{PREFIX}_identity_adam_logk", "id-STE+Adam (log $k$)"),
]
METHODS = [m for _, m in SOURCES]

rows = []
for fname, method in SOURCES:
    runs = pickle.load(open(R / f"{fname}.pkl", "rb"))["runs"]
    by = {}
    for r in runs:
        for st, a in zip(r["steps"], r["loss_auc"]):
            by.setdefault((r["n"], st), []).append(a)
    for (n, st), vals in by.items():
        vals = np.array(vals)
        rows.append({"method": method, "n": n, "step": st,
                     "auc": vals.mean(), "sd": vals.std()})

df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
df["lo"] = df["auc"] - df["sd"]
df["hi"] = df["auc"] + df["sd"]
ns = sorted(df["n"].unique())
df["facet"] = pd.Categorical([f"n = {n}" for n in df["n"]],
                             categories=[f"n = {n}" for n in ns], ordered=True)

p = (
    ggplot(df, aes("step", "auc", color="method", fill="method"))
    + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.12, color="none")
    + geom_line(size=0.5)
    + facet_wrap("facet", ncol=4)
    + scale_color_manual(values=[COLOR_MAP[m] for m in METHODS])
    + scale_fill_manual(values=[COLOR_MAP[m] for m in METHODS])
    + labs(x="Steps (counterfactual pairs seen)",
           y="Loss AUC (lower is better)", color="", fill="")
)
p.save(OUT / OUTNAME, verbose=False)
print(f"Saved {OUT / OUTNAME}")
