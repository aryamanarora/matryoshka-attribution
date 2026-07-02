"""Final recovery (Spearman of learned scores vs true node importance) as a function of n
for the bilinear toy, one Set1 line per method. IxG is flat near 0 (it can't see the
product circuit under a zero-ablation CF); MAttr recovers the ordering. Companion to the
convrate facet, which can't show IxG (it never reaches any rho). Full-width -> paper/figs/.
Regenerate: uv run python scripts/plot_toy_bilinear_recovery.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_ribbon, labs,
    scale_color_brewer, scale_fill_brewer, scale_x_log10,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.0),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
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
]
METHODS = [m for _, m in SOURCES]

rows = []
for fname, method in SOURCES:
    runs = pickle.load(open(R / f"{fname}.pkl", "rb"))["runs"]
    by_n = {}
    for r in runs:
        by_n.setdefault(r["n"], []).append(r["spearman"][-1])
    for n, finals in by_n.items():
        finals = np.array(finals)
        rows.append({"method": method, "n": n,
                     "rho": finals.mean(), "sd": finals.std()})

df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], categories=METHODS, ordered=True)
df["lo"] = df["rho"] - df["sd"]
df["hi"] = df["rho"] + df["sd"]
ns = sorted(df["n"].unique())

p = (
    ggplot(df, aes("n", "rho", color="method", fill="method"))
    + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.15, color="none")
    + geom_line(size=0.5)
    + geom_point(size=1.1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_fill_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + labs(x="Number of Terms $n$",
           y=r"Final Spearman(scores, importance)", color="", fill="")
)
p.save(OUT / "toy_bilinear_recovery.pdf", verbose=False)
print(f"Saved {OUT / 'toy_bilinear_recovery.pdf'}")
