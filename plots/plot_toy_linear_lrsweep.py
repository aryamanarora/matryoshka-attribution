"""Paper-ready plots for the linear-toy MAttr LR sweep (results/toy_linear_mattr_lrsweep.pkl).

Two subfigure-sized PDFs:
  1. toy_linear_lr_recovery.pdf  -- final Spearman recovery vs LR, one line per n
  2. toy_linear_lr_convrate.pdf  -- steps to Spearman>=thresh vs LR, one line per n

Regenerate with: uv run python plots/plot_toy_linear_lrsweep.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_errorbar, labs,
    scale_color_brewer, scale_x_log10, scale_y_log10, scale_y_continuous,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(1.85, 1.6),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
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
INPUT = sys.argv[1] if len(sys.argv) > 1 else str(R / "toy_linear_mattr_lrsweep.pkl")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "toy_linear_lr"

with open(INPUT, "rb") as f:
    data = pickle.load(f)
runs = data["runs"]
thresh = data["args"]["thresh"]
ns = sorted(set(r["n"] for r in runs))
lrs = sorted(set(r["lr"] for r in runs))
n_levels = [str(n) for n in ns]


# ----- 1. final recovery vs LR, per n -----
rows = [{"n": r["n"], "lr": r["lr"], "final": r["spearman"][-1]} for r in runs]
df = pd.DataFrame(rows)
agg = df.groupby(["n", "lr"])["final"].agg(["mean", "std"]).reset_index().fillna(0)
agg["n"] = pd.Categorical(agg["n"].astype(str), categories=n_levels, ordered=True)

p1 = (
    ggplot(agg, aes("lr", "mean", color="n"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5)
    + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Learning Rate", y=r"Final Spearman($s$, $|a|$)", color="$n$")
)
p1.save(OUT / f"{PREFIX}_recovery.pdf", verbose=False)


# ----- 2. convergence rate vs LR, per n -----
rows = [{"n": r["n"], "lr": r["lr"], "tts": r["tts"]}
        for r in runs if r["tts"] is not None]
df2 = pd.DataFrame(rows)
agg2 = df2.groupby(["n", "lr"])["tts"].agg(["mean", "std"]).reset_index().fillna(0)
agg2["n"] = pd.Categorical(agg2["n"].astype(str), categories=n_levels, ordered=True)

p2 = (
    ggplot(agg2, aes("lr", "mean", color="n"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5)
    + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_log10(labels=label_log(base=10))
    + labs(x="Learning Rate", y=fr"Steps to Spearman $\geq {thresh}$", color="$n$")
)
p2.save(OUT / f"{PREFIX}_convrate.pdf", verbose=False)

print("Saved:")
for f in [f"{PREFIX}_recovery", f"{PREFIX}_convrate"]:
    print(f"  {OUT / (f + '.pdf')}")
