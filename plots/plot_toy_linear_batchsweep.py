"""Paper-ready plots for the linear-toy MAttr batch-size sweep
(results/toy_linear_mattr_batchsweep.pkl), LR fixed at 0.05.

Two subfigure-sized PDFs:
  1. toy_linear_batch_recovery.pdf  -- final Spearman recovery vs batch size, per n
  2. toy_linear_batch_convrate.pdf  -- SAMPLES (steps*batch) to Spearman>=thresh vs batch
     (fair-compute view: y is total counterfactual pairs seen, not optimizer steps)

Regenerate with: uv run python plots/plot_toy_linear_batchsweep.py
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
INPUT = sys.argv[1] if len(sys.argv) > 1 else str(R / "toy_linear_mattr_batchsweep.pkl")
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "toy_linear_batch"

with open(INPUT, "rb") as f:
    data = pickle.load(f)
runs = data["runs"]
thresh = data["args"]["thresh"]
ns = sorted(set(r["n"] for r in runs))
batches = sorted(set(r["batch"] for r in runs))
n_levels = [str(n) for n in ns]


def by_n(df):
    df["n"] = pd.Categorical(df["n"].astype(str), categories=n_levels, ordered=True)
    return df


# ----- 1. final recovery vs batch, per n -----
df = pd.DataFrame([{"n": r["n"], "batch": r["batch"], "final": r["spearman"][-1]} for r in runs])
agg = by_n(df.groupby(["n", "batch"])["final"].agg(["mean", "std"]).reset_index().fillna(0))

p1 = (
    ggplot(agg, aes("batch", "mean", color="n"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5) + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=batches, labels=[str(b) for b in batches])
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Batch Size", y=r"Final Spearman($s$, $|a|$)", color="$n$")
)
p1.save(OUT / f"{PREFIX}_recovery.pdf", verbose=False)


# ----- 2. convergence rate in SAMPLES (steps*batch) to threshold vs batch, per n -----
df2 = pd.DataFrame([{"n": r["n"], "batch": r["batch"], "samples": r["tts"] * r["batch"]}
                    for r in runs if r["tts"] is not None])
agg2 = by_n(df2.groupby(["n", "batch"])["samples"].agg(["mean", "std"]).reset_index().fillna(0))

p2 = (
    ggplot(agg2, aes("batch", "mean", color="n"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5) + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=batches, labels=[str(b) for b in batches])
    + scale_y_log10(labels=label_log(base=10))
    + labs(x="Batch Size", y=fr"Samples to Spearman $\geq {thresh}$", color="$n$")
)
p2.save(OUT / f"{PREFIX}_convrate.pdf", verbose=False)

print("Saved:")
for f in [f"{PREFIX}_recovery", f"{PREFIX}_convrate"]:
    print(f"  {OUT / (f + '.pdf')}")
