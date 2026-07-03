"""Paper-ready plots for the linear-toy MAttr experiment (results/toy_linear_mattr.pkl).

Three subfigure-sized PDFs (side-by-side in a \textwidth row):
  1. toy_linear_convergence.pdf  -- Spearman recovery vs training step, by n
  2. toy_linear_recovery_vs_n.pdf -- final recovery metrics vs n
  3. toy_linear_convrate_vs_n.pdf -- steps to Spearman>=thresh vs n (convergence rate)

Run on sc (or locally; CPU-only). Regenerate with: uv run python plots/plot_toy_linear_mattr.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_ribbon, geom_errorbar, geom_hline,
    labs, scale_color_brewer, scale_fill_brewer, scale_x_continuous, scale_y_continuous,
    scale_x_log10, scale_y_log10, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(1.85, 1.6),  # ~3 across a 5.5in \textwidth row
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

with open(R / "toy_linear_mattr.pkl", "rb") as f:
    data = pickle.load(f)
runs = data["runs"]
thresh = data["args"]["thresh"]
ns = sorted(set(r["n"] for r in runs))
n_levels = [str(n) for n in ns]


# ----- 1. convergence curves: Spearman vs step, mean +/- sd over seeds, by n -----
rows = []
for r in runs:
    for st, sp in zip(r["steps"], r["spearman"]):
        rows.append({"n": r["n"], "step": st, "spearman": sp})
df = pd.DataFrame(rows)
agg = (df.groupby(["n", "step"])["spearman"]
         .agg(["mean", "std"]).reset_index().fillna(0))
agg["n"] = pd.Categorical(agg["n"].astype(str), categories=n_levels, ordered=True)

p1 = (
    ggplot(agg, aes("step", "mean", color="n", fill="n"))
    + geom_hline(yintercept=thresh, linetype="dotted", color="#888888", size=0.3)
    + geom_ribbon(aes(ymin="mean-std", ymax="mean+std"), alpha=0.12, color="none")
    + geom_line(size=0.5)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_fill_brewer(type="qual", palette="Set1")
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Training Step", y=r"Spearman($s$, $|a|$)", color="$n$", fill="$n$")
)
p1.save(OUT / "toy_linear_convergence.pdf", verbose=False)


# ----- 2. final recovery vs n: Spearman / top-k overlap / pairwise -----
rows = []
label = {"spearman": "Spearman", "topk_overlap": "Top-$k$", "pairwise": "Pairwise"}
for r in runs:
    for key in ["spearman", "topk_overlap", "pairwise"]:
        rows.append({"n": r["n"], "metric": label[key], "val": r[key][-1]})
df2 = pd.DataFrame(rows)
agg2 = (df2.groupby(["n", "metric"])["val"]
          .agg(["mean", "std"]).reset_index())
agg2["metric"] = pd.Categorical(
    agg2["metric"], categories=list(label.values()), ordered=True)

p2 = (
    ggplot(agg2, aes("n", "mean", color="metric"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.08, size=0.3)
    + geom_line(size=0.5)
    + geom_point(size=1.1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Number of Terms $n$", y="Final Recovery", color="")
)
p2.save(OUT / "toy_linear_recovery_vs_n.pdf", verbose=False)


# ----- 3. convergence rate: steps to Spearman>=thresh vs n -----
rows = [{"n": r["n"], "tts": r["tts"]} for r in runs if r["tts"] is not None]
df3 = pd.DataFrame(rows)
agg3 = df3.groupby("n")["tts"].agg(["mean", "std"]).reset_index().fillna(0)

p3 = (
    ggplot(agg3, aes("n", "mean"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.08,
                    size=0.3, color="#377eb8")
    + geom_line(size=0.5, color="#377eb8")
    + geom_point(size=1.1, color="#377eb8")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + scale_y_log10(labels=label_log(base=10))
    + labs(x="Number of Terms $n$",
           y=fr"Steps to Spearman $\geq {thresh}$")
)
p3.save(OUT / "toy_linear_convrate_vs_n.pdf", verbose=False)

print("Saved:")
for f in ["toy_linear_convergence", "toy_linear_recovery_vs_n", "toy_linear_convrate_vs_n"]:
    print(f"  {OUT / (f + '.pdf')}")
