"""Compare uniform-k vs log-uniform-k sampling for MAttr on the linear toy.

Reads results/toy_linear_mattr.pkl (uniform-k, headline) and
results/toy_linear_mattr_logk.pkl (log-uniform k, the "+log k" ablation). Same metrics,
same batch=1 setup, so convergence (vs # CF pairs / steps) is directly comparable.

Three subfigure-sized PDFs in paper/figs/:
  1. toy_linear_logk_curves.pdf    -- Spearman vs step, selected n, uniform vs log
  2. toy_linear_logk_recovery.pdf  -- final Spearman vs n, uniform vs log
  3. toy_linear_logk_convrate.pdf  -- steps to Spearman>=thresh vs n, uniform vs log

Regenerate with: uv run python plots/plot_toy_linear_logk.py
"""
import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_errorbar, geom_hline, labs,
    scale_color_brewer, scale_linetype_manual, scale_x_log10, scale_y_log10,
    scale_y_continuous, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
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


def load(path, method):
    with open(path, "rb") as f:
        d = pickle.load(f)
    for r in d["runs"]:
        r["method"] = method
    return d["runs"], d["args"]["thresh"]

uni, thresh = load(R / "toy_linear_mattr.pkl", "uniform $k$")
log, _ = load(R / "toy_linear_mattr_logk.pkl", "log $k$")
runs = uni + log
ns = sorted(set(r["n"] for r in runs))
methods = ["uniform $k$", "log $k$"]
ltype = {"uniform $k$": "solid", "log $k$": "dashed"}


# ----- 1. convergence curves (selected n) -----
sel = [n for n in [16, 64, 256] if n in ns]
rows = []
for r in runs:
    if r["n"] not in sel:
        continue
    for st, sp in zip(r["steps"], r["spearman"]):
        rows.append({"n": r["n"], "method": r["method"], "step": st, "spearman": sp})
agg = pd.DataFrame(rows).groupby(["n", "method", "step"])["spearman"].mean().reset_index()
agg["n"] = pd.Categorical(agg["n"].astype(str), categories=[str(n) for n in sel], ordered=True)
agg["method"] = pd.Categorical(agg["method"], categories=methods, ordered=True)

p1 = (
    ggplot(agg, aes("step", "spearman", color="n", linetype="method"))
    + geom_hline(yintercept=thresh, color="#888888", linetype="dotted", size=0.3)
    + geom_line(size=0.5)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_linetype_manual(values=ltype)
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Training Step", y=r"Spearman($s$, $|a|$)", color="$n$", linetype="")
)
p1.save(OUT / "toy_linear_logk_curves.pdf", verbose=False)


# ----- 2. final recovery vs n -----
rows = [{"n": r["n"], "method": r["method"], "final": r["spearman"][-1]} for r in runs]
agg2 = pd.DataFrame(rows).groupby(["n", "method"])["final"].agg(["mean", "std"]).reset_index().fillna(0)
agg2["method"] = pd.Categorical(agg2["method"], categories=methods, ordered=True)

p2 = (
    ggplot(agg2, aes("n", "mean", color="method"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5) + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="Number of Terms $n$", y=r"Final Spearman($s$, $|a|$)", color="")
)
p2.save(OUT / "toy_linear_logk_recovery.pdf", verbose=False)


# ----- 3. convergence rate vs n -----
rows = [{"n": r["n"], "method": r["method"], "tts": r["tts"]}
        for r in runs if r.get("tts") is not None]
agg3 = pd.DataFrame(rows).groupby(["n", "method"])["tts"].agg(["mean", "std"]).reset_index().fillna(0)
agg3["method"] = pd.Categorical(agg3["method"], categories=methods, ordered=True)

p3 = (
    ggplot(agg3, aes("n", "mean", color="method"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5) + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + scale_y_log10(labels=label_log(base=10))
    + labs(x="Number of Terms $n$", y=fr"Steps to Spearman $\geq {thresh}$", color="")
)
p3.save(OUT / "toy_linear_logk_convrate.pdf", verbose=False)

print("Saved:")
for f in ["toy_linear_logk_curves", "toy_linear_logk_recovery", "toy_linear_logk_convrate"]:
    print(f"  {OUT / (f + '.pdf')}")
