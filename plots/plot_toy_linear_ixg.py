"""Compare IxG (attribution patching) vs MAttr on the linear toy, same metrics.

Reads results/toy_linear_ixg.pkl (IxG) and results/toy_linear_mattr.pkl (MAttr, batch=1
n-sweep). Both consume one counterfactual pair per step/sample, so the x-axis (# CF pairs)
is directly comparable.

Three subfigure-sized PDFs:
  1. toy_linear_ixg_curves.pdf    -- Spearman vs # CF pairs, selected n, MAttr vs IxG
  2. toy_linear_ixg_recovery.pdf  -- final Spearman vs n, MAttr vs IxG
  3. toy_linear_ixg_convrate.pdf  -- CF pairs to Spearman>=thresh vs n, MAttr vs IxG

Regenerate with: uv run python plots/plot_toy_linear_ixg.py
"""
import pickle
from pathlib import Path

import numpy as np
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
from toy_linear_mattr import steps_to_threshold  # noqa: E402

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

ixg, thresh = load(R / "toy_linear_ixg.pkl", "IxG")
mattr, _ = load(R / "toy_linear_mattr.pkl", "MAttr")
runs = ixg + mattr
ns = sorted(set(r["n"] for r in runs))
methods = ["MAttr", "IxG"]
ltype = {"MAttr": "solid", "IxG": "dashed"}


# ----- 1. convergence curves (selected n), MAttr vs IxG -----
sel = [n for n in [16, 64, 256] if n in ns]
rows = []
for r in runs:
    if r["n"] not in sel:
        continue
    for st, sp in zip(r["steps"], r["spearman"]):
        rows.append({"n": r["n"], "method": r["method"], "pairs": st, "spearman": sp})
df = pd.DataFrame(rows)
agg = df.groupby(["n", "method", "pairs"])["spearman"].mean().reset_index()
agg["n"] = pd.Categorical(agg["n"].astype(str), categories=[str(n) for n in sel], ordered=True)
agg["method"] = pd.Categorical(agg["method"], categories=methods, ordered=True)

p1 = (
    ggplot(agg, aes("pairs", "spearman", color="n", linetype="method"))
    + geom_hline(yintercept=thresh, color="#888888", linetype="dotted", size=0.3)
    + geom_line(size=0.5)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_linetype_manual(values=ltype)
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="# Counterfactual Pairs", y=r"Spearman($s$, $|a|$)",
           color="$n$", linetype="")
)
p1.save(OUT / "toy_linear_ixg_curves.pdf", verbose=False)


# ----- 2. final recovery vs n, MAttr vs IxG -----
rows = [{"n": r["n"], "method": r["method"], "final": r["spearman"][-1]} for r in runs]
df2 = pd.DataFrame(rows)
agg2 = df2.groupby(["n", "method"])["final"].agg(["mean", "std"]).reset_index().fillna(0)
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
p2.save(OUT / "toy_linear_ixg_recovery.pdf", verbose=False)


# ----- 3. convergence rate (CF pairs to threshold) vs n, MAttr vs IxG -----
rows = [{"n": r["n"], "method": r["method"], "tts": r["tts"]}
        for r in runs if r.get("tts") is not None]
df3 = pd.DataFrame(rows)
agg3 = df3.groupby(["n", "method"])["tts"].agg(["mean", "std"]).reset_index().fillna(0)
agg3["method"] = pd.Categorical(agg3["method"], categories=methods, ordered=True)

p3 = (
    ggplot(agg3, aes("n", "mean", color="method"))
    + geom_errorbar(aes(ymin="mean-std", ymax="mean+std"), width=0.06, size=0.3)
    + geom_line(size=0.5) + geom_point(size=1.0)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(breaks=ns, labels=[str(n) for n in ns])
    + scale_y_log10(labels=label_log(base=10))
    + labs(x="Number of Terms $n$", y=fr"Pairs to Spearman $\geq {thresh}$", color="")
)
p3.save(OUT / "toy_linear_ixg_convrate.pdf", verbose=False)

print("Saved:")
for f in ["toy_linear_ixg_curves", "toy_linear_ixg_recovery", "toy_linear_ixg_convrate"]:
    print(f"  {OUT / (f + '.pdf')}")
