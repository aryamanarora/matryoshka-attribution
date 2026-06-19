"""Plot k-vs-accuracy and k-vs-p(correct) for the multitask DAS eval.

Reads results/pythia1b_multitask_das_acc.pkl (from eval_multitask_das_acc.py).
Two plots, faceted by task: accuracy and p(correct) vs absolute k (log x), learned
(top-k by score) vs random dim ordering, with the all-features point marked.
Hybrid-plot styling. Run on sc.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (ggplot, aes, geom_line, geom_point, geom_ribbon, facet_wrap,
                      scale_x_log10, scale_color_manual, scale_fill_manual, labs,
                      theme_bw, theme_set, theme, element_text, element_line, element_blank)

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        legend_text=element_text(size=6),
        legend_title=element_blank(),
        legend_key_size=8,
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=6, face="plain"),
    )
)

PALETTE = {"learned": "#e41a1c", "random": "#888888"}
short = lambda t: t.replace("syntaxgym/", "")

d = pickle.load(open(R / "pythia1b_multitask_das_acc.pkl", "rb"))
pt = d["per_task"]

rows = []
for t, ev in pt.items():
    ks = ev["ks"]
    for oname in ("learned", "random"):
        for metric, key in (("accuracy", f"{oname}_acc"), ("p(correct)", f"{oname}_pc")):
            mean = ev[key]; std = ev[f"{key}_std"]
            n = ev.get("n_eval", 100)
            for j, k in enumerate(ks):
                se = std[j] / np.sqrt(n)
                rows.append({"task": short(t), "k": k, "order": oname, "metric": metric,
                             "y": mean[j], "lo": mean[j] - se, "hi": mean[j] + se})
df = pd.DataFrame(rows)
df["order"] = pd.Categorical(df["order"], categories=["learned", "random"], ordered=True)


def make(metric, fname):
    sub = df[df["metric"] == metric]
    p = (ggplot(sub, aes("k", "y", color="order", fill="order"))
         + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.18, color="none")
         + geom_line(size=0.5) + geom_point(size=0.5)
         + facet_wrap("task", ncol=6)
         + scale_x_log10()
         + scale_color_manual(values=PALETTE) + scale_fill_manual(values=PALETTE)
         + labs(x="k (dims kept clean, log scale)", y=metric)
         + theme(figure_size=(11, 7), legend_position="top"))
    p.save(OUT / fname, dpi=300)
    print("Saved", fname)


make("accuracy", "mtdas_acc_vs_k.pdf")
make("p(correct)", "mtdas_pcorrect_vs_k.pdf")

# aggregate mean curve across tasks (interpolated onto common log-k grid not needed:
# tasks share the small-k values; just average where k present)
agg = (df.groupby(["metric", "order", "k"], observed=True)["y"].mean().reset_index())
for metric, fname in (("accuracy", "mtdas_acc_vs_k_mean.pdf"),
                      ("p(correct)", "mtdas_pcorrect_vs_k_mean.pdf")):
    sub = agg[agg["metric"] == metric]
    p = (ggplot(sub, aes("k", "y", color="order"))
         + geom_line(size=0.6) + geom_point(size=0.8)
         + scale_x_log10() + scale_color_manual(values=PALETTE)
         + labs(x="k (dims kept clean, log scale)", y=f"mean {metric} over tasks")
         + theme(figure_size=(3.6, 2.2), legend_position="top"))
    p.save(OUT / fname, dpi=300)
    print("Saved", fname)
