"""Facet grid: rows = method (uniform / log / sum_pow2), cols = k level.
y = overlap@k (fraction of true top-k recovered), x = # examples (counterfactual pairs
seen = step * batch), one line per swept hyperparameter value (LR or batch). Mean over
seeds. Fixed n. (For the LR sweep batch=1, so examples = steps.)

Reads results/toy_linear_topk_grid_{lr,batch}.pkl. Emits to paper/figs/:
  toy_linear_topk_grid_lr.pdf     (lines = learning rate, batch=1)
  toy_linear_topk_grid_batch.pdf  (lines = batch size, lr=0.05)

Regenerate with: uv run python plots/plot_toy_linear_topk_grid.py
"""
import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, labs, facet_grid,
    scale_color_brewer, scale_x_log10, scale_y_continuous,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 4.2),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
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

METHOD_LABEL = {"uniform": "uniform $k$", "log": "log $k$", "sum_pow2": r"$\sum_k$ pow2"}
METHOD_ORDER = [METHOD_LABEL[m] for m in ["uniform", "log", "sum_pow2"]]


def make_grid(which, line_key, line_lab):
    with open(R / f"toy_linear_topk_grid_{which}.pkl", "rb") as f:
        d = pickle.load(f)
    runs = d["runs"]
    k_levels = d["args"]["k_levels"]
    n = d["args"]["n"]

    rows = []
    for r in runs:
        for k in k_levels:
            trace = r["topk_multi"].get(k)
            if not trace:
                continue
            for st, ov in zip(r["steps"], trace):
                # examples = counterfactual pairs seen = step * batch
                rows.append({"method": METHOD_LABEL[r["method"]],
                             "k": k, "line": r[line_key],
                             "examples": st * r["batch"], "overlap": ov})
    df = pd.DataFrame(rows)
    # mean over seeds
    agg = df.groupby(["method", "k", "line", "examples"])["overlap"].mean().reset_index()
    agg["method"] = pd.Categorical(agg["method"], categories=METHOD_ORDER, ordered=True)
    agg["k"] = pd.Categorical("$k=" + agg["k"].astype(str) + "$",
                              categories=[f"$k={k}$" for k in k_levels], ordered=True)
    agg["line"] = pd.Categorical(agg["line"].astype(str),
                                 categories=[str(v) for v in sorted(df["line"].unique())],
                                 ordered=True)

    p = (
        ggplot(agg, aes("examples", "overlap", color="line"))
        + geom_line(size=0.5)
        + facet_grid("method ~ k")
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.5, 1.0])
        + labs(x="# Examples (CF pairs)", y=r"Overlap@$k$ (frac. of true top-$k$)",
               color=line_lab)
    )
    out = OUT / f"toy_linear_topk_grid_{which}.pdf"
    p.save(out, verbose=False)
    print(f"  {out}  (n={n})")


print("Saved:")
make_grid("lr", "lr", "Learning Rate")
make_grid("batch", "batch", "Batch Size")
