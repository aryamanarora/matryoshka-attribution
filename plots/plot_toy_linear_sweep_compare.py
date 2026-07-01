"""Convergence-rate comparison of the three k-schedules across LR and batch sweeps.

For each sweep (LR, batch) plots # samples to reach top-k overlap >= TOPK_THRESH (fraction
of the true top-25% nodes recovered), one line per k-schedule (uniform / log / sum_pow2),
faceted by n. "# samples" = number of masked loss
evaluations = steps * batch * (masked evals per step). uniform/log do 1 eval per step;
sum_pow2 does |ks| = (#powers of two < n)+1 evals per step, so its count is scaled up by
|ks| -- the compute-fair view that charges sum_pow2 for re-using each pair |ks| times.

All six sweeps share config (n in {16,64,256}, steps=1500, seeds=4, matched grids).

Outputs paper/figs/: toy_linear_lr_convrate_cmp.pdf, toy_linear_batch_convrate_cmp.pdf
Regenerate with: uv run python plots/plot_toy_linear_sweep_compare.py
"""
import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, labs, facet_wrap,
    scale_color_brewer, scale_x_log10, scale_y_log10,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)
from mizani.formatters import label_log

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 1.95),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.04,
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

METHODS = ["uniform $k$", "log $k$", r"$\sum_k$ pow2"]
SEL_N = [16, 64, 256]
# Convergence criterion: samples to reach this top-k overlap (fraction of true top-25%
# nodes recovered). Top-k overlap saturates below 1 at large n, so a 0.9-style bar isn't
# reachable; 0.6 is reachable across essentially all configs.
TOPK_THRESH = 0.6


def reach(steps, trace, thresh):
    for s, v in zip(steps, trace):
        if v >= thresh:
            return s
    return None


def n_pow2(n):
    """Masked forward evals per step for sum_pow2 at width n (matches train_one)."""
    ks, v = [], 1
    while v < n:
        ks.append(v); v *= 2
    if n - 1 >= 1 and (n - 1) not in ks:
        ks.append(n - 1)
    return len(ks)


def gather(sweep, xkey):
    """sweep in {'lrsweep','batchsweep'}; xkey in {'lr','batch'}. Uses module THRESH."""
    files = [
        (f"toy_linear_uniform_{sweep}.pkl", "uniform $k$"),
        (f"toy_linear_log_{sweep}.pkl", "log $k$"),
        (f"toy_linear_sumpow2_{sweep}.pkl", r"$\sum_k$ pow2"),
    ]
    rows, thresh = [], None
    for fname, method in files:
        with open(R / fname, "rb") as f:
            d = pickle.load(f)
        thresh = d["args"]["thresh"]
        for r in d["runs"]:
            if r["n"] not in SEL_N:
                continue
            tts = reach(r["steps"], r["topk_overlap"], TOPK_THRESH)
            if tts is None:
                continue
            batch = r.get("batch", 1)
            # # samples = pairs (steps*batch) * masked evals per step (|ks| for sum_pow2)
            fwd = n_pow2(r["n"]) if method == r"$\sum_k$ pow2" else 1
            rows.append({"n": r["n"], "method": method, "x": r[xkey],
                         "samples": tts * batch * fwd})
    df = pd.DataFrame(rows)
    agg = df.groupby(["n", "method", "x"])["samples"].mean().reset_index()
    agg["method"] = pd.Categorical(agg["method"], categories=METHODS, ordered=True)
    agg["facet"] = pd.Categorical(
        "$n = " + agg["n"].astype(str) + "$",
        categories=[f"$n = {n}$" for n in SEL_N], ordered=True)
    return agg, thresh


def make_plot(sweep, xkey, xlab, outname):
    agg, thresh = gather(sweep, xkey)
    p = (
        ggplot(agg, aes("x", "samples", color="method"))
        + geom_line(size=0.5) + geom_point(size=1.0)
        + facet_wrap("facet", nrow=1)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_log10(labels=label_log(base=10))
        + labs(x=xlab, y=fr"Samples to Top-$k$ Overlap $\geq {TOPK_THRESH}$", color="")
    )
    p.save(OUT / outname, verbose=False)
    print(f"  {OUT / outname}")


print("Saved:")
make_plot("lrsweep", "lr", "Learning Rate", "toy_linear_lr_convrate_cmp.pdf")
make_plot("batchsweep", "batch", "Batch Size", "toy_linear_batch_convrate_cmp.pdf")
