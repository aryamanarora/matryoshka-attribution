"""Sample-/compute-efficiency view of the 4-way toy comparison.

Same convergence data as plot_toy_linear_methods.py, but the x-axis counts MASKED FORWARD
PASSES (the dominant cost), not counterfactual pairs. uniform/log MAttr and IxG do 1 forward
per step/sample; sum_pow2 does |ks| = (#powers of two < n) + 1 forwards per step on the
SAME pair, so its curve is stretched right by that factor. This exposes whether sum_pow2's
per-pair advantage survives once you pay for the extra forwards.

\textwidth single-row figure, faceted by n. Regenerate:
    uv run python plots/plot_toy_linear_sample_eff.py
"""
import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_hline, labs, facet_wrap,
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
        figure_size=(5.5, 1.9),
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


def n_pow2(n):
    """Number of forward passes sum_pow2 does per step at width n (matches train_one)."""
    ks, v = [], 1
    while v < n:
        ks.append(v); v *= 2
    if n - 1 >= 1 and (n - 1) not in ks:
        ks.append(n - 1)
    return len(ks)


SOURCES = [
    (R / "toy_linear_mattr.pkl", "MAttr (uniform $k$)", lambda n: 1),
    (R / "toy_linear_mattr_logk.pkl", "MAttr (log $k$)", lambda n: 1),
    (R / "toy_linear_mattr_sumpow2.pkl", r"MAttr ($\sum_k$ pow2)", n_pow2),
    (R / "toy_linear_ixg.pkl", "IxG", lambda n: 1),
]
METHODS = [m for _, m, _ in SOURCES]
SEL_N = [16, 64, 256]

thresh = None
rows = []
for path, method, fwd_per_step in SOURCES:
    with open(path, "rb") as f:
        d = pickle.load(f)
    thresh = d["args"]["thresh"]
    for r in d["runs"]:
        if r["n"] not in SEL_N:
            continue
        f_per = fwd_per_step(r["n"])
        for st, sp in zip(r["steps"], r["spearman"]):
            rows.append({"n": r["n"], "method": method,
                         "forwards": st * f_per, "spearman": sp})

df = pd.DataFrame(rows)
agg = df.groupby(["n", "method", "forwards"])["spearman"].mean().reset_index()
agg["method"] = pd.Categorical(agg["method"], categories=METHODS, ordered=True)
agg["facet"] = pd.Categorical(
    "$n = " + agg["n"].astype(str) + "$",
    categories=[f"$n = {n}$" for n in SEL_N], ordered=True)

p = (
    ggplot(agg, aes("forwards", "spearman", color="method"))
    + geom_hline(yintercept=thresh, color="#888888", linetype="dotted", size=0.3)
    + geom_line(size=0.6)
    + facet_wrap("facet", nrow=1)
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_x_log10(labels=label_log(base=10))
    + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
    + labs(x="# Masked Forward Passes", y=r"Spearman($s$, $|a|$)", color="")
)
p.save(OUT / "toy_linear_sample_eff.pdf", verbose=False)
print(f"Saved {OUT / 'toy_linear_sample_eff.pdf'}")
