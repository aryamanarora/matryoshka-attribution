"""faith vs accuracy at k=1 (a SINGLE node patched/kept), not integrated over k.

Same layout as plot_accauc_vs_faithauc_cause.py -- rows = intervention direction, columns =
substrate x ±input, colour = method, shape = loss, task-group averaged -- but each point is
the FIRST point of the curve instead of its log-k integral:

  iso   x = iso_metrics["acc_base"][0]      y = iso_metrics["faithfulness"][0]    (↑ good)
  cause x = cause_metrics["acc_base"][0]    y = cause_metrics["faithfulness"][0]  (↓ good)

k=1 is the smallest sparsity in eval_sva's sweep (sparsities start at 1/total, and
_hard_topk_indices takes ki = max(1, int(k))), so exactly one unit is selected. That unit is
NOT the same granularity in every column (models/llama.py:53-65):
  * `node`: total = layers*heads + layers (+1 if include_input) = 1056 for llama3. There is NO
    position axis, so one unit is one whole attention HEAD or one whole MLP LAYER, at every
    position. k=1 here is a big object -- e.g. IG's usual pick, the entire layer-0 MLP.
  * `mlp`: total = layers*spans*d_mlp = 32*6*14336 = 2,752,512 -- one unit is one neuron at
    one span. `mlp+attn_head` adds layers*spans*heads on top (2,758,656).
So only the two right-hand columns are literally single-NEURON, and they are ~0.99 flat
because one neuron of 2.75M does nothing measurable.

This is the least-averaged view of the cause direction, and it is where a ranking's top-1
choice is tested directly: iso k=1 asks "is the single best unit sufficient on its own"
(it is not, for anything -- faithfulness there is ~0), cause k=1 asks "does removing the
single best unit already break the model".

Data: results/sva_sweep (−input) + results/sva_sweep_input (+input).
Run:  uv run python plots/plot_faith_vs_acc_k1.py  ->  plots/faith_vs_acc_k1.pdf
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_color_manual, scale_shape_manual,
    guides, guide_legend, expand_limits,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accauc_vs_faithauc as R          # noqa: E402  (method set / colours / averaging)
import plot_accauc_vs_faithauc_cause as C    # noqa: E402  (load + group_avg over a getter)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 3.4),   # full text width, 2 rows; appendix figure
        axis_title=element_text(size=8),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=8,
        legend_position="bottom",
        legend_direction="horizontal",
        legend_box="horizontal",
        legend_box_margin=0,
        legend_margin=0,
    )
)

OUT = "plots/faith_vs_acc_k1.pdf"
DIRECTIONS = [
    ("iso: keep 1 unit", lambda d: (d["iso_metrics"]["acc_base"][0],
                                    d["iso_metrics"]["faithfulness"][0])),
    ("cause: patch 1 unit", lambda d: (d["cause_metrics"]["acc_base"][0],
                                       d["cause_metrics"]["faithfulness"][0])),
]


def main():
    rows, ks = [], []
    for res, inp_label in R.SWEEPS:
        raw = C.load(res)
        ks += [d["n_nodes"][0] for d in raw.values()]
        for dir_label, get in DIRECTIONS:
            for m, (mlabel, _) in R.METHODS.items():
                for lkey, llabel in R.LOSSES.items():
                    for sub, slabel in R.SUBSTRATES:
                        r = C.group_avg(raw, m, lkey, sub, get)
                        if r is None:
                            continue
                        rows.append(dict(x=r[0], y=r[1], method=mlabel, loss=llabel,
                                         facet=f"{slabel}, {inp_label}", direction=dir_label))
    # the first sweep point must really be one unit, or "k=1" in the title is a lie
    assert max(abs(k - 1.0) for k in ks) < 1e-6, f"first k is not 1: {min(ks)}..{max(ks)}"
    df = pd.DataFrame(rows)

    df["method"] = pd.Categorical(df["method"], [v[0] for v in R.METHODS.values()])
    df["loss"] = pd.Categorical(df["loss"], list(R.LOSSES.values()))
    df["direction"] = pd.Categorical(df["direction"], [lab for lab, _ in DIRECTIONS])
    facet_order = ["Node, −input", "Node, +input", "MLP, −input", "MLP+Attn, −input"]
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in set(df["facet"])])

    p = (
        ggplot(df, aes("x", "y", color="method", shape="loss"))
        + geom_point(size=2.6, alpha=0.85, stroke=0.3)
        # y free per row: the iso row is pinned near 0 at k=1 (one unit is never sufficient),
        # so sharing y with the cause row would collapse it to a single line of dots.
        + facet_grid("direction ~ facet", scales="free_y")
        + expand_limits(x=0, y=0)
        + scale_color_manual(values={lab: col for lab, col in R.METHODS.values()}, name="Method")
        + scale_shape_manual(values=R.LOSS_SHAPE, name="Loss")
        + labs(x="Accuracy at $k{=}1$, base label (iso ↑ / cause ↓)",
               y="Faithfulness at $k{=}1$ (iso ↑ / cause ↓)")
        + guides(color=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    p.save(OUT, dpi=300, verbose=False)
    p.save(OUT.replace(".pdf", ".png"), dpi=200, verbose=False)   # preview only
    print(f"wrote {OUT} ({len(df)} points)")
    print(df.pivot_table(index=["direction", "method"], columns="facet",
                         values=["x", "y"], observed=True).round(3).to_string())


if __name__ == "__main__":
    main()
