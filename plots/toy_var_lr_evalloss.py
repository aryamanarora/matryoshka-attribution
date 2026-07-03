"""LR sweep for the variable-level quadratic toy, on the eval-denoising-loss metric.

Fixes n=16 (the hardest facet from toy_var_evalloss.py) and sweeps the learning rate for
MAttr uniform-k and MAttr log-k, plotting eval denoising loss (hard top-k MSE averaged over
sparsities k=1..n, fixed held-out CF set) vs training step. IxG (no LR) and Random are shown
as reference lines. Tests whether MAttr's poor convergence here is an LR/variance issue or
structural. Reuses run_mattr / run_ixg / make_eval / random_baseline from toy_var_evalloss.

Saves results/toy_var_lr_evalloss.pkl, plots paper/figs/toy_var_lr_evalloss.pdf.
  uv run python scripts/toy_var_lr_evalloss.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, labs, facet_wrap, scale_color_brewer,
    scale_x_log10, scale_y_log10, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

from toy_var_evalloss import run_mattr, run_ixg, random_baseline

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
R.mkdir(exist_ok=True)

N = 16
SEEDS = 3
NUM_STEPS = 3000
EVAL_EVERY = 50
LRS = [0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
LR_ORDER = [f"lr={lr:g}" for lr in LRS]
REFS = ["IxG", "Random"]


def main():
    rows = []
    for seed in range(SEEDS):
        rb = random_baseline(N, seed)
        ix_steps, ix_evals = run_ixg(N, NUM_STEPS, EVAL_EVERY, seed)
        for sched, sched_label in [("uniform", "MAttr (uniform $k$)"),
                                   ("log", "MAttr (log $k$)")]:
            # reference lines, duplicated into each facet for context
            for s, e in zip(ix_steps, ix_evals):
                rows.append({"facet": sched_label, "line": "IxG", "step": s, "eval": e})
            rows.append({"facet": sched_label, "line": "Random", "step": 1, "eval": rb})
            rows.append({"facet": sched_label, "line": "Random", "step": NUM_STEPS, "eval": rb})
            for lr in LRS:
                st, ev = run_mattr(N, sched, lr, NUM_STEPS, EVAL_EVERY, seed)
                for s, e in zip(st, ev):
                    rows.append({"facet": sched_label, "line": f"lr={lr:g}",
                                 "step": s, "eval": e})
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    with open(R / "toy_var_lr_evalloss.pkl", "wb") as f:
        pickle.dump(df, f)

    agg = df.groupby(["facet", "line", "step"])["eval"].mean().reset_index()
    agg["line"] = pd.Categorical(agg["line"], categories=LR_ORDER + REFS, ordered=True)
    agg["facet"] = pd.Categorical(agg["facet"],
                                  categories=["MAttr (uniform $k$)", "MAttr (log $k$)"],
                                  ordered=True)

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(5.5, 2.1),
            axis_title=element_text(size=7), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(), panel_spacing_x=0.04,
            strip_background=element_blank(), strip_text=element_text(size=7),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=6, legend_position="top", legend_direction="horizontal",
            legend_box_margin=0,
        )
    )
    p = (
        ggplot(agg, aes("step", "eval", color="line"))
        + geom_line(size=0.5)
        + facet_wrap("facet", nrow=1)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_log10(labels=label_log(base=10))
        + labs(x="Training Step", y="Eval Denoising Loss", color="")
    )
    p.save(OUT / "toy_var_lr_evalloss.pdf", verbose=False)
    print(f"Saved {OUT / 'toy_var_lr_evalloss.pdf'}")


if __name__ == "__main__":
    main()
