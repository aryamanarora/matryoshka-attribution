"""LR-decay check for MAttr on the variable-level quadratic toy (eval-denoising-loss).

The constant-LR sweep left MAttr oscillating around a good region but never settling.
Maybe a decaying schedule (fast early, settle late) fixes it. n=16; compares constant LR
vs cosine-to-0 vs 1/sqrt(t) decay for MAttr uniform-k and log-k, on the same held-out
all-k denoising loss. IxG and Random shown as references.

Saves results/toy_var_lrdecay_evalloss.pkl, plots paper/figs/toy_var_lrdecay_evalloss.pdf.
  uv run python scripts/toy_var_lrdecay_evalloss.py
"""
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, labs, facet_wrap, scale_color_brewer,
    scale_x_log10, scale_y_log10, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

from learning_to_attribute import learn_scores
from toy_var_mattr import build_model, forward
from toy_var_evalloss import make_eval, run_ixg, random_baseline

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
R.mkdir(exist_ok=True)

N = 16
SEEDS = 3
NUM_STEPS = 3000
EVAL_EVERY = 50

# (label, lr0, decay): decay in {"const","cosine","sqrt"}
SCHEDULES = [
    ("const 0.05", 0.05, "const"),
    ("const 0.2", 0.2, "const"),
    ("cosine 0.2->0", 0.2, "cosine"),
    ("1/sqrt(t) 0.2", 0.2, "sqrt"),
]


def run_mattr_decay(n, k_schedule, lr0, decay, num_steps, eval_every, seed):
    a, cq, qii, qjj, _ = build_model(n, seed)
    eval_loss = make_eval(n, a, cq, qii, qjj, seed=seed, base=10_000)
    steps, evals = [], []

    def loss_fn(mask):
        x, x_cf = torch.randn(1, n), torch.randn(1, n)
        y_clean = forward(x, a, cq, qii, qjj)
        x_eff = mask * x + (1 - mask) * x_cf
        return ((forward(x_eff, a, cq, qii, qjj) - y_clean) ** 2).mean()

    def on_step(step, k, lv, sc):
        if (step + 1) % eval_every == 0 or step == 0:
            steps.append(step + 1)
            evals.append(eval_loss(sc.data.argsort(descending=True)))

    learn_scores(n, loss_fn, steps=num_steps, variant="hard_topk", k_schedule=k_schedule,
                 T=0.5, n_iters=50, lr=lr0, lr_schedule=decay, on_step=on_step)
    return steps, evals


def main():
    line_order = [s[0] for s in SCHEDULES] + ["IxG", "Random"]
    rows = []
    for seed in range(SEEDS):
        rb = random_baseline(N, seed)
        ix_steps, ix_evals = run_ixg(N, NUM_STEPS, EVAL_EVERY, seed)[:2]
        for sched, sched_label in [("uniform", "MAttr (uniform $k$)"),
                                   ("log", "MAttr (log $k$)")]:
            for s, e in zip(ix_steps, ix_evals):
                rows.append({"facet": sched_label, "line": "IxG", "step": s, "loss": e})
            rows.append({"facet": sched_label, "line": "Random", "step": 1, "loss": rb})
            rows.append({"facet": sched_label, "line": "Random", "step": NUM_STEPS, "loss": rb})
            for label, lr0, decay in SCHEDULES:
                st, ev = run_mattr_decay(N, sched, lr0, decay, NUM_STEPS, EVAL_EVERY, seed)
                for s, e in zip(st, ev):
                    rows.append({"facet": sched_label, "line": label, "step": s, "loss": e})
        print(f"seed {seed} done")

    df = pd.DataFrame(rows)
    with open(R / "toy_var_lrdecay_evalloss.pkl", "wb") as f:
        pickle.dump(df, f)

    agg = df.groupby(["facet", "line", "step"])["loss"].mean().reset_index()
    agg["line"] = pd.Categorical(agg["line"], categories=line_order, ordered=True)
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
        ggplot(agg, aes("step", "loss", color="line"))
        + geom_line(size=0.5)
        + facet_wrap("facet", nrow=1)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_log10(labels=label_log(base=10))
        + labs(x="Training Step", y="Held-out Denoising Loss", color="")
    )
    p.save(OUT / "toy_var_lrdecay_evalloss.pdf", verbose=False)
    print(f"Saved {OUT / 'toy_var_lrdecay_evalloss.pdf'}")


if __name__ == "__main__":
    main()
