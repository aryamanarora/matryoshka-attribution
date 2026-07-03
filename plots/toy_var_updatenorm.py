"""Per-step update norm for each method on the variable-level quadratic toy (n=16).

Shows the magnitude of the parameter update at each step, explaining the convergence
behaviour seen in the eval-loss plots:
  - MAttr (constant LR): update norm ~ lr * ||normalized grad||, stays ~flat -> bounces.
  - MAttr (cosine LR decay): update norm decays to 0 -> settles (matches the eval-loss win).
  - IxG: "update" = increment of the running-mean importance estimate ||imp_s - imp_{s-1}||,
    which decays ~1/s automatically (that's why IxG converges with no LR at all).

Per-step norms are noisy (single-sample gradients); we average over seeds then apply a
rolling mean for legibility. Saves results + paper/figs/toy_var_updatenorm.pdf.
  uv run python scripts/toy_var_updatenorm.py
"""
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, labs, scale_color_brewer,
    scale_x_log10, scale_y_log10, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

from learning_to_attribute import learn_scores
from toy_var_mattr import build_model, forward

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
R.mkdir(exist_ok=True)

N = 16
SEEDS = 5
NUM_STEPS = 3000
ROLL = 25


def run_mattr_norm(n, k_schedule, lr0, decay, num_steps, seed):
    a, cq, qii, qjj, _ = build_model(n, seed)
    norms = []
    prev = torch.zeros(n)                              # scores start at 0

    def loss_fn(mask):
        x, x_cf = torch.randn(1, n), torch.randn(1, n)
        y_clean = forward(x, a, cq, qii, qjj)
        x_eff = mask * x + (1 - mask) * x_cf
        return ((forward(x_eff, a, cq, qii, qjj) - y_clean) ** 2).mean()

    def on_step(step, k, lv, sc):                     # per-step update norm ||Δscores||
        nonlocal prev
        cur = sc.detach()
        norms.append((cur - prev).norm().item())
        prev = cur.clone()

    learn_scores(n, loss_fn, steps=num_steps, variant="hard_topk", k_schedule=k_schedule,
                 T=0.5, n_iters=50, lr=lr0, lr_schedule=decay, on_step=on_step)
    return norms


def run_ixg_norm(n, num_steps, seed):
    """Update = increment of the running-mean importance estimate ||imp_s - imp_{s-1}||."""
    a, cq, qii, qjj, _ = build_model(n, seed)
    ap_sum = torch.zeros(n)
    prev = torch.zeros(n)
    norms = []
    for s in range(1, num_steps + 1):
        Xc = torch.randn(1, n); Xcf = torch.randn(1, n, requires_grad=True)
        y_cf = forward(Xcf, a, cq, qii, qjj)
        grad = torch.autograd.grad(y_cf.sum(), Xcf)[0]
        with torch.no_grad():
            y_clean = forward(Xc, a, cq, qii, qjj)
            ap_sum += ((Xc - Xcf) * grad * (2.0 * (y_cf.detach() - y_clean))[:, None])[0]
            imp = -ap_sum / s
            norms.append((imp - prev).norm().item())
            prev = imp.clone()
    return norms


CONFIGS = [
    ("MAttr (uniform $k$)", lambda seed: run_mattr_norm(N, "uniform", 0.05, "const", NUM_STEPS, seed)),
    ("MAttr (log $k$)", lambda seed: run_mattr_norm(N, "log", 0.05, "const", NUM_STEPS, seed)),
    ("MAttr (uniform, cosine)", lambda seed: run_mattr_norm(N, "uniform", 0.2, "cosine", NUM_STEPS, seed)),
    ("IxG", lambda seed: run_ixg_norm(N, NUM_STEPS, seed)),
]
ORDER = [c[0] for c in CONFIGS]


def main():
    rows = []
    for label, fn in CONFIGS:
        for seed in range(SEEDS):
            for step, val in enumerate(fn(seed), start=1):
                rows.append({"method": label, "seed": seed, "step": step, "norm": val})
        print(f"{label} done")
    df = pd.DataFrame(rows)
    with open(R / "toy_var_updatenorm.pkl", "wb") as f:
        pickle.dump(df, f)

    agg = df.groupby(["method", "step"])["norm"].mean().reset_index()
    agg["norm"] = (agg.groupby("method")["norm"]
                   .transform(lambda s: s.rolling(ROLL, min_periods=1).mean()))
    agg["method"] = pd.Categorical(agg["method"], categories=ORDER, ordered=True)

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(3.4, 2.0),
            axis_title=element_text(size=7), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(),
            strip_background=element_blank(), strip_text=element_text(size=7),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=6, legend_position="top", legend_direction="horizontal",
            legend_box_margin=0,
        )
    )
    p = (
        ggplot(agg, aes("step", "norm", color="method"))
        + geom_line(size=0.5)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_log10(labels=label_log(base=10))
        + labs(x="Training Step", y="Update Norm", color="")
    )
    p.save(OUT / "toy_var_updatenorm.pdf", verbose=False)
    print(f"Saved {OUT / 'toy_var_updatenorm.pdf'}")


if __name__ == "__main__":
    main()
