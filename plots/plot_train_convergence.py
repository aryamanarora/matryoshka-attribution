"""k-adjusted training convergence: loss vs k-fraction, colored by training stage.

Single-step loss is dominated by the randomly sampled k, so raw loss-vs-step is
uninformative. Here we put k on the x-axis and split training into stages: if the
loss-at-a-given-k keeps dropping in later stages the model is still learning
(undertrained); if the stage curves overlap, training has converged.

Reads results/<dir>/<task>_<model>_trainlog.csv (step,k,k_frac,loss,bias_step).

Usage:
    uv run python plots/plot_train_convergence.py [results_subdir]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_wrap, labs, scale_x_log10,
    scale_color_brewer, theme_bw, theme_set, theme, element_text, element_blank,
    element_line,
)

R = Path("/tmp/mib_pkls/results")
RB = R if R.exists() else Path("results")
SUBDIR = sys.argv[1] if len(sys.argv) > 1 else "conv_diag"

TASKS = [
    ("ioi", "gpt2", "IOI / GPT-2"),
    ("arithmetic_subtraction", "llama3", "Arith / Llama"),
    ("mcqa", "llama3", "MCQA / Llama"),
    ("arc_easy", "llama3", "ARC-E / Llama"),
    ("arc_challenge", "llama3", "ARC-C / Llama"),
]

N_STAGES = 5      # split training into this many step-bins
N_KBINS = 12      # log-spaced k-fraction bins

theme_set(
    theme_bw(base_size=9)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(11, 3),
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        strip_background=element_blank(),
        strip_text=element_text(size=8),
        legend_position="right",
    )
)


def main():
    rows = []
    facet_order = []
    for task, model, label in TASKS:
        p = RB / SUBDIR / f"{task}_{model}_trainlog.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["bias_step"] == 0]  # only normal score-training steps
        if len(df) < 10:
            continue
        n = df["step"].max() + 1
        df["stage"] = pd.cut(df["step"], bins=N_STAGES,
                             labels=[f"{int(i*n/N_STAGES)}-{int((i+1)*n/N_STAGES)}"
                                     for i in range(N_STAGES)])
        # log-spaced k-fraction bins
        kf = df["k_frac"].clip(lower=1e-4)
        df["kbin"] = pd.cut(np.log10(kf), bins=N_KBINS)
        g = (df.groupby(["stage", "kbin"], observed=True)
               .agg(loss=("loss", "mean"), k_frac=("k_frac", "mean"))
               .reset_index().dropna())
        g["task"] = label
        rows.append(g)
        facet_order.append(label)

    if not rows:
        print(f"No trainlog CSVs found under {RB/SUBDIR}")
        return
    df = pd.concat(rows, ignore_index=True)
    df["task"] = pd.Categorical(df["task"], categories=facet_order, ordered=True)

    p = (
        ggplot(df, aes(x="k_frac", y="loss", color="stage"))
        + geom_line(size=0.6)
        + geom_point(size=0.8)
        + facet_wrap("task", ncol=5, scales="free_y")
        + scale_x_log10()
        + scale_color_brewer(type="seq", palette="YlOrRd", direction=1)
        + labs(x="k (fraction of nodes)", y="loss (k-binned mean)",
               color="train step")
    )
    out = Path("paper/figs/train_convergence.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
