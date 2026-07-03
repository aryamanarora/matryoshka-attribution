"""Learning curves (Spearman vs step) for identity/sigmoid STE x Adam/SGD on the linear toy.

Companion to toy_linear_identity_lr.py: each config at its best LR from that sweep, n=256,
mean over seeds, with IxG as reference. Shows the trajectory behind the final-value sweep:
identity+SGD should climb like IxG (it IS IxG-by-SGD), identity+Adam should stall low.

Saves results/toy_linear_identity_curves.pkl, plots paper/figs/toy_linear_identity_curves.pdf.
  uv run python scripts/toy_linear_identity_curves.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, labs, scale_color_brewer,
    scale_x_log10, scale_y_continuous, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

from toy_linear_mattr import train_one
from toy_linear_ixg import ixg_one

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

N = 256
SEEDS = 5
STEPS = 2000
# (ste, opt, best_lr, label) -- best LR taken from toy_linear_identity_lr.py
CONFIGS = [
    ("identity", "adam", 0.1, "identity STE, Adam"),
    ("identity", "sgd", 0.01, "identity STE, SGD"),
    ("sigmoid", "adam", 0.01, "sigmoid STE, Adam"),
    ("sigmoid", "sgd", 0.003, "sigmoid STE, SGD"),
]
ORDER = [c[3] for c in CONFIGS] + ["IxG"]


def main():
    rows = []
    for ste, opt, lr, label in CONFIGS:
        for seed in range(SEEDS):
            r = train_one(N, lr=lr, num_steps=STEPS, seed=seed,
                          k_schedule="uniform", ste=ste, opt=opt)
            for st, sp in zip(r["steps"], r["spearman"]):
                rows.append({"config": label, "seed": seed, "step": st, "spearman": sp})
        print(f"{label} done")
    for seed in range(SEEDS):
        r = ixg_one(N, num_samples=STEPS, seed=seed)
        for st, sp in zip(r["steps"], r["spearman"]):
            rows.append({"config": "IxG", "seed": seed, "step": st, "spearman": sp})
    print("IxG done")

    df = pd.DataFrame(rows)
    with open(R / "toy_linear_identity_curves.pkl", "wb") as f:
        pickle.dump(df, f)
    agg = df.groupby(["config", "step"])["spearman"].mean().reset_index()
    agg["config"] = pd.Categorical(agg["config"], categories=ORDER, ordered=True)

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(3.4, 2.1),
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
        ggplot(agg, aes("step", "spearman", color="config"))
        + geom_line(size=0.5)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(x="Training Step (= # CF pairs)", y=r"Spearman($s$, $|a|$)", color="")
    )
    p.save(OUT / "toy_linear_identity_curves.pdf", verbose=False)
    print(f"Saved {OUT / 'toy_linear_identity_curves.pdf'}")


if __name__ == "__main__":
    main()
