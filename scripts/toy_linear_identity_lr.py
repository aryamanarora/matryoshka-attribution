"""Does Adam matter for the identity-STE variant? LR sweep x {identity,sigmoid} x {Adam,SGD}.

Hypothesis: identity STE's gradient is g*delta_i = 2(y_mask-y_clean)*a_i*(x_i-x'_i), whose
per-coordinate MAGNITUDE scales with |a_i| -- i.e. the importance signal lives in the
gradient magnitude. Adam normalizes each coordinate by its own RMS gradient (~|a_i|),
dividing that signal out; plain SGD preserves it (score accumulates ~ E[g*delta_i] ~ a_i^2).
So we expect identity+SGD >> identity+Adam, while sigmoid STE (gate-slope reweighting) is
less hurt by Adam.

Fixed n=256 (largest gap). x=LR, y=final Spearman, line per (ste,opt); IxG as reference.
Saves results/toy_linear_identity_lr.pkl, plots paper/figs/toy_linear_identity_lr.pdf.
  uv run python scripts/toy_linear_identity_lr.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, labs, scale_color_brewer,
    scale_x_log10, scale_y_continuous, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

from toy_linear_mattr import train_one

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

N = 256
SEEDS = 3
STEPS = 2000
LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
CONFIGS = [
    ("identity", "adam", "identity STE, Adam"),
    ("identity", "sgd", "identity STE, SGD"),
    ("sigmoid", "adam", "sigmoid STE, Adam"),
    ("sigmoid", "sgd", "sigmoid STE, SGD"),
]
ORDER = [c[2] for c in CONFIGS]


def main():
    rows = []
    for ste, opt, label in CONFIGS:
        for lr in LRS:
            fins = []
            for seed in range(SEEDS):
                r = train_one(N, lr=lr, num_steps=STEPS, seed=seed,
                              k_schedule="uniform", ste=ste, opt=opt)
                fins.append(r["spearman"][-1])
            rows.append({"config": label, "lr": lr, "final": float(np.mean(fins))})
            print(f"{label:22s} lr={lr:<6g}  final Spearman={np.mean(fins):.3f}")

    df = pd.DataFrame(rows)
    with open(R / "toy_linear_identity_lr.pkl", "wb") as f:
        pickle.dump(df, f)
    df["config"] = pd.Categorical(df["config"], categories=ORDER, ordered=True)

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
        ggplot(df, aes("lr", "final", color="config"))
        + geom_line(size=0.5) + geom_point(size=1.0)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_continuous(limits=(0, 1), breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(x="Learning Rate", y=r"Final Spearman($s$, $|a|$)", color="")
    )
    p.save(OUT / "toy_linear_identity_lr.pdf", verbose=False)
    print(f"Saved {OUT / 'toy_linear_identity_lr.pdf'}")


if __name__ == "__main__":
    main()
