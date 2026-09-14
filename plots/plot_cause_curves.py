"""Cause curves: flip-rate to the SOURCE label vs k when the top-k units are patched to the
source (complement clean), for IG vs MAttr, and MAttr trained in the iso vs cause direction.

y = cause_metrics["acc_source"](k) from eval_sva.py jsons (the paper's Cause metric), x = k on a
log axis. Colour = method (plots/palette.py); linetype = the direction MAttr was TRAINED in
(iso = --mode sufficient, results/sva_sweep; cause = --mode necessary and joint = --mode joint,
results/sva_sweep_cause). Gradient baselines have no training direction and are drawn solid.
Rows = substrate, columns = task. One loss per figure (LOSS env, default ce). The x axis is cut
at KMAX (default 10^4): every non-random ranking has flipped >95% of examples by then, and the
grid runs to 2.7M units, so the uncut axis squeezes all the information into the left fifth.

Run:  uv run python plots/plot_cause_curves.py            -> plots/cause_curves_ce.pdf
      LOSS=acc uv run python plots/plot_cause_curves.py   -> plots/cause_curves_acc.pdf
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, facet_grid, labs, theme, theme_set, theme_bw, element_text,
    element_line, element_blank, scale_x_log10, scale_color_manual, scale_linetype_manual,
    guides, guide_legend,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                    # noqa: E402
import plot_accauc_vs_faithauc as R    # noqa: E402  (parse_method, on_model)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 3.0),   # full text width, 2 rows + 2-row legend
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02, panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top", legend_direction="horizontal", legend_box="vertical",
        legend_box_margin=0, legend_margin=0,
    )
)

LOSS = os.environ.get("LOSS", "ce")
KMAX = float(os.environ.get("KMAX", "1e4"))
OUT = f"plots/cause_curves_{LOSS}.pdf"
SOURCES = ["results/sva_sweep", "results/sva_sweep_cause"]
# parse_method key -> legend label. Only the tuned MAttr configs (the ones worth training in the
# cause direction) plus the gradient baselines; default-eps Adam is left out on purpose.
SERIES = {"IG": "IG", "mc_ig": "Expected Gradients", "IxG": "I×G",
          "stopk-log-eps1e-2": "MAttr", "softsgd-log": "MAttr (SGD)", "Random": "Random"}
SUBS = {"mlp": "MLP neurons", "mlp+attn_head": "MLP neurons + attn heads"}
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
TRAINED = {"iso": "iso (keep top-k)", "cause": "cause (patch top-k)", "joint": "joint"}
LINETYPE = {"iso (keep top-k)": "solid", "cause (patch top-k)": "dashed", "joint": "dotted",
            "n/a": "solid"}


def parse(fname, d):
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    mode = tag.split("_")[0]
    if mode in ("necessary", "joint"):
        m = R.parse_method(fname.replace(tag, tag.replace(mode + "_", "sufficient_", 1)), d)
        return m, ("cause" if mode == "necessary" else "joint")
    m = R.parse_method(fname, d)
    return m, ("iso" if m and m.startswith(("stopk", "softsgd", "soft", "idSTE")) else "n/a")


def load():
    rows = []
    for res in SOURCES:
        for f in sorted(glob.glob(res + "/*.json")):
            d = json.load(open(f))
            if d["nodes"] not in SUBS or d["task"] not in TASKS or not R.on_model(d):
                continue
            m, trained = parse(os.path.basename(f), d)
            if m not in SERIES or (d["loss"] != LOSS and m != "Random"):
                continue
            for k, y in zip(d["n_nodes"], d["cause_metrics"]["acc_source"]):
                rows.append(dict(sub=SUBS[d["nodes"]], task=d["task"], method=SERIES[m],
                                 trained=TRAINED.get(trained, "n/a"), k=float(k), y=float(y),
                                 seed=f))
    df = pd.DataFrame(rows)
    # Random has 3 seeds per cell: average them
    return df.groupby(["sub", "task", "method", "trained", "k"], as_index=False)["y"].mean()


def main():
    df = load()
    df = df[df["k"] <= KMAX]
    df["task"] = pd.Categorical(df["task"], TASKS)
    df["sub"] = pd.Categorical(df["sub"], list(SUBS.values()))
    df["method"] = pd.Categorical(df["method"], list(SERIES.values()))
    df["trained"] = pd.Categorical(df["trained"], list(LINETYPE))
    brk = [10.0 ** e for e in range(0, 8) if 10 ** e <= df["k"].max() * 1.5]
    sup = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    p = (
        ggplot(df, aes("k", "y", color="method", linetype="trained"))
        + geom_line(size=0.45)
        + facet_grid("sub ~ task")
        + scale_x_log10(breaks=brk,
                        labels=lambda bs: ["10" + str(int(round(np.log10(b)))).translate(sup)
                                           for b in bs])
        + scale_color_manual(values={lab: P.color(lab) for lab in SERIES.values()},
                             name="Method")
        + scale_linetype_manual(values=LINETYPE, breaks=list(TRAINED.values()),
                                name="MAttr trained for")
        + labs(x="Units patched to the source, k", y="Flip rate to source label")
        + guides(color=guide_legend(order=1, nrow=1), linetype=guide_legend(order=2, nrow=1))
    )
    p.save(OUT, dpi=300, verbose=False)
    p.save(OUT.replace(".pdf", ".png"), dpi=150, verbose=False)
    print(f"wrote {OUT}: {df[['sub','task','method','trained']].drop_duplicates().shape[0]} series")


if __name__ == "__main__":
    main()
