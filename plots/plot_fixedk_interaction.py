"""Focused figure: k-schedule x STE interaction on the node substrate.

Shows that the soft (sigmoid-STE) gate degrades sharply when k is FIXED (no schedule),
while identity-STE is schedule-robust -- because soft's sigma'-gated gradient only trains
nodes near the top-k boundary, so a fixed boundary leaves the rest of the ranking untrained.

x = k-schedule {log, uniform, fixed-10%}, y = Accuracy AUC, colour = STE family,
faceted by task. acc loss. Data: results/sva_sweep/*_node_* MAttr runs.

Run:  uv run python plots/plot_fixedk_interaction.py
"""
import glob
import json
import re
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_wrap, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_color_brewer, scale_x_discrete,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
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

RES = Path("results/sva_sweep")
TASKS = ["nounpp", "rc", "simple", "within_rc",
         "addition", "months", "weekdays", "hours",   # arithmetic-wild
         "arc_easy", "ioi/qwen2.5"]
SCHED = ["log", "uniform", "fixed (10%)"]
LOSS = "acc"


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RES / "*_node_sufficient_hard_topk_*.json"))):
        name = Path(f).name
        tag = name.split("_node_", 1)[1][:-5]
        if re.search(r"_ig\d+", tag):          # skip the mask-space IG variants
            continue
        d = json.load(open(f))
        if d["loss"] != LOSS:
            continue
        fam = "idSTE (identity-STE)" if "identity" in tag else "soft (sigmoid-STE)"
        sched = ("fixed (10%)" if "fixedk" in tag
                 else "uniform" if "uniformk" in tag else "log")
        task = d["task"] if d["model"] == "llama3" else f"{d['task']}/{d['model']}"
        rows.append({"task": task, "family": fam, "schedule": sched, "acc_auc": d["acc_auc"]})
    return pd.DataFrame(rows)


def main():
    df = load()
    df["schedule"] = pd.Categorical(df["schedule"], categories=SCHED, ordered=True)
    df["task"] = pd.Categorical(df["task"], categories=TASKS, ordered=True)
    df["family"] = pd.Categorical(df["family"],
                                  categories=["soft (sigmoid-STE)", "idSTE (identity-STE)"],
                                  ordered=True)
    p = (
        ggplot(df, aes("schedule", "acc_auc", color="family", group="family"))
        + geom_line(size=0.5)
        + geom_point(size=1.1)
        + facet_wrap("~ task", ncol=3)
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_x_discrete(labels=["log", "unif", "fixed"])
        + labs(x="$k$ during training", y="Accuracy AUC ($\\uparrow$)", color="STE")
        + theme(figure_size=(5.5, 3.0))
    )
    out = RES / "fixedk_ste_interaction.pdf"
    p.save(out, verbose=False)
    print(f"wrote {out}  ({len(df)} points)")


if __name__ == "__main__":
    main()
