"""Scatter of accuracy-AUC (x) vs faithfulness-AUC (y) for the three k-schedules.

Companion to plot_accauc_vs_faithauc.py: same axes, but the contrast is the k-schedule
{log, uniform, fixed 10%} instead of the method, and points are NOT averaged over task
groups -- one facet per task, so the schedule effect can be read per task.

Fixed-k (a single training budget, no schedule) is the setting a mask learner like Edge
Pruning is stuck in, so this is the like-for-like version of that comparison inside our own
method. Rows split the STE family, because that is where the interaction lives: the
sigmoid-STE gate's sigma'-gated gradient only trains nodes near the top-k boundary, so a
FIXED boundary leaves the rest of the ranking untrained, while identity-STE is far more
schedule-robust (cf. plot_fixedk_interaction.py).

Coverage caveat: --fixed-k-frac was only ever run for the `hard_topk` (STE) variants, not
for the soft top-k forward that is the MAttr headline -- so this figure is about the STE
families only. All runs are bs=1, so batch size is not confounded with the schedule.

Data: results/sva_sweep/*_node_*.json (input node excluded from scoring/ablation).
Run:  uv run python plots/plot_kschedule_accauc_vs_faithauc.py
        -> plots/kschedule_accauc_vs_faithauc.pdf
"""
import glob
import json
import os
import re

import pandas as pd
from mizani.breaks import breaks_extended
from plotnine import (
    ggplot, aes, geom_point, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_color_manual, scale_shape_manual,
    scale_x_continuous, scale_y_continuous, guides, guide_legend,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 2.9),
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

RESULTS = "results/sva_sweep"
OUT = "plots/kschedule_accauc_vs_faithauc.pdf"

# k-schedule -> (display, colour); order = legend order
SCHEDULES = {
    "log": ("log $k$", "#1f77b4"),
    "unif": ("uniform $k$", "#ff7f0e"),
    "fixed": ("fixed $k{=}10\\%$", "#d62728"),
}
STES = {"soft": "sigmoid-STE", "id": "identity-STE"}
LOSSES = {"acc": "acc", "ce": "CE", "logit_diff": "logit-diff"}
LOSS_SHAPE = {"acc": "o", "CE": "^", "logit-diff": "s"}
# SVA subtasks first, then the two MIB tasks. Strip labels are plain matplotlib text, not
# LaTeX, so an escaped underscore would render its backslash -- use a hyphen instead.
TASK_ORDER = [("nounpp", "nounpp"), ("rc", "rc"), ("simple", "simple"),
              ("within_rc", "within-rc"), ("arc_easy", "ARC-E"), ("ioi", "IOI")]


def parse(fname, d):
    """(ste family, k-schedule) from the filename tag, or None to skip.

    The k-schedule MUST come from the tag, not from d['k_schedule']: --fixed-k-frac swaps
    the sampler but leaves the recorded k_schedule at its default ('log'), so trusting the
    json field silently mislabels every fixed-k run as log.
    """
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    if "hard_topk" not in tag:          # fixed-k only exists for the STE variants
        return None
    if re.search(r"_ig\d+", tag):       # mask-path IG runs are a separate ablation
        return None
    ste = "id" if "identity" in tag else "soft"
    ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
    return ste, ks


def main():
    rows = []
    for f in sorted(glob.glob(RESULTS + "/*_node_*.json")):
        d = json.load(open(f))
        p = parse(os.path.basename(f), d)
        if p is None or d["loss"] not in LOSSES:
            continue
        ste, ks = p
        task = dict(TASK_ORDER).get(d["task"])
        if task is None:
            continue
        rows.append(dict(acc_auc=d["acc_auc"], faith_auc=d["faith_auc"],
                         sched=SCHEDULES[ks][0], ste=STES[ste],
                         loss=LOSSES[d["loss"]], task=task))
    df = pd.DataFrame(rows)

    df["sched"] = pd.Categorical(df["sched"], [v[0] for v in SCHEDULES.values()])
    df["ste"] = pd.Categorical(df["ste"], list(STES.values()))
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()))
    df["task"] = pd.Categorical(df["task"], [lab for _, lab in TASK_ORDER])

    p = (
        ggplot(df, aes("acc_auc", "faith_auc", color="sched", shape="loss"))
        + geom_point(size=2.4, alpha=0.85, stroke=0.3)
        # Shared x across all panels so acc-AUC is directly comparable task-to-task; y is free
        # per row (the two STE families) since only the schedule ordering matters within a row.
        + facet_grid("ste ~ task", scales="free_y")
        # Both axes anchored at 0 so panel-to-panel gaps read as absolute, not zoomed. Nothing
        # is clipped: the data spans acc 0.33-0.60, faith 0.28-1.29 (limits would DROP points
        # below 0, so re-check these ranges before reusing this on runs that can score negative).
        + scale_x_continuous(limits=(0, None), breaks=breaks_extended(3))
        + scale_y_continuous(limits=(0, None), breaks=breaks_extended(4))
        + scale_color_manual(values={lab: col for lab, col in SCHEDULES.values()},
                             name="$k$-schedule")
        + scale_shape_manual(values=LOSS_SHAPE, name="Loss")
        + labs(x="IIA AUC (↑)", y="Faith AUC (↑)")
        + guides(color=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    p.save(OUT, dpi=300, verbose=False)
    p.save(OUT.replace(".pdf", ".png"), dpi=200, verbose=False)   # preview only
    print(f"wrote {OUT} ({len(df)} points)")
    print(df.groupby(["ste", "sched"], observed=True)
          .agg(n=("acc_auc", "size"), acc=("acc_auc", "mean"), faith=("faith_auc", "mean"))
          .round(3).to_string())


if __name__ == "__main__":
    main()
