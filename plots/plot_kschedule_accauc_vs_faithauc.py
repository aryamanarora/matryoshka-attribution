"""Scatter of accuracy-AUC (x) vs faithfulness-AUC (y) for the three k-schedules, ±input.

Companion to plot_accauc_vs_faithauc.py: same axes and the same task-group averaging, but
the contrast is the k-schedule {log, uniform, fixed 10%} instead of the method. One point per
(schedule, STE family, loss); columns split on whether the input-embedding node is scored and
ablated, rows split the STE family. Both rows use the HARD top-k forward (masks.py's
`hard_topk` / `hard_topk_identity`) and differ only in the backward: the sigmoid-top-k STE
(dm/ds = d(soft)/ds) vs the identity STE (dm/ds = 1).

Fixed-k (a single training budget, no schedule) is the setting a mask learner like Edge
Pruning is stuck in, so this is the like-for-like version of that comparison inside our own
method. Rows split the STE family because that is where the interaction lives: the
sigmoid-STE gate's sigma'-gated gradient only trains nodes near the top-k boundary, so a
FIXED boundary leaves the rest of the ranking untrained, while identity-STE is far more
schedule-robust (cf. plot_fixedk_interaction.py).

Averaging, matching plot_accauc_vs_faithauc.py: mean over three task GROUPS -- SVA (itself the
mean of nounpp/rc/simple/within_rc) + ARC-E + IOI -- so the 4 SVA subtasks together carry the
same weight as each MIB task. Note this averages over two MODELS: every task here is llama3
except ioi, which is qwen2.5. The per-task version of this figure (which shows that fixed-k
identity-STE is actually the *best* config on IOI, against the pooled trend) is at a77b510.

Coverage caveat: --fixed-k-frac was only ever run for the `hard_topk` (STE) variants, not
for the soft top-k forward that is the MAttr headline -- so this figure is about the STE
families only. All runs are bs=1, so batch size is not confounded with the schedule.

Data: results/sva_sweep (input excluded) + results/sva_sweep_input (included), *_node_*.json.
  ±input is NOT recorded in the json fields or the filename tag -- the results DIRECTORY is
  the only thing that distinguishes them, so never glob across both.
Run:  uv run python plots/plot_kschedule_accauc_vs_faithauc.py
        -> plots/kschedule_accauc_vs_faithauc.pdf
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_color_manual, scale_shape_manual,
    guides, guide_legend, expand_limits,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 3.4),   # full text width; appendix figure
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

OUT = "plots/kschedule_accauc_vs_faithauc.pdf"
SVA = ["nounpp", "rc", "simple", "within_rc"]
GROUPS = [SVA, ["arc_easy"], ["ioi"]]
# (results dir, input-included label); see the ±input warning in the docstring
SWEEPS = [("results/sva_sweep", "$-$ input"),
          ("results/sva_sweep_input", "$+$ input")]

# k-schedule -> (display, colour); order = legend order
SCHEDULES = {
    "log": ("log $k$", "#1f77b4"),
    "unif": ("uniform $k$", "#ff7f0e"),
    "fixed": ("fixed $k{=}10\\%$", "#d62728"),
}
# BOTH rows are the HARD top-k forward (masks.py: hard_topk / hard_topk_identity); they differ
# only in the backward. Spell that out in the strip -- "sigmoid-STE" alone reads as if it were
# the soft top-k forward, which is the MAttr headline and is absent here (see caveat above).
STES = {"soft": "hard fwd, sigmoid STE", "id": "hard fwd, identity STE"}
LOSSES = {"acc": "acc", "ce": "CE", "logit_diff": "logit-diff"}
LOSS_SHAPE = {"acc": "o", "CE": "^", "logit-diff": "s"}


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


def load(res):
    """(ste, sched, loss, task) -> (acc_auc, faith_auc) for one sweep dir."""
    raw = {}
    for f in sorted(glob.glob(res + "/*_node_*.json")):
        d = json.load(open(f))
        p = parse(os.path.basename(f), d)
        if p is None or d["loss"] not in LOSSES:
            continue
        raw[(p[0], p[1], d["loss"], d["task"])] = (d["acc_auc"], d["faith_auc"])
    return raw


def group_avg(raw, ste, ks, loss):
    """Mean over task groups, or None if any group is missing (no partial averages)."""
    gx, gy = [], []
    for tasks in GROUPS:
        xs = [raw[(ste, ks, loss, t)] for t in tasks if (ste, ks, loss, t) in raw]
        if len(xs) != len(tasks):       # a partial SVA mean is not the same quantity
            return None
        gx.append(np.mean([v[0] for v in xs]))
        gy.append(np.mean([v[1] for v in xs]))
    return float(np.mean(gx)), float(np.mean(gy))


def main():
    rows, missing = [], []
    for res, inp in SWEEPS:
        raw = load(res)
        for ste, ste_lab in STES.items():
            for ks, (ks_lab, _) in SCHEDULES.items():
                for loss, loss_lab in LOSSES.items():
                    r = group_avg(raw, ste, ks, loss)
                    if r is None:
                        missing.append(f"{inp} {ste_lab} {ks_lab} {loss_lab}")
                        continue
                    rows.append(dict(acc_auc=r[0], faith_auc=r[1], sched=ks_lab,
                                     ste=ste_lab, loss=loss_lab, inp=inp))
    df = pd.DataFrame(rows)

    df["sched"] = pd.Categorical(df["sched"], [v[0] for v in SCHEDULES.values()])
    df["ste"] = pd.Categorical(df["ste"], list(STES.values()))
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()))
    df["inp"] = pd.Categorical(df["inp"], [lab for _, lab in SWEEPS])

    p = (
        ggplot(df, aes("acc_auc", "faith_auc", color="sched", shape="loss"))
        + geom_point(size=2.6, alpha=0.85, stroke=0.3)
        # scales fixed: every panel shares both axes, so ±input and the two STE families are
        # directly comparable. expand_limits anchors at 0 WITHOUT dropping points (scale
        # limits=(0, None) would silently discard anything negative).
        + facet_grid("ste ~ inp")
        + expand_limits(x=0, y=0)
        + scale_color_manual(values={lab: col for lab, col in SCHEDULES.values()},
                             name="$k$-schedule")
        + scale_shape_manual(values=LOSS_SHAPE, name="Loss")
        + labs(x="IIA AUC (↑)", y="Faith AUC (↑)")
        + guides(color=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    p.save(OUT, dpi=300, verbose=False)
    p.save(OUT.replace(".pdf", ".png"), dpi=200, verbose=False)   # preview only
    print(f"wrote {OUT} ({len(df)} points)")
    if missing:
        print(f"NO POINT (incomplete task groups) for {len(missing)}: " + "; ".join(missing))
    print(df.groupby(["inp", "ste", "sched"], observed=True)
          .agg(n=("acc_auc", "size"), acc=("acc_auc", "mean"), faith=("faith_auc", "mean"))
          .round(3).to_string())


if __name__ == "__main__":
    main()
