"""iso vs cause curves behind the AUCs: faithfulness and accuracy vs k, both directions.

Rows are the four curves whose log-k integrals are the four AUCs in
plot_accauc_vs_faithauc_cause.py; columns are tasks. Line colour = method (the same four the
scatter plots, via the parent module's parse_method), linetype = loss.

WHY THE CAUSE ROW LOOKS ANTICORRELATED WITH ITS ACCURACY ROW -- it is, by construction, and
these curves show it. Every metric in a direction is a readout of the SAME per-k mean logit
diff `ld` (eval_sva.eval_metrics):
  * faithfulness = (ld − F_patch) / (F_clean − F_patch), monotone increasing in ld, with
    F_clean/F_patch always measured in the iso direction (so the iso curve runs 0 -> 1);
  * acc_base = mean(ld_i > 0) and acc_source = mean(ld_i < 0), i.e. the SAME comparison
    thresholded per example -- so acc_source = 1 − acc_base exactly, up to ties (measured:
    max |acc_base + acc_source − 1| = 0.06 over all 480 node curves).
In the cause direction ld falls as k grows, so faith falls while acc_source rises: measured
r(cause_accsrc_auc, cause_auc) = −0.989. The iso direction is the same construction with the
sign flipped (r = +0.783); it looks looser only because iso faithfulness is UNBOUNDED above
(ld can overshoot F_clean, giving faith AUC up to ~3.4) while accuracy saturates at 1.
So the cause row's two axes carry ~one degree of freedom, not two -- do not read it as two
independent pieces of evidence.

Data: results/sva_sweep (−input) by default; SWEEP=results/sva_sweep_input for the +input
twin. Node substrate only (the substrate with the MIB tasks and the +input variant).
Run:  uv run python plots/plot_sva_curves_iso_vs_cause.py
        -> plots/sva_curves_iso_vs_cause.pdf
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_x_log10, scale_color_manual,
    guides, guide_legend,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accauc_vs_faithauc as R   # noqa: E402  (method set + colours, kept in sync)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 5.6),   # full text width, 4 rows; appendix figure
        axis_title=element_text(size=8),
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
        legend_key_size=8,
        legend_position="bottom",
        legend_direction="horizontal",
        legend_box="horizontal",
        legend_box_margin=0,
        legend_margin=0,
    )
)

SWEEP = os.environ.get("SWEEP", "results/sva_sweep")
OUT = "plots/sva_curves_iso_vs_cause.pdf"
# (metrics dict, key, row label) -- the four curves behind the four AUCs
CURVES = [
    ("iso_metrics", "faithfulness", "iso: faith (↑)"),
    ("iso_metrics", "acc_base", "iso: acc base (↑)"),
    ("cause_metrics", "faithfulness", "cause: faith (↓)"),
    # acc_base in BOTH directions, so rows 2 and 4 are the same quantity and the iso/cause
    # contrast is apples-to-apples: base-label accuracy rises with k when the top-k is kept
    # clean and falls when it is patched. (= 1 − acc_source, so ↓ is good here; the AUCs are
    # exact complements, AUC(acc_base) + AUC(acc_source) = 1 up to ties.)
    ("cause_metrics", "acc_base", "cause: acc base (↓)"),
]
TASK_ORDER = ["nounpp", "rc", "simple", "within_rc", "arc_easy", "ioi/qwen2.5"]


def load():
    rows = []
    for f in sorted(glob.glob(SWEEP + "/*_node_*.json")):
        d = json.load(open(f))
        m = R.parse_method(os.path.basename(f), d)
        if m is None or m not in R.METHODS:
            continue
        # the ioi cell is qwen2.5 while every other task is llama3 -- say so in the strip
        task = d["task"] if d["model"] == "llama3" else f"{d['task']}/{d['model']}"
        for dictname, key, label in CURVES:
            for k, y in zip(d["n_nodes"], d[dictname][key]):
                rows.append({"task": task, "loss": R.LOSSES[d["loss"]],
                             "method": R.METHODS[m][0], "metric": label,
                             "k": float(k), "value": float(y)})
    return pd.DataFrame(rows)


def main():
    df = load()
    df["metric"] = pd.Categorical(df["metric"], [l for _, _, l in CURVES])
    df["task"] = pd.Categorical(df["task"], [t for t in TASK_ORDER if t in set(df["task"])])
    df["method"] = pd.Categorical(df["method"], [v[0] for v in R.METHODS.values()])
    df["loss"] = pd.Categorical(df["loss"], list(R.LOSSES.values()))

    brk = [10.0 ** e for e in range(0, 8) if 10 ** e <= df["k"].max() * 1.5]
    p = (
        ggplot(df, aes("k", "value", color="method", linetype="loss"))
        + geom_line(size=0.4)
        # free y: the four rows are different quantities (iso faith is unbounded above,
        # the accuracies live in [0, 1]), so a shared y would flatten three of them.
        + facet_grid("metric ~ task", scales="free_y")
        + scale_x_log10(breaks=brk, labels=lambda bs: [f"$10^{{{int(round(np.log10(b)))}}}$"
                                                       for b in bs])
        + scale_color_manual(values={lab: col for lab, col in R.METHODS.values()}, name="Method")
        + labs(x="$k$ (nodes selected: kept clean for iso, patched for cause)", y="",
               linetype="Loss")
        + guides(color=guide_legend(order=1, nrow=1), linetype=guide_legend(order=2, nrow=1))
    )
    p.save(OUT, dpi=300, verbose=False)
    p.save(OUT.replace(".pdf", ".png"), dpi=200, verbose=False)   # preview only
    runs = df[["task", "method", "loss"]].drop_duplicates().shape[0]
    print(f"wrote {OUT} from {SWEEP} ({runs} runs, {len(df)} curve points)")


if __name__ == "__main__":
    main()
