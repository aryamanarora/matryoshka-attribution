"""accauc_vs_faithauc with a second facet dimension: the iso vs cause intervention direction.

Same figure as plot_accauc_vs_faithauc.py (which it imports, so the method set, colours,
losses and task-group averaging cannot drift), with rows = intervention direction:

  iso   -- top-k kept CLEAN, complement patched (denoising; the repo's `sufficient` mode).
           x = acc_auc          = AUC of iso_metrics["acc_base"]     (base label wins, ↑)
           y = faith_auc        = AUC of iso_metrics["faithfulness"] (↑)
  cause -- top-k PATCHED with the source activations, complement clean (noising).
           x = cause_accsrc_auc = AUC of cause_metrics["acc_source"] (source label wins, ↑)
           y = cause_auc        = AUC of cause_metrics["faithfulness"]

BEWARE THE ARROWS: they do not agree across rows. eval_sva.py normalises faithfulness the
same way in both directions -- (logit_diff − F_patch) / (F_clean − F_patch), with F_clean /
F_patch always measured in the iso direction -- so in the cause row faithfulness starts at 1
(k=0 corrupts nothing) and FALLS as k grows. A good ranking therefore MINIMISES cause_auc:
the y axis is ↑-better on the top row and ↓-better on the bottom row. cause_accsrc_auc is
↑-better in both. Values are plotted raw (no 1−x flip) so they match the json fields; y is
free per row because the two rows are different quantities on different ranges.

`sufficient` inside eval_sva.eval_metrics is the RAW hooker flag, whose sense is the opposite
of the repo-level convention in CLAUDE.md: iso passes sufficient=False, cause passes True.
Don't map these row labels onto `--mode sufficient` without re-reading that note.

Data: results/sva_sweep (−input) + results/sva_sweep_input (+input); all 720 jsons carry the
cause fields, so nothing here needs re-running.
Run:  uv run python plots/plot_accauc_vs_faithauc_cause.py
        -> plots/accauc_vs_faithauc_cause.pdf
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_color_manual, scale_shape_manual,
    guides, guide_legend, expand_limits,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accauc_vs_faithauc as R   # noqa: E402  (method set / colours / averaging)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(6.5, 3.4),   # full text width, 2 rows; appendix figure
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

OUT = "plots/accauc_vs_faithauc_cause.pdf"


def auc_of(ys, ks):
    """eval_sva.py's auc_of: trapezoid over log10(k), normalised by the log-k span."""
    lx = np.log10(np.asarray(ks, float)); ya = np.asarray(ys, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


# row label -> (x, y) getter. Both rows use acc_base on x and faithfulness on y, i.e. the SAME
# two quantities measured under the two interventions -- so the axes mean the same thing in
# both rows and only the arrow flips (iso ↑↑, cause ↓↓). cause acc_base has no precomputed
# scalar in the json (only cause_accsrc_auc, its complement), so integrate the curve here.
DIRECTIONS = [
    ("iso: keep top-$k$", lambda d: (d["acc_auc"], d["faith_auc"])),
    ("cause: patch top-$k$",
     lambda d: (auc_of(d["cause_metrics"]["acc_base"], d["n_nodes"]), d["cause_auc"])),
]


def load(res):
    """(method, loss, substrate, task) -> full json dict, for the methods the parent plots."""
    raw = {}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        m = R.parse_method(os.path.basename(f), d)
        if m is None or m not in R.METHODS:
            continue
        # results/sva_sweep also holds a llama3 IOI wave; this key has no model in it, so without
        # the gate the two files for a cell collide and glob order picks the model. See R.on_model.
        if not R.on_model(d):
            continue
        raw[(m, d["loss"], d["nodes"], d["task"])] = d
    return raw


def group_avg(raw, m, loss, sub, get):
    """R.group_avg for an arbitrary (x, y) getter; None if no group has data."""
    gx, gy = [], []
    for tasks in [R.SVA, ["arc_easy"], ["ioi"]]:
        vs = [get(raw[(m, loss, sub, t)]) for t in tasks if (m, loss, sub, t) in raw]
        vs = [v for v in vs if v[0] is not None and v[1] is not None]
        if vs:
            gx.append(np.mean([v[0] for v in vs]))
            gy.append(np.mean([v[1] for v in vs]))
    if not gx:
        return None
    return float(np.mean(gx)), float(np.mean(gy))


def main():
    rows = []
    # R.SWEEPS was renamed SOURCES and gained a third field when the parent grew its zero-ablation
    # row. This figure is the PATCHED sweep only (see the docstring), so the zero dirs are skipped
    # rather than silently folded in as extra facets.
    for res, inp_label, abl in R.SOURCES:
        if abl != "Patched":
            continue
        raw = load(res)
        for dir_label, get in DIRECTIONS:
            for m, (mlabel, _) in R.METHODS.items():
                for lkey, llabel in R.LOSSES.items():
                    for sub, slabel in R.SUBSTRATES:
                        r = group_avg(raw, m, lkey, sub, get)
                        if r is None:
                            continue
                        rows.append(dict(x=r[0], y=r[1], method=mlabel, loss=llabel,
                                         facet=f"{slabel}, {inp_label}", direction=dir_label))
    df = pd.DataFrame(rows)

    df["method"] = pd.Categorical(df["method"], [v[0] for v in R.METHODS.values()])
    df["loss"] = pd.Categorical(df["loss"], list(R.LOSSES.values()))
    df["direction"] = pd.Categorical(df["direction"], [lab for lab, _ in DIRECTIONS])
    facet_order = ["Node, −input", "Node, +input", "MLP, −input", "MLP+Attn, −input"]
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in set(df["facet"])])

    p = (
        ggplot(df, aes("x", "y", color="method", shape="loss"))
        + geom_point(size=2.6, alpha=0.85, stroke=0.3)
        # x shared (both rows are the base-label accuracy AUC, in [0,1]); y free per row
        # because iso faithfulness is unbounded above while cause faithfulness lives near 0.
        + facet_grid("direction ~ facet", scales="free_y")
        + expand_limits(x=0, y=0)   # anchor at 0 without dropping out-of-range points
        + scale_color_manual(values={lab: col for lab, col in R.METHODS.values()}, name="Method")
        + scale_shape_manual(values=R.LOSS_SHAPE, name="Loss")
        # Same two quantities in both rows, so "good" is top-right on the iso row and
        # bottom-left on the cause row (patching a good top-k should destroy both).
        + labs(x="Accuracy AUC, base label (iso ↑ / cause ↓)",
               y="Faith AUC (iso ↑ / cause ↓)")
        + guides(color=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    p.save(OUT, dpi=300, verbose=False)
    p.save(OUT.replace(".pdf", ".png"), dpi=200, verbose=False)   # preview only
    print(f"wrote {OUT} ({len(df)} points)")
    print(df.groupby(["direction", "facet"], observed=True).size().to_string())
    print(df.groupby(["direction", "method"], observed=True)
          .agg(acc=("x", "mean"), faith=("y", "mean")).round(3).to_string())


if __name__ == "__main__":
    main()
