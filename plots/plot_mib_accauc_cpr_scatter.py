"""Compact MIB scatter: acc-AUC (x) vs CPR/logit-diff AUC (y), one point per node method.

For every node-level MIB method (gradient baselines + all MAttr ablations) we computed both
metrics on the validation set. This shows how the two agree across methods (Spearman rho in the
title) — a companion to the MLP/Attn Spearman heatmap. Sized ~1/3 text width (5.5in full).

acc-AUC sources mirror make_mib_accauc_table; CPR = `area_under` (mirrors make_mib_table).
Run:  uv run python plots/plot_mib_accauc_cpr_scatter.py  ->  plots/mib_accauc_cpr_scatter.pdf
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, labs, theme, theme_set, theme_bw, element_text,
    element_line, element_blank, scale_color_manual, expand_limits,
    guides, guide_legend,
)

sys.path.insert(0, "scripts")
import make_mib_table as M            # noqa: E402  COLUMNS, OUR_METHODS, load_cpr_auc
import make_mib_accauc_table as A     # noqa: E402  acc_mattr / acc_base / BASELINES

COLS = M.COLUMNS
RB = Path("results")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(1.65, 2.0),   # display size at 0.30*textwidth (5.5in); matches heatmap fonts
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        plot_title=element_text(size=7, ha="center"),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_position="bottom",
        legend_direction="horizontal",
        legend_title=element_blank(),
        legend_text=element_text(size=5.5),
        legend_key_size=7,
        legend_box_margin=0,
        legend_margin=0,
    )
)

# Colour = method; grey "Other" for the un-highlighted gradient baselines.
# colours matched to accauc_vs_faithauc.pdf: MAttr=blue, +hard=green, IG=brown, I×G=pink
# MAttr headline = soft top-k fwd (log k); "+hard" = sigmoid-STE hard forward ablation.
COLORS = {"MAttr": "#1f77b4", "+hard": "#2ca02c",
          "IG": "#8c564b", "I×G": "#e377c2", "Other": "#cccccc"}
COLOR_ORDER = ["MAttr", "+hard", "IG", "I×G", "Other"]

# the two MAttr methods we keep (drop all other MAttr ablations); IG/I×G among the baselines
HL_DIR = {"topklog_lr_0.05": "MAttr", "htklog_lr_0.05": "+hard"}
HL_BASE = {"NAP-IG": "IG", "I$\\times$G": "I×G"}


def avg(d):
    vs = [v for v in d.values() if v is not None]
    return float(np.mean(vs)) if vs else None


def cpr_base(dirn, sub):
    out = {}
    for t, m, _ in COLS:
        p = RB / dirn / sub / f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl"
        if p.exists():
            try:
                out[(t, m)] = pickle.load(open(p, "rb"))["area_under"]
            except Exception:
                pass
    return out


# gradient baseline CPR dirs (mirror make_mib_table.EXTRA_NODE_BASELINES + NAP-IG repro)
BASE_CPR = {
    "NAP-IG": ("napig_repro_eval", "EAP-IG-inputs_patching_node"),
    "Conductance": ("napig_local_eval", "EAP-IG-inputs-local_patching_node"),
    "I$\\times$G": ("ig1_eval", "EAP-IG-inputs_patching_node"),
    "RelP": ("relp_eval", "RelP_patching_node"),
    "RelP+QK": ("relp_qkgrad_eval", "RelP-qkgrad_patching_node"),
    "AttnRLP": ("attnrlp_eval", "AttnRLP_patching_node"),
    "GIM": ("gim_eval", "GIM_patching_node"),
}


def main():
    rows = []
    # gradient baselines (all kept; IG / I×G highlighted, rest grey)
    for disp, dacc, sub in A.BASELINES:
        acc = avg({(t, m): A.acc_base(dacc, sub, t, m) for t, m, _ in COLS})
        dn, subn = BASE_CPR[disp]
        cpr = avg(cpr_base(dn, subn))
        if acc is not None and cpr is not None:
            rows.append(dict(acc=acc, cpr=cpr, method=HL_BASE.get(disp, "Other")))
    # keep ONLY the two headline MAttr methods (drop all other MAttr ablations)
    for n, d, l, g in M.OUR_METHODS:
        if l != "node" or d not in HL_DIR:
            continue
        acc = avg({(t, m): A.acc_mattr(d, t, m) for t, m, _ in COLS})
        cpr = avg({(t, m): M.load_cpr_auc(d, t, m) for t, m, _ in COLS})
        if acc is not None and cpr is not None:
            rows.append(dict(acc=acc, cpr=cpr, method=HL_DIR[d]))

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], COLOR_ORDER)
    # draw grey "Other" first so the highlighted points sit on top
    df = df.sort_values("method", ascending=False, key=lambda s: s.cat.codes)

    p = (
        ggplot(df, aes("acc", "cpr", color="method"))
        + geom_point(size=2.6, alpha=0.9, stroke=0.4)
        + expand_limits(x=0, y=0)
        + scale_color_manual(values=COLORS, name="")
        + labs(x="acc-AUC (↑)", y="CPR AUC (↑)")
        + guides(color=guide_legend(nrow=2))
    )
    out = "plots/mib_accauc_cpr_scatter.pdf"
    p.save(out, dpi=300, verbose=False)
    print(f"wrote {out} ({len(df)} methods)")


if __name__ == "__main__":
    main()
