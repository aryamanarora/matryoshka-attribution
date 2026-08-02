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
import palette as P
from plotnine import (
    ggplot, aes, geom_point, geom_path, labs, theme, theme_set, theme_bw, element_text,
    element_line, element_blank, scale_fill_manual, scale_shape_manual, expand_limits,
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
        legend_box="vertical",   # method legend above the gradient/mask shape legend
        legend_title=element_blank(),
        legend_text=element_text(size=5.5),
        legend_key_size=7,
        legend_box_margin=0,
        legend_margin=0,
    )
)

# Colour = method; grey "Other" for the un-highlighted gradient baselines. Colours from
# plots/palette.py (single source of truth across all figures) -- no local hex codes.
# MAttr headline = soft top-k fwd (log k); "+hard" = sigmoid-STE hard forward ablation.
COLORS = {**P.METHOD, "Other": P.OTHER}
COLOR_ORDER = ["MAttr", "+hard", "IG", "I×G", "Node Pruning", "Other"]

# Node Pruning is the only mask-learning baseline that covers all 11 cells, so it is the closest
# comparator to MAttr and gets its own colour rather than the grey "Other". Only its best budget
# (s=0.9) is plotted -- see M.EPRUN_SPARSITIES; widen that list and the extra points reappear,
# joined by the dashed path below. Both metrics come from the SAME pkl as every other point here
# (area_under + acc_auc), so nothing extra was run.

# the two MAttr methods we keep (drop all other MAttr ablations); IG/I×G among the baselines
HL_DIR = {"topklog_lr_0.05": "MAttr", "htklog_lr_0.05": "+hard"}
HL_BASE = {"NAP-IG": "IG", "I$\\times$G": "I×G"}

# Shape = how the circuit is OBTAINED, which is the axis this figure is really about: score
# every node with a gradient and rank, vs optimize a mask against an objective. Note this cuts
# ACROSS ours/baseline -- MAttr, +hard and Node Pruning share a shape, and the split is what
# makes the upper-right cluster read as "mask learning wins acc-AUC" rather than "ours wins".
GRADIENT, MASK = "Gradient", "Mask learning"
FAMILY_SHAPE = {GRADIENT: "o", MASK: "s"}   # both fillable: black edge + method fill
MASK_METHODS = {"MAttr", "+hard", "Node Pruning"}


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
    # Node Pruning: one point per target sparsity in M.EPRUN_SPARSITIES (just s=0.9 as
    # shipped), both metrics out of the same pkl. The list is ordered sparse-ward, which is
    # the order the dashed path below connects (geom_path follows frame order).
    ep = []
    for si, (label, dirn) in enumerate(M.EPRUN_SPARSITIES):
        sub = "EdgePruning_patching_node"
        acc = avg({(t, m): A._acc(RB / dirn / sub /
                                  f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
                   for t, m, _ in COLS})
        cpr = avg(cpr_base(dirn, sub))
        if acc is not None and cpr is not None:
            ep.append(dict(acc=acc, cpr=cpr, method="Node Pruning", s=si))
    rows += ep

    df = pd.DataFrame(rows)
    df["family"] = np.where(df.method.isin(MASK_METHODS), MASK, GRADIENT)
    df["method"] = pd.Categorical(df["method"], COLOR_ORDER)
    df["family"] = pd.Categorical(df["family"], [GRADIENT, MASK])
    # draw grey "Other" first so the highlighted points sit on top
    df = df.sort_values("method", ascending=False, key=lambda s: s.cat.codes)
    epdf = pd.DataFrame(ep).sort_values("s") if ep else None

    p = ggplot(df, aes("acc", "cpr", fill="method", shape="family"))
    if epdf is not None and len(epdf) > 1:
        # dashed guide across the sparsity budgets (drawn only if >1 is plotted), same
        # visual language as the loss
        # guide in accauc_vs_faithauc. Added BEFORE geom_point so the markers sit on top of
        # it; adds no legend entry (constant colour, inherit_aes=False).
        p += geom_path(epdf, aes("acc", "cpr"), color=COLORS["Node Pruning"],
                       linetype="dashed", size=0.3, alpha=0.6, inherit_aes=False)
    p = (
        p
        # Method on FILL with a black edge (matching accauc_vs_faithauc): shape is now spoken
        # for by the family split, and an edge keeps the crowded 0.28-0.35 baseline cluster
        # readable at this size. alpha=1 -- translucent fill under a black edge muddies the
        # colour exactly where points overlap.
        + geom_point(size=1.9, color="#000000", stroke=0.3)
        + expand_limits(x=0, y=0)
        + scale_fill_manual(values=COLORS, name="")
        + scale_shape_manual(values=FAMILY_SHAPE, name="")
        + labs(x="acc-AUC (↑)", y="CPR AUC (↑)")
        # nrow=3 (2 columns), not 2: at 1.65in wide a 3-column legend clips "Node Pruning".
        # The two legends stack (legend_box="vertical" in the theme).
        + guides(fill=guide_legend(order=1, nrow=3, override_aes={"shape": "o"}),
                 shape=guide_legend(order=2, nrow=1))
    )
    out = "plots/mib_accauc_cpr_scatter.pdf"
    p.save(out, dpi=300, verbose=False)
    print(f"wrote {out} ({len(df)} points, {df['method'].nunique()} series; "
          f"Node Pruning at {len(ep)} sparsities)")
    # The figure's caption quotes this rho, so print it rather than leaving it hand-maintained
    # -- it drifts with every re-eval, and Node Pruning pulls it down (its two metrics rank
    # its own sparsity budgets in OPPOSITE directions, so extra budgets cost more than one).
    from scipy.stats import spearmanr
    r_all = spearmanr(df.acc, df.cpr)[0]
    o = df[df.method != "Node Pruning"]
    print(f"Spearman rho: {r_all:.3f} (all {len(df)}), "
          f"{spearmanr(o.acc, o.cpr)[0]:.3f} (excl. Node Pruning, {len(o)})")


if __name__ == "__main__":
    main()
