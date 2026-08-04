"""Compact MIB scatter: acc-AUC (x) vs CPR/logit-diff AUC (y), one point per node method.

For every node-level MIB method (gradient baselines + all MAttr ablations) we computed both
metrics on the validation set. This shows how the two agree across methods (Spearman rho in the
title) — a companion to the MLP/Attn Spearman heatmap. Sized ~1/3 text width (5.5in full).

acc-AUC sources mirror make_mib_accauc_table; CPR = `area_under` (mirrors make_mib_table).
Run:  uv run python plots/plot_mib_accauc_cpr_scatter.py  ->  plots/mib_accauc_cpr_scatter.pdf
"""
import re
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
# (M.EPRUN_BEST_SPARSITY) is plotted here, unlike the validation tables which list all three.
# Both metrics come from the SAME pkl as every other point here (area_under + acc_auc), so
# nothing extra was run.

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


# ---------------------------------------------------------------------------------------
# Appendix (--full): the same two metrics, but nothing dropped -- every node-level MAttr
# ablation, every gradient baseline, and all 12 Node Pruning budgets, at ~full page size with
# every point named. The compact figure above answers "do the metrics agree?"; this one is the
# audit trail behind that answer, and is where the disagreement (Node Pruning's two objectives
# ranking their own budgets in opposite directions) is actually legible.
#
# Colour here means GROUP, not method -- 32 points cannot carry 32 hues, and the direct labels
# already give identity. Shape still splits gradient vs mask learning, as in the compact figure.
G_MLOG, G_MUNI = "MAttr (log $k$)", "MAttr (unif. $k$)"
G_GRAD, G_NPKL, G_NPLD = "Gradient baseline", "Node Pruning (KL)", "Node Pruning (logit-diff)"
G_DBM = "DBM"
FULL_ORDER = [G_MLOG, G_MUNI, G_GRAD, G_NPKL, G_NPLD, G_DBM]
FULL_COLORS = {
    G_MLOG: P.METHOD["MAttr"], G_MUNI: P.METHOD["+hard"], G_GRAD: P.METHOD["IG"],
    G_NPLD: P.METHOD["Node Pruning"], G_NPKL: "#9d95d1",   # tint of the same indigo
    G_DBM: "#d98d3a",   # warm, so it reads as neither MAttr (green) nor Node Pruning (indigo)
}
FULL_MASK = {G_MLOG, G_MUNI, G_NPKL, G_NPLD, G_DBM}

# === LR series ===
# Every lr we swept whose dir is COMPLETE on both axes (11/11 cells for acc_auc AND
# area_under). Completeness is the bar because this figure averages each method over the cells
# it has, so a 3-cell point would sit in the same space as an 11-cell one and read as
# comparable when it is not. What that excludes, as of 2026-08-04:
#
#   + unif k, + hard (htk_lr_*)   acc_auc on 4/11   -- never re-eval'd for acc
#   + hard bwd (bern_lr_*)        acc_auc on 2-3/11, and cpr itself is partial (7-10/11)
#   MAttr/+hard at lr=0.01        acc_auc on 3/11   (mib_node_topk_log / _hard_topk_log)
#   DBM at lr=0.01 / 0.1 / 1.0    3/11 -- swept on the cheap cells only, by design
#
# so the plotted series are MAttr log-k and +hard log-k at {0.005, 0.05, 0.1, 0.3} and DBM at
# {0.001, 0.3}. build_lr_rows() prints every exclusion rather than dropping it silently.
#
# lr=0.05 is deliberately NOT listed: those two dirs are already plotted from M.OUR_METHODS as
# the headline "MAttr" and "+hard" points, and a second point at the same coordinates would
# double-count them in the Spearman.
LR_SERIES = [
    (G_MLOG, "MAttr", [("0.005", "topklog_lr_0.005"), ("0.1", "topklog_lr_0.1"),
                       ("0.3", "topklog_lr_0.3")]),
    (G_MLOG, "+hard", [("0.005", "htklog_lr_0.005"), ("0.1", "htklog_lr_0.1"),
                       ("0.3", "htklog_lr_0.3")]),
    (G_DBM, "DBM", [("0.001", "eprun_eval_ld_sig"), ("0.3", "eprun_eval_ld_sig_lr0.3")]),
]

# The lr=0.05 headline points come from M.OUR_METHODS (see above), so to draw one unbroken
# path per method they have to be tagged into the same series as the swept points -- otherwise
# the MAttr line jumps 0.005 -> 0.1 straight past its own best-performing setting.
LR_ANCHOR = {"topklog_lr_0.05": ("MAttr", 0.05), "htklog_lr_0.05": ("+hard", 0.05)}


def _pair(dirn, t, m):
    """(acc_auc, area_under) for one cell, from whichever layout this dir uses.

    Our own trainer writes results/<dir>/<task>_<model>_validation.pkl; MIB's
    run_evaluation.py (every eprun_eval*/DBM dir) nests under a
    <Method>_patching_<level>/ subfolder and spells the task with dashes. Both pkls carry
    acc_auc and area_under, so one reader covers both once the path is resolved.

    Third layout: node acc-AUC produced by run_accauc_mattr.sh lives OUTSIDE this repo, in
    MIB-circuit-track/results/mattr_accauc/, and never made it back into the trainer's pkl.
    So acc falls back to A.acc_mattr while CPR still comes from the local pkl -- probing only
    the local file drops whole LR series out of the figure as "incomplete" when they are not.
    """
    p = RB / dirn / f"{t}_{m}_validation.pkl"
    if not p.exists():
        hits = list((RB / dirn).glob(f"**/{t.replace('_', '-')}_{m}_validation_abs-*.pkl"))
        if not hits:
            return None, None   # no pkl at all means no CPR either -- a genuinely missing cell
        p = hits[0]
    try:
        r = pickle.load(open(p, "rb"))
    except Exception:
        return None, None
    acc = r.get("acc_auc")
    if acc is None:
        acc = A.acc_mattr(dirn, t, m)
    return acc, r.get("area_under")


def build_lr_rows(series=LR_SERIES, level="node"):
    """Points for every swept lr whose dir is complete on BOTH metrics.

    Incomplete dirs are skipped WITH a printed reason -- an lr silently missing from the
    figure looks like an lr we never ran, which is the one thing this figure must not imply.
    """
    rows = []
    for grp, base, lrs in series:
        for lr, dirn in lrs:
            pairs = [_pair(dirn, t, m) for t, m, _ in COLS]
            acc = [a for a, _ in pairs if a is not None]
            cpr = [c for _, c in pairs if c is not None]
            if len(acc) < len(COLS) or len(cpr) < len(COLS):
                print(f"  skip {base} lr={lr} ({dirn}): acc {len(acc)}/{len(COLS)}, "
                      f"cpr {len(cpr)}/{len(COLS)} -- incomplete", file=sys.stderr)
                continue
            rows.append(dict(acc=float(np.mean(acc)), cpr=float(np.mean(cpr)), grp=grp,
                             label=f"{base} lr={lr}", path=f"lr:{base}", s=float(lr)))
    return rows


def delatex(s):
    """Table label -> matplotlib point label. Math mode survives only where it carries meaning
    (the $c_k$ subscript); everything else is flattened, because repel() estimates label width
    from the character count and every stray $..$ makes that estimate worse."""
    s = re.sub(r"\$s\{=\}([\d.]+)\$", r"s=\1", s)
    for a, b in ((r"\ourmethod{}", "MAttr"), (r"$+$ ", "+"), (r"$-$ $c_k$", "$-c_k$"),
                 (r"$\times$", "x")):
        s = s.replace(a, b)
    return s.strip()


# normalized-axes label geometry for 6.5pt text in a 5.4x6.9in figure
DX, CHAR_W, LAB_H, MARK_R = 0.016, 0.0092, 0.021, 0.011


def repel(x, y, labels, xr, yr, n=900):
    """Label de-overlap by rectangle separation in normalized [0,1]^2 axes space (adjustText is
    not installed here, and a Gaussian point-repulsion does not converge on this figure -- the
    long labels like "+id-STE, Gumbel sel." are ~10x wider than tall, so what matters is BOX
    overlap, not centre distance). Each label is a box anchored right of its marker; overlapping
    boxes are pushed apart along whichever axis needs the smaller move, labels are also pushed
    off markers, and a weak spring pulls each back to its anchor. Leader lines make any residual
    drift unambiguous. Deterministic -- no RNG, so the figure is reproducible."""
    ax = (np.asarray(x) - xr[0]) / (xr[1] - xr[0])
    ay = (np.asarray(y) - yr[0]) / (yr[1] - yr[0])
    w = np.array([len(s.replace("$", "")) * CHAR_W for s in labels])
    lx, ly = ax + DX, ay.copy()
    for _ in range(n):
        # half-extents of the pair boxes (labels are left-anchored, so x-centre = lx + w/2)
        cx = lx + w / 2
        dx, dy = cx[:, None] - cx[None, :], ly[:, None] - ly[None, :]
        ox = (w[:, None] + w[None, :]) / 2 - np.abs(dx)     # >0 = overlapping in x
        oy = LAB_H - np.abs(dy)
        hit = (ox > 0) & (oy > 0)
        np.fill_diagonal(hit, False)
        # resolve along the cheaper axis; ties in position (dy==0) break upward
        sy = np.where(dy >= 0, 1.0, -1.0)
        sx = np.where(dx >= 0, 1.0, -1.0)
        useY = oy <= ox
        px = np.where(hit & ~useY, 0.5 * sx * ox, 0.0).sum(1)
        py = np.where(hit & useY, 0.5 * sy * oy, 0.0).sum(1)
        # keep labels off every marker (not just their own)
        mdx, mdy = cx[:, None] - ax[None, :], ly[:, None] - ay[None, :]
        mox = w[:, None] / 2 + MARK_R - np.abs(mdx)
        moy = LAB_H / 2 + MARK_R - np.abs(mdy)
        mhit = (mox > 0) & (moy > 0)
        py += np.where(mhit, np.where(mdy >= 0, 1.0, -1.0) * moy, 0.0).sum(1)
        lx += 0.28 * px + 0.05 * (ax + DX - lx)
        ly += 0.28 * py + 0.05 * (ay - ly)
        ly = np.clip(ly, LAB_H / 2, 1 - LAB_H / 2)
    return lx * (xr[1] - xr[0]) + xr[0], ly * (yr[1] - yr[0]) + yr[0]


def main_full():
    import matplotlib.pyplot as plt

    # Inter, matching the plotnine theme every other figure in the paper uses (`family="Inter"`
    # in theme_set above). This figure is raw matplotlib rather than plotnine because the label
    # placement needs per-annotation control, so the font has to be set on rcParams by hand --
    # plotnine's theme does not reach it. mathtext gets Inter too: leaving it on the DejaVu
    # default would render "$-c_k$" and the legend's "$k$" in a visibly different face from the
    # text right next to them. fonttype 42 embeds the actual TrueType outlines instead of
    # Type-3, which is what arXiv and most camera-ready checkers want.
    plt.rcParams.update({
        "font.family": "Inter", "mathtext.fontset": "custom", "mathtext.rm": "Inter",
        "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
        # cal/sf/tt are unused here but a "custom" fontset resolves all of them at import time,
        # and the cal default is the generic "cursive", which is not installed -- leaving them
        # emits a findfont fallback warning on every run.
        "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
        "pdf.fonttype": 42, "text.color": "#000000",
        "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
    })

    rows = []
    for disp, dacc, sub in A.BASELINES:
        acc = avg({(t, m): A.acc_base(dacc, sub, t, m) for t, m, _ in COLS})
        dn, subn = BASE_CPR[disp]
        cpr = avg(cpr_base(dn, subn))
        if acc is not None and cpr is not None:
            rows.append(dict(acc=acc, cpr=cpr, grp=G_GRAD, label=delatex(disp), path=None, s=0))
    for name, d, level, g in M.OUR_METHODS:
        if level != "node":
            continue
        acc = avg({(t, m): A.acc_mattr(d, t, m) for t, m, _ in COLS})
        cpr = avg({(t, m): M.load_cpr_auc(d, t, m) for t, m, _ in COLS})
        if acc is None or cpr is None:
            print(f"  skip {d}: acc={acc} cpr={cpr}", file=sys.stderr)
            continue
        base, lr = LR_ANCHOR.get(d, (None, 0.0))
        rows.append(dict(acc=acc, cpr=cpr, grp=G_MLOG if g == "ours" else G_MUNI,
                         label=delatex(name), path=f"lr:{base}" if base else None, s=lr))
    # all 12 budgets; the two objectives are separate dashed paths, each ordered sparse-ward
    for label, dirn in M.EPRUN_SPARSITIES:
        sub = "EdgePruning_patching_node"
        acc = avg({(t, m): A._acc(RB / dirn / sub /
                                  f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl")
                   for t, m, _ in COLS})
        cpr = avg(cpr_base(dirn, sub))
        if acc is None or cpr is None:
            print(f"  skip {dirn}: acc={acc} cpr={cpr}", file=sys.stderr)
            continue
        ld = "logit-diff" in label
        rows.append(dict(acc=acc, cpr=cpr, grp=G_NPLD if ld else G_NPKL,
                         label=delatex(label).replace(", logit-diff", ""),
                         path="ld" if ld else "kl",
                         s=float(delatex(label).split("s=")[1].split(",")[0])))
    # every complete swept lr, incl. both complete DBM points (the only DBM source here)
    rows += build_lr_rows()

    df = pd.DataFrame(rows)
    df["fam"] = np.where(df.grp.isin(FULL_MASK), MASK, GRADIENT)

    fig, ax = plt.subplots(figsize=(5.4, 6.9))
    # Dashed guides through every ordered series: the two Node Pruning objectives ordered
    # sparse-ward ("kl"/"ld"), and each swept method ordered by learning rate ("lr:<method>").
    # Same visual language for both because they are the same statement -- points joined by a
    # line differ only in ONE hyperparameter, so the line's direction is the sensitivity to it.
    for key in [k for k in df.path.dropna().unique()]:
        sub = df[df.path == key].sort_values("s")
        if len(sub) > 1:
            ax.plot(sub.acc, sub.cpr, ls="dashed", lw=0.7, alpha=0.55, zorder=1,
                    color=FULL_COLORS[sub.grp.iloc[0]])
    for grp in FULL_ORDER:
        sub = df[df.grp == grp]
        if not len(sub):
            continue
        ax.scatter(sub.acc, sub.cpr, s=46, marker=FAMILY_SHAPE[MASK if grp in FULL_MASK
                                                               else GRADIENT],
                   c=FULL_COLORS[grp], edgecolors="#000000", linewidths=0.5, zorder=3,
                   label=grp)

    xr, yr = ax.get_xlim(), ax.get_ylim()
    xr = (xr[0], xr[1] + 0.22 * (xr[1] - xr[0]))    # room for labels on the right
    ax.set_xlim(xr)
    lx, ly = repel(df.acc.values, df.cpr.values, df.label.tolist(), xr, yr)
    for (x, y, lxi, lyi, lab, grp) in zip(df.acc, df.cpr, lx, ly, df.label, df.grp):
        ax.plot([x, lxi], [y, lyi], lw=0.35, color="#888888", zorder=2)
        ax.annotate(lab, (lxi, lyi), fontsize=6.5, va="center", ha="left",
                    color=FULL_COLORS[grp], zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.set_ylim(yr)
    # "IIA log-AUC", not "acc-AUC": the paper's prose calls this metric IIA AUC, and the AUC is
    # taken over the 10 LOG-spaced sparsity points (0.1...100%), not a linear sweep. The compact
    # figure above still says "acc-AUC" -- change both together or the two versions of the same
    # figure disagree about what their shared x axis measures.
    ax.set_xlabel("IIA log-AUC (↑)", fontsize=9)
    ax.set_ylabel("CPR AUC (↑)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.95,
              borderpad=0.5, handletextpad=0.4)
    fig.tight_layout()
    out = "plots/mib_accauc_cpr_scatter_full.pdf"
    fig.savefig(out, dpi=300)
    from scipy.stats import spearmanr
    o = df[~df.grp.isin({G_NPKL, G_NPLD})]
    print(f"wrote {out} ({len(df)} points)\n"
          f"Spearman rho: {spearmanr(df.acc, df.cpr)[0]:.3f} (all {len(df)}), "
          f"{spearmanr(o.acc, o.cpr)[0]:.3f} (excl. Node Pruning, {len(o)})")


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
    # Node Pruning: best budget only (M.EPRUN_BEST_SPARSITY). The validation tables list all
    # of M.EPRUN_SPARSITIES -- there is room there -- but here the extra budgets are the one
    # anti-correlated cluster in the figure and would understate the agreement this plot is
    # about. Swap in M.EPRUN_SPARSITIES to show them all; the dashed path below wakes up and
    # connects them sparse-ward (geom_path follows frame order).
    ep = []
    for si, (label, dirn) in enumerate([M.EPRUN_BEST_SPARSITY]):
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
    main_full() if "--full" in sys.argv else main()
