"""Compact MIB scatter: acc-AUC (x) vs CPR/logit-diff AUC (y), one point per node method.

For every node-level MIB method (gradient baselines + all MAttr ablations) we computed both
metrics on the validation set. This shows how the two agree across methods (Spearman rho is
printed, not drawn — the caption quotes it) — a companion to the MLP/Attn Spearman heatmap.

Three variants of one plot, all raw matplotlib with direct point labels:
  (no flag)  main text, 0.30\textwidth, eight curated points   -> mib_accauc_cpr_scatter.pdf
  --full     appendix, full page, every node point             -> ..._full.pdf
  --edge     appendix, full page, every edge point             -> ..._edge_full.pdf
  --both     appendix, full page, node over edge in one float  -> ..._both.pdf

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

sys.path.insert(0, "scripts")
import make_mib_table as M            # noqa: E402  COLUMNS, OUR_METHODS, load_cpr_auc
import make_mib_accauc_table as A     # noqa: E402  acc_mattr / acc_base / BASELINES

COLS = M.COLUMNS
RB = Path("results")

# Every figure here is raw matplotlib (see RC below): all three variants are the same plot at
# different sizes and point counts, and the direct labelling needs per-annotation control that
# plotnine does not expose. Colours still come from plots/palette.py, the single source of
# truth across the paper's figures -- no local hex codes except the two tints noted below.

# Shape = how the circuit is OBTAINED, which is the axis this figure is really about: score
# every node with a gradient and rank, vs optimize a mask against an objective. Note this cuts
# ACROSS ours/baseline -- MAttr, +hard and Node Pruning share a shape, and the split is what
# makes the upper-right cluster read as "mask learning wins acc-AUC" rather than "ours wins".
GRADIENT, MASK = "Gradient", "Mask learning"
FAMILY_SHAPE = {GRADIENT: "o", MASK: "s"}   # both fillable: black edge + method fill


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
    "RelP+Shapley": ("relpshapley_eval", "RelPShapley_patching_node"),
    "AttnLRP": ("attnlrp_eval", "AttnLRP_patching_node"),
    "GIM": ("gim_eval", "GIM_patching_node"),
}


# ---------------------------------------------------------------------------------------
# Appendix (--full): the same two metrics, but nothing dropped -- every node-level MAttr
# ablation, every gradient baseline, and all 12 Node Pruning budgets, at ~full page size with
# every point named. The compact figure above answers "do the metrics agree?"; this one is the
# audit trail behind that answer, and is where the disagreement (Node Pruning's two objectives
# ranking their own budgets in opposite directions) is actually legible.
#
# Colour here means GROUP, not method -- ~50 points cannot carry ~50 hues, and the direct labels
# already give identity. Shape still splits gradient vs mask learning, as in the compact figure.
G_MLOG, G_MUNI = "MAttr (log $k$)", "MAttr (unif. $k$)"
G_GRAD, G_NPKL, G_NPLD = "Gradient baseline", "Node Pruning (KL)", "Node Pruning (logit-diff)"
G_DBM = "DBM"
FULL_ORDER = [G_MLOG, G_MUNI, G_GRAD, G_NPKL, G_NPLD, G_DBM]
FULL_COLORS = {
    G_MLOG: P.METHOD["MAttr"], G_MUNI: P.METHOD["+hard"], G_GRAD: P.METHOD["IG"],
    G_NPLD: P.METHOD["Node Pruning"], G_NPKL: "#9d95d1",   # tint of the same indigo
    # Wong reddish purple. It was a warm #d98d3a, which is a near-twin of the gradient
    # baselines' Wong orange (#e69f00) -- survivable at 50 labelled points, not in the
    # eight-point main-text cut, where DBM sits four points from RelP+QK in the same hue and
    # only the marker SHAPE says they are different families. Purple keeps it in the
    # mask-learning family with Node Pruning's indigo while staying well clear of it in
    # lightness (L* ~60 vs ~24).
    G_DBM: "#cc79a7",
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
    # Node Pruning was swept on LR too (submit_node_pruning_lr.sh), at its two best logit-diff
    # budgets. Without these the indigo cloud is a pure SPARSITY sweep, which quietly credits
    # the baseline's single default LR (0.8, the hard-concrete default) with being a good one.
    # lr=1.5 (8/11) and lr=3.0 (3/11) are still filling and drop out on the completeness bar.
    (G_NPLD, "NP s=0.5", [("0.1", "eprun_eval_s0.5_ld_lr0.1"), ("0.3", "eprun_eval_s0.5_ld_lr0.3"),
                          ("1.5", "eprun_eval_s0.5_ld_lr1.5"), ("3.0", "eprun_eval_s0.5_ld_lr3.0")]),
    (G_NPLD, "NP s=0.8", [("0.1", "eprun_eval_s0.8_ld_lr0.1"), ("0.3", "eprun_eval_s0.8_ld_lr0.3"),
                          ("1.5", "eprun_eval_s0.8_ld_lr1.5"), ("3.0", "eprun_eval_s0.8_ld_lr3.0")]),
]

# === DBM sparsity-penalty series ===
# The DBM points above are trained with NO sparsity term, which is why they sit in a narrow
# density band whatever the lr; submit_dbm_l1.sh adds the L1 that pyvene's own tutorial (and
# Boundless DAS) trains this mask with. Plotted as a second dashed path off the same lr=0.3
# point, so the figure separates the two knobs: the "lr:DBM" path is "tune the optimiser", the
# "l1:DBM" path is "give the baseline a sparsity objective at its best lr".
#
# Same completeness bar as everything else here. All six lambdas are 11/11 as of 2026-08-07, so
# the whole path is drawn -- including 6.0, which the 2026-08-06 render excluded at 6/11 and
# which is now the headline DBM row in both MIB tables.
# build_lr_rows prints each exclusion, so a lambda missing from the figure is never silent.
L1_SERIES = [
    (G_DBM, "DBM", [("0.2", "eprun_eval_ld_sig_lr0.3_l10.2"),
                    ("0.6", "eprun_eval_ld_sig_lr0.3_l10.6"),
                    ("2.0", "eprun_eval_ld_sig_lr0.3_l12.0"),
                    ("6.0", "eprun_eval_ld_sig_lr0.3_l16.0"),
                    ("20.0", "eprun_eval_ld_sig_lr0.3_l120.0")]),
]

# lambda=0 IS the unpenalised lr=0.3 run, already plotted by the LR series -- the same trick as
# EPRUN_LR_ANCHOR below. Without this the L1 path floats free of the point it departs from and
# the figure cannot show whether the penalty helped relative to no penalty.
DBM_L1_ANCHOR = {"eprun_eval_ld_sig_lr0.3": ("l1:DBM", 0.0)}

# The lr=0.05 headline points come from M.OUR_METHODS (see above), so to draw one unbroken
# path per method they have to be tagged into the same series as the swept points -- otherwise
# the MAttr line jumps 0.005 -> 0.1 straight past its own best-performing setting.
LR_ANCHOR = {"topklog_lr_0.05": ("MAttr", 0.05), "htklog_lr_0.05": ("+hard", 0.05)}
# Same trick for Node Pruning, except the anchor is a point that ALREADY sits on another path:
# eprun_eval_s0.5_ld is the s=0.5 node of the sparsity path AND the lr=0.8 node of its own LR
# path. That is why rows carry a list of (path, sort-key) pairs rather than one of each.
EPRUN_LR_ANCHOR = {"eprun_eval_s0.5_ld": ("NP s=0.5", 0.8),
                   "eprun_eval_s0.8_ld": ("NP s=0.8", 0.8)}


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


def build_lr_rows(series=LR_SERIES, level="node", key="lr", knob="lr"):
    """Points for every swept value of one hyperparameter whose dir is complete on BOTH metrics.

    `key` names the dashed path the points join ("lr" or "l1"); `knob` is what the label says.
    They are separate arguments only because a point can sit on more than one path, and the path
    key is what identifies it there.

    Incomplete dirs are skipped WITH a printed reason -- a swept value silently missing from the
    figure looks like one we never ran, which is the one thing this figure must not imply.
    """
    rows = []
    for grp, base, vals in series:
        for v, dirn in vals:
            pairs = [_pair(dirn, t, m) for t, m, _ in COLS]
            acc = [a for a, _ in pairs if a is not None]
            cpr = [c for _, c in pairs if c is not None]
            if len(acc) < len(COLS) or len(cpr) < len(COLS):
                print(f"  skip {base} {knob}={v} ({dirn}): acc {len(acc)}/{len(COLS)}, "
                      f"cpr {len(cpr)}/{len(COLS)} -- incomplete", file=sys.stderr)
                continue
            paths = [(f"{key}:{base}", float(v))]
            if dirn in DBM_L1_ANCHOR:      # unpenalised lr=0.3 run = the L1 path's lambda=0 node
                paths.append(DBM_L1_ANCHOR[dirn])
            rows.append(dict(acc=float(np.mean(acc)), cpr=float(np.mean(cpr)), grp=grp,
                             label=f"{base} {knob}={v}", paths=paths))
    return rows


def delatex(s):
    """Table label -> matplotlib point label. Math mode survives only where it carries meaning
    (the $c_k$ subscript and the "unif $k$" prefix); everything else is flattened, because a
    stray $..$ buys nothing and mathtext sets a different face from the surrounding label."""
    s = re.sub(r"\$s\{=\}([\d.]+)\$", r"s=\1", s)
    for a, b in ((r"\ourmethod{}", "MAttr"), (r"$+$ ", "+"), (r"$-$ $c_k$", "$-c_k$"),
                 (r"$\times$", "x")):
        s = s.replace(a, b)
    return s.strip()


# Figure size. Both full-page variants go in at width=\\linewidth (~5.5in), so the width is
# fixed and the HEIGHT is the only free parameter -- and height is what buys label room, since
# the binding constraint at ~50 points is vertical crowding, not horizontal (measured: widening
# XPAD from 0.34 to 0.60 changes the residual overlap count by one, raising FIG_H from 6.9 to
# 8.4 removes a third of them). 8.4in renders at ~8.5in after the width scale-up, which still
# leaves room for the caption inside ICLR's ~9in text height.
FIG_W, FIG_H = 5.4, 8.4
# --both stacks both levels in one float, so the two panels have to share one page. The node
# panel carries 49 labelled points against edge's 9, hence height_ratios=[2, 1]; 8.5in total
# leaves the node panel ~5.5in, i.e. LESS room than the standalone 8.4in figure, which is why
# the ladder pass in place_labels matters more here than it does for --full.
# This is a HARD ceiling, not a preference. ICLR's \\textwidth is 5.5in and \\textheight is
# 9.0in; the float goes in at width=\\linewidth, so LaTeX scales it by 5.5/FIG_W (= 1.019) and
# the height rides along -- every 0.1in here is ~7.3pt on the page. 8.6in overfull'd by 3.6pt.
# 8.5in fits, but only just: it leaves 0.34in = ~25pt for \\abovecaptionskip plus the caption,
# i.e. ONE line of caption and nothing more, which is a trap for a figure that needs to explain
# six series. 8.2in scales to 8.35in and leaves ~47pt, enough for a three-line caption.
FIG_H_BOTH = 8.2

# normalized-axes label geometry. Widths are MEASURED, not estimated (see label_boxes) -- the
# old len(label)*CHAR_W estimate ran 15-30% narrow at 6.5pt Inter, so repel() would report a
# clean layout while "MAttr" and "+id-STE" visibly sat on top of each other in the PDF.
LAB_PT = 6.5
# Fallbacks only. The live values are measured off the rendered panel in place_labels(), since
# both are axes FRACTIONS of a physical marker: 0.011 is the right half-extent for a 46pt^2
# marker on a ~4.9in-wide panel and three times too small on the 1.65in main-text one, where it
# let every label sit on top of its own marker.
DX, MARK_R = 0.016, 0.011
# Blank space added to the right of the data as a fraction of the x range, for the labels.
XPAD = 0.34


def label_boxes(ax, labels, fig, pt=LAB_PT):
    """(widths, height) of each rendered label in axes-fraction units.

    Draws each annotation with the exact fontsize/bbox the real call uses, measures it through
    the renderer, and removes it. Costs one throwaway draw of ~50 short strings and removes the
    only calibration constant in this file: nothing here has to be re-tuned when the font, the
    figure size or a label's text changes.
    """
    r = fig.canvas.get_renderer()
    axb = ax.get_window_extent(renderer=r)
    ws, hs = [], []
    for lab in labels:
        t = ax.annotate(lab, (0.5, 0.5), fontsize=pt, va="center", ha="left",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        bb = t.get_window_extent(renderer=r)
        ws.append(bb.width / axb.width)
        hs.append(bb.height / axb.height)
        t.remove()
    return np.array(ws), float(max(hs))


def repel(x, y, w, lab_h, xr, yr, n=900, anchor_dx=DX, mark_r=MARK_R):
    """Label de-overlap by rectangle separation in normalized [0,1]^2 axes space (adjustText is
    not installed here, and a Gaussian point-repulsion does not converge on this figure -- the
    long labels like "+id-STE, Gumbel sel." are ~10x wider than tall, so what matters is BOX
    overlap, not centre distance). Each label is a box anchored right of its marker; overlapping
    boxes are pushed apart along whichever axis needs the smaller move, labels are also pushed
    off markers, and a weak spring pulls each back to its anchor. Leader lines make any residual
    drift unambiguous. Deterministic -- no RNG, so the figure is reproducible.

    `w` are the measured label widths and `lab_h` the label height, both axes-fraction
    (label_boxes). `anchor_dx` is how far right of a marker its label is anchored and `mark_r`
    the marker half-extent to keep labels clear of, both also axes-fraction and therefore both
    dependent on the panel's physical size -- place_labels() measures them per figure rather
    than passing the full-page constants down to the 1.65in main-text panel, where a marker is
    three times bigger relative to the axes."""
    ax = (np.asarray(x) - xr[0]) / (xr[1] - xr[0])
    ay = (np.asarray(y) - yr[0]) / (yr[1] - yr[0])
    LAB_H = lab_h
    lx, ly = ax + anchor_dx, ay.copy()
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
        mox = w[:, None] / 2 + mark_r - np.abs(mdx)
        moy = LAB_H / 2 + mark_r - np.abs(mdy)
        mhit = (mox > 0) & (moy > 0)
        py += np.where(mhit, np.where(mdy >= 0, 1.0, -1.0) * moy, 0.0).sum(1)
        lx += 0.28 * px + 0.05 * (ax + anchor_dx - lx)
        ly += 0.28 * py + 0.05 * (ay - ly)
        ly = np.clip(ly, LAB_H / 2, 1 - LAB_H / 2)

    # Greedy sweep to finish the job. The simultaneous update above stalls at a handful of
    # residual overlaps no matter how long it runs (measured: identical at n=900, 3000 and 8000)
    # because a label sandwiched between two others receives equal and opposite pushes that
    # cancel exactly. Placing labels one at a time, bottom-up, cannot hit that symmetry: each
    # label only ever moves against boxes already fixed. It is what takes the count to 0.
    # Markers are obstacles too, and fixed ones -- a label that clears every other label but
    # sits on a marker is just as unreadable, and that was the failure left in the dense lr=0.05
    # cluster. Each label scans a ladder of offsets around where the relaxation left it and takes
    # the first slot that clears BOTH sets of obstacles, rather than being nudged off whichever
    # one it currently touches: nudging can cycle (clear the label, land on a marker, clear the
    # marker, land back on the label), and did, on the edge figure's tight upper-right cluster.
    # The ladder scans candidate offsets, so it has to respect the same panel bounds the final
    # clip enforces -- otherwise it happily "resolves" a label to y=1.08, the clip drags it back
    # to 1-LAB_H/2, and it lands right back on the neighbour it was supposed to clear. That is
    # invisible in a tall panel (few labels ever reach the edge) and dominates a short one: it is
    # what left 6 of the edge panel's 9 labels stacked in the top-right corner under --both.
    LO, HI = LAB_H / 2, 1 - LAB_H / 2
    sep = LAB_H / 2 + mark_r
    ladder = [0.0] + [s * d * LAB_H * 0.6 for s in range(1, 40) for d in (1, -1)]
    order = np.argsort(ly)
    placed = []
    for i in order:
        y0, ci = min(max(ly[i], LO), HI), lx[i] + w[i] / 2
        best, best_cost = y0, None
        for off in ladder:
            cand = y0 + off
            if cand < LO or cand > HI:
                continue
            cost = sum(1 for j in placed
                       if abs(ci - (lx[j] + w[j] / 2)) < (w[i] + w[j]) / 2
                       and abs(cand - ly[j]) < LAB_H)
            cost += sum(1 for j in range(len(ax))
                        if abs(ci - ax[j]) < w[i] / 2 + mark_r and abs(cand - ay[j]) < sep)
            if cost == 0:
                best = cand
                break
            if best_cost is None or cost < best_cost:   # fall back to the least-bad slot
                best, best_cost = cand, cost
        ly[i] = best
        placed.append(i)
    ly = np.clip(ly, LAB_H / 2, 1 - LAB_H / 2)
    return lx * (xr[1] - xr[0]) + xr[0], ly * (yr[1] - yr[0]) + yr[0]


# Font setup, shared by every raw-matplotlib figure here. Inter matches the plotnine theme the
# rest of the paper's figures use (`family="Inter"` in theme_set above); these figures are raw
# matplotlib rather than plotnine because the label placement needs per-annotation control, so
# the font has to be set on rcParams by hand -- plotnine's theme does not reach it. mathtext
# gets Inter too: on the DejaVu default, "$-c_k$" and the legend's "$k$" would render in a
# visibly different face from the text right next to them. cal/sf/tt are unused, but a "custom"
# fontset resolves all of them at import time and the cal default ("cursive") is not installed,
# so leaving them emits a findfont warning on every run. fonttype 42 embeds real TrueType
# outlines instead of Type-3, which is what arXiv and most camera-ready checkers want.
RC = {
    "font.family": "Inter", "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42, "text.color": "#000000",
    "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
}


def node_rows():
    """One row per node-level point: gradient baselines, MAttr ablations, Node Pruning, DBM."""
    rows = []
    for disp, dacc, sub in A.BASELINES:
        acc = avg({(t, m): A.acc_base(dacc, sub, t, m) for t, m, _ in COLS})
        dn, subn = BASE_CPR[disp]
        cpr = avg(cpr_base(dn, subn))
        if acc is not None and cpr is not None:
            # avg() means over whatever is on disk, so a still-running method plots a mean over
            # FEWER cells than the points next to it -- invisible on the figure. Say so.
            n = len([v for v in cpr_base(dn, subn).values() if v is not None])
            if n < len(COLS):
                print(f"  NOTE {disp}: CPR mean over {n}/{len(COLS)} cells", file=sys.stderr)
            rows.append(dict(acc=acc, cpr=cpr, grp=G_GRAD, label=delatex(disp), paths=[]))
    for name, d, level, g in M.OUR_METHODS:
        if level != "node":
            continue
        acc = avg({(t, m): A.acc_mattr(d, t, m) for t, m, _ in COLS})
        cpr = avg({(t, m): M.load_cpr_auc(d, t, m) for t, m, _ in COLS})
        if acc is None or cpr is None:
            print(f"  skip {d}: acc={acc} cpr={cpr}", file=sys.stderr)
            continue
        base, lr = LR_ANCHOR.get(d, (None, 0.0))
        # Every ablation name in OUR_METHODS appears TWICE -- once in the log-k block, once in
        # the uniform-k block -- and the table tells them apart by which block the row sits in.
        # A scatter has no blocks, so without this prefix the figure carries two points labelled
        # "MAttr" and two labelled "+hard" whose only distinction is a legend colour. It became
        # load-bearing when the uniform rows were repointed to the lr=0.05 dirs, which put both
        # copies of each name at plotted-and-complete status.
        label = delatex(name) if g == "ours" else "unif $k$, " + delatex(name)
        rows.append(dict(acc=acc, cpr=cpr, grp=G_MLOG if g == "ours" else G_MUNI,
                         label=label,
                         paths=[(f"lr:{base}", lr)] if base else []))
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
        paths = [("ld" if ld else "kl",
                  float(delatex(label).split("s=")[1].split(",")[0]))]
        if dirn in EPRUN_LR_ANCHOR:           # also the lr=0.8 node of its own LR path
            b, lr = EPRUN_LR_ANCHOR[dirn]
            paths.append((f"lr:{b}", lr))
        rows.append(dict(acc=acc, cpr=cpr, grp=G_NPLD if ld else G_NPKL,
                         label=delatex(label).replace(", logit-diff", ""), paths=paths))
    # every complete swept lr, incl. both complete DBM points (the only DBM source here)
    rows += build_lr_rows()
    # ...and the DBM sparsity-penalty sweep at that best lr ("L1=" rather than "lr=")
    rows += build_lr_rows(L1_SERIES, key="l1", knob="L1")
    return rows


def edge_rows():
    """One row per edge-level point: 7 MAttr variants + EAP-IG-inp.

    Far thinner than node_rows(), and not because anything was left out. There is no Edge
    Pruning or DBM at edge level (no results/*/EdgePruning_patching_edge anywhere on disk) and
    no edge lr sweeps, so the sparsity paths, the DBM series and build_lr_rows all have nothing
    to contribute; UGS exists but covers 3/11 cells (gpt2/qwen2.5 on ioi + mcqa only) and falls
    to the completeness bar below, which is printed rather than silent. The honest reading of
    the edge panel is therefore "our ablations against ONE baseline", not a survey -- but where
    that one baseline lands is the point.
    """
    def pair_avg(dirn):
        pairs = [_pair(dirn, t, m) for t, m, _ in COLS]
        return ([a for a, _ in pairs if a is not None],
                [c for _, c in pairs if c is not None])

    rows = []
    # Same completeness bar as build_lr_rows: 11/11 on BOTH axes, because every point here is a
    # mean over cells and a 3-cell mean is not comparable to an 11-cell one.
    EDGE_BASELINES = [("EAP-IG-inp", "eapig_repro_accauc", G_GRAD),
                      ("UGS", "ugs_eval", G_GRAD)]
    for disp, dirn, grp in EDGE_BASELINES:
        acc, cpr = pair_avg(dirn)
        if len(acc) < len(COLS) or len(cpr) < len(COLS):
            print(f"  skip {disp} ({dirn}): acc {len(acc)}/{len(COLS)}, cpr {len(cpr)}/"
                  f"{len(COLS)} -- incomplete", file=sys.stderr)
            continue
        rows.append(dict(acc=float(np.mean(acc)), cpr=float(np.mean(cpr)), grp=grp,
                         label=delatex(disp), paths=[]))
    for name, d, level, g in M.OUR_METHODS:
        if level != "edge":
            continue
        acc, cpr = pair_avg(d)
        if len(acc) < len(COLS) or len(cpr) < len(COLS):
            print(f"  skip {delatex(name)} ({d}): acc {len(acc)}/{len(COLS)}, cpr "
                  f"{len(cpr)}/{len(COLS)} -- incomplete", file=sys.stderr)
            continue
        label = delatex(name) if g == "ours" else "unif $k$, " + delatex(name)
        rows.append(dict(acc=float(np.mean(acc)), cpr=float(np.mean(cpr)),
                         grp=G_MLOG if g == "ours" else G_MUNI, label=label, paths=[]))
    return rows


def draw_points(ax, rows, xpad=XPAD, legend=True, title=None, xlabel=True,
                fs=(9, 8, 7.5), msize=46):
    """Markers, dashed series, axes furniture. Returns the frame; labels come later.

    Split from place_labels() because label geometry is measured in axes-fraction units, so it
    is only valid once the axes has its final size -- i.e. after tight_layout(). Drawing points
    for every panel first, then laying out, then labelling is the only order that gets the same
    answer in a one-panel and a two-panel figure.

    `fs` = (axis-title, tick, legend) point sizes and `msize` the marker area. They are
    arguments, not constants, because the compact figure goes in at 0.30\\textwidth (1.65in)
    against the full-page variants' 5.5in: point sizes are absolute, so the same numbers that
    read correctly on a full page render ~3x oversized in the small float.
    """
    df = pd.DataFrame(rows)
    # Dashed guides through every ordered series: the two Node Pruning objectives ordered
    # sparse-ward ("kl"/"ld"), each swept method ordered by learning rate ("lr:<method>"), and
    # DBM ordered by sparsity coefficient ("l1:DBM"). Same visual language for all three because
    # they are the same statement -- points joined by a line differ in ONE hyperparameter, so the
    # line's direction is the sensitivity to it. A point may belong to more than one series
    # (eprun_eval_s0.5_ld is both the s=0.5 node of the sparsity path and the lr=0.8 node of its
    # LR path; the unpenalised DBM lr=0.3 run is also the L1 path's lambda=0 node), so paths are
    # collected off the raw rows rather than by grouping the frame on a single column.
    segs = {}
    for r in rows:
        for key, sval in r["paths"]:
            segs.setdefault(key, []).append((sval, r["acc"], r["cpr"], r["grp"]))
    for key, pts in segs.items():
        pts.sort()
        if len(pts) > 1:
            ax.plot([p[1] for p in pts], [p[2] for p in pts], ls="dashed", lw=0.7,
                    alpha=0.55, zorder=1, color=FULL_COLORS[pts[0][3]])
    for grp in FULL_ORDER:
        sub = df[df.grp == grp]
        if not len(sub):
            continue
        ax.scatter(sub.acc, sub.cpr, s=msize,
                   marker=FAMILY_SHAPE[MASK if grp in FULL_MASK else GRADIENT],
                   c=FULL_COLORS[grp], edgecolors="#000000", linewidths=0.5, zorder=3,
                   label=grp)

    xr, yr = ax.get_xlim(), ax.get_ylim()
    # Room for labels on the right. 0.22 was enough at 32 points; the "unif k, " prefix and the
    # DBM L1 points push the widest labels past the frame at that value, and repel() has nowhere
    # to send the seven-deep blue cluster at acc~0.47 when its boxes are already at the edge.
    ax.set_xlim((xr[0], xr[1] + xpad * (xr[1] - xr[0])))
    ax.set_ylim(yr)
    if xlabel:
        # "IIA log-AUC", not "acc-AUC": the paper's prose calls this metric IIA AUC, and the AUC
        # is taken over the 10 LOG-spaced sparsity points (0.1...100%), not a linear sweep. The
        # compact figure still says "acc-AUC" -- change both together or the two versions of the
        # same figure disagree about what their shared x axis measures.
        ax.set_xlabel("IIA log-AUC (↑)", fontsize=fs[0])
    ax.set_ylabel("CPR AUC (↑)", fontsize=fs[0])
    if title:
        ax.set_title(title, fontsize=fs[0], loc="left", pad=4)
    ax.tick_params(labelsize=fs[1])
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if legend:
        ax.legend(fontsize=fs[2], loc="lower right", frameon=True, framealpha=0.95,
                  borderpad=0.5, handletextpad=0.4)
    return df


def place_labels(fig, ax, df, pt=LAB_PT, msize=46):
    """Direct labels with leader lines. Call AFTER the figure is laid out (see draw_points).

    `msize` must match the marker area draw_points() used: matplotlib's `s` is an area in
    points^2, i.e. physical, so the same marker covers three times more of the 1.65in main-text
    panel than of a full-page one, and repel()'s obstacle radius has to be measured here rather
    than fixed. The anchor offset rides on the same measurement (DX/MARK_R = 1.45 on the
    full-page figure these were tuned on), so a label always clears its own marker by the same
    visible gap at any figure size.
    """
    xr, yr = ax.get_xlim(), ax.get_ylim()
    w, h = label_boxes(ax, df.label.tolist(), fig, pt=pt)
    axb = ax.get_window_extent(renderer=fig.canvas.get_renderer())
    half = 0.5 * np.sqrt(msize) * fig.dpi / 72.0        # marker half-extent, px
    mark_r = max(half / axb.width, half / axb.height)
    lx, ly = repel(df.acc.values, df.cpr.values, w, h, xr, yr,
                   anchor_dx=1.45 * mark_r, mark_r=mark_r)
    for (x, y, lxi, lyi, lab, grp) in zip(df.acc, df.cpr, lx, ly, df.label, df.grp):
        ax.plot([x, lxi], [y, lyi], lw=0.35, color="#888888", zorder=2)
        ax.annotate(lab, (lxi, lyi), fontsize=pt, va="center", ha="left",
                    color=FULL_COLORS[grp], zorder=4,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    # Report what the layout could not solve. A label sitting under another one is the failure
    # mode this whole file exists to avoid, and it is invisible in the console otherwise -- the
    # PDF just quietly ships with two names on top of each other, as it did before widths were
    # measured rather than estimated.
    nx = (lx - xr[0]) / (xr[1] - xr[0]) + w / 2
    ny = (ly - yr[0]) / (yr[1] - yr[0])
    hit = ((np.abs(nx[:, None] - nx[None, :]) < (w[:, None] + w[None, :]) / 2)
           & (np.abs(ny[:, None] - ny[None, :]) < h))
    np.fill_diagonal(hit, False)
    n = int(np.triu(hit).sum())
    print(f"  label overlaps after layout: {n}", file=sys.stderr)


def node_rho(df):
    from scipy.stats import spearmanr
    o = df[~df.grp.isin({G_NPKL, G_NPLD})]
    return (f"Spearman rho (node): {spearmanr(df.acc, df.cpr)[0]:.3f} (all {len(df)}), "
            f"{spearmanr(o.acc, o.cpr)[0]:.3f} (excl. Node Pruning, {len(o)})")


def edge_rho(df):
    # Three rhos, because one would be misleading. "+hard bwd" (REINFORCE) sits at the origin on
    # BOTH axes -- it is the collapsed run, not a point on the trade-off -- and a single far
    # outlier consistent on both axes manufactures a high rank correlation on its own. Dropping
    # it is what shows whether the remaining points agree at all.
    from scipy.stats import spearmanr
    nb = df[df.label != "+hard bwd"]
    om = df[df.grp != G_GRAD]
    out = (f"Spearman rho (edge): {spearmanr(df.acc, df.cpr)[0]:.3f} (all {len(df)}), "
           f"{spearmanr(nb.acc, nb.cpr)[0]:.3f} (excl. +hard bwd, {len(nb)}), "
           f"{spearmanr(om.acc, om.cpr)[0]:.3f} (MAttr only, {len(om)})")
    g = df[df.grp == G_GRAD]
    if len(g):
        out += (f"\n  EAP-IG-inp at acc={g.acc.iloc[0]:.3f} cpr={g.cpr.iloc[0]:.2f}; "
                f"MAttr variants span acc {om.acc.min():.3f}-{om.acc.max():.3f}, "
                f"cpr {om.cpr.min():.2f}-{om.cpr.max():.2f}")
    return out


def main_full():
    """--full: node level alone, full page, every point named."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    df = draw_points(ax, node_rows())
    fig.tight_layout()
    place_labels(fig, ax, df)
    out = "plots/mib_accauc_cpr_scatter_full.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(df)} points)\n{node_rho(df)}")


def main_full_edge():
    """--full --edge: the same two metrics at edge level, full page, every point named."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    df = draw_points(ax, edge_rows(), xpad=0.22)
    fig.tight_layout()
    place_labels(fig, ax, df)
    out = "plots/mib_accauc_cpr_scatter_edge_full.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(df)} points)\n{edge_rho(df)}")


def main_both():
    """--both: node and edge in ONE full-page figure, node on top.

    The two panels answer the same question at the two granularities MIB scores, and reading
    them against each other is the whole point (mask learning dominates acc-AUC at node level;
    at edge level the one complete gradient baseline lands at comparable acc-AUC but a quarter
    of the CPR). Two separate float environments put them on different pages as often as not.

    Height is split 2:1, not evenly. The node panel carries 49 labelled points against the edge
    panel's 9, and vertical room is what label placement is actually short of -- an even split
    would spend half the page resolving nine labels that have never collided.

    The legend lives on the node panel only: the edge panel's groups are a subset of it, and the
    colours and shapes mean the same thing in both.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    fig, (ax_n, ax_e) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H_BOTH), gridspec_kw=dict(height_ratios=[2, 1]))
    dfn = draw_points(ax_n, node_rows(), title="(a) Node level", xlabel=False)
    dfe = draw_points(ax_e, edge_rows(), xpad=0.22, legend=False, title="(b) Edge level")
    fig.tight_layout()
    place_labels(fig, ax_n, dfn)
    place_labels(fig, ax_e, dfe)
    out = "plots/mib_accauc_cpr_scatter_both.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(dfn)} node + {len(dfe)} edge points)\n"
          f"{node_rho(dfn)}\n{edge_rho(dfe)}")


# === compact figure: which points survive ===
# The main-text scatter is the same plot as --full, cut to eight named points. Selection rule,
# so it is re-derivable rather than taste: one point per METHOD FAMILY at that family's best
# setting, plus the endpoints of the gradient spread.
#
#   MAttr / +hard          the headline and its one forward-pass ablation (lr=0.05, log k)
#   Node Pruning s=0.5     best mask baseline by CPR (1.67), and the one the tables report
#   DBM (L1=6.0)           best DBM setting on both sweeps (lr 0.3, lambda 6.0)
#   GIM, AttnLRP,          the strongest gradient baselines by CPR. GIM and AttnLRP are a tie
#   RelP+QK, NAP-IG        (1.31 each on the 10 matched cells) and near-duplicates by
#                          construction -- they share the same q/4,k/4,v/2 attention rule and
#                          norm freeze, differing only in the tempered softmax and the MLP
#                          activation derivative (rho = 0.96; see MIB-circuit-track's
#                          gim_attnlrp_decomp.py). Both are plotted anyway: the point of this
#                          panel is the CPR/acc-AUC frontier, and two methods landing on top of
#                          each other IS the finding. Drop one only if the overplotting hurts.
#   IxG                    the weakest, so the gradient cloud shows its full range
#
# Keys are (group, node_rows() label), so this list cannot drift away from the full figure: a
# point that stops existing there raises here instead of silently dropping out. The group is
# part of the key because the label alone is not unique -- Node Pruning's two objectives sweep
# the same budgets, so "s=0.5" names a KL point and a logit-diff point, and keying on the label
# alone silently plotted both. Everything else (the other 11 budgets, the lr and L1 paths, the
# remaining ablations) is exactly what the appendix --full version is for.
COMPACT = {
    (G_MLOG, "MAttr"): None, (G_MLOG, "+hard"): None,
    # M.EPRUN_BEST_SPARSITY; drop the bare "s=" (and the objective) for the main text
    (G_NPLD, "s=0.5"): "Node Pruning",
    (G_DBM, "DBM L1=6.0"): "DBM",       # the sweep value is an appendix detail
    (G_GRAD, "GIM"): None, (G_GRAD, "AttnLRP"): None,
    (G_GRAD, "RelP+QK"): None, (G_GRAD, "NAP-IG"): None,
    (G_GRAD, "IxG"): "I$\\times$G",
}

# 0.30\textwidth = 1.65in on the page, and the float goes in at width=\linewidth, so authoring
# at exactly that width renders 1:1 -- fonts here are page points, and the height set here is
# the height on the page.
#
# 2.0in is not free choice: this is the right-hand subfigure of fig:mib-combined, and the
# left-hand one (method_corr_heatmap_bytype, 3.69 x 2.0in at 0.67\textwidth) renders 1.997in
# tall. Matching it means the two panels' frames line up instead of one floating 0.2in above
# the other over a shared row of captions. The 0.003in residual is 0.2pt -- below anything
# visible, and not worth carrying an odd number for. Re-measure if either subfigure's width
# fraction changes.
FIG_W_C, FIG_H_C = 1.65, 2.0
LAB_PT_C, MSIZE_C = 5.5, 18
# Labels are ~as wide as they are on the full page but the panel is a third the width, so they
# need proportionally far more room: 0.34 leaves "Node Pruning" hanging off the frame.
XPAD_C = 0.70


def main():
    """The main-text figure: --full's plot, eight named points, no legend."""
    import matplotlib.pyplot as plt
    rows = [r for r in node_rows() if (r["grp"], r["label"]) in COMPACT]
    missing = set(COMPACT) - {(r["grp"], r["label"]) for r in rows}
    if missing:
        raise SystemExit(f"compact figure: no node_rows() point at {sorted(missing)} "
                         f"-- renamed upstream, or its dir went incomplete")
    for r in rows:
        r["label"] = COMPACT[(r["grp"], r["label"])] or r["label"]
        r["paths"] = []   # one point per series here, so every dashed guide would be a no-op

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(FIG_W_C, FIG_H_C))
    # No legend: with eight points every one is named, so a group legend would spend a third of
    # a 1.65in panel restating what the labels already say. Colour still encodes the group and
    # shape the gradient/mask split, both consistent with the appendix figure.
    df = draw_points(ax, rows, xpad=XPAD_C, legend=False, fs=(7, 6, 6), msize=MSIZE_C)
    fig.tight_layout(pad=0.4)
    place_labels(fig, ax, df, pt=LAB_PT_C, msize=MSIZE_C)
    out = "plots/mib_accauc_cpr_scatter.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(df)} points)")
    # The figure's caption quotes this rho, so print it rather than leaving it hand-maintained
    # -- it drifts with every re-eval, and the mask baselines pull it down (they buy CPR at
    # markedly lower IIA than any gradient method, so the two metrics rank them differently).
    from scipy.stats import spearmanr
    o = df[~df.grp.isin({G_NPKL, G_NPLD, G_DBM})]
    print(f"Spearman rho: {spearmanr(df.acc, df.cpr)[0]:.3f} (all {len(df)}), "
          f"{spearmanr(o.acc, o.cpr)[0]:.3f} (excl. mask baselines, {len(o)})")


if __name__ == "__main__":
    if "--both" in sys.argv:
        main_both()           # always full-page: node and edge panels in one figure
    elif "--edge" in sys.argv:
        main_full_edge()      # always full-page; there is no compact edge variant
    elif "--full" in sys.argv:
        main_full()
    else:
        main()
