"""Compact MIB scatter: acc-AUC (x) vs CPR/logit-diff AUC (y), one point per node method.

For every node-level MIB method (gradient baselines + all MAttr ablations) we computed both
metrics on the validation set. This shows how the two agree across methods (Spearman rho is
printed, not drawn — the caption quotes it) — a companion to the MLP/Attn Spearman heatmap.

Three variants of one plot, all raw matplotlib with direct point labels:
  (no flag)  main text, 0.30\textwidth, eight curated points   -> mib_accauc_cpr_scatter.pdf
  --full     appendix, full page, every node point             -> ..._full.pdf
  --edge     appendix, full page, every edge point             -> ..._edge_full.pdf
  --both     appendix, full page, node over edge in one float  -> ..._both.pdf
  --lr       appendix, full page, ONLY the LR sweeps (node)    -> ..._lr.pdf

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
    """area_under per cell, searching BOTH results roots (A.ROOTS = L2A, then MIB).

    Dual-root for the same reason acc_base is: a dir lands MIB-side and is only sometimes
    copied over. The IG step dirs (napig{10,30}_eval) exist ONLY under MIB-circuit-track, so an
    RB-only read returned an empty dict and the point silently vanished -- the acc axis would
    have resolved fine through A.acc_base, which already searched both, and the mismatch between
    the two readers is exactly the kind of half-fix that plots a method at the wrong place.
    """
    out = {}
    fn_for = lambda t, m: f"{t.replace('_', '-')}_{m}_validation_abs-False.pkl"
    for t, m, _ in COLS:
        for root in A.ROOTS:
            p = root / dirn / sub / fn_for(t, m)
            if not p.exists():
                continue
            try:
                out[(t, m)] = pickle.load(open(p, "rb"))["area_under"]
                break
            except Exception:
                pass
    return out


# gradient baseline CPR dirs (mirror make_mib_table.EXTRA_NODE_BASELINES + NAP-IG repro).
# MUST cover every entry in A.BASELINES: node_rows() indexes this dict by the acc-table's display
# name, so a baseline added there and not here is a KeyError, not a missing point. (That is not
# hypothetical -- adding the IG step rows to the acc table is what broke this figure.)
BASE_CPR = {
    "NAP-IG": ("napig_ref_eval", "EAP-IG-inputs_patching_node"),
    # The IG step ladder. MIB ships --ig-steps 5 ("NAP-IG" above) and that integral is not
    # converged: 10 steps moves the row average 0.85 -> 1.31 CPR and 0.30 -> 0.46 acc-AUC, so on
    # this figure the 5-step point sits far down-left of where the same method lands once it is
    # integrated properly. 30 steps then lands essentially on top of 10 (rho 0.994, zero sign
    # flips between the two rungs), which is what makes the pair worth plotting: the visible gap
    # is 5 -> 10 and the visible non-gap is 10 -> 30. Both dirs are MIB-side only, hence the
    # dual-root cpr_base above.
    "$+$ 10 IG steps": ("napig10_eval", "EAP-IG-inputs_patching_node"),
    "$+$ 30 IG steps": ("napig30_eval", "EAP-IG-inputs_patching_node"),
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

# === IG integration-step ladder ===
# One method at three budgets, so it gets a dashed path like every other one-knob sweep here --
# the line's direction is the sensitivity to --ig-steps, and that is the whole claim. Keyed by
# the acc table's display name (A.BASELINES), which is what node_rows() iterates.
#
# Labels are rewritten because the table's "$+$ 10 IG steps" reads as an ablation OF NAP-IG when
# the rows are stacked under it, and as a separate method once they are scattered. Naming the
# budget on all three -- including the 5-step default -- is what makes the path self-explaining
# without a caption. COMPACT whitelists the 5- and 10-step names too, shortened to "IG-5" /
# "IG-10": that is exactly what the sibling panel of fig:mib-combined
# (method_corr_heatmap_bytype) calls those two columns, and one figure using two names for one
# method is a referee-visible inconsistency for no gain. 30 stays out of both: rho 0.994 with
# 10 and zero sign flips, so it lands on top of it.
#
# "NAP-IG" survives only as a DICT KEY here -- it is the acc table's display name, which is what
# node_rows() iterates. Nothing rendered says "NAP-IG" any more.
IG_STEPS = {"NAP-IG": 5.0, "$+$ 10 IG steps": 10.0, "$+$ 30 IG steps": 30.0}
IG_STEP_LABEL = {"NAP-IG": "IG (5 steps)",
                 "$+$ 10 IG steps": "IG (10 steps)",
                 "$+$ 30 IG steps": "IG (30 steps)"}

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
    # MAttr's optimizer ablation. Unlike the MAttr/+hard ladders above, 0.01 IS listed: that LR
    # has its own dir here (softlog_sgd_lr_0.01) evaluated with the rest of the sweep, rather
    # than being the old lr=0.01 main run whose acc_auc covers only 3/11 cells.
    # 0.05 is excluded for the usual reason -- it is the headline point from M.OUR_METHODS and
    # is stitched back into this path by LR_ANCHOR below.
    # 1.0 is excluded here for the SAME reason 0.05 is excluded from the Adam ladder above: it
    # is SGD's own best LR, so it comes in from M.OUR_METHODS as the plotted "MAttr (SGD)"
    # point, and LR_ANCHOR stitches it back onto this path. 0.05 IS listed, because repointing
    # the OUR_METHODS row from 0.05 to 1.0 left that rung with no other source.
    (G_MLOG, "MAttr (SGD)", [("0.005", "softlog_sgd_lr_0.005"), ("0.01", "softlog_sgd_lr_0.01"),
                             ("0.05", "softlog_sgd_lr_0.05"),
                             ("0.1", "softlog_sgd_lr_0.1"), ("0.3", "softlog_sgd_lr_0.3"),
                             ("3.0", "softlog_sgd_lr_3.0"), ("10.0", "softlog_sgd_lr_10.0")]),
    # The uniform-k twin. G_MUNI carried no LR path at all until now, so its SGD point
    # (softuni_sgd_lr_3.0, the OUR_METHODS row) floated with nothing to read its LR against.
    (G_MUNI, "unif $k$, MAttr (SGD)",
     [("0.05", "softuni_sgd_lr_0.05"), ("0.1", "softuni_sgd_lr_0.1"),
      ("0.3", "softuni_sgd_lr_0.3"), ("1.0", "softuni_sgd_lr_1.0"),
      ("10.0", "softuni_sgd_lr_10.0")]),
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
LR_ANCHOR = {"topklog_lr_0.05": ("MAttr", 0.05), "htklog_lr_0.05": ("+hard", 0.05),
             # each SGD arm's anchor is its OWN optimum, not a shared 0.05 -- that is the whole
             # point of make_mib_table's repoint (log-k peaks at 1.0, uniform-k at 3.0)
             "softlog_sgd_lr_1.0": ("MAttr (SGD)", 1.0),
             "softuni_sgd_lr_3.0": ("unif $k$, MAttr (SGD)", 3.0)}
# Same trick for Node Pruning, except the anchor is a point that ALREADY sits on another path:
# eprun_eval_s0.5_ld is the s=0.5 node of the sparsity path AND the lr=0.8 node of its own LR
# path. That is why rows carry a list of (path, sort-key) pairs rather than one of each.
EPRUN_LR_ANCHOR = {"eprun_eval_s0.5_ld": ("NP s=0.5", 0.8),
                   "eprun_eval_s0.8_ld": ("NP s=0.8", 0.8)}


# =========================================================================================
# --lr: every LR sweep in the paper on one frame, and NOTHING else
# =========================================================================================
# The --full/--both figures answer "do the two metrics agree across methods?", and the LR paths
# are a minor part of that picture -- 59 points, of which the swept ones are a minority, and the
# gradient cloud sets the axis limits. This variant asks the other question: how much of the gap
# between any two methods here is just learning rate? Dropping the fixed-setting points lets the
# axes zoom onto the swept region, which is where that question is legible.
#
# NODE LEVEL ONLY, and not by choice: there is no edge-level LR sweep on disk. Every edge dir is
# a single setting at lr=0.05 (mib_edge_{topk,hard_topk}_{log,uniform}_lr05) plus the two
# gradient baselines, so an edge panel here would be four points with no path through them --
# not a sweep. That is a gap in the experiments, not in this figure; if edge LR sweeps land,
# add an EDGE_LR_SERIES and switch this to the two-panel layout main_both() already implements.
#
# Colour follows palette.py's rule -- a colour is a METHOD, a hyperparameter variant is a
# LINETYPE -- so all four MAttr-family paths take MAttr's blue and separate by dash pattern.
# That is why draw_points() takes `colors`/`order`/`path_style`: the group key here names a
# series, not a family, so it can no longer double as the hue the way FULL_COLORS does.
LR_SERIES_STYLE = {}          # filled below, keyed "lr:<short>" to match build_lr_rows
LR_COLORS, LR_ORDER = {}, []

# (legend name, short label used on the points, colour, linestyle, [(lr, dir), ...])
# Points are labelled "<short> lr=<v>" by build_lr_rows, so `short` is kept to a few characters:
# 34 labels on one panel and the long legend names would not fit even at XPAD 1.0.
#
# COMPLETENESS (11/11 cells on BOTH axes) is what decides membership, as everywhere else here,
# and build_lr_rows prints every exclusion. Verified 2026-08-20; what that currently drops:
#   htk_lr_{0.005,0.1,0.3}  acc 10/11 -- one missing cell kills the whole "+unif k, +hard" path
#   bern_lr_0.05            acc 10/11 (0.01 and 0.3 survive, so the path is drawn through two)
#   bern_lr_0.1_2k          complete, but 2000 steps against everything else's 500 -- excluded
#                           deliberately: it is a step-count point, not an LR point
#   softlog_sgd_lr_{0.005,0.01}   5/11
#   softlog_sgd_lr_{3.0,10.0}, softuni_sgd_lr_{3.0,10.0}   still running (submitted 2026-08-20);
#                           they join automatically on the next render, no edit needed
#   eprun_eval_ld_dcm_*     3/11 across all 15 dirs -- DCM was swept on the cheap cells only
LR_ONLY_SERIES = [
    ("MAttr (log $k$, Adam)", "MAttr", P.METHOD["MAttr"], "solid",
     [("0.005", "topklog_lr_0.005"), ("0.05", "topklog_lr_0.05"),
      ("0.1", "topklog_lr_0.1"), ("0.3", "topklog_lr_0.3")]),
    ("MAttr (log $k$, SGD)", "M-SGD", P.METHOD["MAttr"], "dashed",
     [("0.05", "softlog_sgd_lr_0.05"), ("0.1", "softlog_sgd_lr_0.1"),
      ("0.3", "softlog_sgd_lr_0.3"), ("1.0", "softlog_sgd_lr_1.0"),
      ("3.0", "softlog_sgd_lr_3.0"), ("10.0", "softlog_sgd_lr_10.0")]),
    ("MAttr (unif. $k$, SGD)", "M-SGDu", P.METHOD["MAttr"], "dotted",
     [("0.05", "softuni_sgd_lr_0.05"), ("0.1", "softuni_sgd_lr_0.1"),
      ("0.3", "softuni_sgd_lr_0.3"), ("1.0", "softuni_sgd_lr_1.0"),
      ("3.0", "softuni_sgd_lr_3.0"), ("10.0", "softuni_sgd_lr_10.0")]),
    ("$+$ hard (log $k$, Adam)", "+hard", P.METHOD["+hard"], "solid",
     [("0.005", "htklog_lr_0.005"), ("0.05", "htklog_lr_0.05"),
      ("0.1", "htklog_lr_0.1"), ("0.3", "htklog_lr_0.3")]),
    # REINFORCE backward, uniform k. Two surviving points is barely a path, but it is the only
    # evidence we have that this variant's collapse is not an LR artifact, so it is plotted.
    ("$+$ hard bwd (unif. $k$)", "+hbwd", P.METHOD["+hard"], "dashdot",
     [("0.01", "bern_lr_0.01"), ("0.3", "bern_lr_0.3")]),
    # NOTE the lr=0.001 node is the dir with no lr suffix -- eprun_eval_ld_sig IS the default-LR
    # run, the same anchoring trick LR_ANCHOR plays for MAttr. The --full figure draws this path
    # through TWO points (0.001 and 0.3) because LR_SERIES lists only those; the other three
    # dirs are complete and have been on disk all along, so that path is under-drawn there.
    ("DBM", "DBM", P.METHOD["DBM"], "solid",
     [("0.001", "eprun_eval_ld_sig"), ("0.01", "eprun_eval_ld_sig_lr0.01"),
      ("0.1", "eprun_eval_ld_sig_lr0.1"), ("0.3", "eprun_eval_ld_sig_lr0.3"),
      ("1.0", "eprun_eval_ld_sig_lr1.0")]),
    ("Node Pruning $s{=}0.5$", "NP.5", P.METHOD["Node Pruning"], "solid",
     [("0.1", "eprun_eval_s0.5_ld_lr0.1"), ("0.3", "eprun_eval_s0.5_ld_lr0.3"),
      ("0.8", "eprun_eval_s0.5_ld"), ("1.5", "eprun_eval_s0.5_ld_lr1.5"),
      ("3.0", "eprun_eval_s0.5_ld_lr3.0")]),
    ("Node Pruning $s{=}0.8$", "NP.8", P.METHOD["Node Pruning"], "dashed",
     [("0.1", "eprun_eval_s0.8_ld_lr0.1"), ("0.3", "eprun_eval_s0.8_ld_lr0.3"),
      ("0.8", "eprun_eval_s0.8_ld"), ("1.5", "eprun_eval_s0.8_ld_lr1.5"),
      ("3.0", "eprun_eval_s0.8_ld_lr3.0")]),
]
for _leg, _short, _col, _ls, _ in LR_ONLY_SERIES:
    LR_ORDER.append(_leg)
    LR_COLORS[_leg] = _col
    LR_SERIES_STYLE[f"lr:{_short}"] = _ls


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
# panel carries 59 labelled points against edge's 9, hence height_ratios=[2, 1]; 8.5in total
# leaves the node panel ~5.5in, i.e. LESS room than the standalone 8.4in figure, which is why
# the ladder pass in place_labels matters more here than it does for --full.
# This is a HARD ceiling, not a preference. ICLR's \\textwidth is 5.5in and \\textheight is
# 9.0in; the float goes in at width=\\linewidth, so LaTeX scales it by 5.5/FIG_W (= 1.019) and
# the height rides along -- every 0.1in here is ~7.3pt on the page. 8.6in overfull'd by 3.6pt.
# 8.5in fits, but only just: it leaves 0.34in = ~25pt for \\abovecaptionskip plus the caption,
# i.e. ONE line of caption and nothing more, which is a trap for a figure that needs to explain
# six series. 8.2in scales to 8.35in and leaves ~47pt, enough for a three-line caption.
FIG_H_BOTH = 8.2
# --lr carries ~34 points against --full's 49, and all of them sit in the swept region rather
# than being spread by a gradient cloud, so it needs less vertical room. 6.5in is set by the
# label-overlap diagnostic in place_labels(), not by taste: at 5.5in the dense lr=0.05 cluster
# leaves residual overlaps, at 6.5in it reports 0. XPAD is likewise measured -- the short point
# labels ("M-SGDu lr=0.05") are ~half the width of --full's, so 0.34 is more than they need.
FIG_H_LR, XPAD_LR = 6.5, 0.26

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


# Font setup, shared by every raw-matplotlib figure here -- MOVED TO palette.py, which is where
# the rest of this file's shared styling already lived. Other figures had started copying this
# dict out of here to match, which is the same duplication the hexes were centralised to stop.
# Aliased rather than replaced at the call sites so `S.RC` keeps working for importers.
RC = P.RC


# =========================================================================================
# test_only: collapse every swept family to the ONE setting carried to the test split
# =========================================================================================
# The full-page figures were 68 node points, most of them rungs of an LR or sparsity ladder:
# 12 Node Pruning budgets x 2 objectives, 8 NP learning rates, 7 MAttr-SGD LRs, 5 DBM lambdas.
# That cloud answers "how much of the gap is hyperparameters?" -- which is exactly what the
# --lr variant exists for -- while crowding out the question this figure asks, "do the two
# metrics agree across METHODS?". Worse, it lets a reader pick any rung as the baseline's
# score, including ones we never committed to.
#
# So under test_only each swept family shows the setting we actually took to the test split,
# read from the same constants the test table uses rather than re-argmaxed here:
#
#   Node Pruning  M.EPRUN_BEST_SPARSITY        (s=0.5, logit-diff)  == make_mib_test_table
#                                                                      .NODE_PRUNING
#   DBM           eprun_eval_ld_sig_lr0.3_l16.0 (lr=0.3, lambda=6)  == the headline DBM row of
#                                                                      both MIB tables
#   MAttr         nothing to do -- the four test-carried settings (log/unif x Adam lr=0.05 /
#                 SGD lr=1.0,3.0) are already the M.OUR_METHODS rows, and the ladders around
#                 them lived entirely in LR_SERIES.
#
# Verified against disk: the dirs holding test-split pkls are eprun_eval_s0.5_ld,
# eprun_eval_ld_sig_lr0.3_l16.0, eprun_eval (s=0.9 KL) and the test_node_*/test_edge_* trees.
# eprun_eval is the ONE case where disk is broader than the claim -- s=0.9 KL was the old
# headline budget and its test pkls predate the repoint to s=0.5 logit-diff. Following
# EPRUN_BEST_SPARSITY keeps this figure agreeing with the test table instead of with history.
#
# The MAttr ablations (+hard, -c_k, +hard bwd, id-STE, Gumbel) are NOT filtered: they are
# separate methods at the selected LR, not rungs of a ladder, and dropping them would delete
# the ablation story rather than de-duplicate a sweep. Note this is a weaker bar than the test
# TABLE applies -- it ships soft-forward rows only, and -c_k/+hard bwd/Gumbel have no test run
# at all. Tighten here if the figure should mirror the table exactly.
TEST_ONLY_DBM = ("DBM ($\\lambda{=}6$)", "eprun_eval_ld_sig_lr0.3_l16.0")


def node_rows(test_only=False):
    """One row per node-level point: gradient baselines, MAttr ablations, Node Pruning, DBM.

    test_only=True keeps just the test-carried setting of each swept family (see above).
    """
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
            rows.append(dict(acc=acc, cpr=cpr, grp=G_GRAD,
                             label=IG_STEP_LABEL.get(disp, delatex(disp)),
                             paths=[("ig", IG_STEPS[disp])] if disp in IG_STEPS else []))
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
        # The k-schedule is not the only axis OUR_METHODS reuses a name along: since the two
        # optimizer rows were repointed to their own best LRs (softlog_sgd_lr_1.0,
        # softuni_sgd_lr_3.0), "\ourmethod{}" names FOUR dirs -- {log, unif} x {Adam, SGD} --
        # and the table again tells them apart by a block header (emit_ours files on opt_of).
        # Without this suffix the log-k pair plotted as two points both labelled "MAttr" at
        # (0.499, 1.879) and (0.504, 1.886), i.e. the SGD point was on the main-text figure
        # already, unnamed and indistinguishable from the Adam one.
        if M.opt_of(d) == "sgd" and name == "\\ourmethod{}":
            label += " (SGD)"
        rows.append(dict(acc=acc, cpr=cpr, grp=G_MLOG if g == "ours" else G_MUNI,
                         label=label,
                         paths=[(f"lr:{base}", lr)] if base else []))
    # all 12 budgets; the two objectives are separate dashed paths, each ordered sparse-ward
    for label, dirn in ([M.EPRUN_BEST_SPARSITY] if test_only else M.EPRUN_SPARSITIES):
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
        # With the ladder gone there is no path for the point to sit on, and a bare "s=0.5"
        # stops being self-explanatory once it is the only budget on the frame.
        rows.append(dict(acc=acc, cpr=cpr, grp=G_NPLD if ld else G_NPKL,
                         label=("Node Pruning (" + delatex(label).replace(", logit-diff", "") + ")"
                                if test_only else delatex(label).replace(", logit-diff", "")),
                         paths=[] if test_only else paths))
    if test_only:
        # DBM has NO source outside the LR/L1 series, so dropping those would delete the
        # baseline from the figure rather than thin it. Re-added here as its single test point.
        lab, dirn = TEST_ONLY_DBM
        pairs = [_pair(dirn, t, m) for t, m, _ in COLS]
        acc = [a for a, _ in pairs if a is not None]
        cpr = [c for _, c in pairs if c is not None]
        if len(acc) < len(COLS) or len(cpr) < len(COLS):
            print(f"  skip {lab} ({dirn}): acc {len(acc)}/{len(COLS)}, cpr {len(cpr)}/"
                  f"{len(COLS)} -- incomplete", file=sys.stderr)
        else:
            rows.append(dict(acc=float(np.mean(acc)), cpr=float(np.mean(cpr)),
                             grp=G_DBM, label=delatex(lab), paths=[]))
        return rows
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
    def cells(dirn):
        """{(task, model): (acc_auc, area_under)} for the cells this dir has BOTH metrics on."""
        out = {}
        for t, m, _ in COLS:
            a, c = _pair(dirn, t, m)
            if a is not None and c is not None:
                out[(t, m)] = (a, c)
        return out

    # Candidates, in plot order: (label, dir, group).
    EDGE_BASELINES = [("EAP-IG-inp", "eapig_clean_eval", G_GRAD),
                      ("UGS", "ugs_eval", G_GRAD)]
    cand = [(delatex(disp), dirn, grp) for disp, dirn, grp in EDGE_BASELINES]
    for name, d, level, g in M.OUR_METHODS:
        if level != "edge":
            continue
        label = delatex(name) if g == "ours" else "unif $k$, " + delatex(name)
        # Same disambiguation the node loop above needs: the edge block gained soft-fwd SGD rows
        # (mib_edge_soft{log,uni}_sgd_lr_*), so "\ourmethod{}" names four edge dirs as well as
        # four node ones. Without the suffix the SGD point plots as a second unnamed "MAttr" on
        # top of the Adam one, which is exactly how the node collision hid.
        if M.opt_of(d) == "sgd" and name == "\\ourmethod{}":
            label += " (SGD)"
        cand.append((label, d, G_MLOG if g == "ours" else G_MUNI))

    data = {d: cells(d) for _, d, _ in cand}

    # COMPARABILITY IS ENFORCED BY A COMMON CELL SET, NOT BY AN 11/11 BAR.
    #
    # The bar used to be 11/11 on both axes, which is the right instinct -- every point here is a
    # mean over cells, so a 3-cell mean next to an 11-cell one reads as comparable when it is not.
    # But applied to the soft-SGD dirs it silently deleted them: they cover 9/11, missing exactly
    # arc_easy/llama3 and arc_challenge/llama3, and those two are the HIGHEST-scoring columns in
    # this section. So the bar dropped the points, and simply lowering it would have plotted them
    # over a subset biased against them -- both failure modes, in opposite directions.
    #
    # Instead every point is averaged over the INTERSECTION of the plotted dirs' cells. That is a
    # paired comparison at whatever coverage the thinnest plotted dir has, and it self-heals:
    # when submit_edge_arc_llama3.sh's four jobs land the intersection becomes 11/11 and every
    # coordinate returns to its full-coverage value with no edit here.
    #
    # EDGE_MIN_CELLS keeps a badly-covered dir from dragging the intersection down for everyone:
    # 9 is "may be missing only the two ARC/llama3 cells that no edge dir had before that script",
    # so UGS (3/11 -- gpt2/qwen2.5 on ioi+mcqa only) still falls out rather than cutting the
    # panel to three cells. Every exclusion is printed, never silent.
    EDGE_MIN_CELLS = 9
    kept = []
    for label, dirn, grp in cand:
        if len(data[dirn]) < EDGE_MIN_CELLS:
            print(f"  skip {label} ({dirn}): {len(data[dirn])}/{len(COLS)} cells on both "
                  f"axes -- below EDGE_MIN_CELLS={EDGE_MIN_CELLS}", file=sys.stderr)
            continue
        kept.append((label, dirn, grp))
    if not kept:
        return []

    common = set.intersection(*(set(data[d]) for _, d, _ in kept))
    if len(common) < len(COLS):
        missing = sorted(f"{t}/{m}" for t, m, _ in COLS if (t, m) not in common)
        print(f"  NOTE edge panel: every point averaged over the {len(common)}/{len(COLS)} cells "
              f"common to all plotted dirs; dropped {', '.join(missing)}", file=sys.stderr)

    rows = []
    for label, dirn, grp in kept:
        vals = [data[dirn][c] for c in common]
        rows.append(dict(acc=float(np.mean([a for a, _ in vals])),
                         cpr=float(np.mean([c for _, c in vals])),
                         grp=grp, label=label, paths=[]))
    return rows


def draw_points(ax, rows, xpad=XPAD, legend=True, title=None, xlabel=True,
                fs=(9, 8, 7.5), msize=46, colors=None, order=None, maskset=None,
                path_style=None):
    """Markers, dashed series, axes furniture. Returns the frame; labels come later.

    Split from place_labels() because label geometry is measured in axes-fraction units, so it
    is only valid once the axes has its final size -- i.e. after tight_layout(). Drawing points
    for every panel first, then laying out, then labelling is the only order that gets the same
    answer in a one-panel and a two-panel figure.

    `fs` = (axis-title, tick, legend) point sizes and `msize` the marker area. They are
    arguments, not constants, because the compact figure goes in at 0.30\\textwidth (1.65in)
    against the full-page variants' 5.5in: point sizes are absolute, so the same numbers that
    read correctly on a full page render ~3x oversized in the small float.

    `colors`/`order`/`maskset` default to the FULL_* group encoding shared by the three figures
    above. --lr overrides them because it groups by SERIES rather than by family: it plots four
    MAttr-family paths that all take MAttr's blue (palette.py's rule -- colour is a method,
    a hyperparameter variant is a linetype), so the group key can no longer double as the hue.
    `path_style` is that linetype, keyed by the same path name build_lr_rows() joins points on.
    """
    colors = FULL_COLORS if colors is None else colors
    order = FULL_ORDER if order is None else order
    maskset = FULL_MASK if maskset is None else maskset
    path_style = path_style or {}
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
            ax.plot([p[1] for p in pts], [p[2] for p in pts],
                    ls=path_style.get(key, "dashed"), lw=0.7,
                    alpha=0.55, zorder=1, color=colors[pts[0][3]])
    for grp in order:
        sub = df[df.grp == grp]
        if not len(sub):
            continue
        ax.scatter(sub.acc, sub.cpr, s=msize,
                   marker=FAMILY_SHAPE[MASK if grp in maskset else GRADIENT],
                   c=colors[grp], edgecolors="#000000", linewidths=0.5, zorder=3,
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


def place_labels(fig, ax, df, pt=LAB_PT, msize=46, colors=None):
    """Direct labels with leader lines. Call AFTER the figure is laid out (see draw_points).

    `msize` must match the marker area draw_points() used: matplotlib's `s` is an area in
    points^2, i.e. physical, so the same marker covers three times more of the 1.65in main-text
    panel than of a full-page one, and repel()'s obstacle radius has to be measured here rather
    than fixed. The anchor offset rides on the same measurement (DX/MARK_R = 1.45 on the
    full-page figure these were tuned on), so a label always clears its own marker by the same
    visible gap at any figure size.
    """
    colors = FULL_COLORS if colors is None else colors
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
                    color=colors[grp], zorder=4,
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
    # The other failure mode, and the one XPAD exists to prevent: a label running past the right
    # frame. Without this the only way to pick XPAD was to overshoot, which is how it reached
    # 0.70 -- most of that padding is empty axis. `slack` is how much of the padded range is
    # unused, i.e. how far XPAD can come down before labels start clipping.
    right = nx + w / 2
    off = int((right > 1.0).sum())
    if off:
        print(f"  labels past the right frame: {off}", file=sys.stderr)
    print(f"  x headroom: rightmost label ends at {right.max():.3f} of the frame", file=sys.stderr)


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
    df = draw_points(ax, node_rows(test_only=True))
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

    Height is split 2:1, not evenly. The node panel carries 59 labelled points against the edge
    panel's 9, and vertical room is what label placement is actually short of -- an even split
    would spend half the page resolving nine labels that have never collided.

    The legend lives on the node panel only: the edge panel's groups are a subset of it, and the
    colours and shapes mean the same thing in both.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    fig, (ax_n, ax_e) = plt.subplots(
        2, 1, figsize=(FIG_W, FIG_H_BOTH), gridspec_kw=dict(height_ratios=[2, 1]))
    dfn = draw_points(ax_n, node_rows(test_only=True), title="(a) Node level", xlabel=False)
    dfe = draw_points(ax_e, edge_rows(), xpad=0.22, legend=False, title="(b) Edge level")
    fig.tight_layout()
    place_labels(fig, ax_n, dfn)
    place_labels(fig, ax_e, dfe)
    out = "plots/mib_accauc_cpr_scatter_both.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(dfn)} node + {len(dfe)} edge points)\n"
          f"{node_rho(dfn)}\n{edge_rho(dfe)}")


def main_lr():
    """--lr: node level, LR sweeps only, full page, every point named.

    Every point is a mask-learning method, so the marker SHAPE that carries gradient-vs-mask in
    the other variants is uninformative here and every point is a square. The legend is rebuilt
    by hand rather than taken from ax.legend()'s scatter handles: three of the eight series are
    MAttr blue and separate only by dash pattern, so marker-only handles would show three
    identical blue squares. Line2D handles carry both.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update(RC)
    series = [(leg, short, vals) for leg, short, _, _, vals in LR_ONLY_SERIES]
    rows = build_lr_rows(series)
    if not rows:
        raise SystemExit("--lr: no complete LR series on disk")
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H_LR))
    df = draw_points(ax, rows, xpad=XPAD_LR, legend=False, colors=LR_COLORS, order=LR_ORDER,
                     maskset=set(LR_ORDER), path_style=LR_SERIES_STYLE)
    drawn = set(df.grp)
    ax.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=0.9, marker="s", ms=4.5,
                              mec="#000000", mew=0.5, label=leg)
                       for leg, _, c, ls, _ in LR_ONLY_SERIES if leg in drawn],
              fontsize=7.5, loc="lower right", frameon=True, framealpha=0.95,
              borderpad=0.5, handletextpad=0.4, handlelength=2.4)
    fig.tight_layout()
    place_labels(fig, ax, df, colors=LR_COLORS)
    out = "plots/mib_accauc_cpr_scatter_lr.pdf"
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({len(df)} points, {len(drawn)} series)")
    # Per-series LR spread: the number this figure exists to make visible. A method whose range
    # here is wider than its gap to a rival is a method whose ranking is an LR artifact.
    for leg in LR_ORDER:
        s = df[df.grp == leg]
        if len(s) < 2:
            continue
        print(f"  {leg:<26} n={len(s)}  acc {s.acc.min():.3f}-{s.acc.max():.3f} "
              f"(spread {s.acc.max()-s.acc.min():.3f})  "
              f"cpr {s.cpr.min():.2f}-{s.cpr.max():.2f} "
              f"(spread {s.cpr.max()-s.cpr.min():.2f})")


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
    # The optimizer ablation at ITS own best LR (softlog_sgd_lr_1.0), which the tables report as
    # a second \ourmethod{} row. It was already being drawn -- the label collision above meant
    # this point rendered on top of the Adam one under the same name -- so naming it does not
    # add ink, it stops two different circuits reading as one.
    #
    # LOG-k ONLY, deliberately. The uniform-k SGD point (0.482 / 2.031) is the better-CPR half
    # of the SGD pair, but its Adam twin (0.475 / 2.092) is the highest-CPR point on the whole
    # node figure and is NOT in this cut -- so putting unif-k SGD in alone would place the
    # weaker of the two uniform points on the frontier with nothing to read it against. Both
    # uniform points are in --full, where the pair is legible.
    #
    # Shortened to "+SGD" for the same reason "IG (10 steps)" is shortened to "IG-10": at 5.5pt
    # on a 1.65in panel, "MAttr (SGD)" is the widest label on the figure AND sits at the largest
    # x (acc-AUC 0.504), so it runs 16% past the right frame and only XPAD_C 0.70 clears it --
    # which spends a third of the panel on empty axis. "+SGD" clears at the existing 0.28. It
    # also reads as what it is: MAttr with one knob changed, exactly like the "+hard" beside it.
    # Caveat that the label cannot carry: this point is at SGD's own best LR (1.0), not the
    # headline's 0.05, per the OUR_METHODS repoint -- prose or caption has to say so.
    (G_MLOG, "MAttr (SGD)"): "+SGD",
    # M.EPRUN_BEST_SPARSITY; drop the bare "s=" (and the objective) for the main text
    (G_NPLD, "s=0.5"): "Node Pruning",
    (G_DBM, "DBM L1=6.0"): "DBM",       # the sweep value is an appendix detail
    (G_GRAD, "GIM"): None, (G_GRAD, "AttnLRP"): None,
    # BOTH IG budgets, joined by the dashed "ig" guide -- the one place this panel plots a
    # method twice. It earns the second point because the move is larger than the gaps the
    # panel is otherwise asking the reader to judge: 5 -> 10 steps takes NAP-IG from
    # 0.304/0.854 to 0.465/1.306, i.e. from worst gradient baseline to the best acc-AUC of any
    # of them, past GIM and AttnLRP. Plotting only the 5-step point (what this figure did until
    # now) puts a baseline on the frontier at a setting we know is unconverged, which flatters
    # us; plotting only the 10-step point hides that the converged number is not the one MIB
    # reports. 30 steps stays out: it lands on top of 10 (rho 0.994, zero sign flips) and would
    # be a third label in a 1.65in panel for no visible movement.
    # Labels are "(5)" / "(10)" rather than "(5 steps)" -- the long form is ~1/3 of the panel
    # width at 5.5pt. The dashed segment carries the "same method" reading; the appendix
    # --full figure spells the budgets out.
    (G_GRAD, "RelP+QK"): None,
    (G_GRAD, "IG (5 steps)"): "IG-5",
    (G_GRAD, "IG (10 steps)"): "IG-10",
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
# need proportionally more room to the right. This was 0.70, which left 23% of the panel as
# empty axis -- on a 1.65in figure that is ~0.35in of nothing, next to a heatmap using its full
# width. Measured with the "x headroom" diagnostic in place_labels(): labels first cross the
# right frame between 0.20 and 0.22, so 0.28 keeps them clear with slack for points moving
# under re-evaluation. The old comment claimed 0.34 left "Node Pruning" hanging off; that is
# not reproducible -- 0.35 ends at 0.917, well inside.
#
# The 10-step IG point is now the panel's rightmost (acc-AUC 0.465, past every other gradient
# method), so this briefly went to 0.50 while its label was the long "NAP-IG (10)". Shortening
# to "IG-10" bought it back: swept at the current labels, 0.28 -> 0.956 of the frame, 0.34 ->
# 0.922, 0.50 -> 0.846, none past. Re-tune only if the diagnostic reports "labels past the
# right frame"; do not raise it on suspicion.
#
# 0.34 rather than 0.28 since "+SGD" joined the cut: it lands at the panel's largest x, and at
# 0.28 its label ends at 0.966 of the frame -- inside, but with no slack for the point moving
# under re-evaluation. 0.34 puts it at 0.933.
XPAD_C = 0.34


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
        # Every sweep series is cut to a single point here, so its dashed guide would be a
        # no-op -- except "ig", the one series with two survivors (5 and 10 steps). Keeping it
        # is what makes those read as one method at two budgets rather than two rival methods.
        r["paths"] = [p for p in r["paths"] if p[0] == "ig"]

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
    if "--lr" in sys.argv:
        main_lr()             # always full-page, node only: no edge LR sweep exists
    elif "--both" in sys.argv:
        main_both()           # always full-page: node and edge panels in one figure
    elif "--edge" in sys.argv:
        main_full_edge()      # always full-page; there is no compact edge variant
    elif "--full" in sys.argv:
        main_full()
    else:
        main()
