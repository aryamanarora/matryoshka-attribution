"""Scatter of accuracy-AUC (x) vs faithfulness-AUC (y), one point per (method, loss).

The exception is `Random`, the random-ranking floor, which has no training loss and so draws ONE
point per panel under its own shape -- see LOSSLESS. It is the reference the ordering of every
other series should be read against; without it a panel shows which method wins but not whether
any of them beat chance, which on the zero-ablation row (where x carries a ~0.5 baseline) is the
whole question.

Each point is averaged over TASK-GROUPS: SVA (mean of its 4 subtasks) + Arith (mean of its 4)
+ the 2 MIB tasks (ARC-E, IOI) when present. A panel is (ablation setting) x (substrate x
whether the input node is included in scoring/ablation); only the `node` substrate has the MIB
tasks and the +input variant, so mlp / mlp+attn_head carry SVA and Arith only, no-input.

The layout is a WRAP, ncol=4, ordered so the `Patched` panels fill the first row and the
`Zero-abl.` ones the second -- it reads as a grid but is not one, because facet_grid can
only free scales per row/column and every panel here needs its OWN y (faith-AUC spans 0.6 in
the patched Node panel and 3.0 in the zeroed MLP one). `facet_order` fixes the sequence; the
ablation is the first line of each strip rather than a row label.

`Patched` sets every non-top-k unit to its counterfactual source activation, `Zero-abl.` sets
it to 0. Read the ordering WITHIN a setting and never a point's position across settings --
they are different experiments, not two scorings of one. MAttr is retrained through whichever
intervention it is scored under, and the gradient baselines change estimator outright (I×G ->
Gradient×Input, IG -> textbook zero-baseline IG). The settings agree at only Spearman ~0.44 on
matched cells, which is the reason the zero panels are worth drawing at all. Both axes of the
zero panels are on their own scale: x carries a ~0.5 baseline (a destroyed model still wins the
binary base-vs-source comparison about half the time, and `load` explains why correcting for
that per-run is worse than living with it), and y is inflated ~1.9x because faith-AUC's
(F_clean - F_patch) denominator shrinks when the ablated model is destroyed rather than
flipped. Neither is comparable across settings.

Data: one dir per (ablation x input) cell -- results/sva_sweep, sva_sweep_input, sva_zeroabl,
sva_zeroabl_input. See SOURCES for which methods each carries.
Run:  uv run python plots/plot_accauc_vs_faithauc.py        -> plots/accauc_vs_faithauc.pdf
      uv run python plots/plot_accauc_vs_faithauc.py --all  -> plots/accauc_vs_faithauc_all.pdf
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import palette as P
from plotnine import (
    ggplot, aes, geom_point, geom_path, facet_wrap, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_fill_manual, scale_shape_manual,
    scale_color_manual, guides, guide_legend, expand_limits,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.3),
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
        legend_position="top",
        legend_direction="horizontal",
        legend_box="horizontal",
        legend_box_margin=0,
        legend_margin=0,
    )
)

SVA = ["nounpp", "rc", "simple", "within_rc"]
# goodfire-ai/arithmetic-wild, same model (llama3) and same three substrates as SVA. Kept as a
# SEPARATE group rather than folded into SVA: these are not agreement tasks, and averaging them
# into SVA would hide that they are where the methods separate most (I×G floors at acc-AUC 0.022
# on all four while MAttr reaches ~0.50). As a fourth group each contributes 1/4 of every point.
ARITH = ["addition", "months", "weekdays", "hours"]
# The four task-groups a point may average over, and which of them each SUBSTRATE is REQUIRED to
# have. Every (method, loss) point inside a panel must carry the panel's full required set or it
# is dropped -- see `group_avg`. Without that rule group_avg silently skips a group with no runs,
# so a series whose sweep is still in flight lands on the same axis as a complete one with no
# visible sign of it, and the panel compares two different task populations.
#
# ARC-E and IOI are `node` ONLY, and that is structural, not a gap to be filled. The mlp and
# mlp+attn_head substrates are per-POSITION layouts, so eval_sva.py filters every pair to the
# modal clean-prompt token length (`VARLEN = SPAN or nodes == "node"`, eval_sva.py:579). The MIB
# tasks are variable-length, and measured on llama3 the filter keeps 3.8% of ARC-E (15 of 400
# pairs, 83 distinct lengths) and 28.5% of IOI. A 15-example "ARC-E" would read in the figure as
# a task-group average while being a single-length fluke, so those cells are not run.
GROUPS = [("SVA", SVA), ("Arith", ARITH), ("ARC-E", ["arc_easy"]), ("IOI", ["ioi"])]
# Which MODEL each task is this sweep family's cell for. IOI is qwen2.5 and everything else is
# llama3 -- the pin every submitter in the family carries (submit_input_replication.sh:39,
# submit_sva_cause.sh:42 "ioi is qwen2.5, everything else llama3", submit_sva_dbm.sh:73,
# submit_sva_node_pruning.sh:64), and sva_sweep_input / sva_zeroabl / sva_zeroabl_input hold
# qwen2.5 IOI runs and nothing else.
#
# It has to be ENFORCED here, not assumed. results/sva_sweep also holds a wave of llama3 IOI
# runs (64 files, 2026-08-21), and `load`'s key has no model in it, so before this pin the two
# files for a cell collided and glob order -- the filesystem -- picked the winner. It kept
# qwen2.5 for every series that has both, but `softsgd-log` had ONLY the llama3 run, so this
# figure's headline arm was drawn from a different model than the baselines beside it, with
# nothing on the figure to show it. On IOI that is worth 0.443 vs 0.495 acc-AUC for IG and
# 0.022 vs 0.256 for I×G. The 3 missing qwen2.5 cells were submitted by
# scripts/submit_softsgd_ioi_qwen.sh; until they land, group_avg drops the series and the
# panel report's MISSING column names it.
TASK_MODEL = dict.fromkeys(SVA + ARITH + ["arc_easy"], "llama3")
TASK_MODEL["ioi"] = "qwen2.5"


def on_model(d):
    """True if this run is the canonical model for its task. Every consumer of results/sva_sweep
    must gate on this, whatever its key -- a model-less key COLLIDES (glob order picks a model)
    and a model-bearing one DOUBLE-COUNTS (ioi contributes twice to any task average). It is
    exported rather than restated so the pin cannot drift between the figure and its consumers;
    scripts/method_winrate.py already imports this module for exactly that reason."""
    return d["model"] == TASK_MODEL.get(d["task"], d["model"])


REQUIRED = {"node": ["SVA", "Arith", "ARC-E", "IOI"],
            "mlp": ["SVA", "Arith"],
            "mlp+attn_head": ["SVA", "Arith"]}
# (results dir, input-included label, ablation label). The ablation is the first strip line of
# each panel: `Patched` ablates non-top-k units to the counterfactual source activation,
# `Zero-abl.` sets them to 0. That is a different SETTING, not a rescoring -- MAttr trains
# through it, and the gradient baselines change estimator (I×G -> Gradient×Input, IG ->
# zero-baseline IG) -- so read the ORDERING within a setting, never a point's position across
# them. The two settings agree at only Spearman ~0.44 on matched cells, which is why the zero
# panels are worth drawing.
#
# sva_zeroabl_input carries ONLY this figure's three default series (IG, I×G, MAttr-SGD),
# submitted 2026-08-21 by `ONLY="ig ixg softsgd" OUT=results/sva_zeroabl_input ABLATION=zero
# bash scripts/submit_input_replication.sh`. So `--all` will report the zeroed `+input` panel
# as short: the registry's other methods (MAttr-Adam, Node Pruning, DBM, AttnLRP) were never
# run there. That is a scope choice, not a stalled wave -- the MISSING column in main()'s
# panel report names them, and the same ONLY= line with more arms fills them in.
SOURCES = [("results/sva_sweep", "−input", "Patched"),
           ("results/sva_sweep_input", "+input", "Patched"),
           ("results/sva_zeroabl", "−input", "Zero-abl."),
           ("results/sva_zeroabl_input", "+input", "Zero-abl.")]
SUBSTRATES = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]

# method key -> (display label, colour); order = legend order.
# Colours come from plots/palette.py -- the single source of truth for every figure. Do not
# write hex codes here; plot_accauc_vs_faithauc_cause.py and plot_faith_vs_acc_k1.py read this
# dict directly, and three more figures read palette.py, so a local override desyncs the paper.
METHODS = {
    "IG":         ("IG",           P.color("IG")),
    "IxG":        ("I×G",          P.color("I×G")),
    # "Stepless" IG: alpha ~ U(0,1) drawn per example at m=1, instead of IG's fixed grid. Same
    # integral, unbiased at every m, and at m=1 it costs exactly what I×G costs -- so it sits
    # BETWEEN the two baselines above by construction and the three-way ordering is the point.
    # Tag on disk is `mc_ig_m{draws}_s{seed}`; only the m=1/s=42 arm is drawn (see parse_method).
    "mc_ig":      ("Stepless IG",  P.color("Stepless IG")),
    # Single-pass like I×G (only the backward RULES change): LN-freeze, gated-MLP secant +
    # half-rule, and the uniform half-rule on the QK/OV matmuls. The HF-side implementation is
    # src/learning_to_attribute/grad_attribution.py, verified against vanilla eager attention
    # by scripts/test_attnlrp_hf.py; on MIB the equivalent TransformerLens path is within
    # Spearman 0.96 of GIM (MIB-circuit-track/gim_attnlrp_decomp.py), so this series stands in
    # for the whole LRP family here.
    "AttnLRP":    ("AttnLRP",      P.color("AttnLRP")),
    "stopk-log":  ("MAttr (log)",  P.color("MAttr")),   # headline = soft top-k fwd, log k
    # Same forward and same backward as the headline; Adam -> SGD is the only change. Has its
    # own hex in palette.py (black) rather than sharing MAttr's blue, so it is an ordinary
    # series here; drawn under --sgd only, to keep the default panels legible -- see
    # FIGURE_METHODS. Runs at lr=1.0, off this sweep's shared 0.05 protocol (documented in
    # submit_sva_sweep.sh), which is a real confound with the Adam row and not just a label.
    "softsgd-log": ("MAttr (SGD)", P.color("MAttr (SGD)")),
    "soft-log":   ("+hard (log)",  P.color("+hard")),   # sigmoid-STE hard forward ablation
    # Node Pruning trained through the SAME loss_fn as MAttr (eval_sva.py --method edge_pruning),
    # so its points move along the loss axis like every other series here and the ONLY difference
    # from MAttr is how the mask is parameterized: hard-concrete gates under an annealed L0
    # budget vs top-k. The budget is in the key rather than hidden -- s=0.9 of the SUBSTRATE
    # (MLP neurons, or +attn heads), a far larger unit count than MIB's ~156 nodes, so this is
    # NOT the same absolute circuit size as the results/eprun_node_s0.9 rows in the tables.
    "eprun-s090": ("Node Pruning", P.color("Node Pruning")),
    # The pyvene sigmoid-mask baseline, likewise trained through eval_sva.py's own loss_fn, so
    # the same "only the mask parameterization differs" reading applies: deterministic
    # sigmoid(mask/temp) with temp annealed 50 -> 0.1, vs top-k. The key spells the recipe
    # (lr 0.3, L1 6.0) because that pair, not the method name, decides the circuit -- it is the
    # MIB validation argmax carried over, and a re-swept lr would be a DIFFERENT series.
    "sig_lr0.3_l16.0": ("DBM", P.color("DBM")),
    # The random-ranking floor: score every unit i.i.d. uniform, then run the same eval sweep.
    # Not a competitor -- it is the reference the other series are only interesting relative to,
    # which is why it is grey (see palette.py) and why it sits last in the legend. 3 seeds per
    # cell, averaged in `load`. Filled in for all 68 previously-missing cells by
    # scripts/submit_random_baseline.sh; before that it existed only for the 2 MIB tasks in the
    # 2 patched dirs, so group_avg's all-or-nothing rule dropped it from every panel.
    "Random": ("Random", P.color("Random")),
}
LOSSES = {"acc": "acc", "ce": "CE", "logit_diff": "logit-diff"}
# Methods with NO training loss. A random ranking is not an optimisation, so it exists once per
# cell rather than once per loss -- on disk it carries eval_sva's default `logit_diff`, which is
# a filename artefact, not a fact about the run. Drawing it as a logit-diff square would claim it
# was trained with logit-diff and would put it on the loss trajectory the dashed guide traces, so
# it gets its own shape and is excluded from the guide. One extra legend key, no caption change.
LOSSLESS = {"Random"}
NO_LOSS = "n/a"
# All FILLABLE, and that has to be checked rather than assumed: method is carried by fill, so a
# shape plotnine will not fill silently drops the method encoding. matplotlib lists "X" and "P"
# among its filled markers, but plotnine renders both solid in `color` and ignores `fill` -- the
# first cut used "X" here and Random came out solid BLACK, i.e. indistinguishable from
# MAttr (SGD), the one series it must not be confused with. Verified: o ^ s * p h D d 8 v fill,
# X and P do not. "*" also reads as a footnote mark, which is the right connotation for "n/a".
LOSS_SHAPE = {"acc": "o", "CE": "^", "logit-diff": "s", NO_LOSS: "*"}
# Order the dashed guide visits a method's three points. NOT the legend order (that stays
# LOSSES order) and not sorted by x -- it is the loss's own sharpness ordering, CE (softest
# training signal) -> acc -> logit-diff (hardest), so the line reads as a trajectory rather
# than a shape. geom_path honours row order, which is why the frame is sorted by it.
LOSS_PATH = ["CE", "acc", "logit-diff"]

# Methods drawn in THIS figure. METHODS itself stays the full registry -- it is the shared
# method set/colour map that plot_accauc_vs_faithauc_cause.py and plot_faith_vs_acc_k1.py
# iterate, so deleting a key there would silently drop the series from those figures too.
#
# The default is now the THREE-method cut: our headline arm (MAttr under SGD, the optimiser the
# SVA+ story is about) against the two gradient baselines it is claimed to Pareto-dominate.
# Everything else the registry knows -- MAttr (log) under Adam, +hard, Node Pruning, DBM -- is
# still drawn by `--all`, and is still what the tables report; it is dropped HERE because at
# seven series the facets carried 21 markers each, the legend overflowed \textwidth (the
# logit-diff shape entry clipped), and the mask baselines sat on top of each other.
#
# COVERAGE HAZARD in this cut: "softsgd-log" has no results/sva_sweep_input runs, so the
# `Node, +input` panel draws IG and I×G only. group_avg cannot catch that (it guards missing
# TASKS within a method, not a missing method), so main() checks it explicitly and warns.
FIGURE_METHODS = ["IG", "IxG", "softsgd-log", "Random"]
# `--adam`: the default cut plus MAttr under Adam, i.e. the optimiser contrast on the same axes
# the SVA+ claim is made on. Five series still fits the one-row legend; the full registry does
# not (that is what --all is for, and why it is documented as overflowing).
#
# COVERAGE, and it is asymmetric by DESIGN, not a stalled wave: results/sva_zeroabl_input was
# submitted with ONLY="ig ixg softsgd" (see SOURCES), so `Zero-abl. / Node, +input` draws no
# MAttr-Adam point. main()'s MISSING column names it every run -- do not read that panel's
# absence as Adam failing there.
ADAM_METHODS = ["IG", "IxG", "stopk-log", "softsgd-log", "Random"]
ALL_METHODS = [k for k in METHODS if k not in ("soft-log", "mc_ig")]
# `--stepless`: the default cut plus Stepless IG. It is a SEPARATE cut, and a narrowed one, for a
# coverage reason that cannot be fixed by adding a key to FIGURE_METHODS.
#
# Stepless IG exists ONLY in results/sva_sweep (patched, −input) and ONLY for the four SVA tasks:
# 36 runs = 4 tasks x 3 substrates x 3 losses, submitted 2026-08-22. There are no Arith, ARC-E or
# IOI runs, and none in the other three SOURCES dirs. Under the normal REQUIRED sets every panel
# demands SVA+Arith (and ARC-E+IOI at `node`), so group_avg would drop Stepless IG from all seven
# panels and the figure would come out looking exactly like the default one -- a silent no-op.
#
# So this cut narrows the figure to what the arm actually covers, and says so on the figure: one
# ablation, one input setting, three substrates, task-group = SVA alone. Every OTHER series is
# narrowed with it, so the panels still compare one task population. Read it as a preview of the
# arm, not as a drop-in for the paper figure -- the numbers are not comparable to the default
# cut's, whose points average four task-groups. Backfilling is one submitter away:
#   ARITH_TASKS="addition months weekdays hours" bash scripts/submit_sva_sweep.sh
#   MIB_TASKS=arc_easy bash scripts/submit_sva_sweep.sh
#   MIB_TASKS=ioi MODEL=qwen2.5 bash scripts/submit_sva_sweep.sh
# (each also re-emits the other GRAD arms, but the submitter skips runs already on disk).
STEPLESS_METHODS = ["IG", "IxG", "mc_ig", "softsgd-log", "Random"]
STEPLESS_SOURCES = [("results/sva_sweep", "−input", "Patched")]
STEPLESS_REQUIRED = {sub: ["SVA"] for sub in REQUIRED}


def parse_method(fname, d):
    """Method label from filename tag (mirrors make_fingerprint_tables.parse_method)."""
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    tag = tag.replace("_zeroabl", "")   # ablation is a facet ROW, not a method
    if tag.startswith("conductance") or "fixedk" in tag:
        return None
    # `random_s42` / `random_s43` / `random_s44` -- the seed lives in the TAG and not in the key,
    # so all three land on one key and `load` averages them. Kept ahead of every other branch for
    # the same reason `conductance` is dropped there: no later branch should see these tags.
    if tag.startswith("random"):
        return "Random"
    if tag.startswith("eprun_s"):        # eprun_s090[_ce|_acc] -> one key per budget
        return "eprun-s" + tag.split("_")[1][1:]
    if tag.startswith("sig_"):           # sig_lr0.3_l16.0[_ce|_acc] -> one key per recipe
        return re.sub(r"_(ce|acc)$", "", tag)
    if "hard_topk" in tag:
        if re.search(r"_ig\d+", tag):
            return None
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "unif" if "uniformk" in tag else "log"
        return f"{fam}-{ks}"
    if "sufficient_topk_" in tag:   # soft top-k forward (differentiable, no STE)
        if re.search(r"_ig\d+", tag):
            return None
        ks = "unif" if "uniformk" in tag else "log"
        # The OPTIMIZER has to be in the key. Until 2026-08-21 this branch keyed on the gate and
        # k-schedule only, so when the `topk:sgd` arm landed (72 runs, submit_sva_sweep.sh) every
        # one of them parsed to `stopk-log`/`stopk-unif` and was averaged into the MAttr headline
        # series by group_avg -- 36 SGD runs silently pooled with 78 Adam ones per key, in this
        # figure and in every consumer that imports parse_method. Same failure mode the strict
        # catch-all below was written to prevent, one branch up.
        return f"{'softsgd' if '_topk_sgd' in tag else 'stopk'}-{ks}"
    # Be STRICT here. This used to fall through to "IG" for anything unrecognised, which meant a
    # cause-trained MAttr run (tag `necessary_topk_adam_bs1`, from --mode necessary) would be
    # silently relabelled "IG" and averaged into the IG points. Unknown tags must drop out, not
    # masquerade as a baseline. All `necessary_*` runs are therefore invisible to these figures
    # by design -- they belong in a cause-trained figure of their own.
    # `mc_ig_m{draws}_s{seed}`. Draws and seed are BOTH in the key, so the m=1 arm (the only one
    # that is compute-matched to I×G) can never be averaged with a 10-draw run, and seed
    # replicates for the noise floor stay separate points rather than silently pooling the way
    # Random's three seeds deliberately do. Only m=1/s=42 is in METHODS, so anything else drops.
    # Must precede the `ig` branch below only in spirit -- "mc_ig" does not start with "ig" -- but
    # it is placed with the other gradient tags so the family reads together.
    if tag.startswith("mc_ig"):
        return "mc_ig" if tag.startswith("mc_ig_m1_s42") else None
    if tag.startswith("ixg"):
        return "IxG"
    if tag.startswith("attnlrp"):
        return "AttnLRP"
    if tag.startswith("ig"):
        return "IG"
    return None


def _auc_of(xs, ya):
    """Trapezoid on a log-x grid, normalised by the log span. Mirrors eval_sva.py:738."""
    lx = np.log10(np.asarray(xs, float))
    ya = np.asarray(ya, float)
    return float(np.sum((lx[1:] - lx[:-1]) * (ya[1:] + ya[:-1]) / 2) / (lx[-1] - lx[0]))


def load(res):
    """(method, loss, substrate, task) -> (acc_auc, faith_auc), both as stored.

    NO chance correction, deliberately. The zero row used to be rescaled by

        acc' = (acc - acc[0]) / (1 - acc[0])

    on the theory that acc[0] is the setting's chance floor: zeroing every non-top-k unit
    destroys the model to logit_diff ~ 0, so `acc_base` -- a binary base-vs-source preference --
    sits near 0.5 rather than patching's 0.0. That premise is FALSE, and the correction was
    removed on 2026-08-19.

    `acc[0]` is the accuracy with ONE unit of ~1056 kept clean and the rest zeroed, i.e. a dead
    model, and it is a per-RUN quantity, not a per-cell constant. On SVA/arith it happens to be
    stable (a0 in [0.48, 0.68] across all 99 runs of a task), which is why the correction looked
    benign. On arc_easy under zeroing it is BIMODAL -- a0 takes 0.00, 0.18, 0.30, 0.52, 0.82,
    0.92, 0.93 and 1.00 across the 33 runs, and jumps discontinuously at the next sparsity point
    (0.84 -> 0.11) -- because a destroyed model emits near-identical logits for every example, so
    the `lb > ls` comparison flips coherently for all 100 at once. Dividing by (1 - a0) then
    divides each curve by its own noise draw, and blows up as a0 -> 1: MAttr-CE on arc_easy/zero
    has raw acc-AUC 0.853 with a0 = 0.93, which the correction mapped to -1.10, i.e. it punished
    the method for finding a top-1 node that alone recovers 93% accuracy.

    Any floor worth subtracting has to be SHARED across the methods in a cell (then it is an
    affine map that leaves within-cell ordering alone); a per-run one is not. Raw acc-AUC is
    that, trivially. The cost is that the zero row's x axis carries a ~0.5 baseline and so
    visually flatters it next to the patched row -- which is why the rows must be read
    separately, as the module docstring says. See scripts/method_winrate.py for the same
    reasoning applied to the win-rate tables.

    Files are filtered to TASK_MODEL[task] first -- see that comment for why the model cannot be
    left out of the identity of a cell -- and anything still sharing a key is AVERAGED. The only
    intended collision is Random's 3 seeds; averaging rather than last-wins means a repeated run
    can never depend on glob order again, and a `print` of the group sizes below would show 1
    everywhere but Random.
    """
    runs = {}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        m = parse_method(os.path.basename(f), d)
        if m is None or m not in METHODS:
            continue
        if not on_model(d):
            continue
        runs.setdefault((m, d["loss"], d["nodes"], d["task"]), []).append(
            (d["acc_auc"], d["faith_auc"]))
    return {k: (float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs])))
            for k, vs in runs.items()}


def group_avg(raw, m, loss, sub, required=None):
    """Macro-average over the task-groups required[sub] (default REQUIRED); None if incomplete.

    `required` is a parameter rather than a global read so `--stepless` can narrow every series
    in the figure to one task-group at once. Narrowing it for ONE series would be the exact
    failure this function exists to prevent, so the caller passes one dict for the whole figure.

    Macro-average over groups, not over tasks, so the eight subtasks that come in fours do not
    outvote the two single-task MIB cells.

    All-or-nothing on purpose. This used to average whichever groups happened to be on disk,
    which meant a series whose sweep was still running silently became e.g. an SVA-only point
    plotted on the same axis as a four-group one -- the panel then compared two different task
    populations with nothing on the figure to show it. A group also counts as missing when only
    SOME of its subtasks are present (`set(tasks) <= have`), since a 2-of-4 Arith mean is the
    same failure one level down. Callers report what was dropped rather than swallowing it.
    """
    required = REQUIRED if required is None else required
    have = {t for (mm, ll, ss, t) in raw if (mm, ll, ss) == (m, loss, sub)}
    gx, gy = [], []
    for gname, tasks in GROUPS:
        if gname not in required[sub]:
            continue
        if not set(tasks) <= have:
            return None
        vs = [raw[(m, loss, sub, t)] for t in tasks]
        gx.append(np.mean([v[0] for v in vs]))
        gy.append(np.mean([v[1] for v in vs]))
    if not gx:
        return None
    return float(np.mean(gx)), float(np.mean(gy)), tuple(required[sub])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", dest="draw_all",
                    help="draw the full registry (MAttr-Adam, Node Pruning, DBM) instead of the "
                         "three-method cut; legend overflows \\textwidth at this width")
    ap.add_argument("--adam", action="store_true",
                    help="default cut plus MAttr (Adam), for the optimiser contrast")
    ap.add_argument("--stepless", action="store_true",
                    help="default cut plus Stepless IG; NARROWS the figure to patched/−input and "
                         "to the SVA task-group, which is all that arm has been run on")
    a = ap.parse_args()
    figure_methods = (ALL_METHODS if a.draw_all
                      else STEPLESS_METHODS if a.stepless
                      else ADAM_METHODS if a.adam else FIGURE_METHODS)
    suffix = "_all" if a.draw_all else "_stepless" if a.stepless else "_adam" if a.adam else ""
    sources = STEPLESS_SOURCES if a.stepless else SOURCES
    required = STEPLESS_REQUIRED if a.stepless else REQUIRED

    rows, dropped = [], []
    for res, inp_label, abl in sources:
        raw = load(res)
        for m in figure_methods:
            mlabel = METHODS[m][0]
            for lkey, llabel in LOSSES.items():
                # A lossless method (Random) has one point per cell, not three. It is stored
                # under eval_sva's default logit_diff, so ride that pass and relabel; the other
                # two passes would emit the same point three times under three shapes.
                if m in LOSSLESS:
                    if lkey != "logit_diff":
                        continue
                    llabel = NO_LOSS
                for sub, slabel in SUBSTRATES:
                    # Second strip line names the task-groups the panel averages. It differs by
                    # substrate (node has ARC-E/IOI, the per-position ones structurally cannot),
                    # so putting it in the strip is what stops the two column families from
                    # being read as the same average. Abbreviated to fit the panel width.
                    # The ablation is IN the strip, not a facet_grid row label, because the
                    # grid is now a wrap -- see the facet_wrap comment in the plot spec. THREE
                    # lines, not two with the ablation prefixed: "Patched   MLP+Attn, -input"
                    # is 26 characters and clipped past the right edge of the last panel at
                    # \textwidth/4. Stacked, the longest line is the group list (19), which
                    # already fit.
                    facet = f"{abl}\n{slabel}, {inp_label}\n{'·'.join(required[sub])}"
                    r = group_avg(raw, m, lkey, sub, required)
                    if r is None:
                        have = {t for (mm, ll, ss, t) in raw if (mm, ll, ss) == (m, lkey, sub)}
                        miss = [g for g, ts in GROUPS
                                if g in required[sub] and not set(ts) <= have]
                        if have:   # nothing at all on disk = not submitted; only flag partials
                            dropped.append((abl, facet, mlabel, llabel, "+".join(miss)))
                        continue
                    rows.append(dict(acc_auc=r[0], faith_auc=r[1], method=mlabel,
                                     loss=llabel, facet=facet, ablation=abl,
                                     groups="+".join(r[2])))
    df = pd.DataFrame(rows)

    # ordering for consistent legends / facets (only 4 non-empty substrate x input combos)
    df["method"] = pd.Categorical(df["method"], [METHODS[m][0] for m in figure_methods])
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()) + [NO_LOSS])
    node_g, mlp_g = "·".join(required["node"]), "·".join(required["mlp"])
    # Wrap order, read left-to-right: all four Patched panels, then the three Zero-abl. ones
    # (the zero sweep has no +input arm). ncol=4 below therefore reproduces the old grid's
    # rows without reserving a framed empty cell for the combination that does not exist.
    facet_order = [f"{abl}\n{sub_in}\n{g}"
                   for abl in ("Patched", "Zero-abl.")
                   for sub_in, g in (("Node, −input", node_g), ("Node, +input", node_g),
                                     ("MLP, −input", mlp_g), ("MLP+Attn, −input", mlp_g))]
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in set(df["facet"])])
    df["ablation"] = pd.Categorical(df["ablation"], ["Patched", "Zero-abl."])
    # geom_path connects rows in FRAME order, so the sort below is what defines the line, not
    # a plotnine setting. Sorting by facet/method too keeps each method's three rows contiguous.
    # `ablation` leads the sort so a method's path never runs between the two settings.
    # NO_LOSS is appended rather than left out: pandas warns (and will raise) on values outside
    # the category list, and a lossless method sorts last within its method block -- which costs
    # nothing, since it is one row and the path layer never sees it.
    df["_path"] = pd.Categorical(df["loss"], LOSS_PATH + [NO_LOSS]).codes
    df = df.sort_values(["ablation", "facet", "method", "_path"])

    colors = {METHODS[m][0]: METHODS[m][1] for m in figure_methods}
    lossless = df["method"].isin([METHODS[m][0] for m in LOSSLESS])
    p = (
        ggplot(df, aes("acc_auc", "faith_auc", fill="method", shape="loss"))
        # Dashed guide joining a method's three losses, drawn BEFORE the points so markers sit
        # on top. It carries no information the markers do not -- it groups them, so it is thin,
        # dashed and semi-transparent, and adds no legend entry (the colour scale has guide=None;
        # method is already keyed by fill).
        # Lossless methods are excluded from the frame this layer sees: the guide traces a
        # method's path ACROSS losses, and a one-point group has no path to trace (plotnine would
        # emit a zero-length segment, and ggplot2 the "each group consists of only one
        # observation" warning). Passing filtered data is what keeps the guide's meaning exact.
        + geom_path(aes(color="method", group="method"), data=df[~lossless],
                    linetype="dashed", size=0.3, alpha=0.55, show_legend=False)
        # Black edge on every marker: method is carried by FILL, not colour, so points stay
        # legible where two methods land on top of each other and against the grid lines.
        # alpha=1 -- a translucent fill under a black edge reads as a different, muddier colour
        # wherever markers overlap, which is exactly where the distinction has to hold.
        + geom_point(data=df[~lossless], size=1.9, color="#000000", stroke=0.3)
        # Random gets its OWN layer purely for marker geometry. A star packs less fill area into
        # its bounding box than o/s/^, so at the shared 1.9pt its #cccccc would read darker than
        # the other series rather than lighter -- backwards for a marker that is meant to read as
        # hollow. Size is not an aesthetic here (nothing is mapped to it), so a second layer is
        # the only way to vary it per series; both layers keep show_legend on so the Method and
        # Loss keys are still assembled from the shared scales.
        + geom_point(data=df[lossless], size=3.6, color="#000000", stroke=0.2)
        # WRAP, not grid, and that is the whole point of the layout. Under facet_grid,
        # `scales="free"` frees x per COLUMN and y per ROW -- it is never per panel -- so all
        # four Patched panels shared one y axis, and the single largest point in the row
        # (MAttr's ~2.2 on MLP) set the scale for the Node panels where nothing exceeds 0.9.
        # facet_wrap's free scales ARE per panel. The cost is losing the row/column strips;
        # `facet` now carries the ablation in its own label and `facet_order` fixes the
        # left-to-right sequence so the wrap still reads as the old 4+3 grid.
        + facet_wrap("~facet", ncol=min(4, df["facet"].nunique()), scales="free")
        + expand_limits(x=0, y=0)  # anchor each free axis at 0 (upper stays per-facet)
        + scale_fill_manual(values=colors, name="Method")
        + scale_color_manual(values=colors, guide=None)   # line colour only; no second legend
        + scale_shape_manual(values=LOSS_SHAPE, name="Loss")
        + labs(x="IIA AUC (↑)", y="Faith AUC (↑)")
        + guides(fill=guide_legend(order=1, nrow=1), shape=guide_legend(order=2, nrow=1))
    )
    # The global figure_size is sized for the default TWO rows of panels. `--stepless` draws one
    # row (patched/−input only), so keeping 3.3in would stretch three panels to twice the height
    # of every other version of this figure and make the same points look like a different result.
    if df["facet"].nunique() <= 4:
        p += theme(figure_size=(5.5, 2.1))
    out = f"plots/accauc_vs_faithauc{suffix}.pdf"
    p.save(out, dpi=300, verbose=False)
    # PNG sibling for eyeballing the result without a PDF viewer, as the cause figure and the
    # iso-vs-cause curves already do. Only the PDF is copied into paper/figs.
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print("wrote", out, f"({len(df)} points)")
    # Every panel must show ONE group set (group_avg enforces it) and the full method x loss
    # grid. A short count is a coverage hole, not a styling choice, so print both.
    # Per-method, not len(methods) x len(LOSSES): a lossless method contributes ONE point, so a
    # flat product would report every complete panel as permanently one short.
    n_full = sum(1 if m in LOSSLESS else len(LOSSES) for m in figure_methods)
    cov = df.groupby(["ablation", "facet"], observed=True).agg(
        n=("groups", "size"), groups=("groups", lambda s: " / ".join(sorted(set(s)))))
    cov["of"] = n_full
    # A method with NO runs at all for a substrate/input combo is invisible to group_avg (which
    # guards missing tasks within a method, not a missing method), so a panel can silently draw
    # a smaller method set than its neighbours. With the three-method cut that is not cosmetic:
    # MAttr (SGD) has no sva_sweep_input runs, so `Node, +input` would show the two BASELINES
    # and no MAttr, i.e. exactly the panel a reader would misread as a loss.
    cov["methods"] = df.groupby(["ablation", "facet"], observed=True)["method"].agg(
        lambda s: ",".join(m for m in [METHODS[k][0] for k in figure_methods]
                           if m not in set(s)) or "-")
    cov = cov.rename(columns={"methods": "MISSING"})
    print("\npoints and task-groups per panel:")
    print(cov.to_string())
    if dropped:
        # Partial cells: runs exist for this method/loss/substrate but not for every subtask of
        # every required group, so the point would have been an average over a smaller task
        # population than its neighbours. Listed rather than silently omitted -- this is the
        # to-run list, and an empty list is the signal the figure is ready for the paper.
        print(f"\nDROPPED {len(dropped)} partial cells (missing task-groups):")
        for abl, facet, m, loss, miss in sorted(dropped):
            # facet carries the multi-line strip label; flatten it so the report stays tabular.
            print(f"  {abl:10s} {facet.split(chr(10))[1]:18s} {m:14s} {loss:11s} missing {miss}")
    else:
        print("\nno partial cells: every panel is complete.")


if __name__ == "__main__":
    main()
