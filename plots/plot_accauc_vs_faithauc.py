"""Scatter of IIA-AUC (x) vs faithfulness-AUC (y), one point per (method, loss).

THE DEFAULT CUT IS THE PAPER FIGURE and was narrowed on 2026-08-29: PATCHED ablation only,
LOGIT-DIFF loss only, six methods (two gradient baselines, both Adam arms, SGD, Random). That
makes it six panels (3x2) of 5-7 points -- the two SAE columns carry 5, since Node Pruning and
DBM have no SAE runs. The other flags (--adam / --all / --stepless) keep the old
4-source x 3-loss grid and are exploration; see FIGURE_METHODS for what was dropped and why,
which is worth reading before widening it back.

`Random`, the random-ranking floor, has no training loss and draws ONE point per panel under its
own star -- see LOSSLESS. It is the reference the ordering of every other series should be read
against; without it a panel shows which method wins but not whether any of them beat chance.
In the patched half that floor is ~0.03, so the margins here are real; in the zero-ablation half
it is ~0.51, which is the main reason those panels are no longer in the default cut.

Each point is averaged over TASK-GROUPS: SVA (mean of its 4 subtasks) + Arith (mean of its 4)
+ the 2 MIB tasks (ARC-E, IOI) when present. A panel is (ablation setting) x (substrate x
whether the input node is included in scoring/ablation); only the `node` substrate has the MIB
tasks and the +input variant, so mlp / mlp+attn_head carry SVA and Arith only, no-input.

The layout is a WRAP (LAB_NCOL) -- it reads as a grid but is not one, because facet_grid can only
free scales per row/column and every panel here needs its OWN y (faith-AUC spans 0.9 in the Node
panels and 2.4 on MLP+Attn). `facet_order` fixes the sequence. The ablation is the first line of
each strip only when more than one is drawn, so the default cut does not print "Patched" four
times.

READ THE ORDERING WITHIN A PANEL, never a point's position across panels. The substrates are
different experiments with different unit counts, and under --all/--adam the two ablation
settings are different experiments too: MAttr is retrained through whichever intervention it is
scored under, and the gradient baselines change estimator outright (I×G -> Gradient×Input, IG ->
textbook zero-baseline IG).

MARKERS ARE UNIFORM CIRCLES in the default cut and the series are named by direct labels rather
than a legend, matching plots/plot_mib_accauc_cpr_scatter.py, whose label machinery this file
imports. Where two points coincide -- the MAttr arms do, at the node substrate -- they draw as
one circle with two leader lines out of it, which is the honest picture.

Data: one dir per (ablation x input) cell -- results/sva_sweep, sva_sweep_input, sva_zeroabl,
sva_zeroabl_input. See SOURCES for which methods each carries.
Run:  uv run python plots/plot_accauc_vs_faithauc.py        -> plots/accauc_vs_faithauc.pdf
      uv run python plots/plot_accauc_vs_faithauc.py --adam -> ..._adam.pdf   (all 4 settings,
                                                               all 3 losses, both Adam arms)
      uv run python plots/plot_accauc_vs_faithauc.py --all  -> plots/accauc_vs_faithauc_all.pdf
"""
import argparse
import glob
import json
import os
import re
import sys

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
# scripts/sva/launch/submit_softsgd_ioi_qwen.sh; until they land, group_avg drops the series and the
# panel report's MISSING column names it.
TASK_MODEL = dict.fromkeys(SVA + ARITH + ["arc_easy"], "llama3")
TASK_MODEL["ioi"] = "qwen2.5"


def on_model(d):
    """True if this run is the canonical model for its task. Every consumer of results/sva_sweep
    must gate on this, whatever its key -- a model-less key COLLIDES (glob order picks a model)
    and a model-bearing one DOUBLE-COUNTS (ioi contributes twice to any task average). It is
    exported rather than restated so the pin cannot drift between the figure and its consumers;
    scripts/sva/method_winrate.py already imports this module for exactly that reason."""
    return d["model"] == TASK_MODEL.get(d["task"], d["model"])


REQUIRED = {"node": ["SVA", "Arith", "ARC-E", "IOI"],
            "mlp": ["SVA", "Arith"],
            "mlp+attn_head": ["SVA", "Arith"],
            # Llama-Scope residual SAE latents, one score per (layer, position, latent). SVA and
            # Arith only, for the same structural reason the per-position substrates are: it is a
            # span layout, so eval_sva filters to the modal prompt length and the two
            # variable-length MIB tasks are not run there.
            "resid_sae_span": ["SVA", "Arith"],
            # Llama-Scope MLP-OUTPUT SAE (LXM). Same span layout and therefore the same task
            # coverage as the residual one -- keep the two lists identical, since the whole point
            # of having both columns is that they differ ONLY in the site.
            "mlp_sae_span": ["SVA", "Arith"]}
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
# bash scripts/sva/launch/submit_input_replication.sh`. So `--all` will report the zeroed `+input` panel
# as short: the registry's other methods (MAttr-Adam, Node Pruning, DBM, AttnLRP) were never
# run there. That is a scope choice, not a stalled wave -- the MISSING column in main()'s
# panel report names them, and the same ONLY= line with more arms fills them in.
SOURCES = [("results/sva_sweep", "−input", "Patched"),
           ("results/sva_sweep_input", "+input", "Patched"),
           ("results/sva_zeroabl", "−input", "Zero-abl."),
           ("results/sva_zeroabl_input", "+input", "Zero-abl.")]
SUBSTRATES = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]
# The SAE column is added to the DEFAULT cut only, not to the module-level SUBSTRATES, because
# that list is iterated by plot_accauc_vs_faithauc_cause.py and scripts/sva/method_winrate.py --
# widening it there would silently add an SAE column to two other artifacts.
#
# NOT EVERY METHOD IS ON IT. Node Pruning and DBM have no SAE runs at all, so group_avg drops
# them from this panel and main()'s MISSING column names them every run. That is the documented
# behaviour for a method with no data in a cell, and it is why the panel shows 5 series where
# its neighbours show 7 -- do not read the absence as a score.
FIGURE_SUBSTRATES = SUBSTRATES + [("mlp_sae_span", "SAE (MLP out)")]
# THE SAE COLUMN READS A DIFFERENT RESULTS TREE, keyed by substrate. results/sva_sweep's SAE runs
# use the LEGACY `absorb` error intervention, under which keeping one reconstruction-error node
# restores its whole (layer, position) site regardless of the latents -- a master switch, and
# MAttr found it (100% of its top 10 were error nodes; acc-AUC 0.771 on addition against 0.445
# under `frozen`). So that column was scoring a shortcut, not an attribution.
#
# results/sva_sweep_ferr is the same 8 tasks and the same 5 series re-run with --sae-error frozen,
# where each error term is measured against its OWN reconstruction and the node is an ordinary
# scored unit. `frozen` shares `absorb`'s endpoints exactly (F_clean 6.672 / F_patch -6.651 on
# addition), so the column stays comparable to its neighbours -- which the `none` setting would
# NOT have been, since dropping the node moves F_patch and inflates every score (Random alone
# goes 0.023 -> 0.142 acc-AUC on nothing but an easier k=0 state).
#
# The override applies to EVERY source dir. The zero-ablation and +input sweeps have no frozen
# SAE arm, so the SAE column is simply absent from those panels rather than silently falling back
# to absorb data -- which is the behaviour we want: a missing panel is readable, a mixed one is
# not.
# 5,000 TRAINING STEPS, NOT THE SWEEP'S 2,000 (2026-09-03, requested). Measured on this basis,
# 2k undertrains MAttr on the ARITHMETIC half: acc-AUC 0.563 -> 0.580 (Adam) and 0.574 -> 0.584
# (SGD), with the gain concentrated in months (+0.047/+0.044), weekdays and hours while the four
# SVA tasks move within single-seed noise.
#
# *** THE COST IS ON THE FAITH AXIS AND IT IS NOT SYMMETRIC BETWEEN THE ARMS. *** Adam pays
# faith-AUC 1.49 -> 2.06 for its +0.017 (rc reaches 3.66, i.e. recovering 3.7x the clean logit
# difference); SGD pays 0.76 -> 0.84 for +0.011 and stays under 1. Read the extra Adam accuracy
# with that in mind -- it is bought partly with over-recovery.
#
# The tree holds the 5k MAttr runs plus SYMLINKS to the untrained baselines, which have no step
# count and so are the same runs as the 2k tree's. One budget per method per tree is what makes
# parse_method's `_s\d+` strip safe.
SUBSTRATE_RES = {"mlp_sae_span": "results/sva_sweep_ferr5k",
                 # MLP and MLP+Attn bumped to 5k too (2026-09-03), so the three TRAINED-substrate
                 # columns share a budget. Same finding as the SAE basis and now confirmed three
                 # times: 2k undertrains MAttr on the ARITHMETIC tasks only. Per-arm means over 8
                 # tasks, 2k -> 5k: mlp Adam 0.602 -> 0.614, mlp SGD 0.596 -> 0.612, mlp+attn Adam
                 # 0.606 -> 0.617, mlp+attn SGD 0.610 -> 0.615 -- with all 16 SVA cells inside
                 # +-0.008 and 13 of 16 arithmetic cells gaining (weekdays +0.067, hours +0.040).
                 #
                 # ADAM PAYS FOR IT ON THE FAITH AXIS AND SGD LARGELY DOES NOT: mlp Adam faith
                 # 1.25 -> 2.33 and mlp+attn Adam 1.72 -> 2.83 (one cell at 4.16), against SGD's
                 # 0.91 -> 1.04 and 0.86 -> 1.29. Above 1 is over-recovery, so read Adam's extra
                 # accuracy as partly bought with gap padding.
                 #
                 # BUDGETS ARE NOW MATCHED ACROSS TRAINED METHODS AND ATTRIBUTION PASSES
                 # (2026-09-06): MAttr, Node Pruning and DBM all train 5k steps in this tree
                 # (`_s5000` tags; the NP/DBM 2k logit-diff symlinks were removed when those
                 # landed), and the gradient baselines attribute at the same PASS budget
                 # (5,000, capped at the train pool; IG spends it as 500 ex x 10 path points).
                 # The acc/ce arms of every trained method remain 2k. `node` stays at 2k
                 # throughout -- its eps x lr grid is flat (0.477-0.526 over 24 cells), so it
                 # is not budget-limited.
                 "mlp": "results/sva_sweep_5k",
                 "mlp+attn_head": "results/sva_sweep_5k"}
# The 10x-steps twin of the headline (scripts/sva/launch/submit_unifk_eps_50k_sc.sh, 2026-09-17):
# 50k steps, everything else the headline's, in its own trees for the reason SUBSTRATE_RES gives.
# Only these three substrates were run; the key is absent from the node column.
TENX_KEY, TENX_BASE = "stopk-unif-eps1e-2-10x", "stopk-unif-eps1e-2"
TENX_KEYS = {"stopk-unif-eps1e-2": TENX_KEY, "stopk-unif": "stopk-unif-10x"}   # base key -> 10x key
TENX_RES = {"mlp": "results/sva_sweep_50k", "mlp+attn_head": "results/sva_sweep_50k",
            "mlp_sae_span": "results/sva_sweep_ferr50k"}
# SAE (resid) WAS THE SECOND COLUMN HERE AND WAS DROPPED (2026-09-03, requested). It is still in
# REQUIRED and in --all; only the default cut lost it. The old note read:
#
# The two SAE columns are NOT interchangeable and the figure is worth reading for their contrast.
# Same Llama-Scope 8x dictionary, same 5,243,040 units, same tasks -- only the site differs, and
# MAttr's verdict flips with it: on addition it scores 0.111 vs IG's 0.302 on the residual stream
# and 0.774 vs IG's 0.506 on MLP outputs. The residual cell is depressed by a heavy-tail gradient
# outlier that freezes the mask (one 17,000x spike at a single k-draw); the MLP-output cell has no
# such spike and MAttr wins there on plain defaults. So do not average the two into one "SAE"
# facet -- that would hide the single largest substrate effect in this figure.

# method key -> (display label, colour); order = legend order.
# Colours come from plots/palette.py -- the single source of truth for every figure. Do not
# write hex codes here; plot_accauc_vs_faithauc_cause.py and plot_faith_vs_acc_k1.py read this
# dict directly, and three more figures read palette.py, so a local override desyncs the paper.
METHODS = {
    "IG":         ("IG",           P.color("IG")),
    "IxG":        ("I×G",          P.color("I×G")),
    # Expected Gradients: alpha ~ U(0,1) drawn per example at m=1, instead of IG's fixed grid. Same
    # integral, unbiased at every m, and at m=1 it costs exactly what I×G costs -- so it sits
    # BETWEEN the two baselines above by construction and the three-way ordering is the point.
    # Tag on disk is `mc_ig_m{draws}_s{seed}`; only the m=1/s=42 arm is drawn (see parse_method).
    "mc_ig":      ("Expected Gradients",  P.color("Expected Gradients")),
    # Single-pass like I×G (only the backward RULES change): LN-freeze, gated-MLP secant +
    # half-rule, and the uniform half-rule on the QK/OV matmuls. The HF-side implementation is
    # src/learning_to_attribute/grad_attribution.py, verified against vanilla eager attention
    # by scripts/tools/test_attnlrp_hf.py; on MIB the equivalent TransformerLens path is within
    # Spearman 0.96 of GIM (MIB-circuit-track/gim_attnlrp_decomp.py), so this series stands in
    # for the whole LRP family here.
    "AttnLRP":    ("AttnLRP",      P.color("AttnLRP")),
    # GIM has NO SVA+ runs; it is in the registry only so the --cpr cut's MIB panels can draw
    # it (gim_eval on the test split). parse_method never returns it, and it is kept out of
    # ALL_METHODS below, so no other cut is affected.
    "GIM":        ("GIM",          P.color("GIM")),
    # Uniform-k twin of the eps arm, 5k steps (scripts/sva/launch/submit_unifk_eps_5k.sh). In
    # the --cpr cut only; the bar chart's "unif k" green (P.METHOD["+hard"], the hue
    # plot_mib_test_avg / plot_mib_accauc_cpr_scatter give uniform-k). Kept out of ALL_METHODS
    # like GIM so no other cut changes.
    "stopk-unif-eps1e-2": ("MAttr (Adam, unif k)", P.METHOD["+hard"]),
    # The headline trained 10x longer (50k steps on the SVA+ substrates, 5k on MIB node): a
    # SYNTHETIC key -- parse_method strips the budget suffix, so these runs parse as the
    # headline key and are told apart by TREE (TENX_RES), not by tag. See tenx_for() in main.
    "stopk-unif-eps1e-2-10x": ("MAttr (10×)", P.METHOD["+hard"]),
    # Same 50k budget at Adam's DEFAULT eps (EPS=1e-8 in the launcher). Registered so it is one
    # list entry from any cut; not in CPR_METHODS.
    "stopk-unif-10x": ("MAttr (10×, ε=10⁻⁸)", P.METHOD["MAttr (Adam, default eps)"]),
    # MAttr without learning (optimizer=none): the k-averaged gradient at zero scores, i.e. the
    # first-step update of the gradient appendix by Monte Carlo. Uniform k = the headline twin.
    # frozenid-*: soft forward + IDENTITY backward (masks.py topk_identity) -- the sigma' gate
    # slope and the centring are dropped, so this IS IG along the mask path. frozen-*: the same
    # with the soft top-k Jacobian (MAttr's literal first-step gradient).
    "frozenid-unif": ("MAttr (no learning)", P.METHOD["MAttr (Adam, default eps)"]),
    "frozenid-log":  ("MAttr (no learning, log k)", P.METHOD["MAttr (Adam, default eps)"]),
    "frozen-unif": ("MAttr (no learning, soft bwd)", P.METHOD["MAttr (Adam, default eps)"]),
    "frozen-log":  ("MAttr (no learning, soft bwd, log k)", P.METHOD["MAttr (Adam, default eps)"]),
    "stopk-log":  ("MAttr (log)",  P.color("stopk-log")),   # default-eps Adam: violet
    # Same forward, same k-schedule, same lr (0.05, the sweep's shared protocol), same optimizer
    # as "MAttr (log)" directly above -- Adam's eps raised 1e-8 -> 1e-2 is the ONLY difference,
    # which is what makes the gap between the two attributable to eps and nothing else.
    #
    # It is an identity change, not a numerical one. At the neuron substrates nearly every
    # per-step |grad| exceeds the default eps, so Adam's update collapses to ~sign(g)*lr: every
    # unit takes the same size step whatever its effect size, and the score degenerates into a
    # signed count of steps. Above the typical |g| magnitude weighting comes back. Measured on
    # addition/llama3/mlp that is acc-AUC 0.388 -> 0.490, with the top-k overlap against IG
    # going 0.08 -> 0.73 (scripts/sva/launch/submit_adam_eps_followup.sh). The prediction this figure
    # tests is that it is NEUTRAL at the node substrate, where per-unit gradients are far
    # larger -- the node control already on disk reads 0.525 either way.
    #
    # LOGIT-DIFF ONLY: one point per panel, not the three-loss trajectory the other MAttr
    # series draw, so it carries the logit-diff square and no dashed guide (see `pathable`).
    "stopk-log-eps1e-2": ("MAttr (Adam, ε=10⁻²)", P.color("stopk-log-eps1e-2")),
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
    # DBM's WHOLE lambda ladder scored at each run's own converged L0 (scripts/mib/
    # dbm_multisparsity.py), not one run swept over MIB's grid. MIB-panel-only, like GIM: there
    # is no SVA+ equivalent and it is kept out of ALL_METHODS so no other cut sees it.
    #
    # *** IT IS NOT THE SAME KIND OF OBJECT AS EVERY OTHER POINT ON THIS PANEL, and the caption
    # must say so. *** Eight separately trained masks supplying one budget each, against every
    # other point's single ranking that supplies all ten. The table carries that as a 21k cost
    # against 3k. Plotted here because the comparison is worth making, not because it is like
    # for like.
    "dbm-multisp": ("DBM (multi-sparsity)", P.color("DBM")),
    # The random-ranking floor: score every unit i.i.d. uniform, then run the same eval sweep.
    # Not a competitor -- it is the reference the other series are only interesting relative to,
    # which is why it is grey (see palette.py) and why it sits last in the legend. 3 seeds per
    # cell, averaged in `load`. Filled in for all 68 previously-missing cells by
    # scripts/sva/launch/submit_random_baseline.sh; before that it existed only for the 2 MIB tasks in the
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
# Methods run at ONE loss (logit_diff) rather than the full three. Unlike LOSSLESS this is a
# COVERAGE fact, not a fact about the method: these runs have a real training loss and carry its
# square, there simply is no ce/acc wave for them. The distinction matters in exactly one place
# -- main()'s per-panel completeness count, which must expect 1 point from them and 3 from a
# three-loss method, or every complete panel reports as short forever. That is the same trap the
# count already sidesteps for LOSSLESS; it is spelled out as its own set because the reasons
# differ and because "run it at the other two losses" would empty this one, whereas nothing will
# ever give Random a loss.
SINGLE_LOSS = {"stopk-log-eps1e-2"}
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
# *** THE DEFAULT CUT WAS NARROWED ON 2026-08-29 (requested) TO: patched only, logit-diff only,
# six methods. *** It is the figure that ships as fig:acc-faith; every other cut below is an
# exploration and keeps the full 4-source / 3-loss grid.
#
# What went and why:
#
#  ZERO-ABLATION PANELS, dropped outright. Its IIA AUC does not measure much: the Random floor
#  there is 0.513, against 0.027 in the patched half, because zeroing every non-top-k unit drives
#  the model to near-constant logits and the binary base-vs-source comparison the metric
#  integrates becomes a coin flip. Over its 78 cells per method the BEST method clears that floor
#  by +0.09 on average and I x G by +0.00-0.03, with 22-34 cells per method AT OR BELOW their own
#  floor. Those points sat on the same axes as the patched ones with nothing marking the
#  difference in floor, so the row read as a second result when it was mostly headroom. It is
#  still drawn by --all / --adam, where the strip names the ablation.
#
#  THE OTHER TWO LOSSES, dropped. With one loss each method is ONE point per panel, so the four
#  panels carry 6 markers instead of 18 and the loss-robustness story moves to the figure that
#  can actually show it (plots/plot_sva_robustness_grid.py, which has a column per loss). This
#  also retires the dashed guide and the Loss legend from this figure -- see `losses` in main().
#
# ONE ADAM POINT, and it is the eps=1e-2 one. The default-eps arm (`stopk-log`) was drawn here
# briefly so the eps gap sat on the main axes, and was dropped again on 2026-08-29: this figure
# compares METHODS, and default-eps Adam is a broken configuration of one of them rather than a
# method in its own right, so a point for it invites the reader to rank it. The gap it shows is
# a claim about our own optimiser setting, which belongs in the optimiser section -- `--adam`
# draws both arms side by side and is the cut for it.
#
# WHICH MEANS THE ADAM POINT HERE IS AT eps=1e-2, NOT THE LIBRARY DEFAULT. That is a real
# hyperparameter choice and it is invisible on the figure (the label is just MAttr^A), so the
# caption has to say it. At the neuron substrates it is worth +0.10 IIA AUC over the default.
# 2026-08-29: AttnLRP, Node Pruning and DBM added (requested). All three already had 36/36 of
# this cut's cells, so nothing had to be run -- they were absent because the cut had been
# narrowed for legibility back when it carried three losses and every method cost three
# markers per panel. At one loss they cost one each, so the mask-learning family and the
# LRP baseline fit. Expected Gradients is the one registry method still missing (12/36; the backfill
# is scripts/sva/launch/submit_stepless_backfill.sh) -- add it here once those land.
# AttnLRP dropped again 2026-08-29 (requested). It has full coverage and is still in the
# registry, so `--all` keeps drawing it; it is out of THIS cut only.
# Expected Gradients is deliberately OUT of this cut (2026-08-29, requested) even though its backfill
# landed and it now has 36/36 patched logit-diff cells. It tracks IG to within 0.009-0.018 IIA
# AUC in every panel, so it costs an eighth marker and 0.25in of height to draw a point that
# sits on top of one already there. The numbers belong in the prose. `--all` still draws it.
FIGURE_METHODS = ["IG", "IxG", "eprun-s090", "sig_lr0.3_l16.0",
                  "stopk-log-eps1e-2", "softsgd-log", "Random"]
# Sources and losses for the default cut. The other cuts fall back to the module-level SOURCES
# and LOSSES, so widening this one is a two-line change and cannot silently widen those.
# DEFAULT CUT IS -input ONLY (2026-08-30, requested). The +input arm was dropped rather than
# kept as extra panels because for the GRADIENT baselines it is not a different attribution at
# all -- verified bit-identical on node / resid_sae_span / mlp_sae_span, it only appends one more
# scored unit -- so those panels duplicated their -input twins. It still exists on disk
# (results/sva_sweep_input) and --all / --adam still draw it.
# `--cpr`: the default cut plus the three remaining node-level gradient baselines of the MIB
# test table. Expected Gradients and AttnLRP have SVA+ runs on mlp / mlp+attn_head only (no SAE
# runs), GIM has none, so those panels carry fewer points -- the report says which.
# ONE MAttr OPTIMISER (2026-09-08, requested): the SGD arm is dropped, so the two remaining
# ours-points are Adam at the two k-schedules, drawn and labelled exactly as the bar chart
# plots/plot_mib_test_avg.py draws them. `softsgd-log` stays in METHODS and in MIB_TEST -- it
# is one list entry away from coming back -- it is simply not in this cut.
# "dbm-multisp" was in this cut 2026-09-09 to 2026-09-11 (beside the single-lambda DBM point:
# same method, one run vs eight, +0.38 CPR / +0.09 IIA at node level) and was dropped again
# on request. It stays in the registry, MIB_TEST and FAMILY_COLOR, one list entry from
# coming back.
#
# WHICH MAttr ARM IS "MAttr" HERE (2026-09-11, requested): the UNIFORM-k Adam arm is the
# figure's MAttr -- it carries the star and the plain label -- and the log-k Adam arm is
# drawn as its ablation, a circle labelled "+log k". That matches plot_mib_test_avg.py,
# where the uniform-k bar is the only MAttr bar since the same day. It is the OPPOSITE of
# the default cut (accauc_vs_faithauc.pdf), where "MAttr" is still the log-k eps arm; the
# swap is applied inside main()'s --cpr branch (CPR_STARS / CPR_POINT_LABEL) precisely so
# the default cut does not move.
# EG ("mc_ig") was in this cut for a few hours on 2026-09-18 and was dropped (requested): it
# sits on top of IG in every panel. One list entry to bring back.
CPR_METHODS = ["IG", "IxG", "eprun-s090", "sig_lr0.3_l16.0",
               "stopk-log-eps1e-2", "stopk-unif-eps1e-2", "Random"]
# Per-method markers for the --cpr cut (2026-09-18, requested): IG vs I x G and NP vs DBM
# were the same glyph in the same family colour and needed point labels to tell apart; with
# distinct glyphs the legend carries the names and only the MAttr star keeps a label
# (2026-09-19, requested: the "+log k" label went too -- the filled circle is in the legend).
CPR_MARKERS = {"IG": "o", "IxG": "^", "eprun-s090": "s", "sig_lr0.3_l16.0": "D", "Random": "o",
               "stopk-log-eps1e-2": "o", "AttnLRP": "v"}   # AttnLRP: --full only (2026-09-21)
CPR_LABELLED = {"stopk-unif-eps1e-2"}
# The 10x-steps twin ("stopk-unif-eps1e-2-10x") was in this cut 2026-09-17 and was dropped the
# same day (requested): its row lives in the SVA+ / MIB tables and its bar in plot_mib_test_avg.
# Everything below still supports it -- one list entry to bring it back. Both budgets are stars.
CPR_STARS = {"stopk-unif-eps1e-2", "stopk-unif-eps1e-2-10x"}
CPR_FILLED = {"stopk-log-eps1e-2"}
# Labels anchored LEFT of their marker, key -> facet-label prefixes where it applies. The
# MIB panels stack five points up the right side; at the reduced --cpr height the star's
# "MAttr" label was repelled a third of the axis below its marker on a leader line, while
# the top-left of both panels is empty. "NP" moves left too so it does not take the space
# the star's label needs. Set in main()'s --cpr branch via LABEL_LEFT; default cut untouched.
CPR_LABEL_LEFT = {"stopk-unif-eps1e-2": ("MIB (node", "MIB (edge"), "eprun-s090": ("MIB (node",)}
LABEL_LEFT = {}
CPR_POINT_LABEL = {"stopk-unif-eps1e-2": "MAttr", "stopk-log-eps1e-2": "$+$log $k$",
                   "stopk-unif-eps1e-2-10x": "$+$10$\\times$ steps"}
# Mask-learning methods draw as SQUARES (gradient = circles, MAttr = stars), the same
# gradient/mask shape split plot_mib_accauc_cpr_scatter.FAMILY_SHAPE uses.
MASK_KEYS = {"eprun-s090", "sig_lr0.3_l16.0", "dbm-multisp"}
# The keys drawn as STARS in draw_labelled. Module-level so a cut can narrow it (the --cpr
# branch sets it to CPR_STARS); everything not a star and not in MASK_KEYS is a circle.
STAR_KEYS = {"stopk-log-eps1e-2", "softsgd-log", "stopk-unif-eps1e-2", "stopk-unif-eps1e-2-10x"}
# When True, every NON-star marker is drawn as an outline in its method colour and only the
# stars are filled -- the star is then the one filled thing on the panel (2026-09-11,
# requested, --cpr cut only; set in main()'s --cpr branch). The "+log k" circle counts as
# non-star here on purpose: the point is that ONE marker per panel is filled.
OUTLINE_NON_STAR = False
OUTLINE_LW = 0.8
# Non-star keys that stay FILLED when OUTLINE_NON_STAR is on. The --cpr branch sets it to
# CPR_FILLED: the "+log k" circle is MAttr's own ablation and reads as ours, so it is solid
# like the star; every baseline is an outline.
FILLED_KEYS = set()
# Methods that exist on SOME panels only, by construction rather than by coverage gap:
# key -> the facet-label prefixes where the method IS expected. report() uses this so a
# panel that cannot draw a method does not report itself short forever and name it MISSING
# -- the "report that is never green" report()'s own comment warns about.
# dbm-multisp: no SVA+ runs at all (it is a MIB ladder), and NODE level only -- there is no
# entry in MIB_TEST_EDGE and no edge DBM on disk. A key whose method DOES exist on a panel
# listed here would silently stop that panel reporting a real gap, so keep this exact.
MIB_ONLY = {"dbm-multisp": ("MIB (node",)}


def expected_on(m, facet):
    """Is method `m` supposed to appear on `facet`? True for everything not in MIB_ONLY."""
    return m not in MIB_ONLY or facet.startswith(MIB_ONLY[m])
FIGURE_SOURCES = [s for s in SOURCES if s[2] == "Patched" and "_input" not in s[0]]
FIGURE_LOSSES = {"logit_diff": LOSSES["logit_diff"]}
# Point labels for the directly-labelled default cut. SHORT on purpose: the labels sit inside
# the panel and four panels share \textwidth, so "MAttr (Adam, ε=10⁻²)" at 6.5pt is ~0.75in
# against a ~2.2in panel -- a third of the axis spent on one word. The legend spelled the
# optimiser out because it had a whole strip; a label has to earn its width, and colour still
# carries the identity. Keys not listed fall back to the legend label.
# The eps arm WRAPS. In a 1x4 full-width layout each panel gets ~1.1in of axis, and
# "MAttr+Adam, ε=10⁻²" at 6pt Inter is ~0.79in of it -- anchored at a point that already sits
# 60% across, it runs off the frame at any padding worth spending.
#
# TWO lines, never three. repel() separates labels by the TALLEST one in the panel (lab_h is a
# single max, not per-label), so a 3-line label inflates the required vertical gap for all six
# and drags them off their own markers on long leader lines -- measured, it put "MAttr+Adam"
# 0.55 of the axis below its point. Do not "tidy" the wrap back to one line either, without
# re-reading the `labels past the right frame` count place_labels prints every run.
# Mathtext, not Unicode superscripts: palette.RC points every mathtext family at Inter, so
# "$^\\mathrm{S}$" sets in the same face as the surrounding label, whereas U+02E2/U+1D2C are
# absent from many faces and would silently fall back. Short single-line labels are also what
# make the 1x4 layout work at all -- see LAB_XPAD.
POINT_LABEL = {
    "eprun-s090": "NP",          # point label only; the registry name stays "Node Pruning"
    # Kept for when Expected Gradients is drawn (it is out of FIGURE_METHODS, in --all only).
    # "EG", not "Expected Gradients": the same abbreviation plot_mib_accauc_cpr_scatter.py
    # uses in its compact cut, and for the same reason -- the full name is wider than any
    # other label on a ~1.1in panel. One abbreviation across both figures, not two.
    "mc_ig": "EG",
    "stopk-log": "MAttr$^\\mathrm{A}$ (def. ε)",
    # No optimiser superscript now that only one arm is drawn: with no SGD point on the panel
    # "MAttr$^A$" marks a contrast that is not there. Restore both superscripts together if
    # softsgd-log goes back into CPR_METHODS.
    "stopk-log-eps1e-2": "MAttr",
    "softsgd-log": "MAttr$^\\mathrm{S}$",
    "stopk-unif-eps1e-2": "$+$unif. $k$",
    # "DBM x8", not "DBM (multi-sparsity)": the full name is ~2x the widest label this panel
    # carries. The x8 names the thing that differs from the "DBM" point beside it -- eight
    # trained masks instead of one -- which is the comparison the two points exist to make.
    "dbm-multisp": "DBM $\\times$8",
}
# Panel grid for the labelled cut: ONE ROW of 5 at full \textwidth, ~1.1in per panel.
#
# WAS one row of 5 until the second SAE column landed (2026-08-30). Six in a row does NOT fit:
# the renderer reports `x headroom 1.091`, i.e. the rightmost labels run 9% PAST the panel frame
# and clip -- and the overlap counter stays 0 throughout, so it gives no warning. 5 leaves the
# sixth panel alone on its own row with 4/5 of it empty. 3x2 costs ~3.5in of page height and
# brings headroom back to 0.66-0.78, with the two SAE panels adjacent so their reversal reads
# directly. Check BOTH diagnostics after changing this, not just the overlap count.
LAB_NCOL = 5
# 2.0in of height is TUNED, not chosen: vertical crowding is what binds label placement (the
# same finding plot_mib_accauc_cpr_scatter records), so shrinking this does not shrink the
# figure gracefully -- at 1.85 repel starts pushing labels off their own markers onto long
# leader lines in the two Node panels, while the overlap count stays 0 and gives no warning.
# Check the rendered PNG, not just the diagnostics, after changing it.
# 1.95in is the FLOOR AT SEVEN SERIES, found by bisection and by looking at the render -- not by
# the diagnostics. Each added series costs vertical room: at eight (with Expected Gradients) 1.95 was
# too tight and 2.2 was the floor, so re-check this after adding anything.
# Found by bisection and by looking at the render -- not by the
# diagnostics, which report 0 overlaps at every height tried and give no warning at all.
# Vertical crowding is what binds label placement, and below 1.95 repel starts pushing labels
# off their own markers onto long leader lines in the two Node panels (where NP/DBM/I×G
# cluster): at 1.85 MAttr^S lands ~0.43 against its point at 0.90, at 1.7 ~0.37. Cut further
# only if you also drop a series or shorten a label, and check the PNG when you do.
LAB_FIG = (5.5, 1.70)
# Right-hand padding as a fraction of the x range, for labels that hang off the last point.
# NOT the scatter's 0.34: these labels are longer relative to a 2.4in panel, and at 0.34
# place_labels reported the rightmost label ending at 1.13 of the frame in three of the four
# panels. Tuned against that diagnostic, which prints the number every run -- if it ever
# reports >1.0 again, raise this rather than shortening a label to fit.
LAB_XPAD = 0.74
LAB_PT = 5.6            # four panels across \textwidth, not the scatter's single one
LAB_MSIZE = 15          # marker AREA in pt^2, uniform across methods -- see draw_labelled
# `--adam`: the default cut plus MAttr under Adam, i.e. the optimiser contrast on the same axes
# the SVA+ claim is made on. Five series still fits the one-row legend; the full registry does
# not (that is what --all is for, and why it is documented as overflowing).
#
# COVERAGE, and it is asymmetric by DESIGN, not a stalled wave: results/sva_zeroabl_input was
# submitted with ONLY="ig ixg softsgd" (see SOURCES), so `Zero-abl. / Node, +input` draws no
# MAttr-Adam point. main()'s MISSING column names it every run -- do not read that panel's
# absence as Adam failing there.
#
# The eps arm is in this cut too, and this is the cut where it MEANS something: `stopk-log` and
# `stopk-log-eps1e-2` are the same run with one number changed, so the two sit here as a matched
# pair and the distance between them IS the eps effect. Six series, one more than the comment
# above says fits on one legend row, which is why the fill guide wraps to two rows when the cut
# is this wide (see `legend_rows`).
ADAM_METHODS = ["IG", "IxG", "stopk-log", "stopk-log-eps1e-2", "softsgd-log", "Random"]
ALL_METHODS = [k for k in METHODS
               if k not in ("soft-log", "mc_ig", "GIM", "stopk-unif-eps1e-2", "dbm-multisp")]
# `--stepless`: the default cut plus Expected Gradients. It is a SEPARATE cut, and a narrowed one, for a
# coverage reason that cannot be fixed by adding a key to FIGURE_METHODS.
#
# Expected Gradients exists ONLY in results/sva_sweep (patched, −input) and ONLY for the four SVA tasks:
# 36 runs = 4 tasks x 3 substrates x 3 losses, submitted 2026-08-22. There are no Arith, ARC-E or
# IOI runs, and none in the other three SOURCES dirs. Under the normal REQUIRED sets every panel
# demands SVA+Arith (and ARC-E+IOI at `node`), so group_avg would drop Expected Gradients from all seven
# panels and the figure would come out looking exactly like the default one -- a silent no-op.
#
# So this cut narrows the figure to what the arm actually covers, and says so on the figure: one
# ablation, one input setting, three substrates, task-group = SVA alone. Every OTHER series is
# narrowed with it, so the panels still compare one task population. Read it as a preview of the
# arm, not as a drop-in for the paper figure -- the numbers are not comparable to the default
# cut's, whose points average four task-groups. Backfilling is one submitter away:
#   ARITH_TASKS="addition months weekdays hours" bash scripts/sva/launch/submit_sva_sweep.sh
#   MIB_TASKS=arc_easy bash scripts/sva/launch/submit_sva_sweep.sh
#   MIB_TASKS=ioi MODEL=qwen2.5 bash scripts/sva/launch/submit_sva_sweep.sh
# (each also re-emits the other GRAD arms, but the submitter skips runs already on disk).
STEPLESS_METHODS = ["IG", "IxG", "mc_ig", "softsgd-log", "Random"]
STEPLESS_SOURCES = [("results/sva_sweep", "−input", "Patched")]
STEPLESS_REQUIRED = {sub: ["SVA"] for sub in REQUIRED}


def parse_method(fname, d):
    """Method label from filename tag (mirrors make_fingerprint_tables.parse_method)."""
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    tag = tag.replace("_zeroabl", "")   # ablation is a facet ROW, not a method
    # `_ferr` marks the frozen SAE error intervention. It is a property of the SUBSTRATE COLUMN
    # (SUBSTRATE_RES routes those runs to their own tree), not of the method, so it is stripped
    # here -- otherwise every frozen run would parse as an unknown method and vanish.
    tag = tag.replace("_ferr", "")
    # `_s<steps>` is run_tag's marker for a non-default training budget. Stripped for the same
    # reason as `_ferr`: the budget is a property of the results TREE, which SUBSTRATE_RES routes,
    # not of the method. Each tree holds exactly ONE budget per method, so nothing collides --
    # mixing a 2k and a 5k run of the same method in one tree WOULD silently average them.
    #
    # *** APPLIED TO MAttr TAGS ONLY, because `_s<n>` is not a step count anywhere else. ***
    # Two other families end in it and both broke when the strip was unconditional:
    #   eprun_s090      `_s090` is a SPARSITY budget; stripping gave `eprun`, which matches no
    #                   branch, and Node Pruning vanished from all four panels (26 -> 23 points)
    #   mc_ig_m1_s42    `_s42` is a SEED; stripping cost Expected Gradients its series (8 -> 7)
    # `random_s42` happens to survive on its prefix branch, but the rule is the same: only a
    # trained MAttr run carries a step suffix, so only those are stripped.
    if "sufficient_" in tag or "hard_topk" in tag:
        tag = re.sub(r"_s\d+$", "", tag)
    # Node Pruning / DBM at a bumped budget (the 2026-09-06 5k runs) carry the same marker,
    # but their tags ALSO end in _s090 (a sparsity) / seeds elsewhere, so the strip demands
    # >= 4 digits: _s5000 strips, _s090 and _s42 survive. Step budgets below 1000 never
    # reach a figure tree.
    if tag.startswith(("eprun", "sig_")):
        tag = re.sub(r"_s\d{4,}$", "", tag)
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
        # "_topk_none" = no learning (trainer.learn_scores optimizer="none", 2026-09-18): the
        # mean negated gradient at zero scores. Its own family so it cannot collide with the
        # default-eps Adam key (neither tag carries an eps mark).
        opt = ("softsgd" if "_topk_sgd" in tag else "frozenid" if "_topk_identity_none" in tag
               else "frozen" if "_topk_none" in tag else "stopk")
        # ADAM'S EPS TOO, and for the identical reason. `_eps1e-2` is an Adam run like the
        # default-eps one, shares the whole `sufficient_topk_adam` prefix, and at neuron scale is
        # a DIFFERENT ranking (acc-AUC 0.388 vs 0.490 on addition/mlp) -- so without this the
        # 2026-08-28 eps wave would pool into `stopk-log` exactly as the SGD wave did.
        # eval_sva.run_tag writes the value in normalised sci notation (eval_sva.eps_tag), so the
        # spelling here is one-to-one with the value; the key keeps it rather than collapsing to
        # a boolean, so a future eps=1e-1 arm gets its own key and, being absent from METHODS,
        # DROPS instead of joining this one.
        eps = re.search(r"_eps([0-9.]+e[+-]?[0-9]+)", tag)
        return f"{opt}-{ks}" + (f"-eps{eps.group(1)}" if eps else "")
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


def _cpr_of(d):
    """MIB-style CPR from the stored curve: LINEAR trapezoid of faithfulness over the kept
    PROPORTION p = k / total, unnormalised (the span is ~1, as MIB's .001..1 grid is).

    Same integrand as `faith_auc`, different measure -- log-x normalised there, linear-p here.
    eval_sva's 24 sparsities are log-spaced from 1/total to 1 (eval_sva.py:1248), so this is
    the number MIB's `area_under` would report if scored on this grid: the linear weight puts
    almost all of the mass on the last three or four points (p >= 0.1), which is exactly the
    property the paper's CPR critique is about and what the --cpr cut exists to show.
    """
    p = np.asarray(d["n_nodes"], float) / float(d["total"])
    f = np.asarray(d["faithfulness"], float)
    return float(np.sum((p[1:] - p[:-1]) * (f[1:] + f[:-1]) / 2))


def load(res):
    """(method, loss, substrate, task) -> (acc_auc, faith_auc, cpr); the first two as stored,
    CPR recomputed from the curve by `_cpr_of`.

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
    separately, as the module docstring says. See scripts/sva/method_winrate.py for the same
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
        # TENX_KEYS' base keys pass too: "stopk-unif" (default-eps uniform k) has no METHODS entry
        # of its own, and load() of a 10x tree must keep it so tenx_for() can re-key it.
        if m is None or (m not in METHODS and m not in TENX_KEYS):
            continue
        if not on_model(d):
            continue
        runs.setdefault((m, d["loss"], d["nodes"], d["task"]), []).append(
            (d["acc_auc"], d["faith_auc"], _cpr_of(d)))
    return {k: tuple(float(np.mean([v[i] for v in vs])) for i in range(3))
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
    gx, gy, gz = [], [], []
    for gname, tasks in GROUPS:
        if gname not in required[sub]:
            continue
        if not set(tasks) <= have:
            return None
        vs = [raw[(m, loss, sub, t)] for t in tasks]
        gx.append(np.mean([v[0] for v in vs]))
        gy.append(np.mean([v[1] for v in vs]))
        gz.append(np.mean([v[2] for v in vs]))
    if not gx:
        return None
    return float(np.mean(gx)), float(np.mean(gy)), tuple(required[sub]), float(np.mean(gz))


def draw_labelled(df, figure_methods, out, ycol="faith_auc", ylabel="Faith log-AUC (↑)",
                  xlabel="Compactness (↑)", colors=None, hlines=(), figsize=None,
                  markers=None, legend=False, free_lims=False, label_keys=None, ncol=None):
    """The default cut, raw matplotlib: uniform circles + direct point labels, no legend.

    `ycol`/`ylabel` select the y metric: the stored log-AUC (default) or the MIB-style CPR the
    --cpr cut recomputes from the same curves (see _cpr_of / mib_rows). Both AUCs on the
    default cut are LOG-weighted and the labels say so; CPR is a linear AUC and does not.

    `colors` overrides the per-method palette (the --cpr cut colours by FAMILY, as
    plot_mib_test_avg.py does). `hlines` = [(facet, y, label, colour)] draws a reference line
    in one panel for a method with a y value but no x -- MIB's Random control.

    WHY NOT PLOTNINE, which every other cut here uses. Direct labelling needs the rendered
    geometry of each string -- its width in axes fractions, measured through the renderer after
    layout -- so labels can be pushed off each other and off the markers. That is per-annotation
    control the grammar does not expose, and it is exactly why plot_mib_accauc_cpr_scatter.py
    (the node+edge CPR-vs-IIA summary this figure is being matched to) is raw matplotlib too.
    Its label machinery is IMPORTED rather than reimplemented, so the two figures place labels
    by the same rules and a fix to the repel pass reaches both.

    UNIFORM CIRCLES. An earlier version varied marker size so that the MAttr arms, which
    coincide at the node substrate, did not hide each other. Direct labels make that
    unnecessary and the sizing actively misleading -- a big marker reads as emphasis. Coincident
    points now draw as one circle with a leader line per label, which is a truer picture: the
    figure says "these are the same point here" instead of implying a resolvable ordering.
    """
    import matplotlib.pyplot as plt                       # local: the other cuts never need it
    import plot_mib_accauc_cpr_scatter as S               # label_boxes / repel / place_labels

    facets = list(df["facet"].cat.categories)
    # LAB_NCOL is a CEILING, not a fixed column count. It was fixed at 5 when the default cut had
    # five substrate panels; dropping SAE (resid) left the grid still reserving a fifth slot, so
    # 20% of the canvas was blank on the right and every panel was drawn 4/5 as wide as it needed
    # to be. Panels get wider as a result, which is the direction that HELPS label placement --
    # but the docstring's rule still applies: check the printed overlap count AND the x-headroom
    # diagnostic after changing this, not just one of them.
    ncol = ncol or min(LAB_NCOL, len(facets))   # --cpr passes 4: two rows of four, no blanks
    nrow = int(np.ceil(len(facets) / ncol))
    plt.rcParams.update(P.RC)
    # Height scales with the row count: LAB_FIG is sized for one row, and a second row of
    # panels needs its own height rather than half of the first row's.
    base = figsize or LAB_FIG
    figsize = base if nrow == 1 else (base[0], base[1] * nrow * 0.92)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()
    # A trailing slot can still exist when the facet count is not a multiple of ncol (e.g. 6
    # facets at ncol=5). Hide it rather than leaving an empty framed panel, which reads as a
    # cell whose runs all failed.
    for ax in axes[len(facets):]:
        ax.set_visible(False)
    colors = colors or {METHODS[m][0]: METHODS[m][1] for m in figure_methods}

    for ax, facet in zip(axes, facets):
        sub = df[df["facet"] == facet]
        if sub.empty:                       # a placeholder slot (no rows yet): title + note
            ax.set_title("\n".join(facet.split("\n")[:-1]), fontsize=6.2, pad=2.5)
            ax.text(0.5, 0.5, PLACEHOLDER_NOTE.get(facet, "no runs"), ha="center", va="center",
                    fontsize=LAB_PT, color="#8a8a8a", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue
        for hf, hy, hlab, hc in hlines:
            if hf == facet:
                ax.axhline(hy, ls=(0, (3, 2)), lw=0.7, color=hc, zorder=2)
                ax.annotate(hlab, (0.03, hy), xycoords=("axes fraction", "data"),
                            fontsize=LAB_PT, color=hc, va="bottom", ha="left")
        # The MAttr arms draw as STARS, everything else as the uniform circles the docstring
        # argues for -- "ours" carries a shape as well as its colours, matching the starred
        # Pareto frontier of plot_mib_accauc_cpr_scatter's compact cut. A star packs less fill
        # area into its bounding box than a circle (same fact the plotnine cuts handle for
        # Random), so it gets ~2x the marker area to read at the same visual weight.
        star = sub["_key"].isin(STAR_KEYS)
        sq = sub["_key"].isin(MASK_KEYS)
        circ = ~star & ~sq
        outline = OUTLINE_NON_STAR & ~star & ~sub["_key"].isin(FILLED_KEYS)

        def style(mask, hollow):
            cols = [colors[m] for m in sub["method"][mask]]
            if hollow:
                return dict(facecolor="none", edgecolor=cols, linewidth=OUTLINE_LW)
            return dict(c=cols, edgecolor="#000000", linewidth=0.3)
        if markers:
            # Per-KEY markers (the --cpr cut, 2026-09-18): one glyph per method so a legend
            # can carry the names and the point labels can be dropped for everything but ours.
            # Hollow/filled follows the same outline rule as before; the star stays a star.
            for key, mk in markers.items():
                m = (sub["_key"] == key) & ~star
                if not m.any():
                    continue
                hollow = bool(outline[m].iloc[0])
                ax.scatter(sub["acc_auc"][m], sub[ycol][m], marker=mk, zorder=3,
                           s=LAB_MSIZE * (0.85 if mk in "sD" else 1.0), **style(m, hollow))
        else:
          for hollow in (False, True):
            # A square packs more ink into its box than a circle of equal `s`; scale it down
            # a touch so the two read at the same weight.
            for mask, marker, size in ((circ, "o", LAB_MSIZE), (sq, "s", LAB_MSIZE * 0.85)):
                m = mask & (outline if hollow else ~outline)
                if m.any():
                    ax.scatter(sub["acc_auc"][m], sub[ycol][m], s=size, marker=marker,
                               zorder=3, **style(m, hollow))
        ax.scatter(sub["acc_auc"][star], sub[ycol][star], s=LAB_MSIZE * 2.2,
                   marker="*", c=[colors[m] for m in sub["method"][star]],
                   edgecolor="#000000", linewidth=0.3, zorder=4)
        # Anchor at 0 on both axes, as the plotnine version does via expand_limits: the Random
        # point is the floor and a panel that crops it loses the only absolute reference.
        # XPAD then adds room on the right for labels that hang off the last point.
        xs, ys = sub["acc_auc"], sub[ycol]
        if free_lims:
            # Axes fit the data (2026-09-18, requested): autoscale with a margin instead of
            # anchoring both axes at 0 -- the chance-corrected zero row has negative
            # compactness, and a fixed 0 floor hid the spread within a panel.
            ax.margins(x=0.34, y=0.15)   # x room for the two MAttr labels off the rightmost point
            ax.autoscale(enable=True, axis="both", tight=False)
            for hf, hy, _, _ in hlines:
                if hf == facet:
                    lo, hi = ax.get_ylim()
                    ax.set_ylim(min(lo, hy - 0.1 * (hi - lo)), hi)
        else:
            ax.set_xlim(0, max(xs) * (1 + LAB_XPAD))
            ax.set_ylim(0, max([max(ys)] + [hy for hf, hy, _, _ in hlines if hf == facet]) * 1.12)
        # Title DROPS the strip's last line, which is the task-group list. The faceted cuts
        # keep it because there it stops the node and per-position column families being read
        # as the same average; here the panels are named on the figure and the datasets belong
        # in the caption, not on four repeated sub-headings. Dropping it also takes the title
        # to one short line, which is what stops four titles overrunning each other at ~1.1in
        # of panel.
        ax.set_title("\n".join(facet.split("\n")[:-1]), fontsize=6.2, pad=2.5)
        # Full rectangle, matching theme_bw and therefore the faceted cuts of this same figure
        # -- P.furnish already sets every spine to SPINE_LW, so the border comes for free and
        # the L-shape the first cut of this renderer drew was the deviation, not this.
        P.furnish(ax)
        ax.tick_params(labelsize=5.5)

    for ax in axes[len(facets):]:
        ax.set_visible(False)
    fig.supxlabel(xlabel, fontsize=7, y=0.015)
    fig.supylabel(ylabel, fontsize=7, x=0.012)
    if legend and markers:
        # One legend at the right, in figure_methods order, with the marker each method draws.
        from matplotlib.lines import Line2D
        hs = []
        order = [k for k in figure_methods if k in STAR_KEYS] + \
                [k for k in figure_methods if k in FILLED_KEYS and k not in STAR_KEYS] + \
                [k for k in figure_methods if k not in STAR_KEYS and k not in FILLED_KEYS]
        for key in order:
            if key not in df["_key"].values:
                continue
            name = METHODS[key][0]; col = colors[name]
            if key in STAR_KEYS:
                hs.append(Line2D([], [], marker="*", ls="", color=col, markeredgecolor="#000000",
                                 markeredgewidth=0.3, markersize=8, label=POINT_LABEL.get(key, name)))
            else:
                hollow = OUTLINE_NON_STAR and key not in FILLED_KEYS
                hs.append(Line2D([], [], marker=markers.get(key, "o"), ls="",
                                 markerfacecolor="none" if hollow else col, color=col,
                                 markeredgecolor=col if hollow else "#000000",
                                 markeredgewidth=OUTLINE_LW if hollow else 0.3, markersize=5,
                                 label=POINT_LABEL.get(key, name)))
        fig.legend(handles=hs, loc="center right", fontsize=LAB_PT + 0.4, frameon=False,
                   handletextpad=0.4, labelspacing=0.9, borderaxespad=0.2)
        # The legend is a fixed physical width (longest label "Random" at LAB_PT+0.4), so the
        # panel area it takes is inversely proportional to the figure width: 0.095 of the
        # 1.1*LAB_FIG[0] five-column figure it was tuned on. Scaling it keeps the legend clear
        # of the last column at other widths (the 2x4 cut is 0.9*LAB_FIG[0]).
        legend_frac = 0.095 * (LAB_FIG[0] * 1.1) / fig.get_figwidth()
        fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.5, rect=(0.013, 0.02, 1 - legend_frac, 1))
    else:
        fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.5, rect=(0.013, 0.02, 1, 1))
    # AFTER tight_layout: place_labels measures the marker half-extent and the label widths off
    # the laid-out panel, so calling it earlier would size every offset against a panel geometry
    # that is about to change.
    for ax, facet in zip(axes, facets):
        sub = df[df["facet"] == facet]
        if label_keys is not None:
            sub = sub[sub["_key"].isin(label_keys)]
        if sub.empty:
            continue
        lab = pd.DataFrame(dict(acc=sub["acc_auc"].values, cpr=sub[ycol].values,
                                label=[POINT_LABEL.get(m, METHODS[m][0])
                                       for m in sub["_key"]],
                                grp=sub["method"].astype(str).values,
                                side=["left" if facet.startswith(LABEL_LEFT.get(m, ()))
                                      else "right" for m in sub["_key"]]))
        print(f"  {facet.splitlines()[0]}:", file=sys.stderr)
        S.place_labels(fig, ax, lab, pt=LAB_PT, msize=LAB_MSIZE, colors=colors)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)


# The MIB (node) panel of the --cpr cut, on the TEST split so it is the same population as
# plots/plot_mib_test_avg.py's node panel and as the Random control below (a literal transcribed
# from MIB's Table 1, test set). Six methods, the same as the SVA+ panels; the dirs are the ones
# make_mib_test_table.py names for each row. Read directly off the pkls: CPR = `area_under`
# (linear trapezoid over MIB's ten proportions), IIA = `acc_auc` (log-weighted over the same
# grid). Two caveats the caption must carry: MIB's MAttr+Adam is at the DEFAULT eps
# (topk_log_lr05), whereas the SVA+ Adam arm is eps=1e-2; and the llama3 cells are full-split
# here (test is uncapped, see CLAUDE.md), so this is not the validation panel
# plot_mib_accauc_cpr_scatter.py draws.
MIB_FACET = "MIB (node, test)\nMIB"
# Sentinel standing where a results dir goes in MIB_TEST, for the one row whose numbers are
# computed from a ladder rather than read off a pkl. A unique object rather than a string so it
# cannot collide with a real dir name.
DBMMS = object()
# Edge level, same split. Only the two MAttr arms have BOTH metrics on the test split; the
# edge gradient baseline EAP-IG-inp has a Table 1 test CPR (make_mib_test_table.EDGE_BASELINES)
# but no IIA anywhere (our own edge EAP-IG evals, results/eapig_clean_eval, are validation),
# so it draws as a reference line like Random does at node level. No mask baseline exists at
# edge level on any split (no EdgePruning_patching_edge on disk).
MIB_EDGE_FACET = "MIB (edge, test)\nMIB"
MIB_TEST_EDGE = {
    "stopk-log-eps1e-2": ("test_edge_topk_log_lr05", None),
    "softsgd-log":       ("test_edge_softlog_sgd_lr_3.0", None),   # SGD's own edge optimum
    "stopk-unif-eps1e-2": ("test_edge_topk_uniform_lr05", None),
    # Our own edge-level test evals (2026-09-17, submit_eapig_mc_edge_test_sc.sh /
    # submit_eprun_edge_sc.sh), replacing the "EAP-IG-inp" reference line (MIB's published
    # number, no IIA). IG is m=10 as in the node panel; mib_rows holds a row to 12/12 cells.
    "IG":                ("eapig_clean10_test", "EAP-IG-inputs_patching_edge"),
    "mc_ig":             ("eapig_mc_test", "EAP-IG-inputs-mc_patching_edge"),
    "eprun-s090":        ("eprun_eval_s0.99_ld", "EdgePruning_patching_edge"),
}
# ZERO ABLATION, node level, test split (submit_mib_zero_sc.sh, 2026-09-18): every method
# attributed under zero and scored with intervention='zero'. Same keys as MIB_TEST.
MIB_ZERO_FACET = "MIB (node, test), zero\nMIB"
MIB_EDGE_ZERO_FACET = "MIB (edge, test), zero\nMIB"   # no edge-level zero runs: drawn blank
# Placeholder panels: kept in the grid even with no rows, with this note in the middle, so the
# two rows' columns stay aligned. The node-zero note is transient (its wave is landing).
# 2026-09-19: empty. The edge column was dropped from the cut (requested), which removed the
# edge-zero blank, and the node-zero wave has landed 12/12. Kept as the hook for the next
# transient panel; `keep` below still unions it in.
PLACEHOLDER_NOTE = {}
MIB_TEST_ZERO = {
    "IxG":               ("mib_zero_test/ixg", "EAP-IG-inputs_zero_node"),
    "IG":                ("mib_zero_test/ig", "EAP-IG-inputs_zero_node"),
    "mc_ig":             ("mib_zero_test/eg", "EAP-IG-inputs-mc_zero_node"),
    "AttnLRP":           ("mib_zero_test/attnlrp", "AttnLRP_zero_node"),
    "eprun-s090":        ("eprun_eval_s0.5_ld_zero", "EdgePruning_zero_node"),
    "sig_lr0.3_l16.0":   ("eprun_eval_ld_sig_lr0.3_l16.0_zero", "EdgePruning_zero_node"),
    "stopk-log-eps1e-2": ("test_node_topk_log_lr05_zero", None),
    "stopk-unif-eps1e-2": ("test_node_topk_uniform_lr05_zero", None),
    "Random":            ("mib_zero_test/random", "Random_zero_node"),
}
# registry key -> (results dir, subfolder or None). None = MAttr layout `{task}_{model}_test.pkl`;
# a subfolder = run_evaluation.py layout `{task-dashed}_{model}_test_abs-False.pkl`.
MIB_TEST = {
    "IxG":               ("ig1_test", "EAP-IG-inputs_patching_node"),
    "IG":                ("napig10_test", "EAP-IG-inputs_patching_node"),   # SVA+ IG is m=10 too
    "mc_ig":             ("napig_mc_test", "EAP-IG-inputs-mc_patching_node"),
    "AttnLRP":           ("attnlrp_eval", "AttnLRP_patching_node"),
    "GIM":               ("gim_eval", "GIM_patching_node"),
    "eprun-s090":        ("eprun_eval_s0.5_ld", "EdgePruning_patching_node"),   # NODE_PRUNING
    "sig_lr0.3_l16.0":   ("eprun_eval_ld_sig_lr0.3_l16.0", "EdgePruning_patching_node"),
    "stopk-log-eps1e-2": ("test_node_topk_log_lr05", None),
    "softsgd-log":       ("test_node_softlog_sgd_lr_1.0", None),
    # "+ Adam, unif k" of the test table. DEFAULT eps, like the Adam log-k row above.
    "stopk-unif-eps1e-2": ("test_node_topk_uniform_lr05", None),
    "stopk-unif-eps1e-2-10x": ("test_node_topk_uniform_lr05_5k", None),   # 5000 steps; no edge twin
    # Our own random node ordering on the test split (MIB-circuit-track/run_random_test.sh,
    # U(0,1) node scores, seed 42+cell). It supplies the IIA log-AUC that MIB's Table 1 does
    # not report; until all 11 cells land, Random falls back to a reference line at the
    # Table 1 CPR (mib_random_cpr).
    "Random":            ("random_test", "Random_patching_node"),
    # NOT A PKL. The multi-sparsity row has no `area_under`/`acc_auc` on disk to read: its two
    # scalars are integrated over MIB's grid FROM the ladder by scripts/mib/dbm_multisparsity.py,
    # which is also what make_mib_test_table.py calls, so the point and the table row cannot
    # disagree. The DBMMS sentinel routes it there in mib_rows.
    "dbm-multisp":       (DBMMS, None),
}
# Colour by FAMILY, exactly plot_mib_test_avg.FAMILY: the two gradient baselines share IG's
# orange, the two mask baselines Node Pruning's indigo, both MAttr arms MAttr's blue, Random
# grey. Labels carry the within-family identity.
FAMILY_COLOR = {
    "IG": P.METHOD["IG"], "IxG": P.METHOD["IG"], "mc_ig": P.METHOD["IG"],
    "AttnLRP": P.METHOD["IG"], "GIM": P.METHOD["IG"],
    "eprun-s090": P.METHOD["Node Pruning"], "sig_lr0.3_l16.0": P.METHOD["Node Pruning"],
    # Same indigo as the single-lambda DBM point: it is the SAME METHOD at a different
    # protocol, and giving it its own hue would read as a fourth family.
    "dbm-multisp": P.METHOD["Node Pruning"],
    # BOTH k-SCHEDULES IN MAttr BLUE, matching plot_mib_test_avg.FAMILY after its own
    # 2026-09-08 recolour: there the uniform-k bars stopped carrying their own hue, so a
    # separate colour here would make one method two colours across two figures on one page.
    # The two are told apart by their labels, as they are in the bar chart.
    "stopk-log-eps1e-2": P.METHOD["MAttr"], "softsgd-log": P.METHOD["MAttr"],
    "stopk-unif-eps1e-2": P.METHOD["MAttr"], "stopk-unif-eps1e-2-10x": P.METHOD["MAttr"],
    "Random": P.METHOD["Random"],
}


def mib_rows(figure_methods, table=None, facet=MIB_FACET):
    """Rows for one MIB test panel, shaped like main()'s SVA+ rows (faith_auc NaN: not on disk).

    Every method must have all 11 cells or it is skipped, printed -- the same all-or-nothing
    rule group_avg applies to the SVA+ task-groups.
    """
    import pickle
    sys.path.insert(0, "scripts/mib")
    import make_mib_test_table as T
    table = MIB_TEST if table is None else table
    out = []
    for m in figure_methods:
        if m not in table:
            continue
        d, sub = table[m]
        accs, cprs = [], []
        for task, model, _ in T.COLUMNS:
            if d is DBMMS:
                # Same reader make_mib_test_table.py uses, so this point and the table's
                # "DBM (multi-sparsity)" row are the same two numbers by construction.
                import dbm_multisparsity as _DBMMS
                got = _DBMMS.cell(task, model, "test")
                if got is None:
                    continue
                cpr, iia, _n = got
                accs.append(iia); cprs.append(cpr)
                continue
            f = (f"results/{d}/{task}_{model}_test.pkl" if sub is None else
                 f"results/{d}/{sub}/{task.replace('_', '-')}_{model}_test_abs-False.pkl")
            if not os.path.exists(f):
                continue
            r = pickle.load(open(f, "rb"))
            accs.append(r["acc_auc"]); cprs.append(r["area_under"])
        if len(accs) < len(T.COLUMNS):
            print(f"  MIB panel {facet.splitlines()[0]}: {m} has {len(accs)}/{len(T.COLUMNS)} "
                  f"test cells, skipped", file=sys.stderr)
            continue
        out.append(dict(acc_auc=float(np.mean(accs)), faith_auc=float("nan"),
                        cpr=float(np.mean(cprs)), method=METHODS[m][0], _key=m,
                        loss=LOSSES["logit_diff"], facet=facet, ablation="Patched",
                        groups="MIB"))
    return out


def mib_edge_eapig_cpr():
    """Mean test CPR of EAP-IG-inp at edge level (MIB Table 1 literal). No IIA on disk."""
    sys.path.insert(0, "scripts/mib")
    import make_mib_test_table as T
    return float(np.mean(list(T.EDGE_BASELINES["EAP-IG-inp (CF)"].values())))


def mib_random_cpr():
    """Mean test CPR of MIB's Random control (Table 1 literal, via make_mib_test_table).
    There is no IIA number for it anywhere on disk, so it draws as a reference line."""
    sys.path.insert(0, "scripts/mib")
    import make_mib_test_table as T
    return float(np.mean(list(T.NODE_BASELINES["Random"].values())))


def report(df, figure_methods, losses, dropped):
    """Per-panel coverage, printed by every cut. Factored out so the raw-matplotlib
    default and the plotnine cuts cannot end up reporting different things."""
    # Every panel must show ONE group set (group_avg enforces it) and the full method x loss
    # grid. A short count is a coverage hole, not a styling choice, so print both.
    # Per-method, not len(methods) x len(LOSSES): a method that contributes ONE point -- because
    # it has no loss (LOSSLESS) or was run at only one (SINGLE_LOSS) -- would make a flat product
    # report every complete panel as permanently short, and a report that is never green is a
    # report nobody reads.
    # `losses`, not the module-level LOSSES: the single-loss default cut expects one point from
    # EVERY method, so counting against all three would report every complete panel as short.
    # MIB_ONLY methods are expected on the MIB panels and NOWHERE ELSE, so the expected count
    # is per-facet rather than one number. Without this, adding a MIB-only row makes every SVA+
    # panel report n/of short forever and name the row as MISSING -- the "report that is never
    # green" this function's own comment warns about, arrived at by a different route.
    def expected(facet):
        ms = [m for m in figure_methods if expected_on(m, facet)]
        return sum(1 if m in LOSSLESS or m in SINGLE_LOSS else len(losses) for m in ms)

    cov = df.groupby(["ablation", "facet"], observed=True).agg(
        n=("groups", "size"), groups=("groups", lambda s: " / ".join(sorted(set(s)))))
    cov["of"] = [expected(f) for _, f in cov.index]
    # A method with NO runs at all for a substrate/input combo is invisible to group_avg (which
    # guards missing tasks within a method, not a missing method), so a panel can silently draw
    # a smaller method set than its neighbours. With the three-method cut that is not cosmetic:
    # MAttr (SGD) has no sva_sweep_input runs, so `Node, +input` would show the two BASELINES
    # and no MAttr, i.e. exactly the panel a reader would misread as a loss.
    def missing_for(facet, present):
        ks = [k for k in figure_methods if expected_on(k, facet)]
        return ",".join(m for m in [METHODS[k][0] for k in ks] if m not in present) or "-"

    # dict, not a Series of sets: .loc on a Series whose VALUES are sets makes pandas try to
    # hash them as an indexer and raises "unhashable type: 'set'".
    grp = {k: set(v) for k, v in
           df.groupby(["ablation", "facet"], observed=True)["method"]}
    cov["methods"] = [missing_for(f, grp[(a, f)]) for a, f in cov.index]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", dest="draw_all",
                    help="draw the full registry (MAttr-Adam, Node Pruning, DBM) instead of the "
                         "three-method cut; legend overflows \\textwidth at this width")
    ap.add_argument("--adam", action="store_true",
                    help="default cut plus MAttr (Adam), for the optimiser contrast")
    ap.add_argument("--zero", action="store_true",
                    help="the default cut PLUS the zero-ablation panels, same labelled style. "
                         "Exploratory, not the paper figure: the zero half's IIA AUC sits on a "
                         "~0.51 random floor against the patched half's ~0.03, so the two rows "
                         "are not on a comparable scale -- read each against its own Random "
                         "point, which is labelled in every panel for exactly that reason.")
    ap.add_argument("--stepless", action="store_true",
                    help="default cut plus Expected Gradients; NARROWS the figure to patched/−input and "
                         "to the SVA task-group, which is all that arm has been run on")
    ap.add_argument("--cpr", action="store_true",
                    help="default cut with MIB-style CPR (linear trapezoid over the kept "
                         "proportion, recomputed from the stored curves) on y instead of the "
                         "log-weighted Faith AUC, plus a leading panel of the same methods on "
                         "MIB node-level validation read from the MIB pkls. Writes "
                         "accauc_vs_cpr.pdf; the paper figure is untouched.")
    ap.add_argument("--by-loss", action="store_true",
                    help="--cpr only: CPR vs Compactness with ONE ROW PER TRAINING LOSS (logit-diff, "
                         "CE, soft-acc) and one column per trained SVA+ substrate (node, MLP, "
                         "MLP+Attn); no MIB panel, no SAE column (its CE/acc runs are IG/IxG only). "
                         "Writes accauc_vs_cpr_byloss.pdf (2026-09-21).")
    ap.add_argument("--full", action="store_true",
                    help="--cpr only: ALSO draw the second row (MIB node + SVA+ substrates under "
                         "zero ablation), 2x4 panels. Writes accauc_vs_cpr_full.pdf (appendix). "
                         "Without it --cpr is the one-row patching figure, accauc_vs_cpr.pdf "
                         "(main text, 2026-09-20).")
    ap.add_argument("--y-cpr", action="store_true",
                    help="THE PAPER'S DEFAULT CUT, unchanged in panels, methods and renderer, with "
                         "the linear CPR on y instead of the log-weighted Faith AUC (CPR is the "
                         "linear AUC always, 2026-09-17). Writes accauc_vs_cpr_sva.pdf, the file "
                         "sections/sva+.tex includes since that day. Unlike --cpr it adds no MIB "
                         "panels and drops no arm.")
    a = ap.parse_args()
    if a.y_cpr and (a.cpr or a.draw_all or a.stepless or a.adam):
        raise SystemExit("--y-cpr is a variant of the default cut only")
    figure_methods = (ALL_METHODS if a.draw_all
                      else STEPLESS_METHODS if a.stepless
                      else ADAM_METHODS if a.adam
                      else CPR_METHODS if a.cpr else FIGURE_METHODS)
    if a.cpr and a.full:
        # AttnLRP in the appendix (2x4) figure only (requested 2026-09-21): it has node, MLP and
        # MLP+Attn runs under both ablations and a MIB test row, but no SAE run, so that panel
        # simply lacks the point. Slotted after IxG so the gradient family stays contiguous.
        figure_methods = [m for m in figure_methods if m != "AttnLRP"]
        figure_methods.insert(figure_methods.index("IxG") + 1, "AttnLRP")
    if a.full and not (a.cpr and not a.zero):
        raise SystemExit("--full is a variant of --cpr only")
    if a.by_loss and not (a.cpr and not a.zero and not a.full):
        raise SystemExit("--by-loss is a variant of plain --cpr only")
    suffix = ("_all" if a.draw_all else "_stepless" if a.stepless
              else "_adam" if a.adam else "_zero" if a.zero else "_full" if a.full
              else "_byloss" if a.by_loss else "")
    if a.cpr and (other_cut_flags := (a.draw_all or a.stepless or a.adam)):
        raise SystemExit("--cpr is a variant of the default cut (optionally --zero) only")
    other_cut = a.draw_all or a.stepless or a.adam
    # --zero is a DEFAULT-cut variant, not an `other_cut`: same five methods, same one loss, same
    # labelled renderer -- only the source list grows. So it must not flip other_cut, which is
    # what routes a run to the plotnine path.
    # --cpr --zero (2026-09-18): the CPR cut on the ZERO-ablation runs alone -- results/sva_zeroabl,
    # 2k steps, three substrates (no SAE zero runs exist), no MIB panels (MIB is patching-only).
    # The node column stays, since nothing replaces it here. Writes accauc_vs_cpr_zero.pdf.
    cpr_zero = a.cpr and a.zero
    sources = (STEPLESS_SOURCES if a.stepless
               else [s for s in SOURCES if s[2] == "Zero-abl." and "_input" not in s[0]] if cpr_zero
               else SOURCES if (other_cut or a.zero) else FIGURE_SOURCES)
    losses = LOSSES if (other_cut or a.by_loss) else FIGURE_LOSSES
    substrates = SUBSTRATES if other_cut else FIGURE_SUBSTRATES
    if cpr_zero:
        substrates = [(sub, lab) for sub, lab in substrates if sub != "mlp_sae_span"]
    elif a.by_loss:
        # The three TRAINED substrates that carry every loss for every method (node from the
        # 2k tree, MLP / MLP+Attn from the 5k tree via SUBSTRATE_RES). The node column stays:
        # there is no MIB panel to replace it (MIB has no CE / soft-acc runs).
        substrates = list(SUBSTRATES)
    elif a.cpr:
        # The MIB test panel IS the node-level comparison, on 11 cells rather than the SVA+
        # node column's 4 groups; keeping both would draw the same ordering twice side by side.
        substrates = [(sub, lab) for sub, lab in substrates if sub != "node"]
    required = STEPLESS_REQUIRED if a.stepless else REQUIRED
    # With one loss and one ablation the shape aesthetic and the ablation strip line each carry
    # a constant, so both are dropped rather than drawn as a legend/label with one value in it.
    # Derived from the cut rather than hardcoded to the default, so a widened FIGURE_LOSSES
    # brings the Loss legend back on its own.
    show_loss = len([k for k in losses if k != "logit_diff"]) > 0
    show_abl = len({abl for _, _, abl in sources}) > 1
    # Same rule as show_abl: with a single source the input label carries no information, so it
    # would print "−input" identically on every panel and spend a line of strip height on it.
    show_inp = len({inp for _, inp, _ in sources}) > 1

    rows, dropped = [], []
    # Cache per (source dir, substrate-override) so a dir is globbed once even when several
    # substrates share it -- load() reads every json in the tree.
    cache = {}

    def raw_for(res, sub):
        """load() of the tree this (source, substrate) draws from.

        SUBSTRATE_RES overrides the source dir for substrates whose runs live elsewhere -- see
        that dict for why the SAE column does. A substrate with an override reads ONLY from the
        override; it never falls back to `res`, because a silent fallback is exactly how the
        column would end up mixing two error interventions.
        """
        # The override is for the PATCHED default source only: sva_zeroabl carries the same
        # substrates at 2k, and routing its mlp column to the 5k patched tree would label patched
        # runs as zero-ablation (the --zero cut did exactly that before 2026-09-18).
        d = SUBSTRATE_RES.get(sub, res) if res == FIGURE_SOURCES[0][0] else res
        if d not in cache:
            cache[d] = load(d)
        return cache[d]

    def tenx_for(sub):
        """The 10x tree for `sub`, its runs re-keyed base -> 10x key (TENX_KEYS); {} off-substrate."""
        d = TENX_RES.get(sub)
        if d is None:
            return {}
        if d not in cache:
            cache[d] = {(TENX_KEYS[m], l, s, t): v for (m, l, s, t), v in load(d).items()
                        if m in TENX_KEYS}
        return cache[d]

    # --cpr (2026-09-18): a SECOND ROW of panels for zero ablation -- the SVA+ substrates from the
    # 5k zero tree (results/sva_zeroabl_5k, submit_zero_5k_sc.sh) -- facet-suffixed ", zero".
    # A source tagged ZERO_ROW is routed to that tree for every substrate (no SUBSTRATE_RES / 10x).
    # Since 2026-09-20 the zero row is --full only: the main-text figure is the patching row alone,
    # and the 2x4 version lives in the appendix next to the zero-ablation tables.
    ZERO_ROW = "Zero-abl. (5k)"
    if a.cpr and not a.zero and a.full:
        sources = list(sources) + [("results/sva_zeroabl_5k", "−input", ZERO_ROW)]
    for res, inp_label, abl in sources:
        for m in figure_methods:
            mlabel = METHODS[m][0]
            for lkey, llabel in losses.items():
                # A lossless method (Random) has one point per cell, not three. It is stored
                # under eval_sva's default logit_diff, so ride that pass and relabel; the other
                # two passes would emit the same point three times under three shapes.
                if m in LOSSLESS:
                    if lkey != "logit_diff":
                        continue
                    llabel = NO_LOSS
                for sub, slabel in substrates:
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
                    zero_row = abl == ZERO_ROW
                    facet = ((f"{abl}\n" if show_abl and not zero_row else "")
                             + (f"{slabel}, {inp_label}" if show_inp else slabel)
                             + (", zero" if zero_row else "")
                             + (f", {llabel}" if a.by_loss else "")
                             + f"\n{'·'.join(required[sub])}")
                    raw = (tenx_for(sub) if m in TENX_KEYS.values() and res == FIGURE_SOURCES[0][0]
                           else raw_for(res, sub))
                    r = group_avg(raw, m, lkey, sub, required)
                    if r is None:
                        have = {t for (mm, ll, ss, t) in raw if (mm, ll, ss) == (m, lkey, sub)}
                        miss = [g for g, ts in GROUPS
                                if g in required[sub] and not set(ts) <= have]
                        if have:   # nothing at all on disk = not submitted; only flag partials
                            dropped.append((abl, facet, mlabel, llabel, "+".join(miss)))
                        continue
                    # `_key` is the registry key, kept alongside the display label so
                    # draw_labelled can look up POINT_LABEL without reverse-mapping a label
                    # string back to its method.
                    # The zero row's compactness is CHANCE-CORRECTED, 2 AUC - 1, exactly as
                    # scripts/sva/make_sva_table.py --zero prints it (the lb > ls indicator has a
                    # 0.5 floor under zeroing). The MIB zero panel is NOT corrected -- its accuracy
                    # has no such floor -- again matching its table.
                    acc = 2 * r[0] - 1 if zero_row else r[0]
                    rows.append(dict(acc_auc=acc, faith_auc=r[1], cpr=r[3], method=mlabel,
                                     _key=m, loss=llabel, facet=facet, ablation=abl,
                                     groups="+".join(r[2])))
    if a.cpr and not a.zero and not a.by_loss:
        # MIB edge (MIB_TEST_EDGE / MIB_EDGE_FACET) dropped from the cut 2026-09-19 (requested):
        # the figure is node-level throughout; the edge numbers stay in the test table.
        rows = (mib_rows(figure_methods)
                + (mib_rows(figure_methods, MIB_TEST_ZERO, MIB_ZERO_FACET) if a.full else [])
                + rows)
    df = pd.DataFrame(rows)

    # ordering for consistent legends / facets (only 4 non-empty substrate x input combos)
    df["method"] = pd.Categorical(df["method"], [METHODS[m][0] for m in figure_methods])
    df["loss"] = pd.Categorical(df["loss"], list(LOSSES.values()) + [NO_LOSS])
    node_g, mlp_g = "·".join(required["node"]), "·".join(required["mlp"])
    sae_g = "·".join(required.get("resid_sae_span", required["mlp"]))
    # Wrap order, read left-to-right: all four Patched panels, then the three Zero-abl. ones
    # (the zero sweep has no +input arm). ncol=4 below therefore reproduces the old grid's
    # rows without reserving a framed empty cell for the combination that does not exist.
    # Iterate only the ablations this cut actually draws. With show_abl False the prefix is
    # dropped, so looping both would emit each panel label TWICE and pd.Categorical rejects
    # duplicate categories -- the failure is loud, but the fix belongs here rather than in a
    # dedupe downstream, because the order is the thing being defined.
    abls = ["Patched", "Zero-abl."] if show_abl else [None]
    SUB_IN = [("Node", node_g), ("MLP", mlp_g), ("MLP+Attn", mlp_g),
              ("SAE (resid)", sae_g), ("SAE (MLP out)", sae_g)]
    facet_order = [(f"{abl}\n" if show_abl else "")
                   + (f"{sub}, {inp}" if show_inp else sub) + f"\n{g}"
                   for abl in abls
                   for inp in (["−input", "+input"] if show_inp else [None])
                   for sub, g in SUB_IN]
    if a.by_loss:
        # rows = losses, logit-diff (the default) first; columns = the three trained substrates.
        facet_order = [f.replace("\n", f", {ll}\n", 1)
                       for ll in [LOSSES["logit_diff"], LOSSES["ce"], LOSSES["acc"]]
                       for f in facet_order if f.split("\n")[0] in ("Node", "MLP", "MLP+Attn")]
    elif a.cpr:
        # Row 1: MIB node, then SVA+; row 2 the same four columns under zero ablation.
        zero_facets = [f.replace("\n", ", zero\n", 1) for f in facet_order]
        facet_order = [MIB_FACET] + facet_order + ([MIB_ZERO_FACET] + zero_facets if a.full else [])
    # This list is the RENDER WHITELIST, not just a sort key: pd.Categorical maps anything absent
    # from it to NaN, and the panel then vanishes with no warning -- the point count in the
    # "wrote ..." line still includes it, which is the only visible trace. Adding a substrate to
    # FIGURE_SUBSTRATES and REQUIRED is therefore NOT enough; it must be added here too.
    keep = set(df["facet"]) | (set(PLACEHOLDER_NOTE) if a.cpr and not a.zero and not a.by_loss else set())
    df["facet"] = pd.Categorical(df["facet"], [f for f in facet_order if f in keep])
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
    # Method keys per row. The Method and Loss guides sit SIDE BY SIDE across one \textwidth, so
    # what has to fit is (widest method row) + (the 3-or-4 loss keys); wrapping the method guide
    # is the only lever, since the loss keys are one row by construction. 5 fits, 6-7 needs two
    # rows, and the full registry needs three -- and even then --all stays tight, which is why
    # its docstring calls it a debug cut rather than a paper figure.
    # Method keys per legend row. The budget depends on whether the Loss guide is beside it:
    # with it, the two share one \textwidth and 5 method keys is the limit; without it the
    # method guide has the strip to itself and 6 fit, which is what the single-loss default cut
    # needs so its legend does not wrap for no reason.
    _cap = 5 if show_loss else 6
    legend_rows = 1 if len(figure_methods) <= _cap else 2 if len(figure_methods) <= 7 else 3
    lossless = df["method"].isin([METHODS[m][0] for m in LOSSLESS])
    # Which rows the dashed guide may connect. The rule is geom_path's own precondition -- a
    # group needs at least two points to have a path -- evaluated per (facet, method) rather
    # than assumed from the method's name.
    #
    # It used to be `~lossless`, i.e. "Random is the one series with a single point". That was
    # true only while Random was the only method not run at all three losses, and it stopped
    # being true when the eps arm landed: `stopk-log-eps1e-2` is logit-diff-only, has a real
    # loss (so it is not LOSSLESS and must keep its square), and would have handed geom_path a
    # one-row group -- a zero-length segment plus ggplot2's "each group consists of only one
    # observation" warning. Testing the precondition directly also covers the case a partial
    # sweep produces, where a normally-three-loss method is down to one landed cell in a panel.
    out = f"plots/accauc_vs_{'cpr' if a.cpr else 'cpr_sva' if a.y_cpr else 'faithauc'}{suffix}.pdf"
    # The default cut leaves here: it is drawn by raw matplotlib (direct labels need measured
    # per-annotation geometry) and never touches the plotnine spec below. Returning BEFORE that
    # spec is built, rather than building and discarding it, keeps a plotnine change from being
    # able to break the paper figure -- and everything shared between the two renderers (the
    # frame, the categories, the facet order, the coverage report) has already happened above.
    if not other_cut:
        if a.cpr:
            # Uniform-k is this cut's MAttr (star, plain label); log-k is its ablation (circle,
            # "+log k"). Applied here, not at module level, so the default cut keeps its own
            # assignment -- see the comment on CPR_METHODS.
            global STAR_KEYS, OUTLINE_NON_STAR, FILLED_KEYS, LABEL_LEFT
            STAR_KEYS = CPR_STARS
            OUTLINE_NON_STAR = True    # baselines hollow, ours filled (all-filled was tried and reverted)
            FILLED_KEYS = CPR_FILLED
            # Left anchors, 2026-09-18 (checked on the PNG): MIB node's "MAttr" otherwise gets a
            # leader line down into the IG point because "+log k" sits just right of the star;
            # "+log k" on MLP (patched) and MLP+Attn (zero) ran past the right frame.
            # "MIB (edge" added 2026-09-19: the Edge Pruning point widened that panel's x-range
            # and pushed both MAttr labels into each other at the top-right corner.
            LABEL_LEFT = {"stopk-unif-eps1e-2": ("MIB (node", "MIB (edge"),
                          "stopk-log-eps1e-2": ("MLP\n", "MLP+Attn, zero")}
            POINT_LABEL.update(CPR_POINT_LABEL)
            # "log-AUC" vs bare "CPR": the x axis is the log-weighted IIA AUC, the y axis is
            # MIB's CPR, which is a LINEAR AUC over the kept proportion -- the labels are
            # what tell the reader the two axes weight the sparsity grid differently.
            fam = {METHODS[m][0]: FAMILY_COLOR[m] for m in figure_methods}
            if a.zero:
                nf = df["facet"].nunique()
                draw_labelled(df, figure_methods, out, ycol="cpr", ylabel="CPR (↑)",
                              xlabel="Compactness (↑)", colors=fam,
                              figsize=(LAB_FIG[0] * nf / 5, 1.3))
                print("wrote", out, f"({len(df)} points)")
                report(df, figure_methods, losses, dropped)
                return
            if a.by_loss:
                # 3 rows (losses) x 3 columns (substrates); Random has no loss (LOSSLESS) and
                # therefore appears in the logit-diff row only. The node CE / acc panels put the
                # star at the right edge, so its label anchors LEFT there (checked on the PNG).
                LABEL_LEFT = {"stopk-unif-eps1e-2": ("Node, CE", "Node, acc")}
                draw_labelled(df, figure_methods, out, ycol="cpr", ylabel="CPR (↑)",
                              xlabel="Compactness (↑)", colors=fam,
                              figsize=(LAB_FIG[0] * 0.9 * 3 / 4, 1.3), markers=CPR_MARKERS,
                              legend=True, free_lims=True, label_keys=CPR_LABELLED, ncol=3)
                print("wrote", out, f"({len(df)} points)")
                report(df, figure_methods, losses, dropped)
                return
            hl = []   # the EAP-IG-inp reference line is superseded by our own IG edge test point
            have_rnd = ((df["facet"] == MIB_FACET) & (df["_key"] == "Random")).any()
            if not have_rnd:
                rnd = mib_random_cpr()
                print(f"MIB Random control: test CPR {rnd:.2f} from Table 1; our random-ordering "
                      f"eval (results/random_test) is incomplete, so no IIA -- drawn as a line")
                hl.append((MIB_FACET, rnd, "Random", P.METHOD["Random"]))
            # 1.95 -> 1.65 when the SGD arm was dropped, -> 1.30 on 2026-09-11 (requested)
            # once DBM x8 was gone AND the MIB panels' crowded labels were anchored LEFT
            # (CPR_LABEL_LEFT). Without the left anchors 1.55 / 1.45 / 1.30 all pushed the
            # star's "MAttr" label a third of the axis below its marker while the overlap
            # count stayed 0 -- so check the rendered PNG, not just the two diagnostics
            # printed below, before changing this.
            draw_labelled(df, figure_methods, out, ycol="cpr", ylabel="CPR (↑)",
                          xlabel="Compactness (↑)", colors=fam, hlines=hl,
                          figsize=(LAB_FIG[0] * 0.9, 1.3), markers=CPR_MARKERS, legend=True,
                          free_lims=True, label_keys=CPR_LABELLED, ncol=4)
        elif a.y_cpr:
            draw_labelled(df, figure_methods, out, ycol="cpr", ylabel="CPR (↑)")
        else:
            draw_labelled(df, figure_methods, out)
        print("wrote", out, f"({len(df)} points)")
        report(df, figure_methods, losses, dropped)
        return

    pathable = df.groupby(["facet", "method"], observed=True)["loss"].transform("size") >= 2
    p = ggplot(df, aes("acc_auc", "faith_auc", fill="method", shape="loss"))
    # The dashed guide joins a method's points ACROSS losses, so a single-loss cut has nothing
    # for it to join and the layer is omitted entirely rather than handed an empty frame
    # (plotnine builds the layer either way and an all-empty one is a needless failure mode).
    if pathable.any():
        p += geom_path(aes(color="method", group="method"), data=df[pathable],
                       linetype="dashed", size=0.3, alpha=0.55, show_legend=False)
    p = (
        p
        # Black edge on every marker: method is carried by FILL, not colour, so points stay
        # legible where two methods land on top of each other and against the grid lines.
        # alpha=1 -- a translucent fill under a black edge reads as a different, muddier colour
        # wherever markers overlap, which is exactly where the distinction has to hold.
        #
        # (An earlier single-loss cut varied size here so coincident points did not hide each
        # other; the labelled renderer handles that, so sizes are flat.) At the node substrate
        # arms agree to 0.0024 on x and 0.005 on y -- inside the measured reproducibility floor,
        # i.e. genuinely the same point -- so at one size the last one drawn hides the other two
        # and three of the six legend entries simply do not appear. Jitter would be a lie about
        # a quantitative axis; drawing them large-to-small makes a coincident cluster read as
        # concentric rings, which is what "these are indistinguishable here" should look like.
        # The multi-loss cuts keep the single flat layer, where the dashed guide already tells
        # the reader which markers belong together.
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
        # Shape still MAPS to loss in a single-loss cut -- Random keeps its star and the trained
        # methods keep the logit-diff square, so a marker means the same thing in every version
        # of this figure -- but the guide is suppressed, because a "Loss" key listing one loss
        # and "n/a" explains nothing and costs a third of the legend strip.
        + scale_shape_manual(values=LOSS_SHAPE, name="Loss",
                             guide=(True if show_loss else None))
        + labs(x="Compactness (↑)", y="Faith log-AUC (↑)")
        # Method keys wrap to a second row once the cut is wide enough that one row would run
        # past \textwidth -- which is the failure the --all cut is documented as having, with
        # the Loss key's shape entries clipping off the right edge. Five 5.5in-wide keys fit;
        # the six-series --adam cut does not, and its longest label ("MAttr (Adam, ε=10⁻²)") is
        # the one that would be cut. Two rows costs ~7pt of height off panels that have it to
        # spare, which is cheaper than an unreadable key.
        + guides(fill=guide_legend(order=1, nrow=legend_rows),
                 **({"shape": guide_legend(order=2, nrow=1)} if show_loss else {}))
    )
    # The global figure_size is sized for the default TWO rows of panels. `--stepless` draws one
    # row (patched/−input only), so keeping 3.3in would stretch three panels to twice the height
    # of every other version of this figure and make the same points look like a different result.
    if df["facet"].nunique() <= 4:
        p += theme(figure_size=(5.5, 2.1))
    p.save(out, dpi=300, verbose=False)
    # PNG sibling for eyeballing the result without a PDF viewer, as the cause figure and the
    # iso-vs-cause curves already do. Only the PDF is copied into paper/figs.
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print("wrote", out, f"({len(df)} points)")
    report(df, figure_methods, losses, dropped)


if __name__ == "__main__":
    main()
