"""Eval metric vs training step for MAttr under Adam vs SGD, against the IG reference.

The sweep's runners checkpoint an eval every 200 steps into `train_eval_log` (11 points,
step 0..1999, each carrying acc_auc / faith_auc / kstar_50). This plots that, which answers a
question none of the endpoint figures can: whether SGD's advantage at the neuron substrates is
a better OPTIMUM or merely a faster approach to the same one -- i.e. whether the Adam curve is
still climbing at step 2000 or has flattened below SGD.

IG IS NOT TRAINED. It is a single-pass attribution, so it has no trajectory and appears as a
horizontal reference line at its own value on the same cells. Reading it as "IG at step 0" would
be wrong; it is a constant that the trained curves either do or do not cross.

IG ALSO HAS NO LOSS, and this figure gives that fact a job. The sweep carries a `loss` field on
the IG runs because every run has one, but IG's attribution does not use it: k* -- a
deterministic function of the scores -- is IDENTICAL across the three loss labels in 10/12 ARITH
and 8/12 SVA (substrate, task) cells. So the three IG lines in an `--overlay` panel are
REPLICATES OF ONE CIRCUIT, and their spread is the evaluation's own noise floor, not a loss
effect. On ARITH acc-AUC that floor is mean 0.015, max 0.027 (faith-AUC: mean 0.034), which is
the yardstick any small Adam/SGD/IG gap in this figure has to clear. Do not read the IG spread
as IG responding to the loss, and do not report a difference under it.

COVERAGE IS THE BINDING CONSTRAINT HERE, and it is not the same constraint as the other
figures'. `train_eval_log` was added partway through the sweep, so it exists in 81/81 of the
softsgd runs but only 39/81 of the Adam ones. Pairing on it leaves:

    ARITH   36 cells with BOTH arms logged     <- all 4 tasks x 3 substrates x 3 losses
    SVA      0 cells                           <- the Adam SVA runs all predate the logging

so arithmetic is the only place this comparison can be drawn at all. As of 2026-08-25 ARITH is
COMPLETE there: `submit_addition_adam_relog.sh` re-ran the nine `addition` Adam cells (the last
ones that predated the logging), so every panel is n=4 and the old "Node/logit-diff has no paired
cell" hole is closed. Those re-runs are fresh training, not a rescoring, and their endpoints moved
-- the MLP cells by +0.04 to +0.06 acc-AUC in Adam's favour, the node cells by <0.005 (the
pre-re-run values are kept in results/_backup_addition_adam_prelog_20260824). That is run-to-run
variance on one task, so it does not overturn the figure's ordering, but a number quoted off an
`addition`/Adam/MLP cell before that date is stale. Per-task curves are drawn thin underneath the
mean so a thin panel cannot pass for a converged average, and the task list per panel is printed.

ADAM IS BLUE AND SGD IS BLACK, matching plot_optimizer_lr.py's MIB panels. This REVERSES what
this figure used to do -- both arms took MAttr's blue and separated by linetype, on palette.py's
rule that a colour is a method and a hyperparameter variant is a linetype. That rule is right
where the optimizer is incidental; it is wrong here, because in these two figures the optimizer
IS the contrast, and palette.py gives "MAttr (SGD)" its own (measured, CVD-checked) hex for
exactly this case. Linetype is still drawn along with hue, so the pair survives greyscale.

STYLE IS SHARED WITH THE MIB FIGURES, not reimplemented: palette.P.RC for the font and
P.furnish() for the grid/spine treatment. This was a plotnine facet_grid until 2026-08-21 and is
now raw matplotlib for that reason alone -- plotnine's theme cannot reach the mathtext rcParams,
so "$k^\\star$" set in a different face from the label beside it.

Run:  uv run python plots/plot_train_curves.py
      uv run python plots/plot_train_curves.py --metric faith_auc
      uv run python plots/plot_train_curves.py --metric kstar_pct --tasks sva
      uv run python plots/plot_train_curves.py --substrate MLP     # one subfigure-sized panel
      uv run python plots/plot_train_curves.py --overlay           # acc/faith x substrate, losses overlaid
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                          # noqa: E402
from plot_accauc_vs_faithauc import ARITH, SVA, parse_method  # noqa: E402

RES = "results/sva_sweep"
# HIGH-EPS ADAM (eps=1e-2) is the Adam arm as of 2026-08-30, not the default-eps one. At this
# width the default eps=1e-8 makes Adam's update effectively sign(g) -- the learned score becomes
# a signed COUNT of steps with every trace of effect magnitude divided out -- so `stopk-log` was
# measuring that degeneracy rather than Adam. Only logit_diff was run at eps=1e-2, which is all
# these panels use; --all-losses variants still need the default-eps key.
TRAINED = [("stopk-log-eps1e-2", "MAttr (Adam)"), ("softsgd-log", "MAttr (SGD)")]
REF = ("IG", "IG (untrained ref.)")
SUBSTRATES = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]
LOSSES = [("ce", "CE"), ("acc", "acc"), ("logit_diff", "logit-diff")]
# kstar_50 is logged as an absolute unit count and censors at `total`; normalise exactly as
# plot_method_ranks does so the three substrates are on one scale.
#: AXIS LABELS ARE THE PAPER'S METRIC NAMES, not the log keys. `acc_auc` is the compactness
#: measure and `faith_auc` is CPR; they were drawn as "acc-AUC"/"faith-AUC" here while every other
#: figure in the repo had already moved (plot_ablation_bars, plot_dbm_l0_vs_metric,
#: plot_mib_accauc_cpr_scatter, plot_epsgrid_facets), so a reader crossing figures met two names
#: for one quantity. The keys are untouched -- only what is printed on an axis.
METRICS = {"acc_auc": "Compactness (↑)", "faith_auc": "CPR (↑)",
           "kstar_pct": r"$k^\star$ (% of units, ↓)"}
LINETYPE = {"MAttr (Adam)": "solid", "MAttr (SGD)": "dashed", REF[1]: "dotted"}
# See the module docstring: hue is the optimizer here, as in plot_optimizer_lr.py's MIB panels.
ARM_COLOR = {"MAttr (Adam)": P.METHOD["MAttr"], "MAttr (SGD)": P.METHOD["MAttr (SGD)"],
             REF[1]: P.METHOD["IG"]}
# Two sizes. The 3x3 grid is a full-width appendix figure; --substrate emits ONE panel at the
# size plot_optimizer_lr.py uses, so an SVA+ training curve can sit in a subfigure row beside
# the MIB learning-rate panels. Point sizes are absolute, so the small panel needs its own.
FIG_GRID, FIG_ONE = (5.4, 4.6), (2.7, 2.15)
# Per-row height of the --overlay layout, which is as tall as it has rows. 1.55in puts a
# full-width 2-row version at 5.4 x 3.7; the half-width overlay reads ROW_H_NARROW below.
# This module is imported for its constants (plot_train_curves_20k takes the OVERLAY set,
# plot_rank_stability_20k takes ARM_COLOR/FS_GRID), so repointing a name here reshapes a figure
# in another file -- ROW_H was briefly redefined to the narrow value and silently squashed
# train_curves_20k.pdf from 4.25in to 3.25in tall. Add a new name, do not repoint an old one.
ROW_H = 1.55
FS_GRID, FS_ONE = (7.5, 7, 6.5), (8, 7, 5.8)      # (axis label, tick, legend/annotation)
# The --overlay layout is drawn HALF-WIDTH (2.7in, i.e. ~0.48\linewidth) so it can sit in a
# subfigure beside neuron_recall.pdf. That is a re-LAYOUT, not a scale: \includegraphics is
# left at width=\linewidth inside the subfigure, so the point sizes below are what prints.
# Shrinking the old 5.4in figure with a width= key instead would have put 7.5pt labels on the
# page at 3.7pt. If it ever goes back to a full-width float, pass --wide; the two sets of
# constants are kept side by side so neither is a magic number.
FIG_OVERLAY_W, FS_OVERLAY, ROW_H_NARROW = 2.7, (6.5, 5.5, 5.5), 1.05
FIG_OVERLAY_WIDE_W = 5.4
# Header strip reserved above the axes, split the way plot_neuron_recall splits it: LEG_H for
# the handles themselves and TITLE_H for the column titles, which set_title draws ABOVE the
# axes rectangle and therefore INTO the same band. Sizing the strip for the handles alone is
# what put the legend on top of "Node"/"MLP"/"MLP+Attn" -- the arithmetic looks right and the
# two still collide, because the titles are not inside the axes the fraction is measured from.
# The legend is anchored above the title band rather than above the axes for the same reason.
LEG_H, LEG_H_WIDE, TITLE_H = 0.52, 0.6, 0.22
# `--overlay` puts all three losses in one panel, so the loss needs its own channel. It gets
# LINETYPE and the arm keeps ARM_COLOR, so the two layouts are colour-identical: a reader moving
# between them does not have to relearn which line is Adam. The earlier version of this view did
# the opposite (a local viridis ramp for the loss, arm on linetype), which meant the same figure
# script drew Adam in blue in one layout and in three different greens in the other.
#
# Ordered hyperparameter -> ordered channel is preserved: dotted / dashed / solid follows the
# sweep's own softest-signal-to-hardest order, CE -> acc -> logit-diff, with the headline loss
# solid. IG carries a `loss` field it does not use, so it draws three lines here; per the
# docstring those are replicates of one circuit and their spread is the evaluation noise floor.
LOSS_LINETYPE = {"CE": "dotted", "acc": "dashed", "logit-diff": "solid"}


def load(res=RES, tasks=None):
    """(substrate, task, loss) -> {arm: {"curve": [(step, {metric: v})], "final": {...}}}."""
    cells = {}
    keep = {k for k, _ in TRAINED} | {REF[0]}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        m = parse_method(os.path.basename(f), d)
        if m not in keep or (tasks and d["task"] not in tasks):
            continue
        tot = d["total"]

        def pct(e):
            ks = e.get("kstar_50")
            return 100.0 * (tot if ks is None else ks) / tot

        rec = {"final": {"acc_auc": d["acc_auc"], "faith_auc": d["faith_auc"],
                         "kstar_pct": pct(d)}}
        rec["curve"] = [(e["step"], {"acc_auc": e["acc_auc"], "faith_auc": e["faith_auc"],
                                     "kstar_pct": pct(e)}) for e in (d.get("train_eval_log") or [])]
        cells.setdefault((d["nodes"], d["task"], d["loss"]), {})[m] = rec
    return cells


def kstar_axis(ax):
    """k* spans 0.003% to 100% -- 4.5 decades -- because a run that never reaches 50%
    faithfulness is CENSORED to the full unit count. On a linear axis the censored head at 100
    owns the panel and everything after step ~400 is a flat line on the floor. The hairline marks
    that ceiling for what it is: "did not reach 50%", not "selected every unit"."""
    ax.set_yscale("log")
    ax.axhline(100, lw=0.4, ls=(0, (1, 2)), color="#999999", zorder=0)


def dash(arm, loss, overlay):
    """Linetype carries the loss when the losses share a panel, the arm otherwise."""
    return LOSS_LINETYPE[loss] if overlay else LINETYPE[arm]


def panel(ax, mean, per, overlay, fs, note=None):
    """One cell: thin per-task curves under the bold arm means.

    The per-task lines are not decoration. Every panel here averages 4 tasks, so the mean is
    not a converged average and must not be able to look like one -- if the thin lines disagree
    about the sign of the Adam/SGD gap, the panel is not evidence. They are drawn fainter in the
    overlay layout, which stacks three losses into the space the grid gives one: 24 thin lines
    at the grid's weight is a wash the 9 means cannot be read out of.
    """
    lw, al = (0.3, 0.14) if overlay else (0.35, 0.28)
    for (arm, loss, _), g in per.groupby(["arm", "loss", "task"], observed=True):
        g = g.sort_values("step")
        ax.plot(g.step, g.y, ls=dash(arm, loss, overlay), lw=lw, alpha=al,
                color=ARM_COLOR[arm], zorder=1)
    for (arm, loss), g in mean.groupby(["arm", "loss"], observed=True):
        g = g.sort_values("step")
        ax.plot(g.step, g.y, ls=dash(arm, loss, overlay), lw=1.1, color=ARM_COLOR[arm], zorder=3)
    if note:
        ax.annotate(note, (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                    fontsize=fs[2], color="#666666")
    ax.tick_params(labelsize=fs[1])
    P.furnish(ax)


def handles(overlay):
    """Legend handles. Built by hand because in the overlay layout hue and linetype carry
    different variables -- three arm colours and three loss dashes are independent, so one
    combined handle set would imply nine series that do not exist."""
    arms = [Line2D([0], [0], color=ARM_COLOR[lb], lw=1.1,
                   ls="solid" if overlay else LINETYPE[lb], label=lb)
            for lb in [x for _, x in TRAINED] + [REF[1]]]
    if not overlay:
        return arms
    return arms + [Line2D([0], [0], color="#444444", lw=1.1, ls=LOSS_LINETYPE[lb], label=lb)
                   for _, lb in LOSSES]


def render(a, out, mean, per, nlab, mrows, overlay, one):
    plt.rcParams.update(P.RC)
    xlab = "training step"
    subs = [s for _, s in SUBSTRATES]

    if one:
        fs = FS_ONE
        sel = (mean.substrate == a.substrate) & (mean.loss == a.loss) & (mean.metric == a.metric)
        selp = (per.substrate == a.substrate) & (per.loss == a.loss) & (per.metric == a.metric)
        fig, ax = plt.subplots(figsize=FIG_ONE)
        panel(ax, mean[sel], per[selp], False, fs,
              note=f"{a.substrate}, {a.loss}, n={per[selp].task.nunique()} tasks")
        ax.set_xlabel(xlab, fontsize=fs[0])
        ax.set_ylabel(METRICS[a.metric], fontsize=fs[0])
        ax.legend(handles=handles(False), fontsize=fs[2], loc="lower right",
                  frameon=True, framealpha=0.95, borderpad=0.35, handletextpad=0.4,
                  handlelength=2.0, labelspacing=0.3)
        fig.tight_layout()
    elif overlay:
        # metric rows x substrate columns, all three losses in every panel. sharey by ROW only:
        # the substrates within a row are meant to be compared, but acc-AUC and faith-AUC are
        # different scales and forcing them onto one axis flattens whichever has less range.
        fs = FS_GRID if a.wide else FS_OVERLAY
        w = FIG_OVERLAY_WIDE_W if a.wide else FIG_OVERLAY_W
        row_h = ROW_H if a.wide else ROW_H_NARROW
        leg_h = (LEG_H_WIDE if a.wide else LEG_H) + TITLE_H
        nr = len(mrows)
        fig, axes = plt.subplots(nr, len(subs), figsize=(w, row_h * nr + leg_h),
                                 sharex=True, sharey="row", squeeze=False)
        for r, met in enumerate(mrows):
            for i, s in enumerate(subs):
                ax = axes[r][i]
                nn = nlab[nlab.substrate == s]
                # The n-per-loss note goes on the top row only; it is a property of the column,
                # and repeating it under every metric is the clutter this layout is trying to cut.
                panel(ax, mean[(mean.substrate == s) & (mean.metric == met)],
                      per[(per.substrate == s) & (per.metric == met)], True, fs,
                      note=(nn.label.iloc[0] if len(nn) else "n=0") if r == 0 else None)
                if r == 0:
                    ax.set_title(s, fontsize=fs[0], pad=3)
                if i == 0:
                    ax.set_ylabel(METRICS[met], fontsize=fs[0])
                # Once, under the middle column, unless the figure is wide enough for three.
                # "training step" is ~0.75in at 6.5pt against a 0.75in panel, so three copies at
                # half width leave the outer two overlapping their neighbours' tick labels.
                if r == nr - 1 and (a.wide or i == len(subs) // 2):
                    ax.set_xlabel(xlab, fontsize=fs[0])
                if met == "kstar_pct":
                    kstar_axis(ax)
        # w_pad=0: the substrate columns share y within a row and so have no inner tick labels
        # to separate. The gap that survives is the x tick labels' overhang ("2000" sits on the
        # right spine), which tight_layout reserves regardless of the pad.
        fig.tight_layout(w_pad=0.0)
        fh = row_h * nr + leg_h
        top = 1.0 - leg_h / fh
        fig.subplots_adjust(top=top)
        # 6 handles fit one row across 5.4in and need two across 2.7in. Wrapping is not automatic
        # -- ncol=6 at half width silently overruns the figure and the outer handles are clipped
        # by bbox_inches, so the arm the reader most needs to identify goes missing.
        fig.legend(handles=handles(True), fontsize=fs[2], ncol=6 if a.wide else 3,
                   loc="lower center", bbox_to_anchor=(0.5, top + TITLE_H / fh), frameon=False,
                   handletextpad=0.4, handlelength=2.0, columnspacing=1.2)
    else:
        fs = FS_GRID
        cols = [lb for _, lb in LOSSES]
        # Substrates are rows and losses columns. sharey across the WHOLE grid, as the plotnine
        # version did: the substrates are meant to be compared, and a free y-scale per row hides
        # that MLP sits below Node.
        fig, axes = plt.subplots(len(subs), len(cols), figsize=FIG_GRID,
                                 sharex=True, sharey=True, squeeze=False)
        for i, s in enumerate(subs):
            for j, c in enumerate(cols):
                ax = axes[i][j]
                sel = (mean.substrate == s) & (mean.loss == c) & (mean.metric == a.metric)
                selp = (per.substrate == s) & (per.loss == c) & (per.metric == a.metric)
                nn = nlab[(nlab.substrate == s) & (nlab.loss == c)] if len(nlab) else nlab
                panel(ax, mean[sel], per[selp], False, fs,
                      note=(nn.label.iloc[0] if len(nn) else "n=0"))
                if i == 0:
                    ax.set_title(c, fontsize=fs[0], pad=3)
                if j == 0:
                    ax.set_ylabel(METRICS[a.metric], fontsize=fs[0])
                if i == len(subs) - 1:
                    ax.set_xlabel(xlab, fontsize=fs[0])
                if a.metric == "kstar_pct":
                    kstar_axis(ax)
            # Substrate names on the right edge, where facet_grid put them. An extra ylabel on
            # the left would collide with the metric label the first column already owns.
            axes[i][-1].annotate(s, (1.02, 0.5), xycoords="axes fraction", rotation=270,
                                 ha="left", va="center", fontsize=fs[0])
        fig.tight_layout()
        # Shrink the axes block FIRST, then anchor the legend in the strip that opens up. Doing
        # it the other way round (legend at y=1.0, then subplots_adjust) put the handles on top
        # of the column titles, since tight_layout has no idea a figure-level legend exists.
        top = 0.93
        fig.subplots_adjust(top=top)
        fig.legend(handles=handles(False), fontsize=fs[2], ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, top + 0.015), frameon=False,
                   handletextpad=0.4, handlelength=2.0, columnspacing=1.2)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--metric", choices=list(METRICS), default="acc_auc")
    ap.add_argument("--tasks", choices=["arith", "sva", "all"], default="arith")
    # In the 3x3 grid the loss is the facet. `--overlay` is the other view: the three losses share
    # a panel (linetype) so the loss dependence is read WITHIN a panel rather than across columns,
    # which frees the column axis for the substrate and the ROW axis for the METRIC -- all three
    # of them, so --metric does nothing here. Colour stays the arm in both layouts, so a reader
    # moving between them does not have to relearn which line is Adam.
    ap.add_argument("--overlay", action="store_true")
    # The overlay layout is half-width by DEFAULT (see FIG_OVERLAY_W) because that is the size
    # the paper uses it at. Default rather than an opt-in flag so that re-running this script
    # bare reproduces the committed figs/train_curves_arith_overlay.pdf instead of silently
    # overwriting it with a full-width one that then prints at 3.7pt.
    ap.add_argument("--wide", action="store_true",
                    help="full-\\linewidth overlay (the pre-2026-08-24 size)")
    # One substrate x one loss, at plot_optimizer_lr.py's subfigure size. Ignores --overlay,
    # since a single panel has no row axis to spend on a second metric.
    ap.add_argument("--substrate", choices=[s for _, s in SUBSTRATES], default=None)
    ap.add_argument("--loss", choices=[lb for _, lb in LOSSES], default="logit-diff")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = {"arith": ARITH, "sva": SVA, "all": ARITH + SVA}[a.tasks]
    one = a.substrate is not None
    overlay = a.overlay and not one     # a single panel has no row axis to spend on a metric
    # The overlay layout spends its row axis on the metric, so it draws a FIXED set and ignores
    # --metric -- which is also why its filename carries no metric. The two AUCs only: they are
    # one quality axis read together, on the same 0-1-ish scale and the same linear ticks. k* was
    # a third row briefly and is not, because it needs a log axis and a censoring hairline of its
    # own -- three rows where one is a different kind of chart reads as three figures stacked.
    # Re-add it by putting "kstar_pct" back here; kstar_axis() already handles the axis.
    mrows = ["acc_auc", "faith_auc"] if overlay else [a.metric]
    if overlay:
        out = a.out or f"plots/train_curves_{a.tasks}_overlay.pdf"
    else:
        suf = (f"_{a.substrate.lower().replace('+', '')}_{a.loss.replace('-', '')}" if one else "")
        out = a.out or f"plots/train_curves_{a.tasks}_{a.metric}{suf}.pdf"

    cells = load(a.res, tasks)
    lab = dict(TRAINED)
    rows, refs, notes, dropped = [], [], [], []
    for sub, slabel in SUBSTRATES:
        for loss, llabel in LOSSES:
            # Complete-case on the CURVE, not just on the run: an arm with a result but no
            # train_eval_log cannot be drawn, and including its partner alone would compare the
            # two arms on different tasks.
            keys = [k for k in cells
                    if k[0] == sub and k[2] == loss
                    and all(cells[k].get(m, {}).get("curve") for m, _ in TRAINED)
                    and REF[0] in cells[k]]
            miss = [k for k in cells
                    if k[0] == sub and k[2] == loss and k not in keys]
            dropped += [(k, [m for m, _ in TRAINED
                             if not cells[k].get(m, {}).get("curve")]) for k in miss]
            if not keys:
                notes.append((slabel, llabel, 0, []))
                continue
            for met in mrows:
                for arm, alabel in TRAINED:
                    per_step = {}
                    for k in keys:
                        for step, v in cells[k][arm]["curve"]:
                            per_step.setdefault(step, []).append(v[met])
                            rows.append(dict(substrate=slabel, loss=llabel, arm=alabel,
                                             metric=met, task=k[1], step=step, y=v[met],
                                             kind="task"))
                    for step, vs in sorted(per_step.items()):
                        rows.append(dict(substrate=slabel, loss=llabel, arm=alabel, metric=met,
                                         task="__mean__", step=step, y=float(np.mean(vs)),
                                         kind="mean"))
                ref = float(np.mean([cells[k][REF[0]]["final"][met] for k in keys]))
                for step in (0, 1999):
                    refs.append(dict(substrate=slabel, loss=llabel, arm=REF[1], metric=met,
                                     task="__mean__", step=step, y=ref, kind="mean"))
            notes.append((slabel, llabel, len(keys), sorted(k[1] for k in keys)))
    if not rows:
        print("no cell has train_eval_log for both arms -- nothing to draw")
        return

    df = pd.DataFrame(rows + refs)
    df["substrate"] = pd.Categorical(df["substrate"], [s for _, s in SUBSTRATES])
    df["loss"] = pd.Categorical(df["loss"], [lb for _, lb in LOSSES])
    order = [lb for _, lb in TRAINED] + [REF[1]]
    df["arm"] = pd.Categorical(df["arm"], order)
    mean, per = df[df.kind == "mean"], df[df.kind == "task"]

    if overlay:
        # One label per panel, spelling out every loss's n -- including the zeroes, since a loss
        # with no paired cell is simply ABSENT from an overlay panel and would otherwise be
        # invisible. (Node/logit-diff was exactly this case until the addition re-runs landed on
        # 2026-08-25; no panel is short now, but the branch stays -- an SVA overlay would hit it.)
        agg = {}
        for s, l, n, _ in notes:
            agg.setdefault(s, []).append(f"{l} {n}")
        # Collapsed to "n=3" when the three losses agree, which today they all do -- the long
        # form spelled "3" out three times per panel and, at the half-width size this layout is
        # now drawn at, overran the panel and printed across its neighbour. The full per-loss
        # form is kept for the case it exists to cover: when the n's DIFFER, the zeroes are the
        # whole point (a loss with no paired cell is simply absent from the panel), so it is a
        # collapse-when-redundant, not a truncation. Wrapped after the second loss because on
        # one line it is wider than the panel either way.
        def note(v):
            ns = {x.rsplit(" ", 1)[1] for x in v}
            if len(ns) == 1:
                return f"n={ns.pop()}"
            return "n tasks: " + ", ".join(v[:2]) + ",\n" + ", ".join(v[2:])

        nlab = pd.DataFrame([dict(substrate=s, label=note(v)) for s, v in agg.items()])
    else:
        nlab = pd.DataFrame([dict(substrate=s, loss=l, label=f"n={n} tasks")
                             for s, l, n, _ in notes if n]).astype({"loss": df["loss"].dtype})
    nlab = nlab.astype({"substrate": df["substrate"].dtype})

    if one and not len(mean[(mean.substrate == a.substrate) & (mean.loss == a.loss)]):
        raise SystemExit(f"--substrate {a.substrate} --loss {a.loss}: no paired cell "
                         f"(every ARITH combination has one since 2026-08-25; --tasks sva has "
                         f"none at all)")
    render(a, out, mean, per, nlab, mrows, overlay, one)
    print("wrote", out)
    print("\npaired cells per panel (substrate x loss) -- both arms must have train_eval_log:")
    for slabel, llabel, n, ts in notes:
        print(f"  {slabel:<10} {llabel:<12} n={n}  {', '.join(ts)}")
    if dropped:
        import collections
        c = collections.Counter()
        for k, missing in dropped:
            for m in missing:
                c[m] += 1
        print("\nunpaired cells, by the arm whose curve is missing:")
        for m, n in c.most_common():
            print(f"  {m:<14} {n:3d}")
        notask = sorted({k[1] for k, _ in dropped} - {t for _, _, _, ts in notes for t in ts})
        if notask:
            print(f"  tasks with NO paired cell anywhere: {', '.join(notask)}")


if __name__ == "__main__":
    main()
