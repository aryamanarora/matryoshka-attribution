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
and 8/12 SVA (substrate, task) cells. So the three IG lines in a `--color loss` panel are
REPLICATES OF ONE CIRCUIT, and their spread is the evaluation's own noise floor, not a loss
effect. On ARITH acc-AUC that floor is mean 0.015, max 0.027 (faith-AUC: mean 0.034), which is
the yardstick any small Adam/SGD/IG gap in this figure has to clear. Do not read the IG spread
as IG responding to the loss, and do not report a difference under it.

COVERAGE IS THE BINDING CONSTRAINT HERE, and it is not the same constraint as the other
figures'. `train_eval_log` was added partway through the sweep, so it exists in 71/71 of the
softsgd runs but only 27/81 of the Adam ones. Pairing on it leaves:

    ARITH   18 cells with BOTH arms logged
    SVA      0 cells                          <- the Adam SVA runs all predate the logging

so arithmetic is the only place this comparison can be drawn at all, and even there the paired
Adam runs are a NON-RANDOM subset (the re-run ones). `addition` has no paired cell in any
(substrate, loss). Per-task curves are drawn thin underneath the mean so a 2-task panel cannot
pass for a converged average, and the task list per panel is printed.

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
TRAINED = [("stopk-log", "MAttr (Adam)"), ("softsgd-log", "MAttr (SGD)")]
REF = ("IG", "IG (untrained ref.)")
SUBSTRATES = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]
LOSSES = [("ce", "CE"), ("acc", "acc"), ("logit_diff", "logit-diff")]
# kstar_50 is logged as an absolute unit count and censors at `total`; normalise exactly as
# plot_method_ranks does so the three substrates are on one scale.
METRICS = {"acc_auc": "acc-AUC (↑)", "faith_auc": "faith-AUC (↑)",
           "kstar_pct": r"$k^\star$ (% of units, ↓)"}
LINETYPE = {"MAttr (Adam)": "solid", "MAttr (SGD)": "dashed", REF[1]: "dotted"}
# See the module docstring: hue is the optimizer here, as in plot_optimizer_lr.py's MIB panels.
ARM_COLOR = {"MAttr (Adam)": P.METHOD["MAttr"], "MAttr (SGD)": P.METHOD["MAttr (SGD)"],
             REF[1]: P.METHOD["IG"]}
# Two sizes. The 3x3 grid is a full-width appendix figure; --substrate emits ONE panel at the
# size plot_optimizer_lr.py uses, so an SVA+ training curve can sit in a subfigure row beside
# the MIB learning-rate panels. Point sizes are absolute, so the small panel needs its own.
FIG_GRID, FIG_ONE = (5.4, 4.6), (2.7, 2.15)
FS_GRID, FS_ONE = (7.5, 7, 6.5), (8, 7, 5.8)      # (axis label, tick, legend/annotation)
# `--color loss` overlays the three losses in one substrate panel, so the loss needs a hue.
# There is no project-wide loss palette (palette.py is a METHOD palette and deliberately says
# so), and inventing one there would bind every other figure. These are viridis stops, chosen
# because the losses have a natural order -- the sweep's own softest-signal-to-hardest order,
# CE -> acc -> logit-diff -- so an ordered ramp is the honest encoding; it is also CVD-safe and
# stays dark->mid->light in greyscale. Local to this figure on purpose.
LOSS_COLOR = {"CE": "#3b528b", "acc": "#21918c", "logit-diff": "#7ad151"}


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


def panel(ax, mean, per, colors, by_loss, fs, note=None):
    """One substrate x loss cell: thin per-task curves under the bold arm means.

    The per-task lines are not decoration. Every panel here averages 2-3 tasks, so the mean is
    not a converged average and must not be able to look like one -- if the thin lines disagree
    about the sign of the Adam/SGD gap, the panel is not evidence.
    """
    key = "loss" if by_loss else "arm"
    for (kval, arm, _), g in per.groupby([key, "arm", "task"], observed=True):
        g = g.sort_values("step")
        ax.plot(g.step, g.y, ls=LINETYPE[arm], lw=0.35, alpha=0.28,
                color=colors[kval], zorder=1)
    for (kval, arm), g in mean.groupby([key, "arm"], observed=True):
        g = g.sort_values("step")
        ax.plot(g.step, g.y, ls=LINETYPE[arm], lw=1.1, color=colors[kval], zorder=3)
    if note:
        ax.annotate(note, (0.03, 0.97), xycoords="axes fraction", ha="left", va="top",
                    fontsize=fs[2], color="#666666")
    ax.tick_params(labelsize=fs[1])
    P.furnish(ax)


def handles(colors, by_loss):
    """Legend handles. Built by hand because hue and linetype carry different variables: under
    --color loss the three loss colours and the three arm linetypes are independent, so one
    combined handle set would imply nine series that do not exist."""
    if by_loss:
        return ([Line2D([0], [0], color=colors[lb], lw=1.1, label=lb) for _, lb in LOSSES]
                + [Line2D([0], [0], color="#444444", lw=1.1, ls=LINETYPE[lb], label=lb)
                   for lb in [x for _, x in TRAINED] + [REF[1]]])
    return [Line2D([0], [0], color=colors[lb], lw=1.1, ls=LINETYPE[lb], label=lb)
            for lb in [x for _, x in TRAINED] + [REF[1]]]


def render(a, out, mean, per, nlab, colors, by_loss, one):
    plt.rcParams.update(P.RC)
    ylab, xlab = METRICS[a.metric], "training step"

    if one:
        fs = FS_ONE
        m = mean[(mean.substrate == a.substrate) & (mean.loss == a.loss)]
        p = per[(per.substrate == a.substrate) & (per.loss == a.loss)]
        fig, ax = plt.subplots(figsize=FIG_ONE)
        panel(ax, m, p, colors, False, fs,
              note=f"{a.substrate}, {a.loss}, n={p.task.nunique()} tasks")
        ax.set_xlabel(xlab, fontsize=fs[0])
        ax.set_ylabel(ylab, fontsize=fs[0])
        ax.legend(handles=handles(colors, False), fontsize=fs[2], loc="lower right",
                  frameon=True, framealpha=0.95, borderpad=0.35, handletextpad=0.4,
                  handlelength=2.0, labelspacing=0.3)
        fig.tight_layout()
    else:
        fs = FS_GRID
        subs = [s for _, s in SUBSTRATES]
        cols = [None] if by_loss else [lb for _, lb in LOSSES]
        # by_loss puts the three substrates in a row; otherwise substrates are rows and losses
        # columns. sharey across the WHOLE grid, as the plotnine version did: the substrates are
        # meant to be compared, and a free y-scale per row hides that MLP sits below Node.
        nr, nc = (1, len(subs)) if by_loss else (len(subs), len(cols))
        fig, axes = plt.subplots(nr, nc, figsize=(FIG_GRID[0], 2.3 if by_loss else FIG_GRID[1]),
                                 sharex=True, sharey=True, squeeze=False)
        for i, s in enumerate(subs):
            for j, c in enumerate(cols):
                ax = axes[0][i] if by_loss else axes[i][j]
                sel = mean.substrate == s
                selp = per.substrate == s
                if c is not None:
                    sel &= mean.loss == c
                    selp &= per.loss == c
                nn = nlab[(nlab.substrate == s) & ((nlab.loss == c) if c is not None
                                                   else True)] if len(nlab) else nlab
                panel(ax, mean[sel], per[selp], colors, by_loss, fs,
                      note=(nn.label.iloc[0] if len(nn) else "n=0"))
                if by_loss or i == 0:
                    ax.set_title(s if by_loss else c, fontsize=fs[0], pad=3)
                if (by_loss and i == 0) or (not by_loss and j == 0):
                    ax.set_ylabel(ylab, fontsize=fs[0])
                if by_loss or i == len(subs) - 1:
                    ax.set_xlabel(xlab, fontsize=fs[0])
            if not by_loss:
                # Substrate names on the right edge, where facet_grid put them. An extra ylabel
                # on the left would collide with the metric label the first column already owns.
                axes[i][-1].annotate(s, (1.02, 0.5), xycoords="axes fraction", rotation=270,
                                     ha="left", va="center", fontsize=fs[0])
        fig.tight_layout()
        # Shrink the axes block FIRST, then anchor the legend in the strip that opens up. Doing
        # it the other way round (legend at y=1.0, then subplots_adjust) put the handles on top
        # of the column titles, since tight_layout has no idea a figure-level legend exists.
        top = 0.86 if by_loss else 0.93
        fig.subplots_adjust(top=top)
        fig.legend(handles=handles(colors, by_loss), fontsize=fs[2], ncol=6 if by_loss else 3,
                   loc="lower center", bbox_to_anchor=(0.5, top + 0.015), frameon=False,
                   handletextpad=0.4, handlelength=2.0, columnspacing=1.2)
    fig.savefig(out, dpi=300)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--metric", choices=list(METRICS), default="acc_auc")
    ap.add_argument("--tasks", choices=["arith", "sva", "all"], default="arith")
    # In the 3x3 grid the loss is the facet and colour is the arm, which is now the encoding that
    # earns it (Adam blue / SGD black -- see the docstring). `--color loss` is the other view:
    # the losses overlay in one substrate panel, hue carries the loss and linetype the arm, so
    # the loss dependence is read WITHIN a panel rather than across columns. Three panels
    # instead of nine, at the cost of the optimizer contrast this figure is mainly about.
    ap.add_argument("--color", choices=["arm", "loss"], default="arm")
    # One substrate x one loss, at plot_optimizer_lr.py's subfigure size. Ignores --color, since
    # a single panel has no facet to spend hue on.
    ap.add_argument("--substrate", choices=[s for _, s in SUBSTRATES], default=None)
    ap.add_argument("--loss", choices=[lb for _, lb in LOSSES], default="logit-diff")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tasks = {"arith": ARITH, "sva": SVA, "all": ARITH + SVA}[a.tasks]
    one = a.substrate is not None
    suf = (f"_{a.substrate.lower().replace('+', '')}_{a.loss.replace('-', '')}" if one
           else "_byloss" if a.color == "loss" else "")
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
            for arm, alabel in TRAINED:
                per_step = {}
                for k in keys:
                    for step, v in cells[k][arm]["curve"]:
                        per_step.setdefault(step, []).append(v[a.metric])
                        rows.append(dict(substrate=slabel, loss=llabel, arm=alabel, task=k[1],
                                         step=step, y=v[a.metric], kind="task"))
                for step, vs in sorted(per_step.items()):
                    rows.append(dict(substrate=slabel, loss=llabel, arm=alabel, task="__mean__",
                                     step=step, y=float(np.mean(vs)), kind="mean"))
            ref = float(np.mean([cells[k][REF[0]]["final"][a.metric] for k in keys]))
            for step in (0, 1999):
                refs.append(dict(substrate=slabel, loss=llabel, arm=REF[1], task="__mean__",
                                 step=step, y=ref, kind="mean"))
            notes.append((slabel, llabel, len(keys), sorted(k[1] for k in keys)))
    if not rows:
        print("no cell has train_eval_log for both arms -- nothing to draw")
        return

    df = pd.DataFrame(rows + refs)
    df["substrate"] = pd.Categorical(df["substrate"], [s for _, s in SUBSTRATES])
    df["loss"] = pd.Categorical(df["loss"], [lb for _, lb in LOSSES])
    order = [lb for _, lb in TRAINED] + [REF[1]]
    df["arm"] = pd.Categorical(df["arm"], order)
    # plain string column: aes("arm+task") cannot add a categorical to a str. The loss must be
    # in the group key for --color loss, where three losses share one panel.
    df["grp"] = df["arm"].astype(str) + "/" + df["loss"].astype(str) + "/" + df["task"]
    mean, per = df[df.kind == "mean"], df[df.kind == "task"]

    by_loss = a.color == "loss"
    if by_loss:
        # One label per panel, spelling out every loss's n -- including the zeroes, since a loss
        # with no paired cell is simply ABSENT from an overlay panel and would otherwise be
        # invisible (Node/logit-diff is exactly this case).
        agg = {}
        for s, l, n, _ in notes:
            agg.setdefault(s, []).append(f"{l} {n}")
        # Wrapped after the second loss: on one line this label is wider than a 1.8in panel, so
        # it ran under the column title and into the neighbouring panel.
        nlab = pd.DataFrame([dict(substrate=s, label="n tasks: " + ", ".join(v[:2]) + ",\n"
                                  + ", ".join(v[2:]))
                             for s, v in agg.items()])
    else:
        nlab = pd.DataFrame([dict(substrate=s, loss=l, label=f"n={n} tasks")
                             for s, l, n, _ in notes if n]).astype({"loss": df["loss"].dtype})
    nlab = nlab.astype({"substrate": df["substrate"].dtype})
    nlab["step"] = 0
    nlab["y"] = mean["y"].max()

    colors = ({lb: LOSS_COLOR[lb] for _, lb in LOSSES} if by_loss else ARM_COLOR)
    if one and not len(mean[(mean.substrate == a.substrate) & (mean.loss == a.loss)]):
        raise SystemExit(f"--substrate {a.substrate} --loss {a.loss}: no paired cell "
                         f"(Node/logit-diff is the known-empty one)")
    render(a, out, mean, per, nlab, colors, by_loss, one)
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
