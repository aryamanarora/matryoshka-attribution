"""The 20,000-step MAttr runs, drawn beside the training-step window the paper actually reports.

fig:optimiser-curves (plot_train_curves.py --overlay) stops at step 1999 because that is where
the sweep stops: `submit_sva_sweep.sh`'s MATTR_COMMON is 2000 steps, so every Adam-vs-SGD number
in the paper is read off the left edge of this figure. That panel shows Adam below SGD at the
neuron substrates. It cannot show whether Adam is WORSE or merely SLOWER, because it has no
data to the right of the vertical rule -- and the answer turns out to be mostly "slower":

    addition / llama3 / mlp / logit-diff, soft top-k, log-k     acc-AUC @2k -> @20k
      MAttr (Adam)  lr 0.05      0.361 -> 0.465     probe still rising past step ~10k
      MAttr (Adam)  lr 0.005     0.346 -> 0.443
      MAttr (SGD)   lr 1.0       0.496 -> 0.494     flat from step ~1000
    best-vs-best gap             0.135 -> 0.029

so ~78% of the published gap on this cell is training budget, not optimiser. See
`adam-sgd-mlp-gap-is-training-budget` for the full result including the L18 recall inversion.

WHY TWO PANELS AND NOT ONE CURVE EXTENDED. The left column is a DIFFERENT POPULATION and the
figure must not let them blur: it is the 18 paired ARITH cells of fig:optimiser-curves (3 losses
x 6 (substrate, task) cells with `train_eval_log` on both arms), and `addition` is not one of
them -- plot_train_curves.py's docstring says so in as many words ("`addition` has no paired
cell in any (substrate, loss)"), because the Adam runs that predate the logging were never
re-run for it. The right column is ONE cell, ONE loss, ONE seed. Drawing the 20k curves as a
continuation of the left panel's means would claim a 20k measurement for 18 cells that do not
have one. They share a y-axis per row and nothing else.

THE X-AXES ARE NOT ON THE SAME SCALE -- 0-2k on the left, 0-20k on the right, a 10x difference
between adjacent panels. That is deliberate and it is why the right panel shades its first 2000
steps: the shaded strip IS the left panel's full width. A shared 0-20k axis would compress the
published window into 10% of the left panel and make it unreadable, and a log x-axis would hide
exactly the thing being shown (how far past the reported budget the Adam curve keeps climbing).

PROBE FIDELITY DIFFERS BETWEEN THE COLUMNS. Left: --train-eval-examples 20 (the sweep default).
Right: 64, raised because the 20-example probe SATURATES (eval_sva.py:773 -- flat at steps 2000
and 6250 on nounpp/mlp while the real test acc-AUC rose +0.04), and reading "has Adam converged"
off a saturating probe is the exact mistake that comment documents. The 64-example probe tracks
the 100-example test eval well here (Adam lr 0.005: probe 0.323 @2k vs 0.346 test; SGD: 0.486 vs
0.496), so the right panel's endpoint is trustworthy; the left panel's absolute level is not
comparable to it, only its SHAPE is. Do not read a level difference across the two columns.

COLOUR IS THE OPTIMISER IN BOTH COLUMNS (Adam blue, SGD black, IG the untrained reference), which
is the whole point of putting them side by side -- a reader moving from fig:optimiser-curves does
not have to relearn the encoding. Linetype carries the LOSS on the left (as in the overlay
layout) and the LEARNING RATE on the right, where there is only one loss; the legend is split in
two for that reason.

Run:  uv run python plots/plot_train_curves_20k.py
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                                    # noqa: E402
from plot_accauc_vs_faithauc import ARITH                              # noqa: E402
from plot_train_curves import (ARM_COLOR, FIG_GRID, FS_GRID, LOSSES, LOSS_LINETYPE,  # noqa: E402
                               METRICS, REF, ROW_H, TRAINED, load, panel)

# The three 20k runs, from `STEPS=20000 PROBE_EVERY=500 PROBE_EX=64
# OUTBASE=results/sva_mlp_steps20k bash scripts/submit_sva_mlp_lr.sh` (2026-08-22).
# Both Adam LRs are kept: 0.05 is the headline's LR and the arm's 2k argmax, 0.005 is the
# sva_mlp_lr argmax, and they bracket the claim -- if only the better one were drawn, "Adam
# catches up" could be an LR fluke rather than a budget effect. The SGD arm is the CONTROL and is
# not optional: a 20k Adam run that gains 0.1 proves nothing if SGD gains as much over the same
# span, since the claim is about the GAP (submit_sva_mlp_lr.sh's "SUBMIT THE SGD CONTROL TOO").
RES20K = "results/sva_mlp_steps20k"
RUNS = [
    ("MAttr (Adam)", "lr 0.05",  "solid",
     f"{RES20K}/topk_adam/lr_0.05/addition_llama3_mlp_sufficient_topk_adam_bs1_s20000.json"),
    ("MAttr (Adam)", "lr 0.005", "dashed",
     f"{RES20K}/topk_adam/lr_0.005/addition_llama3_mlp_sufficient_topk_adam_bs1_s20000.json"),
    ("MAttr (SGD)",  "lr 1.0",   "solid",
     f"{RES20K}/topk_sgd/lr_1.0/addition_llama3_mlp_sufficient_topk_sgd_bs1_s20000.json"),
]
# IG on the SAME cell (addition / llama3 / mlp / logit_diff), from the sweep dir the paper reads.
# It is a single-pass attribution with no trajectory, so it is a horizontal constant that the
# trained curves either do or do not cross -- not "IG at step 0".
IG_JSON = "results/sva_sweep/addition_llama3_mlp_ig.json"
SUBSTRATE = "MLP"                 # the substrate where the Adam deficit lives, and the 20k cell's
PUBLISHED_STEPS = 2000            # where fig:optimiser-curves (and every reported number) ends
MROWS = ["acc_auc", "faith_auc"]  # same two rows as the overlay layout, same order
BAND = "#dcdcdc"


def left_panel_frame(res="results/sva_sweep"):
    """The MLP column of fig:optimiser-curves, rebuilt: paired ARITH cells, all three losses.

    Complete-case on the CURVE exactly as plot_train_curves.main does -- an arm with a result but
    no `train_eval_log` cannot be drawn, and including its partner alone would compare the two
    arms on different tasks. Returns (mean, per, task list).
    """
    cells = load(res, ARITH)
    rows, refs, kept = [], [], set()
    for loss, llabel in LOSSES:
        keys = [k for k in cells
                if k[0] == "mlp" and k[2] == loss
                and all(cells[k].get(m, {}).get("curve") for m, _ in TRAINED)
                and REF[0] in cells[k]]
        if not keys:
            continue
        kept |= {k[1] for k in keys}
        for met in MROWS:
            for arm, alabel in TRAINED:
                per_step = {}
                for k in keys:
                    for step, v in cells[k][arm]["curve"]:
                        per_step.setdefault(step, []).append(v[met])
                        rows.append(dict(loss=llabel, arm=alabel, metric=met, task=k[1],
                                         step=step, y=v[met], kind="task"))
                for step, vs in sorted(per_step.items()):
                    rows.append(dict(loss=llabel, arm=alabel, metric=met, task="__mean__",
                                     step=step, y=float(np.mean(vs)), kind="mean"))
            ref = float(np.mean([cells[k][REF[0]]["final"][met] for k in keys]))
            for step in (0, PUBLISHED_STEPS - 1):
                refs.append(dict(loss=llabel, arm=REF[1], metric=met, task="__mean__",
                                 step=step, y=ref, kind="mean"))
    df = pd.DataFrame(rows + refs)
    df["loss"] = pd.Categorical(df["loss"], [lb for _, lb in LOSSES])
    df["arm"] = pd.Categorical(df["arm"], [lb for _, lb in TRAINED] + [REF[1]])
    return df[df.kind == "mean"], df[df.kind == "task"], sorted(kept)


def load_20k():
    """(arm, lr label, linestyle, [(step, {metric: v})]) per run, plus the IG constants."""
    out = []
    for arm, lrlab, ls, f in RUNS:
        d = json.load(open(f))
        log = d.get("train_eval_log") or []
        if not log:
            raise SystemExit(f"{f}: no train_eval_log -- nothing to draw")
        out.append((arm, lrlab, ls,
                    [(e["step"], {m: e[m] for m in MROWS}) for e in log],
                    {m: d[m] for m in MROWS}))
    ig = json.load(open(IG_JSON))
    return out, {m: ig[m] for m in MROWS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/sva_sweep")
    ap.add_argument("--out", default="plots/train_curves_20k.pdf")
    a = ap.parse_args()

    mean, per, tasks = left_panel_frame(a.res)
    if not len(mean):
        raise SystemExit("no paired MLP cell has train_eval_log on both arms -- nothing to draw")
    runs, ig = load_20k()

    plt.rcParams.update(P.RC)
    fs = FS_GRID
    nr = len(MROWS)
    # The header strip has to hold TWO legend rows (8 handles at ncol=4) AND the column titles,
    # so it is taller than the overlay layout's 0.6in. At 0.6 the second handle row landed on top
    # of "MLP, ARITH sweep" and the n-tasks note under it.
    head = 1.15
    fig, axes = plt.subplots(nr, 2, figsize=(FIG_GRID[0], ROW_H * nr + head),
                             sharey="row", squeeze=False)
    for r, met in enumerate(MROWS):
        axl, axr = axes[r]
        # LEFT: fig:optimiser-curves' MLP column, verbatim (same panel(), same overlay=True
        # channels), so the two figures are the same chart and the reader can check it by eye.
        panel(axl, mean[mean.metric == met], per[per.metric == met], True, fs)
        axl.set_xlim(0, PUBLISHED_STEPS - 1)
        if r == 0:
            # Bottom-right, not panel()'s default top-left: the top-left of this panel holds the
            # column title and every curve here starts near the origin, so the corner the note
            # can occupy without covering data is the opposite one.
            axl.annotate(f"n={len(tasks)} tasks\n({', '.join(tasks)})", (0.97, 0.03),
                         xycoords="axes fraction", ha="right", va="bottom", fontsize=fs[2],
                         color="#666666")
        axl.set_ylabel(METRICS[met], fontsize=fs[0])

        # RIGHT: the 20k cell. The shaded strip is the left panel's ENTIRE width.
        axr.axvspan(0, PUBLISHED_STEPS, color=BAND, lw=0, zorder=0)
        axr.axvline(PUBLISHED_STEPS, color="#999999", lw=0.5, ls=(0, (1, 2)), zorder=1)
        axr.axhline(ig[met], color=ARM_COLOR[REF[1]], lw=1.0, ls="dotted", zorder=2)
        for arm, _, ls, curve, final in runs:
            xs = [s for s, _ in curve]
            ys = [v[met] for _, v in curve]
            axr.plot(xs, ys, ls=ls, lw=1.1, color=ARM_COLOR[arm], zorder=3)
            # The endpoint marker is the real 100-example test eval, not the 64-example probe;
            # it is what a results table would report, and it sits ~0.01-0.02 above the probe.
            axr.plot([xs[-1]], [final[met]], marker="o", ms=2.6, mew=0,
                     color=ARM_COLOR[arm], zorder=4)
        axr.set_xlim(0, 20000)
        axr.tick_params(labelsize=fs[1])
        P.furnish(axr)
        if r == 0:
            axl.set_title(f"{SUBSTRATE}, ARITH sweep (published window)", fontsize=fs[0], pad=3)
            axr.set_title("addition, 20k steps", fontsize=fs[0], pad=3)
            axr.annotate("reported\nbudget", (PUBLISHED_STEPS, 0.02), xycoords=("data",
                         "axes fraction"), ha="left", va="bottom", fontsize=fs[2],
                         color="#666666", xytext=(3, 0), textcoords="offset points")
        if r == nr - 1:
            for ax in (axl, axr):
                ax.set_xlabel("training step", fontsize=fs[0])

    fig.tight_layout()
    top = 1.0 - head / (ROW_H * nr + head)
    fig.subplots_adjust(top=top)
    # Two handle groups because the two columns spend LINETYPE on different variables: the loss on
    # the left, the learning rate on the right. One merged set would imply series that do not
    # exist (there is no "CE, lr 0.005" run anywhere in this figure).
    arms = [Line2D([0], [0], color=ARM_COLOR[lb], lw=1.1, ls="solid", label=lb)
            for lb in [x for _, x in TRAINED] + [REF[1]]]
    losses = [Line2D([0], [0], color="#444444", lw=1.1, ls=LOSS_LINETYPE[lb], label=f"{lb} (L)")
              for _, lb in LOSSES]
    lrs = [Line2D([0], [0], color="#444444", lw=1.1, ls=ls, label=f"{lab} (R)")
           for lab, ls in dict((lab, ls) for _, lab, ls, _ in RUNS).items()]
    # ncol=3 with 9 handles: matplotlib fills COLUMN-major, so the three groups (arms | losses |
    # learning rates) land one per column instead of wrapping across each other. The anchor clears
    # `top` by a title's height -- the column titles sit ABOVE the axes box, so a legend anchored
    # at top+0 lands on them even though subplots_adjust reserved the strip.
    fig.legend(handles=arms + losses + lrs, fontsize=fs[2], ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, top + 0.055), frameon=False,
               handletextpad=0.4, handlelength=2.0, columnspacing=1.2, labelspacing=0.25)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    print(f"\nleft panel: MLP x {len(tasks)} paired ARITH tasks ({', '.join(tasks)}) x 3 losses")
    print(f"            'addition' paired: {'addition' in tasks}   <- the 20k cell")
    print("\nright panel (addition/llama3/mlp/logit-diff), probe@2000 -> probe@20k -> test:")
    for arm, lrlab, _, curve, final in runs:
        d = dict(curve)
        at2k = max(s for s in d if s <= PUBLISHED_STEPS)
        for met in MROWS:
            print(f"  {arm:<12} {lrlab:<9} {met:<10} "
                  f"{d[at2k][met]:.3f} -> {curve[-1][1][met]:.3f} -> {final[met]:.3f}")
    print(f"  IG (untrained ref.)     " + "  ".join(f"{m} {ig[m]:.3f}" for m in MROWS))


if __name__ == "__main__":
    main()
