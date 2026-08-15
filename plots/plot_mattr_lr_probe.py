"""The MAttr lr x step-budget probe, plotted as convergence curves: training step on the x-axis.

DIAGNOSTIC, not a paper figure -- it exists to answer "is the neuron-substrate MAttr deficit
a tuning artifact?", and the answer is read off the SHAPE of these curves, not their endpoints.
(``plot_mattr_lr_metrics.py`` is the companion with lr on the x-axis.)

Two series per run, and the difference between them is the whole point:
  * line  = the k-independent metric on a FIXED 16-example TRAIN subset, logged every 250 steps.
    The train LOSS cannot be plotted this way -- it is measured at a k that moves over training
    (log-k redraws the budget each step), so loss at step 500 and at step 5000 are not the same
    quantity. These AUCs integrate the whole k grid and ARE comparable across steps.
  * star  = the final TEST value (100 examples). The probe saturates at 16 examples and reads
    LOW: on nounpp it moved +0.005 over a span where test moved +0.043. So a RISING line is
    evidence of under-convergence; a FLAT line is inconclusive, and the star is the number that
    goes in a table.

``--metric all`` gives the full grid, one row per probed metric. Only six of the eval's metrics
are probed during training (the probe runs the sweep on 16 examples every 250 steps and has to
stay cheap); faith-max, cause-AUC, cause p-src and k*90 exist only as final test values, so see
``plot_mattr_lr_metrics.py`` for those.

Reading the rows:
  * **acc-AUC is the one to conclude on.** faith-AUC is unbounded and gap-paddable -- a circuit
    can raise it by widening the clean/patched gap rather than by being the right circuit.
  * **F_clean / F_patch are flat by construction** -- they are the full-model and fully-patched
    endpoints that normalise faithfulness, i.e. properties of the task and the example set, not
    of the circuit. They are drawn to make that concrete (and to show the probe's 16 examples
    sitting at a different F_clean than the test 100, which is part of why the star and the line
    disagree). Since the denominator is fixed, faith-AUC moves only through the circuit's own
    F -- and gap padding shows up as faith exceeding 1, the circuit "outperforming" the full
    model, which is exactly what the nounpp row does at 1.5-2.0.
  * **k* is lower-is-better and log-scaled.** Every other row is higher-is-better.

Runs from before the probe existed (results/sva_sweep, the lr=0.05 @ 2000 headline) carry no
train_eval_log and so appear as a star with no line -- that gap is real, not a loading bug.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

# The six metrics train_eval_log carries, in reading order. (key, row label, log-scale y).
METRICS = [("acc_auc", "acc-AUC $\\uparrow$", False),
           ("faith_auc", "faith-AUC $\\uparrow$\n(gap-paddable)", False),
           ("cause_accsrc_auc", "cause acc-src\nAUC $\\uparrow$", False),
           ("kstar_50", "k* at 50% $\\downarrow$\n(units, log)", True),
           ("F_clean", "F_clean\n(gap endpoint)", False),
           ("F_patch", "F_patch\n(gap endpoint)", False)]

# variant -> panel title. `hard_topk` is the "+hard" ablation, NOT the headline; the headline is
# `topk` (soft top-k forward). Getting this backwards has bitten this repo before.
PANELS = [("addition", "topk", "addition -- soft top-k (headline)"),
          ("addition", "hard_topk", "addition -- + hard (sigmoid STE)"),
          ("addition", "hard_topk_identity", "addition -- identity STE (SGD)"),
          ("nounpp", "topk", "nounpp (SVA control) -- soft top-k")]


def load():
    """Every probe run, carrying the full metric dict rather than one series."""
    runs = []
    for f in sorted(glob.glob(str(ROOT / "results/probe_*/*/*.json"))):
        d = json.load(open(f))
        cfg = d.get("config", {})
        if not cfg:                      # pre-config runs cannot be attributed to an lr
            continue
        runs.append({"task": cfg["task"], "variant": cfg["variant"], "lr": cfg["lr"],
                     "steps": cfg["steps"], "seed": cfg.get("seed", 42),
                     "test": {k: d.get(k) for k, _, _ in METRICS},
                     "probe": d.get("train_eval_log", [])})
    return runs


def ig_ref():
    """The IG baseline per task, `_acc` loss variant to match these runs' --loss acc."""
    ref = {}
    for task, *_ in PANELS:
        try:
            ref[task] = json.load(open(ROOT / f"results/sva_sweep/{task}_llama3_mlp_ig_acc.json"))
        except FileNotFoundError:
            ref[task] = {}
    return ref


def draw_row(axes, runs, metric, logy, ig, color):
    """One metric across the four panels, with the addition panels on a shared y scale."""
    vals = [v for r in runs if r["task"] == "addition"
            for v in [p[metric] for p in r["probe"] if p.get(metric) is not None]
            + ([r["test"][metric]] if r["test"][metric] is not None else [])]
    for ax, (task, variant, _) in zip(axes, PANELS):
        for r in sorted([r for r in runs if r["task"] == task and r["variant"] == variant],
                        key=lambda r: (r["lr"], r["steps"])):
            c = color[r["lr"]]
            pts = [(p["step"], p[metric]) for p in r["probe"] if p.get(metric) is not None]
            if pts:
                ax.plot(*zip(*pts), color=c, lw=1.4, alpha=.85,
                        ls="-" if r["steps"] <= 8000 else "--")
            if r["test"][metric] is not None:
                # replicate seeds hollow: the seed spread on some of these is as large as the
                # lr effect, and a filled star implies more precision than one run has.
                kw = dict(mfc="none", mew=1.4) if r["seed"] != 42 else dict(mec="k", mew=.5)
                ax.plot(r["steps"], r["test"][metric], "*", color=c, ms=13, zorder=5, **kw)
        if ig[task].get(metric) is not None:
            ax.axhline(ig[task][metric], color="crimson", ls=":", lw=1.2, zorder=1)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=.25, lw=.5)
    if vals and not logy:                       # shared scale over the three addition panels
        lo, hi = min(vals), max(vals + [ig["addition"].get(metric, -np.inf)])
        pad = .05 * (hi - lo) or .01
        for ax in axes[:3]:
            ax.set_ylim(lo - pad, hi + pad)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="acc_auc",
                    choices=["all"] + [m for m, _, _ in METRICS])
    metric = ap.parse_args().metric
    runs, ig = load(), ig_ref()
    rows = METRICS if metric == "all" else [m for m in METRICS if m[0] == metric]

    lrs = sorted({r["lr"] for r in runs})
    # Colour by log lr: the sweep spans 100x, so a linear map would put 0.01/0.02/0.05 on top
    # of each other and spend all its contrast on the two values that lose outright.
    norm = plt.Normalize(np.log10(min(lrs)), np.log10(max(lrs)))
    cmap = plt.get_cmap("viridis")
    color = {lr: cmap(norm(np.log10(lr))) for lr in lrs}

    fig, axes = plt.subplots(len(rows), 4, figsize=(19, 2.6 * len(rows) + 1.2),
                             sharex=True, squeeze=False)
    for row, (key, label, logy) in zip(axes, rows):
        draw_row(row, runs, key, logy, ig, color)
        row[0].set_ylabel(label, fontsize=10)
    for ax, (_, _, title) in zip(axes[0], PANELS):
        ax.set_title(title, fontsize=11)
    for ax in axes[-1]:
        ax.set_xlabel("training step")

    handles = [Line2D([], [], color=color[lr], lw=2.5, label=f"lr {lr:g}") for lr in lrs]
    handles += [Line2D([], [], color="grey", lw=1.6, ls="-", label="$\\leq$8k steps"),
                Line2D([], [], color="grey", lw=1.6, ls="--", label="16k steps"),
                Line2D([], [], color="grey", marker="*", ls="", ms=12, label="final test"),
                Line2D([], [], color="grey", marker="*", ls="", ms=12, mfc="none",
                       label="seed replicate"),
                Line2D([], [], color="crimson", ls=":", lw=1.4, label="IG baseline")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9.5,
               frameon=False, bbox_to_anchor=(.5, -.02 if len(rows) > 1 else -.06))
    fig.suptitle("MAttr at the neuron (mlp) substrate, --loss acc: "
                 "training-time convergence, learning rate $\\times$ step budget", fontsize=13)
    fig.tight_layout(rect=(0, .03 if len(rows) > 1 else .06, 1, .97))
    stem = f"plots/mattr_lr_probe{'' if metric == 'acc_auc' else '_' + metric}"
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"{stem}.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote {stem}.{{pdf,png}} from {len(runs)} runs")


if __name__ == "__main__":
    main()
