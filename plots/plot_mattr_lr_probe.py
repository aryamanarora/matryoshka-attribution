"""The MAttr lr x step-budget probe, plotted as convergence curves.

DIAGNOSTIC, not a paper figure -- it exists to answer "is the neuron-substrate MAttr deficit
a tuning artifact?", and the answer is read off the SHAPE of these curves, not their endpoints.

Two series per run, and the difference between them is the whole point:
  * line  = ``probe/acc_auc``, the k-independent metric on a FIXED 16-example TRAIN subset,
    logged every 250 steps. The train LOSS cannot be plotted this way -- it is measured at a k
    that moves over training (log-k redraws the budget each step), so loss at step 500 and at
    step 5000 are not the same quantity. acc-AUC integrates the whole k grid and IS comparable.
  * star  = the final TEST acc-AUC (100 examples). The probe saturates at 16 examples and reads
    LOW: on nounpp it moved +0.005 over a span where test moved +0.043. So a RISING line is
    evidence of under-convergence; a FLAT line is inconclusive, and the star is the number that
    goes in a table.

Reads results/probe_{lr,lr_low,steps}/*/*.json, i.e. arms A-C of scripts/submit_mattr_probes.sh.
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

# Gradient-baseline reference (llama3/mlp, the `_acc` loss variant, from results/sva_sweep): the
# bar MAttr has to clear at this substrate. IG is the strongest baseline in both cells.
#
# faith-AUC is plottable but is NOT the metric to conclude on: it integrates a faithfulness
# ratio that a circuit can inflate by widening the clean/patched gap rather than by being the
# right circuit ("gap padding"), and it is unbounded above -- the ig5 runs in results/sva_sweep
# reach 3.9 on nounpp while scoring WORSE on acc-AUC. Read it as a secondary check on whether an
# acc-AUC change is a real circuit change, not as a target.
IG = {"acc_auc": {"addition": 0.497, "nounpp": 0.689},
      "faith_auc": {"addition": 0.484, "nounpp": 1.050}}

# variant -> (panel title, paper role). `hard_topk` is the "+hard" ablation, NOT the headline;
# the headline is `topk` (soft top-k forward). Getting this backwards has bitten this repo before.
PANELS = [("addition", "topk", "addition -- soft top-k (headline)"),
          ("addition", "hard_topk", "addition -- + hard (sigmoid STE)"),
          ("addition", "hard_topk_identity", "addition -- identity STE (SGD)"),
          ("nounpp", "topk", "nounpp (SVA control) -- soft top-k")]


def load(metric):
    runs = []
    for f in sorted(glob.glob(str(ROOT / "results/probe_*/*/*.json"))):
        d = json.load(open(f))
        cfg, summ = d.get("config", {}), d.get("summary", d)
        if not cfg:                      # pre-config runs cannot be attributed to an lr
            continue
        runs.append({"task": cfg["task"], "variant": cfg["variant"], "lr": cfg["lr"],
                     "steps": cfg["steps"], "seed": cfg.get("seed"),
                     "test": summ.get(metric),
                     "probe": [(r["step"], r[metric]) for r in d.get("train_eval_log", [])
                               if r.get(metric) is not None]})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="acc_auc", choices=sorted(IG))
    metric = ap.parse_args().metric
    runs = load(metric)
    ig = IG[metric]
    lrs = sorted({r["lr"] for r in runs})
    # Colour by log lr: the sweep spans 100x, so a linear map would put 0.01/0.02/0.05 on top
    # of each other and spend all its contrast on the two values that lose outright.
    norm = plt.Normalize(np.log10(min(lrs)), np.log10(max(lrs)))
    cmap = plt.get_cmap("viridis")
    color = {lr: cmap(norm(np.log10(lr))) for lr in lrs}

    # Shared y across the three ADDITION panels: with per-panel autoscale the "+hard" panel came
    # out 0.1-0.4 and the headline panel 0.05-0.5, so two curves 0.3 apart looked identical.
    # nounpp keeps its own scale -- it is a different task, and on faith-AUC it sits near 1.1
    # while addition sits near 0.5, so one scale would flatten every addition panel to a line.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    vals = [v for r in runs if r["task"] == "addition"
            for v in [x[1] for x in r["probe"]] + ([r["test"]] if r["test"] is not None else [])]
    pad = .05 * (max(vals) - min(vals))
    for ax in axes.ravel()[:3]:
        ax.set_ylim(min(vals) - pad, max(max(vals), ig["addition"]) + pad)
    for ax, (task, variant, title) in zip(axes.ravel(), PANELS):
        sel = [r for r in runs if r["task"] == task and r["variant"] == variant]
        for r in sorted(sel, key=lambda r: (r["lr"], r["steps"])):
            c = color[r["lr"]]
            if r["probe"]:
                xs, ys = zip(*r["probe"])
                ax.plot(xs, ys, color=c, lw=1.6, alpha=.9,
                        ls="-" if r["steps"] <= 8000 else "--")
            if r["test"] is not None:
                ax.plot(r["steps"], r["test"], "*", color=c, ms=15, mec="k", mew=.5, zorder=5)
        ax.axhline(ig[task], color="crimson", ls=":", lw=1.4, zorder=1)
        ax.text(.99, ig[task], f" IG {ig[task]:.3f} ", color="crimson", fontsize=8, va="bottom",
                ha="right", transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.25, lw=.5)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    label = metric.replace("_auc", "-AUC")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"{label}\n(line: 16-ex train probe / star: 100-ex test)", fontsize=9)

    handles = [Line2D([], [], color=color[lr], lw=2, label=f"lr {lr:g}") for lr in lrs]
    handles += [Line2D([], [], color="grey", lw=1.6, ls="-", label="$\\leq$8k steps"),
                Line2D([], [], color="grey", lw=1.6, ls="--", label="16k steps"),
                Line2D([], [], color="grey", marker="*", ls="", ms=12, label="final test")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               frameon=False, bbox_to_anchor=(.5, -.04))
    fig.suptitle(f"MAttr at the neuron (mlp) substrate, {label}: "
                 "learning rate $\\times$ step budget", fontsize=12)
    fig.tight_layout(rect=(0, .02, 1, .97))
    stem = f"plots/mattr_lr_probe{'' if metric == 'acc_auc' else '_' + metric}"
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"{stem}.{ext}", bbox_inches="tight", dpi=160)
    print(f"wrote {stem}.{{pdf,png}} from {len(runs)} runs")


if __name__ == "__main__":
    main()
