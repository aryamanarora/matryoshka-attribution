"""The MAttr lr sweep as metric-vs-lr curves: every metric the eval reports, one panel each.

The companion to plot_mattr_lr_probe.py, which plots the same runs against STEP. Here lr is the
x-axis and the step budget is the line style, which is the view that answers "where is the
optimum, and does it move with budget" -- it does, and that interaction is the reason a
single-budget lr grid is misleading.

Read with three warnings:

* **k\\* is lower-is-better and log-scaled**; every other panel is higher-is-better. The panels
  are titled accordingly, but the eye still reads "up = good" and gets k\\* backwards.
* **faith-AUC/faith-max are unbounded and gap-paddable** -- a circuit can raise them by widening
  the clean/patched gap instead of by being the right circuit. On these very runs faith-AUC and
  acc-AUC disagree about which lr wins at 8k. Conclude on acc-AUC, corroborate with k\\*.
* **the seed spread is drawn, not assumed.** Arm D replicated topk 0.02/0.05 at 8k with seed 1;
  those points are hollow. The 0.02-vs-0.05 gap at 8k is comparable to the seed spread, so no
  single-seed point in here supports a ranking on its own.

The IG reference line is `results/sva_sweep/<task>_llama3_mlp_ig_acc.json` -- the `_acc` loss
variant, matching these runs' --loss acc.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent

# (key, panel title, log-scale y). Order = reading order: the two headline scalars, then the
# cause/noising family, then the two sparsity thresholds.
METRICS = [("acc_auc", "acc-AUC $\\uparrow$", False),
           ("faith_auc", "faith-AUC $\\uparrow$ (gap-paddable)", False),
           ("faith_max", "faith-max $\\uparrow$ (gap-paddable)", False),
           # cause-AUC is LOWER-is-better and is the one metric here whose direction inverts.
           # It is the AUC of the NOISING faithfulness curve, normalised identically to the iso
           # one: 1.0 when nothing is corrupted, 0 when everything is. A circuit that destroys
           # the behaviour with few units drops fast and sweeps out a SMALL area. Verified
           # empirically over the 60 addition runs on disk: corr(acc_auc, cause_auc) = -0.44,
           # while the two source-token readouts below run +0.45/+0.51.
           ("cause_auc", "cause-AUC $\\downarrow$", False),
           ("cause_accsrc_auc", "cause acc-src AUC $\\uparrow$", False),
           ("cause_psrc_auc", "cause p-src AUC $\\uparrow$", False),
           ("kstar_50", "k* at 50% $\\downarrow$ (units)", True),
           ("kstar_90", "k* at 90% $\\downarrow$ (units)", True)]

VARIANTS = {"topk": ("soft top-k (headline)", "tab:blue"),
            "hard_topk": ("+ hard (sigmoid STE)", "tab:orange"),
            "hard_topk_identity": ("identity STE (SGD)", "tab:green")}

STEP_STYLE = {2000: (":", 4), 8000: ("-", 7), 16000: ("--", 9)}


def load(task):
    runs = []
    for f in sorted(glob.glob(str(ROOT / "results/probe_*/*/*.json"))):
        d = json.load(open(f))
        cfg = d.get("config", {})
        if not cfg or cfg["task"] != task:
            continue
        runs.append({**{k: d.get(k) for k, _, _ in METRICS},
                     "variant": cfg["variant"], "lr": cfg["lr"], "steps": cfg["steps"],
                     "seed": cfg.get("seed", 42)})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", default="addition")
    task = ap.parse_args().task
    runs = load(task)
    if not runs:
        raise SystemExit(f"no probe runs for task {task!r}")
    try:
        ig = json.load(open(ROOT / f"results/sva_sweep/{task}_llama3_mlp_ig_acc.json"))
    except FileNotFoundError:
        ig = {}

    fig, axes = plt.subplots(2, 4, figsize=(19, 8))
    for ax, (key, title, logy) in zip(axes.ravel(), METRICS):
        for variant, (_, color) in VARIANTS.items():
            for steps, (ls, ms) in STEP_STYLE.items():
                sel = [r for r in runs if r["variant"] == variant and r["steps"] == steps
                       and r[key] is not None]
                main_seed = sorted((r for r in sel if r["seed"] == 42), key=lambda r: r["lr"])
                if main_seed:
                    ax.plot([r["lr"] for r in main_seed], [r[key] for r in main_seed],
                            ls=ls, marker="o", ms=ms, color=color, lw=1.6, alpha=.9)
                # replicates: hollow, unconnected -- they are a spread, not a second curve.
                for r in sel:
                    if r["seed"] != 42:
                        ax.plot(r["lr"], r[key], marker="o", ms=ms, mfc="none", mec=color,
                                mew=1.6, ls="")
        if ig.get(key) is not None:
            ax.axhline(ig[key], color="crimson", ls=":", lw=1.4)
            ax.text(.99, ig[key], f" IG {ig[key]:.3g} ", color="crimson", fontsize=8,
                    va="bottom", ha="right", transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=.25, lw=.5)
    for ax in axes[1]:
        ax.set_xlabel("learning rate")

    handles = [Line2D([], [], color=c, lw=2.5, label=n) for n, c in VARIANTS.values()]
    handles += [Line2D([], [], color="grey", ls=ls, marker="o", ms=ms, label=f"{s} steps")
                for s, (ls, ms) in STEP_STYLE.items()]
    handles += [Line2D([], [], color="grey", marker="o", mfc="none", mew=1.6, ls="",
                       ms=7, label="seed replicate")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9.5,
               frameon=False, bbox_to_anchor=(.5, -.03))
    fig.suptitle(f"MAttr lr sweep, {task}/llama3/mlp, --loss acc: every reported metric",
                 fontsize=13)
    fig.tight_layout(rect=(0, .03, 1, .96))
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"plots/mattr_lr_metrics_{task}.{ext}", bbox_inches="tight", dpi=150)
    print(f"wrote plots/mattr_lr_metrics_{task}.{{pdf,png}} from {len(runs)} runs")


if __name__ == "__main__":
    main()
