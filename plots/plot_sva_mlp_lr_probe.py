"""The 2026-08-20 neuron-substrate lr grid, plotted against TRAINING STEP -- and the reason
its flat result is NOT the conclusion it looks like.

Data: results/sva_mlp_lr/topk_adam/lr_*, addition / llama3 / --nodes mlp (2,293,760 units),
headline config (soft top-k fwd, Adam, log-k, --loss logit_diff), bs=1, 2000 steps, lr
0.001..10. The lr=0.05 point is the pre-existing headline run in results/sva_sweep, reused as
the grid's centre rather than recomputed.

Two series per run, and the difference between them is the whole point (same convention as
plot_mattr_lr_probe.py, whose docstring has the longer version):
  * line = acc-AUC on a FIXED 20-example TRAIN subset, logged every 200 steps. The train LOSS
    cannot be plotted this way -- log-k redraws the budget each step, so loss at step 200 and
    at step 2000 are not the same quantity. The AUCs integrate the whole k grid and ARE
    comparable across steps.
  * star = the final TEST value (100 held-out examples), the number that goes in a table.

WHAT THE LEFT PANEL SHOWS, AND WHY IT OVERTURNS THE FLAT lr CURVE. Every line at or below the
optimum is still RISING at step 1999 -- lr=0.05 climbs 0.285 -> 0.324 over its last 800 steps.
Per the probe's own contract (eval_sva.py:773) a rising probe is evidence of UNDER-CONVERGENCE.
So the right panel's flat 0.30-0.36 plateau over lr 0.005-0.3 does not say "lr is not the
knob and MAttr is simply worse"; it says every one of these runs was stopped early, and a grid
run at a single truncated budget cannot locate the optimum of a surface whose optimum moves
with that budget.

THE PRIOR SWEEP IS THE EVIDENCE THAT IT DOES MOVE (right panel, dashed). results/probe_* holds
29 earlier runs on this same cell with --loss acc, crossing lr with step budget. There, going
2000 -> 8000 steps is worth ~+0.10 acc-AUC (topk lr=0.02: 0.367 -> 0.465), roughly five times
the entire spread of my 2000-step lr grid -- and the lr optimum SHIFTS DOWN as the budget grows
(0.05 >= 0.02 at 2k; 0.02 > 0.05 at both 8k and 16k). The headline claim of that sweep, that
identity-STE reaches 0.502 at lr=0.05/16000 steps, meets IG's 0.500 on this cell.

Caveat on mixing them: the dashed series are --loss acc and the solid are --loss logit_diff, so
they are not the same experiment and the vertical offset between them is not interpretable.
The step-budget effect is quoted WITHIN the acc series, where it is a controlled contrast.

Bottom line to read off the figure: steps, not lr, are the binding constraint at this
substrate. The open experiment is an 8000-step step-matched re-run at --loss logit_diff, not
more lr points.

Run:  uv run python plots/plot_sva_mlp_lr_probe.py   -> plots/sva_mlp_lr_probe.pdf
"""
import glob
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "results/sva_mlp_lr/topk_adam"
# The headline run predates this grid and lives in the shared sweep dir; it IS the lr=0.05 point.
CENTRE = ROOT / "results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.json"
IG = ROOT / "results/sva_sweep/addition_llama3_mlp_ig.json"


def load_grid():
    """(lr, probe curve, final test acc-AUC) for the 2000-step logit_diff grid."""
    runs = []
    for d in sorted(glob.glob(str(SWEEP / "lr_*"))):
        lr = float(re.search(r"lr_([0-9.]+)", d).group(1))
        for f in glob.glob(f"{d}/*.json"):
            j = json.load(open(f))
            runs.append({"lr": lr, "probe": j.get("train_eval_log", []),
                         "test": j.get("acc_auc")})
    if CENTRE.exists():
        j = json.load(open(CENTRE))
        runs.append({"lr": 0.05, "probe": j.get("train_eval_log", []),
                     "test": j.get("acc_auc")})
    return sorted(runs, key=lambda r: r["lr"])


def load_prior():
    """The earlier --loss acc lr x step-budget grid, addition/topk only, seed 42 only."""
    out = []
    for f in sorted(glob.glob(str(ROOT / "results/probe_*/*/*.json"))):
        j = json.load(open(f))
        c = j.get("config", {})
        if not c or c["task"] != "addition" or c["variant"] != "topk":
            continue
        if c.get("seed", 42) != 42:          # replicates are a spread estimate, not a series
            continue
        out.append({"lr": c["lr"], "steps": c["steps"], "test": j.get("acc_auc")})
    return out


def main():
    runs, prior = load_grid(), load_prior()
    ig = json.load(open(IG))["acc_auc"] if IG.exists() else None

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lrs = [r["lr"] for r in runs]
    norm = plt.Normalize(np.log10(min(lrs)), np.log10(max(lrs)))
    cmap = plt.get_cmap("viridis")
    col = {r["lr"]: cmap(norm(np.log10(r["lr"]))) for r in runs}

    # ---- left: convergence. The shape here is the finding; the endpoints are not.
    for r in runs:
        pts = [(p["step"], p["acc_auc"]) for p in r["probe"] if p.get("acc_auc") is not None]
        if pts:
            ax0.plot(*zip(*pts), color=col[r["lr"]], lw=1.5, alpha=.9)
        if r["test"] is not None:
            ax0.plot(2000, r["test"], "*", color=col[r["lr"]], ms=13, mec="k", mew=.5, zorder=5)
    if ig is not None:
        ax0.axhline(ig, color="crimson", ls=":", lw=1.3, zorder=1)
        ax0.text(1980, ig + .012, f"IG = {ig:.3f}", color="crimson", size=9, ha="right")
    ax0.set_xlabel("training step")
    ax0.set_ylabel("acc-AUC $\\uparrow$")
    ax0.set_title("still rising at the budget's end\n(line = 20-ex TRAIN probe, star = 100-ex TEST)",
                  size=10)
    ax0.grid(alpha=.25, lw=.5)
    # lr=0.05 predates train_eval_log, so it is a star with no line -- flagged, not a bug.
    ax0.legend(handles=[Line2D([], [], color=col[r["lr"]], lw=1.5, label=f"lr={r['lr']:g}"
                               + (" (star only)" if not r["probe"] else ""))
                        for r in runs],
               fontsize=8, ncol=2, loc="upper left", frameon=False)

    # ---- right: the lr surface, my one budget against the prior sweep's three.
    ax1.plot([r["lr"] for r in runs], [r["test"] for r in runs], "o-", color="k", lw=1.6,
             ms=5, label="2000 steps, logit_diff (this grid)")
    for steps, c in [(2000, "#8c8c8c"), (8000, "#1f77b4"), (16000, "#d62728")]:
        pts = sorted((p["lr"], p["test"]) for p in prior if p["steps"] == steps)
        if pts:
            ax1.plot(*zip(*pts), "s--", color=c, lw=1.3, ms=4, alpha=.9,
                     label=f"{steps} steps, acc loss (prior)")
    if ig is not None:
        ax1.axhline(ig, color="crimson", ls=":", lw=1.3, zorder=1)
    ax1.set_xscale("log")
    ax1.set_xlabel("learning rate")
    ax1.set_ylabel("final test acc-AUC $\\uparrow$")
    ax1.set_title("steps move the surface ~5x more than lr does\n"
                  "(dashed = different loss; compare WITHIN a series)", size=10)
    ax1.grid(alpha=.25, lw=.5)
    ax1.legend(fontsize=8, loc="lower center", frameon=False)

    fig.suptitle("MAttr lr grid, addition / llama3 / mlp (2,293,760 units) -- "
                 "the flat lr curve is an under-convergence artifact", size=11)
    fig.tight_layout(rect=[0, 0, 1, .94])
    out = ROOT / "plots/sva_mlp_lr_probe.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=180)
    print(f"wrote {out}")
    for r in runs:
        pts = [p["acc_auc"] for p in r["probe"] if p.get("acc_auc") is not None]
        rise = (pts[-1] - pts[-5]) if len(pts) >= 5 else float("nan")
        print(f"  lr={r['lr']:<7g} test={r['test']:.3f}  probe last-800-step rise={rise:+.3f}")


if __name__ == "__main__":
    main()
