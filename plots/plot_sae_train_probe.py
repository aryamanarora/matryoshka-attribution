"""Is MAttr undertrained on the SAE basis? The training probe vs IG's fixed score.

THE QUESTION. On the Llama-Scope residual-SAE substrate, MAttr's curves sit well to the RIGHT of
IG's in plots/plot_sva_curves.py -- it needs a far larger circuit to recover the behaviour --
which is the reverse of every other substrate, where MAttr leads. Three explanations are
consistent with that and are not separable from a final score alone: MAttr is undertrained here
(6.29M units at 2000 steps), the soft-vs-hard mask gap is worse on a sparse latent basis, or
MAttr genuinely loses. This figure settles the FIRST one: if the probe is still climbing at the
last step, the run had not converged and no comparison against it is a method result yet.

WHAT IS PLOTTED. eval_sva's training-time probe (train_eval_log) runs the whole metric suite
over the full k grid every 200 steps, so it is k-INDEPENDENT and comparable across steps -- the
raw training loss is not, because the log-k schedule redraws k every step.

*** THE TWO Y VALUES ARE NOT THE SAME QUANTITY. *** The probe is measured on a FIXED TINY TRAIN
SUBSET (--train-eval-examples, default 20); IG's line and MAttr's end marker are the final
100-example HELD-OUT score. So the vertical gap between a probe curve and its own end marker is
the probe's optimism, not progress, and IG's line is a held-out reference that the probe cannot
be compared to level-for-level. Read the probe for SHAPE (rising vs flat) and the markers for
LEVEL. eval_sva's own comment on this is blunter: a rising probe is evidence of
under-convergence, a flat one is inconclusive, because at 16 examples it saturated while the
held-out score was still climbing.

IG HAS NO CURVE because it does not train -- one closed-form pass, so it is a horizontal line by
construction. That is the comparison: a method with nothing to tune against two that may simply
not have run long enough.

ALL RUNS HERE POST-DATE the convex-blend rewrite of _sae_interchange (2026-08-29). The earlier
base-plus-delta runs are quarantined in results/_stale_saedelta and must not be mixed in: MAttr
trains THROUGH the intervention, so those masks were optimised against a diverging forward.

Run:  uv run python plots/plot_sae_train_probe.py [--metric acc_auc|faith_auc]
Out:  plots/sae_train_probe.pdf  (+ .png)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402
import plot_accauc_vs_faithauc as V                    # task lists, colours  # noqa: E402

RES = "results/sva_sweep"
SUB = "resid_sae_span"
ARMS = [("sufficient_topk_sgd_bs1", "MAttr$^\\mathrm{S}$", P.color("softsgd-log"), "solid"),
        ("sufficient_topk_adam_eps1e-2_bs1", "MAttr$^\\mathrm{A}$",
         P.color("stopk-log-eps1e-2"), "solid")]
REF = [("ig", "IG", P.color("IG"), (0, (3, 1.6))),
       ("ixg", "I×G", P.color("I×G"), (0, (1.2, 1.2)))]
FIG_W = 5.5
FS_TITLE, FS_TICK, FS_LAB, FS_LEG = 6.5, 5.5, 6.5, 5.5


def load(task, tag):
    p = f"{RES}/{task}_llama3_{SUB}_{tag}.json"
    try:
        return json.load(open(p))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="acc_auc", choices=["acc_auc", "faith_auc"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"plots/sae_train_probe{'' if a.metric == 'acc_auc' else '_faith'}.pdf"

    tasks = [t for t in V.SVA + V.ARITH
             if any(load(t, tag) for tag, *_ in ARMS)]
    if not tasks:
        raise SystemExit(f"no MAttr {SUB} runs in {RES}")

    plt.rcParams.update(P.RC)
    ncol = min(4, len(tasks))
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(FIG_W, 1.35 * nrow + 0.75),
                             sharex=True, sharey=True, squeeze=False)
    axf = axes.ravel()
    rising = []
    for ax, task in zip(axf, tasks):
        for tag, lab, col, ls in ARMS:
            d = load(task, tag)
            if not d or not d.get("train_eval_log"):
                continue
            tl = d["train_eval_log"]
            xs = [e["step"] for e in tl]
            ys = [e.get(a.metric) for e in tl]
            ok = [(x, y) for x, y in zip(xs, ys) if y is not None and np.isfinite(y)]
            if not ok:
                continue
            ax.plot(*zip(*ok), lw=0.9, color=col, ls=ls, zorder=3)
            # Held-out final score as a marker at the right edge -- a DIFFERENT quantity from
            # the probe (see the docstring), so it is a marker, never joined to the curve.
            fv = d.get(a.metric)
            if fv is not None and np.isfinite(fv):
                ax.plot([xs[-1]], [fv], marker="o", ms=2.8, color=col,
                        mec="#000000", mew=0.3, ls="none", zorder=4)
            # "still climbing at the end" = last third above the middle third
            n = len(ok)
            if n >= 6:
                mid = np.mean([y for _, y in ok[n // 3:2 * n // 3]])
                end = np.mean([y for _, y in ok[-2:]])
                if end - mid > 0.02:
                    rising.append(f"{task}/{lab}")
        for tag, lab, col, ls in REF:
            d = load(task, tag)
            if d and d.get(a.metric) is not None and np.isfinite(d[a.metric]):
                ax.axhline(d[a.metric], color=col, ls=ls, lw=0.9, zorder=2)
        ax.set_title(task.replace("_", " "), fontsize=FS_TITLE, pad=2.5)
        P.furnish(ax)
        ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1.5)
    for ax in axf[len(tasks):]:
        ax.set_visible(False)

    fig.supxlabel("training step", fontsize=FS_LAB, y=0.015)
    fig.supylabel({"acc_auc": "Compactness (↑)", "faith_auc": "Faith AUC (↑)"}[a.metric],
                  fontsize=FS_LAB, x=0.008)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.5, rect=(0.012, 0.03, 1, 0.93))
    handles = [Line2D([], [], color=c, ls=ls, lw=0.9, label=f"{lab} (probe, train subset)")
               for _, lab, c, ls in ARMS]
    handles += [Line2D([], [], color=c, ls=ls, lw=0.9, label=f"{lab} (held-out, no training)")
                for _, lab, c, ls in REF]
    handles += [Line2D([], [], color="#000000", marker="o", ms=2.8, ls="none", mew=0.3,
                       mfc="#ffffff", label="MAttr final held-out")]
    fig.legend(handles=handles, fontsize=FS_LEG, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), frameon=False, handlelength=2.0,
               handletextpad=0.5, columnspacing=1.2)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)
    print("\nstill climbing at the last probe point (end-third > mid-third by >0.02):")
    print("  " + (", ".join(rising) if rising else "none"))
    print(f"\n{'task':<11}" + "".join(f"{l:>26}" for _, l, _, _ in ARMS) + f"{'IG held-out':>13}")
    for t in tasks:
        row = [f"{t:<11}"]
        for tag, _, _, _ in ARMS:
            d = load(t, tag); tl = (d or {}).get("train_eval_log") or []
            p0 = tl[0].get(a.metric) if tl else None
            pl = tl[-1].get(a.metric) if tl else None
            fv = (d or {}).get(a.metric)
            row.append(f"  probe {p0:.2f}->{pl:.2f}  test {fv:.3f}" if None not in (p0, pl, fv)
                       else f"{'--':>26}")
        d = load(t, "ig")
        row.append(f"{d[a.metric]:>13.3f}" if d and np.isfinite(d.get(a.metric, np.nan)) else f"{'--':>13}")
        print("".join(row))


if __name__ == "__main__":
    sys.exit(main())
