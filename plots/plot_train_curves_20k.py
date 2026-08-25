"""The 20,000-step MAttr runs: does Adam's deficit at the neuron substrates survive a longer budget?

fig:optimiser-curves (plot_train_curves.py --overlay) stops at step 1999 because that is where
the sweep stops: `submit_sva_sweep.sh`'s MATTR_COMMON is 2000 steps, so every Adam-vs-SGD number
in the paper is read off that window. It shows Adam below SGD at the neuron substrates. It cannot
show whether Adam is WORSE or merely SLOWER, because it has no data past the budget -- and the
answer turns out to be mostly "slower":

    addition / llama3 / mlp / logit-diff, soft top-k, log-k     acc-AUC @2k -> @20k
      MAttr (Adam)  lr 0.05      0.361 -> 0.465     probe still rising past step ~10k
      MAttr (Adam)  lr 0.005     0.346 -> 0.443
      MAttr (SGD)   lr 1.0       0.496 -> 0.494     flat from step ~1000
    best-vs-best gap             0.135 -> 0.029

so ~78% of the published gap on this cell is training budget, not optimiser. See
`adam-sgd-mlp-gap-is-training-budget` for the full result including the L18 recall inversion.

THIS IS ONE CELL, ONE LOSS, ONE SEED, and the figure must not be read as a 20k measurement of
fig:optimiser-curves' population. Until 2026-08-24 this script drew that population in a left-hand
column (the 18 paired ARITH cells, rebuilt from the sweep) so the two could be compared side by
side; that column is gone, at the cost that the reader now has to carry the comparison across
figures. The shaded strip is what carries it: it marks step 0-2000, the ENTIRE width of
fig:optimiser-curves, so the part of this curve that the paper's numbers actually cover is
visually separated from the part they do not. Do not remove the band without replacing that cue.

`addition` IS NOW IN fig:optimiser-curves' population (2026-08-25). Its Adam runs predated
`train_eval_log` and the pairing rule dropped them until `submit_addition_adam_relog.sh` re-ran
the nine cells, so that figure went from 3 tasks to 4 and every panel is now n=4. The two figures
are therefore NO LONGER DISJOINT -- this cell is one of the four averaged there -- which cuts both
ways: the 20k curve can now be read as "the addition line of fig:optimiser-curves, continued", but
"an independent cell" is no longer an available defence of it. The 20k runs are a SEPARATE wave
(results/sva_mlp_steps20k, 64-example probe) and were not themselves re-run, so the @2k endpoints
quoted above are still the pre-re-run numbers for the 20k arm and are not expected to equal the
sweep's `addition` cell.

THE PROBE IS NOT THE SWEEP'S PROBE. The sweep uses --train-eval-examples 20; these runs use 64,
raised because the 20-example probe SATURATES (eval_sva.py:773 -- flat at steps 2000 and 6250 on
nounpp/mlp while the real test acc-AUC rose +0.04), and reading "has Adam converged" off a
saturating probe is the exact mistake that comment documents. The 64-example probe tracks the
100-example test eval well here (Adam lr 0.005: probe 0.323 @2k vs 0.346 test; SGD: 0.486 vs
0.496), so the endpoints are trustworthy. The endpoint DOT on each curve is that test eval, not
the probe. Levels are still not comparable to fig:optimiser-curves, which is on the 20-example
probe -- only the shapes are.

COLOUR IS THE OPTIMISER (Adam blue, SGD black, IG the untrained reference), matching
fig:optimiser-curves so a reader moving between them does not relearn the encoding. Linetype is
the learning rate. Both Adam LRs are drawn on purpose: 0.05 is the headline's LR and the arm's 2k
argmax, 0.005 is the sva_mlp_lr argmax, and they bracket the claim -- with only the better one,
"Adam catches up" could be an LR fluke rather than a budget effect.

SIZE: half-width (2.7in, ~0.48\\linewidth), sharing plot_train_curves.py's overlay constants so
this figure and train_curves_arith_overlay.pdf print at the same point sizes and can sit in one
subfigure row. \\includegraphics stays at width=\\linewidth inside the subfigure -- scaling a
5.4in figure down with a width= key instead would put 6.5pt labels on the page at 3.2pt.

Run:  uv run python plots/plot_train_curves_20k.py
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                                    # noqa: E402
from plot_train_curves import (ARM_COLOR, FIG_OVERLAY_W, FS_OVERLAY,   # noqa: E402
                               LEG_H, METRICS, REF, ROW_H_NARROW, TITLE_H)

# The three 20k runs, from `STEPS=20000 PROBE_EVERY=500 PROBE_EX=64
# OUTBASE=results/sva_mlp_steps20k bash scripts/submit_sva_mlp_lr.sh` (2026-08-22).
# The SGD arm is the CONTROL and is not optional: a 20k Adam run that gains 0.1 proves nothing if
# SGD gains as much over the same span, since the claim is about the GAP
# (submit_sva_mlp_lr.sh's "SUBMIT THE SGD CONTROL TOO").
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
PUBLISHED_STEPS = 2000            # where fig:optimiser-curves (and every reported number) ends
MROWS = ["acc_auc", "faith_auc"]  # same two rows as the overlay layout, same order
BAND = "#dcdcdc"
XTICKS = [0, 5000, 10000, 15000, 20000]


def load_20k():
    """(arm, lr label, linestyle, [(step, {metric: v})], final) per run, plus the IG constants."""
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
    ap.add_argument("--out", default="plots/train_curves_20k.pdf")
    a = ap.parse_args()

    runs, ig = load_20k()

    plt.rcParams.update(P.RC)
    fs = FS_OVERLAY
    nr = len(MROWS)
    # Same header split as the overlay layout: LEG_H for the handles, TITLE_H for the title,
    # which set_title draws ABOVE the axes rectangle and therefore into the same band.
    head = LEG_H + TITLE_H
    fh = ROW_H_NARROW * nr + head
    fig, axes = plt.subplots(nr, 1, figsize=(FIG_OVERLAY_W, fh), sharex=True, squeeze=False)
    for r, met in enumerate(MROWS):
        ax = axes[r][0]
        # The shaded strip is fig:optimiser-curves' entire x-range -- see the docstring, it is the
        # only thing left marking where the paper's reported numbers stop.
        ax.axvspan(0, PUBLISHED_STEPS, color=BAND, lw=0, zorder=0)
        ax.axvline(PUBLISHED_STEPS, color="#999999", lw=0.5, ls=(0, (1, 2)), zorder=1)
        ax.axhline(ig[met], color=ARM_COLOR[REF[1]], lw=1.0, ls="dotted", zorder=2)
        for arm, _, ls, curve, final in runs:
            xs = [s for s, _ in curve]
            ys = [v[met] for _, v in curve]
            ax.plot(xs, ys, ls=ls, lw=1.1, color=ARM_COLOR[arm], zorder=3)
            # The endpoint marker is the real 100-example test eval, not the 64-example probe;
            # it is what a results table would report, and it sits ~0.01-0.02 above the probe.
            ax.plot([xs[-1]], [final[met]], marker="o", ms=2.6, mew=0,
                    color=ARM_COLOR[arm], zorder=4)
        ax.set_xlim(0, 20000)
        ax.set_xticks(XTICKS)
        # "20000" at 5.5pt is wider than the tick spacing allows across 2.7in; thousands notation
        # keeps all five decade marks, and dropping to 0/10k/20k would hide where Adam's curve
        # turns over (~10k), which is the reading the figure exists for.
        ax.set_xticklabels(["0" if x == 0 else f"{x // 1000}k" for x in XTICKS])
        ax.tick_params(labelsize=fs[1])
        ax.set_ylabel(METRICS[met], fontsize=fs[0])
        P.furnish(ax)
        if r == 0:
            ax.set_title("addition / llama3 / MLP, logit-diff", fontsize=fs[0], pad=3)
            ax.annotate("reported\nbudget", (PUBLISHED_STEPS, 0.03), xycoords=("data",
                        "axes fraction"), ha="left", va="bottom", fontsize=fs[2],
                        color="#666666", xytext=(2, 0), textcoords="offset points")
        if r == nr - 1:
            ax.set_xlabel("training step", fontsize=fs[0])

    fig.tight_layout()
    top = 1.0 - head / fh
    fig.subplots_adjust(top=top)
    # One handle per RUN rather than the old two-group (arms | learning rates) split. That split
    # existed because the two columns spent linetype on different variables; with one column the
    # groups would be ambiguous anyway -- "lr 0.05" and "lr 1.0" are both solid and only the
    # colour tells them apart, so a linetype-only handle for either would be a lie.
    hs = [Line2D([0], [0], color=ARM_COLOR[arm], lw=1.1, ls=ls, label=f"{arm}, {lrlab}")
          for arm, lrlab, ls, _ in RUNS]
    hs.append(Line2D([0], [0], color=ARM_COLOR[REF[1]], lw=1.0, ls="dotted", label=REF[1]))
    fig.legend(handles=hs, fontsize=fs[2], ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, top + TITLE_H / fh), frameon=False,
               handletextpad=0.4, handlelength=2.0, columnspacing=1.2, labelspacing=0.25)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    print("\n(addition/llama3/mlp/logit-diff), probe@2000 -> probe@20k -> test:")
    for arm, lrlab, _, curve, final in runs:
        d = dict(curve)
        at2k = max(s for s in d if s <= PUBLISHED_STEPS)
        for met in MROWS:
            print(f"  {arm:<12} {lrlab:<9} {met:<10} "
                  f"{d[at2k][met]:.3f} -> {curve[-1][1][met]:.3f} -> {final[met]:.3f}")
    print("  IG (untrained ref.)     " + "  ".join(f"{m} {ig[m]:.3f}" for m in MROWS))


if __name__ == "__main__":
    main()
