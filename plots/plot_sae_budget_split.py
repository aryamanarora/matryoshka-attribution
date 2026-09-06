"""How to spend a FIXED attribution budget: integration steps vs examples, grid IG vs Stepless IG.

THE QUESTION. A gradient attribution's cost is (draws x examples) backward passes over one
example each -- call it example-backwards. At a fixed budget those two factors trade off, and
the sweeps in this repo have only ever sampled one corner of that trade (IG at m=10 over 100
examples, I x G at m=1 over 100). This figure holds the budget at 4,000 example-backwards and
walks the split: m in {1,2,5,10,20,40} with E = 4000/m.

X IS THE TRAINING-SAMPLE COUNT E, not m, so the axis reads "what if I spend the budget on more
data instead of more integration steps". Under that orientation the grid's failure is a CLIFF at
the right-hand end -- all the samples, one integration step -- which is the shape a reader should
take away. x is the REALISED sample count, so the two short points sit at 2,471 / 2,759 rather
than at a 4,000 they never reached.

THE RESULT, and it is the reason the figure is one row rather than a table. The GRID estimator
has exactly one bad operating point and it is the cheapest-looking one: at m=1 the left-endpoint
rule degenerates to the single point alpha=0, which is not an integral estimate at all -- it IS
I x G -- and it costs 0.111 mean IIA AUC. STEPLESS IG, which draws alpha ~ U(0,1) PER EXAMPLE,
has no such point: at m=1 its 4,000 examples give 4,000 independent alpha draws for the same one
backward pass each, so it is already unbiased and already at the plateau. From m=5 on the two
estimators are indistinguishable (|delta| <= 0.005, inside run-to-run noise).

So the honest reading is NOT "stepless beats grid" -- it is "stepless has no setting you can get
wrong, and the grid has one". That is why both curves are drawn over the whole range instead of
quoting the m=1 pair alone.

SHARED Y ACROSS PANELS, deliberately. The four tasks sit at genuinely different levels (rc ~0.5,
months ~0.27) and a free y per panel would rescale each one until every curve looked equally
dramatic, which is the opposite of the point: the claim is that ONE shape (flat, with a hole at
m=1 for the grid) recurs at four different levels. The whole spread fits 0.10-0.53, so nothing
has to be cropped to say that.

CAPPED POINTS ARE MARKED, not silently drawn. `months` and `weekdays` have only 2,471 and 2,759
usable pairs at their modal prompt length, so at m=1 (E=4000) they cannot reach the budget and
ran short. Those markers are hollow. It matters because m=1 is exactly where the grid looks
worst, so an unmarked point would let a reader credit the estimator for a data shortfall. Note
Stepless hits the same ceiling on the same two cells and still scores 0.270/0.331, which is what
says the shortfall is not the explanation.

WHY ONLY FOUR TASKS. The other four cannot afford the low-m end at all -- `simple` has 413
usable pairs, so E=4000/2000/800 are all out of reach and only m>=10 reaches budget. Including
them would mean comparing a capped m=1 against an uncapped m=40 within one panel. That
exclusion is itself a finding about I x G-like methods (their compute ceiling is the dataset,
not a knob) and is stated in the prose, not hidden here.

Data: results/sae_budget_split/m<m>/<task>_llama3_resid_sae_span_{ig,mc_ig_m<m>_s42}.json,
produced at --grad-batch 25 (chunking is exact; it only bounds memory).
ONE SEED (s42). Stepless is stochastic by construction, so its error bar is a seed replicate
that has not been run -- do not read the <=0.005 differences at m>=5 as ordered.

Run:  uv run python plots/plot_sae_budget_split.py
Out:  plots/sae_budget_split.pdf  (+ .png; plots/*.pdf is gitignored)
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                     # noqa: E402

BUDGET = 4000
MS = [1, 2, 5, 10, 20, 40]
TASKS = [("rc", "rc"), ("within_rc", "within_rc"), ("months", "months"),
         ("weekdays", "weekdays")]
ROOT = "results/sae_budget_split"
# REALISED example count per (task, m), read from the run logs' "N examples x D draws = ..."
# line. Only the m=1 column ever falls short: it asks for E=4,000 distinct pairs and
# gradient_scores keeps a pair only when BOTH the clean and the corrupted prompt tokenise to the
# modal length, which leaves months 2,471 and weekdays 2,759. Everything else hit its nominal E
# exactly, and the grid and stepless sweeps realised identical counts (same filter, same data).
#
# HARDCODED rather than re-derived: a count of clean prompts at the modal length gives 2,705 /
# 2,759, not 2,471 / 2,759 -- the pair-level filter is stricter than the prompt-level one, so
# deriving it here would put two points at the wrong x. These are the numbers the runs logged.
REALISED = {("months", 1): 2471, ("weekdays", 1): 2759}


def _flip(v):
    """E <-> m under E*m = BUDGET. Its own inverse, so one function serves both directions of
    the secondary axis. Guarded at 0 because matplotlib probes the transform at the axis
    origin, which would otherwise divide by zero on every draw."""
    v = np.asarray(v, dtype=float)
    return np.divide(BUDGET, v, out=np.full_like(v, np.nan), where=v != 0)


def n_examples(task, m):
    """Training samples this point actually used (nominal E unless the dataset ran out)."""
    return REALISED.get((task, m), BUDGET // m)

# Colour = ESTIMATOR (palette's rule), plus linetype and marker so the pair survives at 1in
# panel width -- IG's orange and Stepless's sand are 39 dE apart but both light and both warm,
# which is the one pairing in this palette where hue alone is thin.
SERIES = [("ig",     "IG (fixed grid)",  P.color("IG"),          "solid",  "o"),
          ("mc_ig",  "Stepless IG",      P.color("Stepless IG"), "dashed", "s")]

FIG_W, FIG_H = 5.5, 2.15
FS_TITLE, FS_TICK, FS_LAB, FS_ANN = 6.5, 5.5, 6.5, 5.5
YLIM = (0.08, 0.55)


def load(task, m, kind):
    fn = "ig.json" if kind == "ig" else f"mc_ig_m{m}_s42.json"
    p = f"{ROOT}/m{m}/{task}_llama3_resid_sae_span_{fn}"
    try:
        with open(p) as f:
            return json.load(f)["acc_auc"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/sae_budget_split.pdf")
    a = ap.parse_args()

    data = {(t, m, k): load(t, m, k) for t, _ in TASKS for m in MS for k, *_ in SERIES}
    missing = [k for k, v in data.items() if v is None]
    if missing:
        raise SystemExit(f"{len(missing)} missing cells, e.g. {missing[:4]} -- "
                         "run scripts/sva/launch/submit_sae_budget.sh's split sweep first")

    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(1, len(TASKS) + 1, figsize=(FIG_W, FIG_H), sharey=True)
    panels = [(t, lab, False) for t, lab in TASKS] + [(None, "mean", True)]

    for ax, (task, lab, is_mean) in zip(axes, panels):
        for key, _, col, ls, mk in SERIES:
            ys = [np.mean([data[(t, m, key)] for t, _ in TASKS]) if is_mean
                  else data[(task, m, key)] for m in MS]
            # x is the number of TRAINING SAMPLES actually used. Plotting the realised count
            # rather than the nominal one puts the two short points where they really are
            # (months m=1 at 2,471, not 4,000) instead of implying they got the full E.
            xs = [np.mean([n_examples(t, m) for t, _ in TASKS]) if is_mean
                  else n_examples(task, m) for m in MS]
            ax.plot(xs, ys, ls=ls, lw=0.9, color=col, zorder=2)
            # Hollow marker = this point still ran short of the BUDGET even after moving it: at
            # m=1 fewer examples means fewer backward passes, so 2,471 x 1 is 62% of the 4,000
            # every other point on the curve got. The x position shows the sample count; the
            # hollow marker is what says the compute was not matched.
            for m, x, y in zip(MS, xs, ys):
                capped = (any((t, m) in REALISED for t, _ in TASKS) if is_mean
                          else (task, m) in REALISED)
                ax.plot([x], [y], marker=mk, ms=2.6, ls="none", zorder=3,
                        mfc=("#ffffff" if capped else col), mec=col if capped else "#000000",
                        mew=0.7 if capped else 0.3)
        ax.set_xscale("log")
        ax.set_xticks([100, 400, 2000])
        ax.set_xticklabels(["100", "400", "2k"], fontsize=FS_TICK)
        ax.set_xlim(70, 6000)
        # Second axis on top for the integration steps. m = BUDGET/E is its own inverse under
        # this parameterisation, so one lambda serves both directions. Ticks are pinned to
        # m = 40/10/2, which land exactly on the E = 100/400/2k ticks below -- the two axes then
        # read as one statement ("40 steps x 100 samples") instead of two scales to reconcile.
        sec = ax.secondary_xaxis("top", functions=(_flip, _flip))
        sec.set_xticks([40, 10, 2])
        sec.set_xticklabels(["40", "10", "2"], fontsize=FS_TICK)
        sec.tick_params(length=1.5, pad=1.0)
        sec.minorticks_off()
        for sp in sec.spines.values():
            sp.set_linewidth(P.SPINE_LW)
        ax.set_ylim(*YLIM)
        ax.minorticks_off()
        ax.set_title(lab, fontsize=FS_TITLE, pad=11,
                     fontweight="bold" if is_mean else "normal")
        P.furnish(ax)
        ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1.5)

    # The gap the figure is about, annotated once on the mean panel rather than in every one.
    axm = axes[-1]
    g = np.mean([data[(t, 1, "ig")] for t, _ in TASKS])
    s = np.mean([data[(t, 1, "mc_ig")] for t, _ in TASKS])
    x1 = np.mean([n_examples(t, 1) for t, _ in TASKS])
    axm.annotate("", xy=(x1, s), xytext=(x1, g),
                 arrowprops=dict(arrowstyle="<->", lw=0.6, color="#000000",
                                 shrinkA=0.6, shrinkB=0.6))
    axm.annotate(f"{s - g:+.2f}", xy=(x1 * 0.85, (s + g) / 2), fontsize=FS_ANN,
                 va="center", ha="right")

    axes[0].set_ylabel("IIA AUC (↑)", fontsize=FS_LAB)
    fig.supxlabel(f"training samples $E$  (bottom)   vs   integration steps "
                  f"$m = {BUDGET:,}/E$  (top)", fontsize=FS_LAB, y=0.02)
    fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0.04, 1, 0.90))
    fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=0.9, marker=mk, ms=2.6,
                               mec="#000000", mew=0.3, label=lab)
                        for _, lab, c, ls, mk in SERIES]
               + [Line2D([], [], color="#666666", ls="none", marker="o", ms=2.6,
                         mfc="#ffffff", mec="#666666", mew=0.7,
                         label="ran short of budget (dataset-bounded)")],
               fontsize=FS_ANN, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.005),
               frameon=False, handlelength=1.8, handletextpad=0.4, columnspacing=1.4)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    print(f"\n{'m':>4}{'E':>7}   {'grid IG':>9}{'stepless':>10}{'delta':>8}")
    for m in MS:
        gg = np.mean([data[(t, m, "ig")] for t, _ in TASKS])
        ss = np.mean([data[(t, m, "mc_ig")] for t, _ in TASKS])
        cap = "  *" if any((t, m) in REALISED for t, _ in TASKS) else ""
        print(f"{m:>4}{BUDGET // m:>7}   {gg:>9.3f}{ss:>10.3f}{ss - gg:>+8.3f}{cap}")
    print("  * at least one task ran short of the budget")


if __name__ == "__main__":
    sys.exit(main())
