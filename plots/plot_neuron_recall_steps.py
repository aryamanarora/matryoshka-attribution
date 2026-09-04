"""Does a 10x longer training budget change WHICH neurons \\ourmethod{} finds?

COMPANION to plots/plot_neuron_recall.py, which asks how much of the published neuron set each
METHOD recovers. This holds the method fixed and varies the TRAINING BUDGET: 2,000 steps (the
budget every reported number in the paper uses) against 20,000, on the one cell where both exist.

WHY ONE TASK. The 20k runs were only ever produced for `addition` (submit_sva_mlp_lr.sh with
STEPS=20000, 2026-08-22), because the question they were built for -- is the Adam-vs-SGD gap at
2k confounded with training budget? -- is a per-cell question and 20k steps is 10x the cost. So
this figure is addition/llama3/mlp/logit-diff and nothing else, and no claim here generalises to
the other three arithmetic tasks without running them.

*** BOTH BUDGETS COME FROM THE SAME SUBMITTER, WITH ONLY --steps DIFFERING. *** The 2k arms are
NOT the ones in results/sva_sweep: those come from a different script with different probe
settings, and pairing them against the 20k runs would compare two things at once. They were
re-run into results/sva_mlp_lr through submit_sva_mlp_lr.sh -- the same script, same lr, same
eps, same everything but STEPS -- which is the only way the difference between two curves here
is attributable to the budget.

ADAM IS AT eps=1e-2, not torch's 1e-8 default. At 2.29M mask logits the default makes the update
~sign(g)*lr and the learned score a signed COUNT of steps, so a "budget" comparison at default
eps would mostly be measuring how long that degeneracy had to accumulate. The SGD control has no
eps and is unchanged between the two budgets.

RECALL, RANKS AND THE GROUND TRUTH are imported from plot_neuron_recall so the two figures cannot
drift apart: same published L18 neuron set, same max-over-positions ranking, same k grid, same
step-function recall. If that file's definition of a hit changes, this figure changes with it.

Run:  uv run python plots/plot_neuron_recall_steps.py
Out:  plots/neuron_recall_steps.pdf  (+ .png)
"""
import os
import sys
from pathlib import Path

import json
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                   # noqa: E402
import plot_neuron_recall as NR                       # hit_ranks / curve / GT / K grid  # noqa: E402

TASK = "addition"
# (label, colour key, linestyle, 2k path, 20k path). The 2k and 20k members of a pair differ in
# --steps and nothing else; `_s20000` is run_tag's own suffix for a non-default step count.
A2K = Path("results/sva_mlp_lr")
A20K = Path("results/sva_mlp_steps20k")
ADAM_T = "sufficient_topk_adam_eps1e-2_bs1"
SGD_T = "sufficient_topk_sgd_bs1"
ARMS = [
    ("MAttr (Adam), lr 0.05", "MAttr", "solid",
     A2K / "topk_adam/lr_0.05", A20K / "topk_adam/lr_0.05", ADAM_T),
    ("MAttr (Adam), lr 0.005", "MAttr", "dashed",
     A2K / "topk_adam/lr_0.005", A20K / "topk_adam/lr_0.005", ADAM_T),
    ("MAttr (SGD), lr 1.0", "MAttr (SGD)", "solid",
     A2K / "topk_sgd/lr_1.0", A20K / "topk_sgd/lr_1.0", SGD_T),
]
# BUDGET IS A PANEL, not an alpha. The first version faded the 2k arms to alpha=0.35 on the same
# axes; at that opacity BLACK (the SGD colour) reads as a distinct grey series rather than as a
# muted version of the black one beside it, so a 3-arm figure looked like it had 5 or 6. Two
# panels keep colour meaning the arm and linestyle meaning the lr, with nothing left over.
# Shared y so the two budgets are directly comparable, which is the whole comparison.
LW = 1.2
FIG_W, FIG_H = 4.4, 1.85
FS_LAB, FS_TICK, FS_LEG = 6.5, 5.5, 5.5


def ranks_for(res, tag, gt_layer, gt_neurons):
    return NR.hit_ranks(TASK, tag, gt_layer, np.array(gt_neurons), res)


def main():
    gt = json.load(open(NR.GT))
    gt_layer = gt["layer"] if isinstance(gt, dict) and "layer" in gt else NR.GT_LAYER
    pub = gt[TASK] if isinstance(gt, dict) and TASK in gt else gt["neurons"][TASK]

    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True, sharex=True)
    PANEL = {"2k": axes[0], "20k": axes[1]}
    # SAME k grid the companion builds (plot_neuron_recall.py:257): log-spaced over 1..total,
    # where total = L*N is returned by hit_ranks. Built after the first arm loads so `total`
    # comes from the data rather than being hardcoded to a width this cell might not have.
    ks = None
    drawn, missing = [], []
    for lab, ckey, ls, d2, d20, tag in ARMS:
        for res, budget in ((d2, "2k"), (d20, "20k")):
            t = tag + ("_s20000" if budget == "20k" else "")
            r, total = ranks_for(res, t, gt_layer, pub)
            if r is None:
                missing.append(f"{lab} @{budget}")
                continue
            if ks is None:
                ks = np.unique(np.round(np.logspace(0, np.log10(total), 240)).astype(int))
            y = NR.curve(r, total, len(pub), ks)
            PANEL[budget].plot(ks, y, color=P.color(ckey), ls=ls, lw=LW, zorder=3)
            drawn.append((lab, budget, r[0] if len(r) else None))
    if not drawn:
        raise SystemExit("no runs found -- are the 2k eps=1e-2 arms still in flight?")

    for b, ax in PANEL.items():
        ax.set_xscale("log")
        ax.set_xticks(NR.XTICKS)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{b} steps", fontsize=FS_LAB, pad=2.5)
        ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1.5)
        P.furnish(ax)
    axes[0].set_ylabel("recall", fontsize=FS_LAB)
    fig.supxlabel("$k$ (neurons)", fontsize=FS_LAB, y=0.02)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=P.color(c), ls=ls, lw=LW, label=lab) for lab, c, ls, *_ in ARMS]
    fig.legend(handles=h, fontsize=FS_LEG, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), frameon=False, handlelength=1.8,
               handletextpad=0.5, columnspacing=1.2)
    fig.tight_layout(pad=0.3, rect=(0, 0.04, 1, 0.88))
    out = "plots/neuron_recall_steps.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out} (+ .png)   task={TASK}, {len(pub)} published neurons")
    if missing:
        print(f"missing: {', '.join(missing)}")
    print(f"\n{'arm':<24}{'budget':>8}{'best rank':>11}")
    for lab, b, r0 in drawn:
        print(f"{lab:<24}{b:>8}{(r0 if r0 else '--'):>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
