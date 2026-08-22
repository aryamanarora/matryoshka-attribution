"""How much of a published neuron set each method recovers, as a function of budget.

Ground truth is \\citet{feucht2026arithmetic}'s layer-18 MLP neurons for the four
arithmetic-wild tasks -- 15-28 neurons each, vendored at
src/learning_to_attribute/data/arith_wild_l18_neurons.json. It is the only unit-level
published circuit any of our tasks has (IOI/GPT-2 has a published HEAD circuit, which is a
different object), so this is the paper's one chance to score attribution against prior work
rather than against faithfulness.

y is RECALL of that set, x the budget k, both over the `mlp` substrate's distinct neurons
(32 layers x 14336 = 458752). A curve rather than a single hit-count because any fixed k is
arbitrary: top-5 and top-200 rank the methods differently for the weaker ones (DBM is 0/12 at
top-5 and 5/12 at top-200), and a reader should be able to see that rather than take our word
for which k we picked.

THE HEADLINE, and it is an OPTIMIZER result, not a gradients-vs-masks one. \\ourmethod{}+SGD
recovers a published neuron in its top 5 in 12/12 cells (4 tasks x 3 losses) -- more than IG
(9/12) and I×G (6/12) -- while \\ourmethod{}+Adam manages 0/12 and does not reach one until
rank 141. Node Pruning never finds one at any k below 200; DBM and the uniform-k Adam arm
manage 5/12 and 3/12. Do not read the flat lines as "mask learning cannot do this".

*** THAT HEADLINE IS PATCHED-ONLY, and the zero row says so. *** Under zero-ablation the SGD
advantage does not survive: top-5 goes 12/12 -> 2/12, which is a TIE with IG (also 2/12), and at
top-200 SGD is BEATEN by the gradient baselines, 8/12 against 12/12 for both IG and I×G. So the
optimizer result is a claim about the patched setting, not about attribution in general, and any
prose saying \\ourmethod{}+SGD "matches or outperforms IG on recall" is scoped to the top row.
The gradient baselines are the more robust ones here, which is the opposite of the acc-AUC story
in plot_accauc_vs_faithauc, where zeroing puts IG/I×G at the random floor -- worth stating rather
than quietly reporting whichever setting flatters the method.

NOT A LAYER ARTIFACT. The obvious deflation is that these methods merely like layer 18 and the
published set is most of what is interesting there. It does not hold: of the L18 neurons each
method puts in its top-n, essentially every one is a published neuron (IG 3/3 on hours, 5/5 on
addition; SGD 5/5 on weekdays, 2/2 on months), against 0.00-0.01 expected if the L18 picks were
uniform inside a 14336-neuron layer. `--layer-control` re-runs that check and prints it.

RANKING IS DESCENDING RAW SCORE, not |score| -- the same order evaluate.sparsity_sweep uses to
build every faithfulness curve in the paper, so these are literally the neurons that enter each
method's circuit first. Ranking by |score| here would score a different set of neurons than our
own numbers were computed from, for IG/I×G especially, whose scores are signed effects.

ONE NEURON, MANY POSITIONS. The score index is per-(layer, position, neuron); we reduce to
distinct neurons by taking each neuron's BEST-scoring position, matching
make_sva_neuron_table.py's dedupe so the two artifacts rank the same objects.

TWO ABLATION ROWS. `Patched` (results/sva_sweep) and `Zero-abl.` (results/sva_zeroabl) are two
SETTINGS, not two scorings of one run: the mask methods train through the ablation and the
gradient baselines change estimator with it. They share a y axis here -- legitimately, unlike the
faithfulness figures, because recall is scored against a FIXED external neuron set rather than
against a within-setting faithfulness curve that the setting itself inflates. Read ordering within
a row first; the cross-row comparison is the secondary question of whether the result survives.

*** The published set is layer-18-only by construction: Feucht et al. searched layer 18. ***
So recall here is recall of "the L18 neurons they name", and a method that localises the task
to a different layer is not thereby wrong -- it disagrees with prior work, which is what the
figure measures. This belongs in the caption; do not drop it.

Run:  uv run python plots/plot_neuron_recall.py                 -> plots/neuron_recall.pdf
      uv run python plots/plot_neuron_recall.py --all-losses    -> plots/neuron_recall_losses.pdf
      uv run python plots/plot_neuron_recall.py --layer-control -> prints the control, no figure
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import palette as P
import torch

GT = Path("src/learning_to_attribute/data/arith_wild_l18_neurons.json")
# (row label, results dir, tag fragment). The ablation is a SETTING, not a rescoring: the mask
# methods train through it and the gradient baselines change estimator (I×G -> Gradient×Input,
# IG -> zero-baseline IG), so the two rows are two experiments and only the ORDERING within a row
# is meaningful. Worth a row here because the setting reorders methods elsewhere (Spearman ~0.44
# on matched cells, see plot_accauc_vs_faithauc), so "does the recall result survive zeroing?" is
# a real question rather than a formality -- and unlike faithfulness, recall is scored against a
# FIXED external neuron set, so the two rows are directly comparable on the same y axis.
ABLATIONS = [("Patched", Path("results/sva_sweep"), ""),
             ("Zero-abl.", Path("results/sva_zeroabl"), "_zeroabl")]
MODEL, SUB = "llama3", "mlp"
TASKS = [("hours", "Hours"), ("months", "Months"), ("weekdays", "Weekdays"),
         ("addition", "Addition")]
# (label, tag template, palette key, linetype). The %s takes the loss fragment. Same runs and
# the same order as make_sva_neuron_table.METHODS, so the table and this figure describe one
# experiment. Colour is the METHOD and linetype the hyperparameter, per palette.py's rule --
# which is why "+ unif k" is MAttr's blue dashed rather than a seventh hue, and why the SGD arm
# gets its own (black) hex: an optimizer swap that beats every other series here is not
# readable as a linetype variant of the series it beats.
METHODS = [
    ("IG", "ig%s", "IG", "solid"),
    ("I×G", "ixg%s", "I×G", "solid"),
    ("Node Pruning", "eprun_s090%s", "Node Pruning", "solid"),
    ("DBM", "sig_lr0.3_l16.0%s", "DBM", "solid"),
    ("MAttr", "sufficient_topk_adam%s_bs1", "MAttr", "solid"),
    ("MAttr + unif $k$", "sufficient_topk_adam%s_uniformk_bs1", "MAttr", "dashed"),
    ("MAttr + SGD", "sufficient_topk_sgd%s_bs1", "MAttr (SGD)", "solid"),
]
LOSSES = [("logit-diff", ""), ("CE", "_ce"), ("Accuracy", "_acc")]
FS = (7.5, 6.5, 6.5)        # (axis label, tick, legend)
PANEL_W, ROW_H = 1.35, 1.45
# Legend strip height, DERIVED from how many rows the legend actually wraps to rather than
# fixed: 8 entries at ncol=4 is two rows, and a hardcoded reserve sized for one row put the
# second row straight through the panel titles. TITLE_H is part of the same reserve because
# set_title draws ABOVE the axes rectangle, i.e. into the very band the legend is anchored in
# -- sizing the band for the legend alone collides them even though the arithmetic looks right.
LEG_NCOL, LEG_ROW_H, LEG_PAD, TITLE_H = 4, 0.20, 0.10, 0.20


def tag_for(tpl, lfrag, zfrag):
    """Fill a METHODS template for one (loss, ablation).

    The sweep's run_tag puts the zero-ablation fragment directly AFTER the loss fragment, i.e.
    before the trailing `_uniformk`/`_bs1` -- `sufficient_topk_sgd_ce_zeroabl_bs1`, never
    `..._bs1_zeroabl`. Appending instead of inserting resolves to nothing on disk for the four
    mask arms, which would silently drop them from the zero row rather than error. Verified
    against all 7 methods x 3 losses in results/sva_zeroabl.
    """
    return tpl.replace("%s", "%s" + zfrag) % lfrag


def hit_ranks(task, tag, gt_layer, gt_neurons, res):
    """Ranks (1-based) at which this run's neuron ordering hits the published set.

    Returns None when the run is absent, so a missing cell is a gap in the figure rather than
    a silently-zero curve -- the two are very different claims about a method.
    """
    j, s = res / f"{task}_{MODEL}_{SUB}_{tag}.json", res / f"{task}_{MODEL}_{SUB}_{tag}.scores.pt"
    if not (j.exists() and s.exists()):
        return None, None
    m = json.load(open(j))
    m = m.get("meta", m)
    L, Pos, N = m["num_layers"], m["seq_len"], m["intermediate_size"]
    sc = torch.load(s, map_location="cpu").float()
    assert sc.numel() == L * Pos * N, f"{tag}: {sc.numel()} != {L}*{Pos}*{N} (not an mlp run?)"
    # max over positions = each neuron scored at its best position, then rank all L*N neurons
    best = sc.view(L, Pos, N).max(dim=1).values.reshape(-1)
    order = torch.argsort(best, descending=True).numpy()
    hit = (order // N == gt_layer) & np.isin(order % N, gt_neurons)
    return np.flatnonzero(hit) + 1, L * N


def curve(ranks, total, ngt, ks):
    """Recall at each k, from the sorted hit ranks. Step function, evaluated on the k grid."""
    return np.searchsorted(ranks, ks, side="right") / ngt


def layer_control(gt_layer, gt):
    """Is the recovery neuron-level, or just 'this method likes layer 18'?

    For each method's top-n (n = |published set|), count how many picks land in layer 18 AT ALL
    versus how many are published. If the two are equal the method is not merely finding the
    layer. E is what you would expect published if the L18 picks were uniform within the layer.
    """
    for abl, res, zfrag in ABLATIONS:
        print(f"\n== {abl} ({res})")
        print(f"{'task':10s}{'n':>4s}  " + "".join(f"{lab:>26s}" for lab, *_ in METHODS))
        for task, _ in TASKS:
            pn = gt[task]
            n = len(pn)
            cells = []
            for lab, tpl, _, _ in METHODS:
                tag = tag_for(tpl, "", zfrag)
                j = res / f"{task}_{MODEL}_{SUB}_{tag}.json"
                if not j.exists():
                    cells.append("--")
                    continue
                m = json.load(open(j))
                m = m.get("meta", m)
                L, Pos, N = m["num_layers"], m["seq_len"], m["intermediate_size"]
                sc = torch.load(res / f"{task}_{MODEL}_{SUB}_{tag}.scores.pt", map_location="cpu")
                best = sc.float().view(L, Pos, N).max(dim=1).values.reshape(-1)
                top = torch.argsort(best, descending=True).numpy()[:n]
                in18 = int((top // N == gt_layer).sum())
                pub = int(((top // N == gt_layer) & np.isin(top % N, pn)).sum())
                cells.append(f"L18 {in18:2d}  pub {pub:2d}  E{in18 * n / N:.2f}")
            print(f"{task:10s}{n:>4d}  " + "".join(f"{c:>26s}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-losses", action="store_true", help="3 loss rows instead of logit-diff")
    ap.add_argument("--layer-control", action="store_true", help="print the control, no figure")
    ap.add_argument("--out")
    a = ap.parse_args()

    g = json.load(open(GT))
    gt_layer, gt = g["layer"], g["neurons"]
    if a.layer_control:
        layer_control(gt_layer, gt)
        return

    # Rows are (ablation x loss), ablation OUTER so the two settings stay adjacent blocks rather
    # than interleaving by loss -- the comparison the figure exists to support is within a column,
    # across settings. Default is one loss, so the default figure is exactly two rows.
    losses = LOSSES if a.all_losses else LOSSES[:1]
    rows = [(abl, res, zfrag, lname, lfrag)
            for abl, res, zfrag in ABLATIONS for lname, lfrag in losses]
    plt.rcParams.update(P.RC)
    nr, nc = len(rows), len(TASKS)
    nleg = LEG_ROW_H * -(-(len(METHODS) + 1) // LEG_NCOL) + LEG_PAD     # +1 = the chance entry
    leg_h = nleg + TITLE_H
    fig, axes = plt.subplots(nr, nc, figsize=(PANEL_W * nc, ROW_H * nr + leg_h),
                             sharex=True, sharey=True, squeeze=False)
    ks, total = None, None
    for r, (abl, res, zfrag, lname, frag) in enumerate(rows):
        for c, (task, tlab) in enumerate(TASKS):
            ax = axes[r][c]
            pn = np.array(gt[task])
            for lab, tpl, ckey, ls in METHODS:
                ranks, tot = hit_ranks(task, tag_for(tpl, frag, zfrag), gt_layer, pn, res)
                if ranks is None:
                    continue
                total = tot
                # log grid over the whole neuron space: the curves must reach 1.0 at the right
                # edge (every method finds everything once k = all neurons), and a truncated x
                # would hide that they do and make the separation look permanent.
                ks = np.unique(np.round(np.logspace(0, np.log10(tot), 240)).astype(int))
                ax.plot(ks, curve(ranks, tot, len(pn), ks), color=P.METHOD[ckey],
                        ls=ls, lw=1.0, solid_joinstyle="round")
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            P.furnish(ax)
            ax.tick_params(labelsize=FS[1], length=2, width=0.5)
            if r == 0:
                ax.set_title(tlab, fontsize=FS[0])
            if c == 0:
                # The ablation always names the row now; the loss only when there is more than
                # one of them, so the default two-row figure is not labelled with a constant.
                stack = abl if not a.all_losses else f"{abl}\n{lname}"
                ax.set_ylabel(f"{stack}\nrecall", fontsize=FS[0])
            if r == nr - 1:
                ax.set_xlabel("$k$ (neurons)", fontsize=FS[0])
            # n varies 15-28 by task, so the reader cannot assume a shared denominator. Sits
            # top-LEFT: that corner is "high recall at tiny k", which nothing reaches, whereas
            # the bottom-right corner it started in is where every curve converges on 1.0.
            ax.annotate(f"$n{{=}}${len(gt[task])}", (0.05, 0.94), xycoords="axes fraction",
                        ha="left", va="top", fontsize=FS[2] - 0.5, color="#666666")
    # Chance: a uniformly random ranking recovers k/total of the set. Drawn because at the k
    # where the good methods are already at 0.2-0.4 it is ~1e-4, which is the whole point, and
    # a reader should not have to compute that to know the flat lines are flat at chance.
    for r in range(nr):
        for c in range(nc):
            axes[r][c].plot(ks, ks / total, color="#999999", lw=0.4, ls=(0, (1, 2)), zorder=0)
    handles = [plt.Line2D([], [], color=P.METHOD[k], ls=ls, lw=1.0, label=lab)
               for lab, _, k, ls in METHODS]
    handles.append(plt.Line2D([], [], color="#999999", lw=0.4, ls=(0, (1, 2)), label="chance"))
    fig.tight_layout()
    fh = ROW_H * nr + leg_h
    top = 1.0 - leg_h / fh
    fig.subplots_adjust(top=top)
    # anchored above the TITLE band, not above the axes -- see TITLE_H
    fig.legend(handles=handles, ncol=LEG_NCOL, loc="lower center",
               bbox_to_anchor=(0.5, top + TITLE_H / fh),
               frameon=False, fontsize=FS[2], handlelength=1.6, columnspacing=1.1,
               handletextpad=0.5, borderpad=0)
    out = a.out or (f"plots/neuron_recall{'_losses' if a.all_losses else ''}.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # Hit counts at two budgets, so the numbers quoted in prose come from this script and not
    # from a probe that is not in the repo.
    # Reported PER ABLATION, never pooled: the headline "12/12 at top-5" is a claim about the
    # patched setting, and averaging the two settings into one fraction would silently redefine
    # it. `ncell` is per-setting for the same reason.
    ncell = len(losses) * len(TASKS)
    for k in (5, 200):
        for abl, res, zfrag in ABLATIONS:
            print(f"\n{abl}: cells (of {ncell}) with a published neuron in the top-{k}:")
            for lab, tpl, _, _ in METHODS:
                n = best = 0
                for _, frag in losses:
                    for task, _ in TASKS:
                        ranks, _t = hit_ranks(task, tag_for(tpl, frag, zfrag), gt_layer,
                                              np.array(gt[task]), res)
                        if ranks is not None and len(ranks) and ranks[0] <= k:
                            n += 1
                            best = ranks[0] if not best else min(best, ranks[0])
                print(f"  {lab:18s} {n:2d}/{ncell}   best rank {best or '--'}")


if __name__ == "__main__":
    main()
