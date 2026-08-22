"""Does 2k -> 20k training reshuffle the WHOLE ranking, or only its middle?

Global Spearman says the ranking is destroyed: MAttr+SGD lr=1.0 on addition/llama3/mlp scores
rho=+0.101 between its 2000-step and 20000-step checkpoints, over 2.29M units. Yet acc-AUC is
0.496 -> 0.494. Those two facts are only compatible if the churn is confined to ranks the metric
does not read, so this figure resolves the ranking BY DEPTH instead of collapsing it to one rho.

THREE PANELS, one claim each:

 (a) chance-corrected top-k agreement, on the 24-point k grid `eval_sva.py:733` actually
     integrates over. kappa = (overlap - k/N) / (1 - k/N); raw overlap is useless at the dense end
     because top-1M-of-2.29M agrees 52% by construction. The curve is a U with a broad stable
     plateau at k~24-1100 (kappa 0.52-0.78, peak 0.784 at k=583) and a minimum at k~26k-95k
     (kappa 0.20-0.23). That is the hypothesis confirmed.

     THE k<=10 ZERO IS NOT INSTABILITY -- read panel (b) before believing it. The 2k run's top-10
     units sit at 20k ranks 9,11,12,13,14,15,16,20,26,40: they did not move, ELEVEN NEW UNITS WERE
     INSERTED ABOVE THEM. Top-k overlap cannot tell insertion from reshuffling and reports both as
     zero, which is exactly the failure mode a rank-displacement view exists to catch.

 (b) median |Delta log10 rank| by 2k-rank band -- the same question asked of the units rather than
     of the set. Flat at 0.19-0.51 decades through rank 1000, spikes to 1.07 decades in the
     1k-10k band (p75 there is rank 2.16M: over a quarter of that band fell to the very bottom),
     then falls back to 0.06 in the tail. Same U, and it separates the head's small constant
     displacement from the middle's genuine churn.

 (c) WHY acc-AUC DID NOT MOVE, and it is a CANCELLATION, not stability. Accuracy is decided in
     k~600-4000 -- everywhere above k=7441 both checkpoints are saturated at 0.94-1.00, and
     everywhere below k=583 both are at 0.00. In that decision band the 20k run is WORSE
     (0.32 vs 0.48 at k=1102, 0.63 vs 0.86 at k=2082) and above it slightly better
     (0.99 vs 0.94), and the log-k trapezoid nets the two to ~zero. Meanwhile faith-AUC in the
     churn band goes 0.93 -> 5.44. The reshuffle did something large; acc-AUC is simply blind to
     it. Do not read "acc-AUC 0.496 -> 0.494" as "the same circuit".

ADAM IS DRAWN FOR CONTRAST AND HAS THE OPPOSITE PROFILE: globally far more stable (rho +0.574 vs
SGD's +0.101) but with a much less stable HEAD (top-1000 kappa 0.125 vs SGD's 0.734). A single
rho would have ranked the two runs the other way round from how their circuits behave, which is
the second reason this figure is not a scatter plot.

Run:  uv run python plots/plot_rank_stability_20k.py
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                          # noqa: E402
from plot_train_curves import ARM_COLOR, FS_GRID             # noqa: E402

CELL = "addition_llama3_mlp_sufficient_topk_{opt}_bs1"
# lr 1.0 for SGD and 0.005 for Adam: the two arms' 2k argmax in results/sva_mlp_lr, which is the
# pair the 20k control was launched on. Adam's headline lr 0.05 has NO 2000-step counterpart with
# a matching checkpoint (it lives in results/sva_sweep), so it cannot be diffed at 2k.
ARMS = [("MAttr (SGD)", "sgd", "1.0"), ("MAttr (Adam)", "adam", "0.005")]
P2K = "results/sva_mlp_lr/topk_{opt}/lr_{lr}/" + CELL + ".json"
P20K = "results/sva_mlp_steps20k/topk_{opt}/lr_{lr}/" + CELL + "_s20000.json"
BANDS = [(0, 10), (10, 100), (100, 1_000), (1_000, 10_000),
         (10_000, 100_000), (100_000, 300_000), (300_000, 1_000_000), (1_000_000, None)]
FIG = (7.2, 2.05)
BAND_C = "#dcdcdc"


def ranks(path):
    """scores -> (order[rank] = unit, rank[unit] = rank). The .scores.pt sits beside the .json."""
    s = torch.load(path.replace(".json", ".scores.pt"), map_location="cpu").float()
    order = torch.argsort(s, descending=True)
    r = torch.empty_like(order)
    r[order] = torch.arange(len(s))
    return order, r


def kgrid(n):
    """eval_sva.py:733's own grid, so panel (a) is sampled where the AUC is."""
    sp = sorted(set(float(10 ** x) for x in np.linspace(np.log10(1.0 / n), 0.0, 24)))
    # k=N is dropped: "the top 2.29M of 2.29M units agree" is true by construction, and plotting
    # it puts a spike to kappa=1 on the right edge that reads as the rankings re-converging.
    return sorted({max(1, int(round(s * n))) for s in sp} - {n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/rank_stability_20k.pdf")
    a = ap.parse_args()

    data = {}
    for label, opt, lr in ARMS:
        f2, f20 = (p.format(opt=opt, lr=lr) for p in (P2K, P20K))
        o2, r2 = ranks(f2)
        o20, r20 = ranks(f20)
        n = len(r2)
        ks = kgrid(n)
        kap = []
        for k in ks:
            ov = len(set(o2[:k].tolist()) & set(o20[:k].tolist())) / k
            ch = k / n
            kap.append(1.0 if ch >= 1 else (ov - ch) / (1 - ch))
        disp = []
        for lo, hi in BANDS:
            hi = n if hi is None else hi
            new = r20[o2[lo:hi]].double() + 1
            old = torch.arange(lo, hi).double() + 1
            disp.append((lo, hi, float((new.log10() - old.log10()).abs().median())))
        data[label] = dict(n=n, ks=ks, kappa=kap, disp=disp,
                           j2=json.load(open(f2)), j20=json.load(open(f20)))

    plt.rcParams.update(P.RC)
    fs = FS_GRID
    fig, axes = plt.subplots(1, 3, figsize=FIG)
    sgd = data["MAttr (SGD)"]
    n = sgd["n"]

    # (a) agreement vs k. The shaded strip is where accuracy is actually decided -- the first and
    # last k at which the two checkpoints disagree by more than the 100-example eval's own noise;
    # outside it both curves are pinned at 0 or at ~1 and agreement there is unfalsifiable.
    ax = axes[0]
    ax.axvspan(583, 3936, color=BAND_C, lw=0, zorder=0)
    for label, _, _ in ARMS:
        d = data[label]
        ax.plot(d["ks"], d["kappa"], lw=1.1, color=ARM_COLOR[label], zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("$k$ (units kept)", fontsize=fs[0])
    ax.set_ylabel("top-$k$ agreement $\\kappa$", fontsize=fs[0])
    ax.set_title("(a) 2k vs 20k, by depth", fontsize=fs[0], pad=3)
    ax.annotate("acc decided\nhere", (1400, 0.02), fontsize=fs[2], color="#666666",
                ha="center", va="bottom")
    ax.tick_params(labelsize=fs[1])
    P.furnish(ax)

    # (b) displacement by band. Bars, not lines: the x axis is a set of bins, not a continuum, and
    # a line between bin centres would invite reading a value at rank 5000 that was never measured.
    ax = axes[1]
    w = 0.38
    for i, (label, _, _) in enumerate(ARMS):
        d = data[label]
        xs = np.arange(len(d["disp"])) + (i - 0.5) * w
        ax.bar(xs, [v for _, _, v in d["disp"]], width=w, color=ARM_COLOR[label],
               lw=0, zorder=3)
    ax.set_xticks(np.arange(len(BANDS)))
    ax.set_xticklabels([("$10^{%d}$" % round(np.log10(max(lo, 1)))) if lo else "1"
                        for lo, _ in BANDS], fontsize=fs[1])
    ax.set_xlabel("2k-step rank band (lower edge)", fontsize=fs[0])
    ax.set_ylabel(r"median $|\Delta \log_{10}$ rank$|$", fontsize=fs[0])
    ax.set_title("(b) how far units move", fontsize=fs[0], pad=3)
    ax.tick_params(labelsize=fs[1])
    P.furnish(ax)

    # (c) the cancellation. acc on the left axis, faith on the right -- they are different scales
    # (0-1 vs 0-5.5) and forcing them onto one flattens acc into the floor, which is the curve the
    # panel exists to show crossing.
    ax = axes[2]
    ks = [max(1, int(round(s * n)))
          for s in sorted(set(float(10 ** x) for x in np.linspace(np.log10(1.0 / n), 0.0, 24)))]
    ax.axvspan(583, 3936, color=BAND_C, lw=0, zorder=0)
    for j, ls in ((sgd["j2"], "dashed"), (sgd["j20"], "solid")):
        ax.plot(ks, j["iso_metrics"]["acc_base"], lw=1.1, ls=ls,
                color=ARM_COLOR["MAttr (SGD)"], zorder=3)
    ax2 = ax.twinx()
    for j, ls in ((sgd["j2"], "dashed"), (sgd["j20"], "solid")):
        ax2.plot(ks, j["faithfulness"], lw=1.0, ls=ls, color=P.METHOD["IG"], zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("$k$ (units kept)", fontsize=fs[0])
    ax.set_ylabel("accuracy", fontsize=fs[0])
    ax2.set_ylabel("faithfulness", fontsize=fs[0], color=P.METHOD["IG"])
    ax2.tick_params(labelsize=fs[1], colors=P.METHOD["IG"])
    ax.set_title("(c) SGD: acc cancels, faith does not", fontsize=fs[0], pad=3)
    ax.tick_params(labelsize=fs[1])
    P.furnish(ax)

    fig.tight_layout()
    fig.subplots_adjust(top=0.72)
    hs = [Line2D([0], [0], color=ARM_COLOR[lb], lw=1.1, label=lb) for lb, _, _ in ARMS]
    hs += [Line2D([0], [0], color="#444444", lw=1.1, ls=d, label=lb)
           for d, lb in (("dashed", "2k steps (c)"), ("solid", "20k steps (c)"))]
    fig.legend(handles=hs, fontsize=fs[2], ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.4,
               handlelength=2.0, columnspacing=1.2)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    for label, _, _ in ARMS:
        d = data[label]
        print(f"\n=== {label}  (N={d['n']})")
        print("  kappa @ k:", "  ".join(f"{k}:{v:.2f}" for k, v in zip(d["ks"], d["kappa"])
                                        if k in (10, 100, 583, 1102, 3936, 26591, 95021, 339556)))
        print("  median |dlog10 rank| by band:",
              "  ".join(f"{lo}-{hi}:{v:.2f}" for lo, hi, v in d["disp"]))


if __name__ == "__main__":
    main()
