"""Which MAttr OBJECTIVE does each baseline implicitly optimise? (2026-09-21, requested)

For every node-substrate SVA+ cell, Spearman rho between a baseline's ranking (its default
logit-difference target) and the headline MAttr arm (uniform k, Adam eps=1e-2) trained under
each attribution loss: logit-diff, CE, soft-acc, KL (to the clean distribution) and CMD
(|1 - faithfulness|). Averaged over the cells that have both runs; the heatmap annotates the
mean rho and the argmax objective per method is printed, i.e. "I x G is closest to MAttr trained
on <objective>". A gradient method is a first-order estimate of SOME objective's change under
patching, so the objective it lands nearest is what it is implicitly optimising.

Cells: the ten node-level SVA+ tasks (results/sva_sweep). KL and CMD arms exist where
submit_unifk_eps_node_sc.sh LOSSES="kl cmd" has landed; the per-column cell count is annotated.

    uv run python plots/plot_method_objective_affinity.py [--out plots/method_objective_affinity.pdf]
"""
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P  # noqa: E402

RES = "results/sva_sweep"
CELLS = [("nounpp", "llama3"), ("rc", "llama3"), ("simple", "llama3"), ("within_rc", "llama3"),
         ("addition", "llama3"), ("months", "llama3"), ("weekdays", "llama3"), ("hours", "llama3"),
         ("arc_easy", "llama3"), ("ioi", "qwen2.5")]
ARM = "sufficient_topk_adam_eps1e-2{loss}_uniformk_bs1"
OBJECTIVES = [("logit_diff", "logit-diff"), ("ce", "CE"), ("acc", "soft-acc"), ("kl", "KL"), ("cmd", "CMD")]
# display name -> file tag (results/sva_sweep/<task>_<model>_node_<tag>.scores.pt)
METHODS = [("I$\\times$G", "ixg"), ("IG ($m{=}10$)", "ig"), ("Expected Gradients", "mc_ig_m1_s42"),
           ("AttnLRP", "attnlrp"), ("Node Pruning", "eprun_s090"), ("DBM", "sig_lr0.3_l16.0"),
           ("MAttr $-$ learning", "sufficient_topk_identity_none_uniformk_bs1"),
           ("Random", "random_s42")]


def load(task, model, tag):
    fs = glob.glob(f"{RES}/{task}_{model}_node_{tag}.scores.pt")
    if not fs:
        return None
    s = torch.load(fs[0], map_location="cpu")
    s = s if torch.is_tensor(s) else s["scores"]
    return s.float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/method_objective_affinity.pdf")
    a = ap.parse_args()
    rho = {}   # (method, objective) -> [per-cell rho]
    for task, model in CELLS:
        arms = {o: load(task, model, ARM.format(loss="" if o == "logit_diff" else f"_{o}"))
                for o, _ in OBJECTIVES}
        for mlabel, tag in METHODS:
            s = load(task, model, tag)
            if s is None:
                continue
            for o, _ in OBJECTIVES:
                if arms[o] is None or len(arms[o]) != len(s):
                    continue
                rho.setdefault((mlabel, o), []).append(spearmanr(s, arms[o]).correlation)
    M = np.full((len(METHODS), len(OBJECTIVES)), np.nan)
    N = np.zeros_like(M, dtype=int)
    for i, (mlabel, _) in enumerate(METHODS):
        for j, (o, _) in enumerate(OBJECTIVES):
            v = rho.get((mlabel, o), [])
            if v:
                M[i, j] = float(np.mean(v)); N[i, j] = len(v)

    print(f"{'method':22s}" + "".join(f"{ol:>12s}" for _, ol in OBJECTIVES) + "   closest objective")
    for i, (mlabel, _) in enumerate(METHODS):
        row = "".join(f"{M[i, j]:8.2f} ({N[i, j]:2d})" if N[i, j] else f"{'---':>12s}" for j in range(len(OBJECTIVES)))
        best = int(np.nanargmax(M[i])) if np.isfinite(M[i]).any() else None
        print(f"{mlabel:22s}{row}   {OBJECTIVES[best][1] if best is not None else '---'}")

    plt.rcParams.update(P.RC)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        best = int(np.nanargmax(M[i])) if np.isfinite(M[i]).any() else -1
        for j in range(M.shape[1]):
            if N[i, j]:
                txt = f"{M[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        fontweight="bold" if j == best else "normal",
                        color="white" if abs(M[i, j]) > 0.55 else "black")
                if N[i, j] < len(CELLS):
                    ax.text(j + 0.42, i - 0.38, f"n={N[i, j]}", ha="right", va="top", fontsize=4.5,
                            color="#555555")
    ax.set_xticks(range(len(OBJECTIVES))); ax.set_xticklabels([ol for _, ol in OBJECTIVES], fontsize=6.5)
    ax.set_yticks(range(len(METHODS))); ax.set_yticklabels([ml for ml, _ in METHODS], fontsize=6.5)
    ax.set_xlabel("MAttr training objective", fontsize=7)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Spearman $\\rho$ (mean over node-level SVA+ cells)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(a.out); fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
