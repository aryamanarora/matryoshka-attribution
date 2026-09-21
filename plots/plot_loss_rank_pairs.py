"""Rank agreement between the attribution LOSSES of one MAttr arm, pair by pair (2026-09-21).

One cell (default ioi / qwen2.5, node substrate, the headline uniform-k eps=1e-2 arm), one
scatter per pair of losses: x = rank under loss A, y = rank under loss B (1 = highest score),
one point per unit, attention heads vs MLP blocks by colour, Spearman rho in the corner, and
the units named in --mark drawn as labelled stars. Lower-triangular grid, so each pair appears
once.

    uv run python plots/plot_loss_rank_pairs.py [--task ioi --model qwen2.5] [--losses logit_diff ce acc kl cmd]
    -> plots/loss_rank_pairs_<task>.pdf (+ .png)
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
ARM = "sufficient_topk_adam_eps1e-2{loss}_uniformk_bs1"
LOSS_LABEL = {"logit_diff": "logit-diff", "ce": "CE", "acc": "soft-acc", "kl": "KL", "cmd": "CMD"}
SHAPE = {"qwen2.5": (24, 14), "llama3": (32, 32), "gpt2": (12, 12), "gemma2": (26, 8)}   # (layers, heads)
C_ATTN, C_MLP = "#0072b2", "#e69f00"


def load(task, model, loss):
    tag = ARM.format(loss="" if loss == "logit_diff" else f"_{loss}")
    fs = glob.glob(f"{RES}/{task}_{model}_node_{tag}.scores.pt")
    if not fs:
        return None
    s = torch.load(fs[0], map_location="cpu")
    s = s if torch.is_tensor(s) else s["scores"]
    return s.float().numpy()


def unit_names(n, L, H):
    """node layout without the input node: [all attn (layer-major) | all MLP]."""
    assert n == L * H + L, f"{n} units != {L}*{H}+{L}"
    return [f"a{i // H}.h{i % H}" for i in range(L * H)] + [f"m{i}" for i in range(L)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--model", default="qwen2.5")
    ap.add_argument("--losses", nargs="+", default=["logit_diff", "ce", "acc", "kl", "cmd"])
    ap.add_argument("--mark", nargs="*", default=["a19.h6"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    L, H = SHAPE[a.model]
    scores = {l: load(a.task, a.model, l) for l in a.losses}
    losses = [l for l in a.losses if scores[l] is not None]
    for l in a.losses:
        if scores[l] is None:
            print(f"  {l}: no run for {a.task}/{a.model}, dropped")
    n = len(scores[losses[0]])
    names = unit_names(n, L, H)
    is_attn = np.array([nm.startswith("a") for nm in names])
    ranks = {l: (-scores[l]).argsort().argsort() + 1 for l in losses}   # 1 = highest score

    k = len(losses)
    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(k - 1, k - 1, figsize=(1.35 * (k - 1), 1.35 * (k - 1)), squeeze=False)
    for i in range(k - 1):
        for j in range(k - 1):
            ax = axes[i][j]
            if j > i:
                ax.set_visible(False)
                continue
            lx, ly = losses[j], losses[i + 1]
            x, y = ranks[lx], ranks[ly]
            ax.scatter(x[~is_attn], y[~is_attn], s=6, c=C_MLP, lw=0, alpha=0.85, zorder=3, label="MLP")
            ax.scatter(x[is_attn], y[is_attn], s=3.5, c=C_ATTN, lw=0, alpha=0.55, zorder=2, label="attn head")
            for nm in a.mark:
                if nm in names:
                    u = names.index(nm)
                    ax.scatter([x[u]], [y[u]], marker="*", s=40, c="#d55e00", edgecolors="black",
                               lw=0.4, zorder=5)
                    ax.annotate(nm, (x[u], y[u]), xytext=(3, 3), textcoords="offset points",
                                fontsize=5, color="#d55e00")
            rho = spearmanr(x, y).correlation
            ax.text(0.04, 0.96, f"$\\rho$={rho:.2f}", transform=ax.transAxes, fontsize=5.5,
                    va="top", ha="left")
            ax.plot([1, n], [1, n], lw=0.4, color="#bbbbbb", zorder=1)
            ax.set_xlim(0, n + 1); ax.set_ylim(0, n + 1)
            ax.set_xticks([1, n]); ax.set_yticks([1, n])
            ax.tick_params(labelsize=5, length=1.5, pad=1)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if i == k - 2:
                ax.set_xlabel(f"rank, {LOSS_LABEL[lx]}", fontsize=6)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f"rank, {LOSS_LABEL[ly]}", fontsize=6)
            else:
                ax.set_yticklabels([])
    h, lab = axes[k - 2][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", fontsize=6, frameon=False, bbox_to_anchor=(0.98, 0.98),
               markerscale=1.8)
    fig.suptitle(f"{a.task} / {a.model}, node substrate, MAttr (uniform $k$, Adam $\\epsilon$=1e-2)",
                 fontsize=6.5, y=0.995)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    out = a.out or f"plots/loss_rank_pairs_{a.task}.pdf"
    fig.savefig(out); fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)
    print("Spearman rho between losses:")
    for i, l1 in enumerate(losses):
        print(f"  {LOSS_LABEL[l1]:10s}" + " ".join(f"{spearmanr(ranks[l1], ranks[l2]).correlation:6.2f}" for l2 in losses))


if __name__ == "__main__":
    main()
