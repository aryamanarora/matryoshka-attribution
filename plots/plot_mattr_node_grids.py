"""MAttr's node ranking drawn on the model: one layer x head grid per MIB task-model pair, with
the layer's MLP as an extra column (and the input embedding as a single cell above it).

Cell = the node's rank among all nodes of that cell (heads + MLPs + input) under the MAttr
headline test run (results/test_node_topk_uniform_lr05), on the same signed scale as
plot_ioi_head_types.py: top-half ranks in blue (rank 1 darkest), bottom-half ranks in red
(last darkest), the median rank white. The scale is log distance from the median, so a
12-layer and a 32-layer model read alike. On ioi/gpt2 the Wang et al. (2023) circuit heads are
outlined in their class colour.

Panels are the test table's twelve columns in its order (IOI x4 models, Arithmetic x2, MCQA x3,
ARC-E x2, ARC-C), 3 x 4. Grids are drawn with a free aspect so every panel has the same size;
the model's (layers x heads) is in the panel title.

    uv run python plots/plot_mattr_node_grids.py   ->  paper/figs/mattr_node_grids.pdf
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams.update({"font.family": "Inter", "font.size": 7, "axes.titlesize": 6.8,
                     "axes.labelsize": 6.5, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                     "legend.fontsize": 6, "axes.linewidth": 0.5, "pdf.fonttype": 42})
R = Path("results/test_node_topk_uniform_lr05")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

CELLS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
         ("arithmetic_addition", "llama3"), ("arithmetic_subtraction", "llama3"),
         ("mcqa", "qwen2.5"), ("mcqa", "gemma2"), ("mcqa", "llama3"),
         ("arc_easy", "gemma2"), ("arc_easy", "llama3"), ("arc_challenge", "llama3")]
TASK = {"ioi": "IOI", "arithmetic_addition": "Arith. (+)", "arithmetic_subtraction": "Arith. (−)",
        "mcqa": "MCQA", "arc_easy": "ARC-E", "arc_challenge": "ARC-C"}
MODEL = {"gpt2": "GPT-2", "qwen2.5": "Qwen", "gemma2": "Gemma", "llama3": "Llama"}

# Wang et al. IOI circuit, for the ioi/gpt2 outlines (same table as plot_ioi_head_types.py).
CLASSES = [
    ("Duplicate token",  ["a0.h1", "a0.h10", "a3.h0"]),
    ("Previous token",   ["a2.h2", "a4.h11"]),
    ("Induction",        ["a5.h5", "a5.h8", "a5.h9", "a6.h9"]),
    ("S-inhibition",     ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("Name mover",       ["a9.h6", "a9.h9", "a10.h0"]),
    ("Backup name mover", ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10", "a11.h2", "a11.h9"]),
    ("Neg. NM / copy suppr.", ["a10.h7", "a11.h10"]),
]
SET1 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#999999"]
COL = {c: SET1[i] for i, (c, _) in enumerate(CLASSES)}
CLASS_OF = {h: c for c, hs in CLASSES for h in hs}


def load(task, model):
    d = json.load(open(R / f"{task}_{model}_importances.json"))
    return {n: v["score"] for n, v in d["nodes"].items() if n != "logits" and "score" in v}


def ranks(s):
    v = np.array(list(s.values()))
    return {n: int((v > s[n]).sum()) + 1 for n in s}


def signed(rank, n):
    half = (n + 1) / 2
    return np.log10(half / rank) if rank <= half else -np.log10(half / (n + 1 - rank))


def grid(task, model):
    """(layers x (heads + 2)) array of signed ranks: heads, a NaN gap column, the MLP column;
    the input node is returned separately. Returns (G, input_u, n_layers, n_heads, n)."""
    rk = ranks(load(task, model)); n = len(rk)
    L = 1 + max(int(k[1:].split(".")[0]) for k in rk if k.startswith("a"))
    H = 1 + max(int(k.split(".h")[1]) for k in rk if k.startswith("a"))
    G = np.full((L, H + 2), np.nan)
    for l in range(L):
        for h in range(H):
            G[l, h] = signed(rk[f"a{l}.h{h}"], n)
        G[l, H + 1] = signed(rk[f"m{l}"], n)
    return G, signed(rk["input"], n), L, H, n


fig, axes = plt.subplots(3, 4, figsize=(5.5, 4.3))
VMAX = max(np.log10((len(load(t, m)) + 1) / 2) for t, m in CELLS)
cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#ffffff")
for ax, (task, model) in zip(axes.ravel(), CELLS):
    G, inp, L, H, n = grid(task, model)
    im = ax.imshow(G, cmap=cmap, vmin=-VMAX, vmax=VMAX, aspect="auto", interpolation="nearest")
    # input embedding: one cell drawn above the MLP column, in the gap row it creates
    ax.add_patch(Rectangle((H + 0.5, -1.5), 1, 1, facecolor=cmap((inp + VMAX) / (2 * VMAX)),
                           edgecolor="none", clip_on=False))
    ax.text(H + 1, L + 0.1, "MLP", ha="center", va="top", fontsize=4.8, color="#555", clip_on=False)
    if (task, model) == ("ioi", "gpt2"):
        for h, c in CLASS_OF.items():
            l, hh = (int(t[1:]) for t in h.split("."))
            ax.add_patch(Rectangle((hh - 0.5, l - 0.5), 1, 1, fill=False, lw=0.9, edgecolor=COL[c]))
    ax.set_title(f"{TASK[task]} / {MODEL[model]}", pad=6)   # pad clears the input cell
    ax.set_xticks([0, H - 1]); ax.set_xticklabels(["0", str(H - 1)])   # dims read off the ticks
    ax.set_yticks([0, L - 1]); ax.set_yticklabels(["0", str(L - 1)])
    ax.tick_params(length=1.5, pad=1.5)
    ax.set_xlim(-0.5, H + 1.5); ax.set_ylim(L - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.4)
for ax in axes[:, 0]:
    ax.set_ylabel("Layer")
for ax in axes[-1]:
    ax.set_xlabel("Head")
fig.tight_layout(h_pad=1.2, w_pad=0.7, rect=(0, 0.075, 0.905, 1))
cax = fig.add_axes([0.925, 0.35, 0.012, 0.4])
cb = fig.colorbar(im, cax=cax, ticks=[VMAX, VMAX - 1, VMAX - 2, 0, -(VMAX - 2), -(VMAX - 1), -VMAX])
cb.set_ticklabels(["1", "10", "100", "med.", "−100", "−10", "−1"])
cb.ax.tick_params(labelsize=5.5, length=1.5); cb.outline.set_linewidth(0.4)
cb.set_label("Rank (− = from the bottom)", fontsize=6)
hs = [Line2D([], [], marker="s", ls="", ms=4.5, mfc="none", mec=COL[c], mew=0.9, label=c)
      for c, _ in CLASSES]
fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.46, 0.0), ncol=7, frameon=False,
           handletextpad=0.3, columnspacing=0.7, fontsize=5.4,
           title="IOI / GPT-2 outlines: Wang et al. (2023) head classes", title_fontsize=5.6)
out = OUT / "mattr_node_grids.pdf"
fig.savefig(out); fig.savefig(out.with_suffix(".png"), dpi=200)
print("wrote", out)
