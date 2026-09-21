"""Exploratory: alternative layouts for the ioi/gpt2 "where does each method rank the IOI circuit
heads" figure (plots/plot_ioi_head_types.py is the shipped per-head heatmap). Same data, five
other encodings, each to its own PNG under plots/ioi_head_types_variants/ so they can be
compared side by side before one is promoted.

    uv run python plots/eda_ioi_head_types_variants.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning_to_attribute.deps import mib_results_dir  # noqa: E402

plt.rcParams.update({"font.family": "Inter", "font.size": 7, "axes.titlesize": 7.5,
                     "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                     "legend.fontsize": 6, "axes.linewidth": 0.5, "xtick.major.width": 0.4,
                     "ytick.major.width": 0.4, "pdf.fonttype": 42})
R = Path("results"); R_MIB = mib_results_dir()
OUT = Path("plots/ioi_head_types_variants"); OUT.mkdir(parents=True, exist_ok=True)

CLASSES = [
    ("Duplicate token",  ["a0.h1", "a0.h10", "a3.h0"]),
    ("Previous token",   ["a2.h2", "a4.h11"]),
    ("Induction",        ["a5.h5", "a5.h8", "a5.h9", "a6.h9"]),
    ("S-inhibition",     ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("Name mover",       ["a9.h6", "a9.h9", "a10.h0"]),
    ("Backup name mover", ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10", "a11.h2", "a11.h9"]),
    ("Neg. NM / copy suppression", ["a10.h7", "a11.h10"]),
]
CLASS_OF = {h: c for c, hs in CLASSES for h in hs}
HEADS = [h for _, hs in CLASSES for h in hs]
NEG = set(CLASSES[-1][1]); POS = [h for h in HEADS if h not in NEG]
SET1 = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#999999"]
COL = {c: SET1[i] for i, (c, _) in enumerate(CLASSES)}
METHODS = [
    ("MAttr",              "test_node_topk_uniform_lr05",             "flat"),
    ("IntInv",             "test_node_actpatch_noise",                "flat"),
    ("Node Pruning",       "eprun_node_s0.5_ld",                      "graph"),
    ("DBM",                "eprun_node_ld_sig_lr0.3_l16.0",           "graph"),
    ("AttnLRP",            "attnlrp/AttnLRP_patching_node",           "nested"),
    ("Expected Gradients", "napig_mc/EAP-IG-inputs-mc_patching_node", "nested"),
    ("IG (m=10)",          "napig10/EAP-IG-inputs_patching_node",     "nested"),
    ("I×G",                "ig1/EAP-IG-inputs_patching_node",         "nested"),
    ("− learning",         "test_node_topkid_uniform_frozen",         "flat"),
    ("Random",             "random_s42/Random_patching_node",         "nested"),
]
N = 157


def load(path):
    d = json.load(open(path)); nodes = d.get("nodes", d)
    s = {n: v["score"] for n, v in nodes.items() if n != "logits" and "score" in v}
    assert len(s) == N
    return s


def scores(spec):
    _, loc, layout = spec
    if layout == "flat":
        return load(R / loc / "ioi_gpt2_importances.json")
    if layout == "graph":
        return load(R / loc / "graph_ioi_gpt2.json")
    return load(R_MIB / loc / "ioi_gpt2" / "importances.json")


def ranks(s):
    v = np.array(list(s.values()))
    return {n: int((v > s[n]).sum()) + 1 for n in s}


RK = {m[0]: ranks(scores(m)) for m in METHODS}
NAMES = [m[0] for m in METHODS]
ALL_HEADS = [f"a{l}.h{h}" for l in range(12) for h in range(12)]


def savefig(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ---------------------------------------------------------------- B: rank strips ----------
# One row per method, x = rank on a log axis. Every non-circuit node is a faint tick; the 26
# circuit heads are coloured points (class colour, filled). The reading: a good method's
# coloured points pile up at the left, and the colour tells which class is left behind.
def variant_strips():
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    y = {m: i for i, m in enumerate(NAMES[::-1])}
    rng = np.random.default_rng(0)
    for m in NAMES:
        rk = RK[m]
        others = [rk[n] for n in rk if n not in CLASS_OF]
        ax.scatter(others, np.full(len(others), y[m]), marker="|", s=18, color="#bbbbbb",
                   linewidths=0.5, zorder=1)
        for h in HEADS:
            jitter = rng.uniform(-0.22, 0.22)
            ax.scatter(rk[h], y[m] + jitter, s=14, color=COL[CLASS_OF[h]], edgecolor="#000",
                       linewidths=0.3, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.85, 190)
    ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["1", "3", "10", "30", "100"])
    ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES[::-1])
    ax.set_ylim(-0.6, len(NAMES) - 0.4)
    ax.set_xlabel("Rank among 157 nodes (1 = most important)")
    ax.grid(axis="x", color="#eeeeee", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hs = [Line2D([], [], marker="o", ls="", ms=4, mfc=COL[c], mec="#000", mew=0.3, label=c)
          for c, _ in CLASSES] + [Line2D([], [], marker="|", ls="", ms=5, color="#bbbbbb",
                                         label="other node")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False,
              handletextpad=0.3, columnspacing=1.0)
    savefig(fig, "B_rank_strips")


# ------------------------------------------------------- C: class-level summary heatmap ----
# methods x classes, cell = median rank of that class's heads (log colour, text). The
# coarsest view: one number per (method, class).
def variant_class_heatmap():
    M = np.array([[np.median([RK[m][h] for h in hs]) for _, hs in CLASSES] for m in NAMES])
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    im = ax.imshow(np.log10(M), cmap="Blues_r", vmin=0, vmax=np.log10(N), aspect="auto")
    for i in range(len(NAMES)):
        for j in range(len(CLASSES)):
            v = M[i, j]
            ax.text(j, i, f"{v:g}", ha="center", va="center", fontsize=6,
                    color="#fff" if v <= 12 else "#000")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels([c.replace(" / ", "/\n").replace("name mover", "NM") for c, _ in CLASSES],
                       rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(len(NAMES))); ax.set_yticklabels(NAMES)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[0, 1, 2])
    cb.set_ticklabels(["1", "10", "100"]); cb.set_label("Median rank", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    savefig(fig, "C_class_heatmap")


# --------------------------------------------------------- D: bump chart across methods ----
# x = methods, y = rank (log), one line per circuit head coloured by class. Shows which heads
# MOVE between methods (a9.h6 is the visible one) and which are pinned everywhere.
def variant_bump():
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    x = np.arange(len(NAMES))
    for h in HEADS:
        ys = [RK[m][h] for m in NAMES]
        ax.plot(x, ys, "-o", color=COL[CLASS_OF[h]], lw=0.8, ms=2.6, mec="#000", mew=0.25,
                alpha=0.9, zorder=3)
    # label the heads whose rank spans more than a decade across the non-random methods
    for h in HEADS:
        ys = [RK[m][h] for m in NAMES[:-1]]
        if max(ys) / min(ys) > 8 and h not in NEG:
            ax.annotate(h, (x[-1] - 0.9, RK[NAMES[-2]][h]), fontsize=5.5, xytext=(3, 0),
                        textcoords="offset points", va="center")
    ax.set_yscale("log"); ax.invert_yaxis(); ax.set_ylim(190, 0.85)
    ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(["1", "3", "10", "30", "100"])
    ax.set_xticks(x); ax.set_xticklabels(NAMES, rotation=30, ha="right")
    ax.set_ylabel("Rank (1 = top)")
    ax.grid(axis="y", color="#eeeeee", lw=0.4); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    hs = [Line2D([], [], color=COL[c], lw=1.2, label=c) for c, _ in CLASSES]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False,
              handlelength=1.2, columnspacing=1.0)
    savefig(fig, "D_bump")


# -------------------------------------------------------- E: circuit recall vs. top-k -----
# For each method, the fraction of the 24 positive circuit heads inside the method's top-k
# nodes as k grows (log k). The single-curve view of "does the method find the circuit", with
# a per-class small-multiple underneath so the class story is not lost.
def variant_recall():
    ks = np.arange(1, N + 1)
    cmap = plt.get_cmap("tab10")
    mcol = {m: ("#000000" if m == "MAttr" else "#999999" if m == "Random" else cmap(i))
            for i, m in enumerate(NAMES)}
    fig, axes = plt.subplots(2, 4, figsize=(5.5, 3.0), gridspec_kw=dict(height_ratios=[1.25, 1]))
    gs = axes[0, 0].get_gridspec()
    for a in axes[0]:
        a.remove()
    big = fig.add_subplot(gs[0, :])
    for m in NAMES:
        r = np.array([RK[m][h] for h in POS])
        big.plot(ks, [(r <= k).mean() for k in ks], color=mcol[m], lw=1.6 if m == "MAttr" else 0.9,
                 ls="--" if m == "Random" else "-", label=m)
    big.set_xscale("log"); big.set_xlim(1, N); big.set_ylim(0, 1.02)
    big.set_xlabel("Top-k nodes"); big.set_ylabel("Recall of 24 circuit heads")
    big.axvline(24, color="#ccc", lw=0.5); big.text(24, 0.03, " k = 24", fontsize=5.5, color="#888")
    big.legend(ncol=5, frameon=False, loc="upper left", fontsize=5.5, handlelength=1.4,
               columnspacing=0.9)
    for s in ("top", "right"):
        big.spines[s].set_visible(False)
    for a, (c, hs) in zip(axes[1], [c for c in CLASSES if c[0] != CLASSES[-1][0]][:4]):
        for m in NAMES:
            r = np.array([RK[m][h] for h in hs])
            a.plot(ks, [(r <= k).mean() for k in ks], color=mcol[m],
                   lw=1.4 if m == "MAttr" else 0.7, ls="--" if m == "Random" else "-")
        a.set_xscale("log"); a.set_xlim(1, N); a.set_ylim(0, 1.05)
        a.set_title(c, fontsize=6.5, pad=2)
        a.tick_params(labelsize=5.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout(h_pad=0.6, w_pad=0.5)
    savefig(fig, "E_recall_curves")


# -------------------------------------------------- F: layer x head grids per method -------
# Small multiples of the 12x12 (layer x head) grid, one per method, cell shade = rank (log),
# circuit heads outlined in their class colour. "Where in the model" each method looks,
# against where the circuit is.
def variant_grids():
    fig, axes = plt.subplots(2, 5, figsize=(5.5, 2.75))
    for a, m in zip(axes.ravel(), NAMES):
        G = np.array([[RK[m][f"a{l}.h{h}"] for h in range(12)] for l in range(12)])
        a.imshow(np.log10(G), cmap="Blues_r", vmin=0, vmax=np.log10(N), aspect="equal")
        for h in HEADS:
            l, hh = (int(t[1:]) for t in h.split("."))
            a.add_patch(Rectangle((hh - 0.5, l - 0.5), 1, 1, fill=False, lw=1.0,
                                  edgecolor=COL[CLASS_OF[h]]))
        a.set_title(m, fontsize=6.5, pad=2)
        a.set_xticks([0, 11]); a.set_yticks([0, 11]); a.tick_params(labelsize=5, length=1.5)
        if m in (NAMES[0], NAMES[5]):
            a.set_ylabel("Layer", fontsize=6)
        if m in NAMES[5:]:
            a.set_xlabel("Head", fontsize=6)
    hs = [Line2D([], [], marker="s", ls="", ms=5, mfc="none", mec=COL[c], mew=1.0, label=c)
          for c, _ in CLASSES]
    fig.legend(handles=hs, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=4,
               frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    savefig(fig, "F_layer_head_grids")


if __name__ == "__main__":
    variant_strips(); variant_class_heatmap(); variant_bump(); variant_recall(); variant_grids()
