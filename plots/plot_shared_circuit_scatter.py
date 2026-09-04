"""Node-level anatomy of the transfer heatmap's star cell: Arith.(sub.) x Addition.

The transfer matrix says MIB's arithmetic task and wild Addition substitute for each other at
~0.98 of own-circuit accuracy across two different benchmarks and harnesses; this panel shows
WHAT that shared circuit is. One point per node (1057, MAttr+Adam scores from
results/transfer_src/adam -- same 18/20 shared top nodes and the same two heads as the SGD
scores, r=0.76 vs 0.75), x = score on Arith.(sub.), y = score on Addition, coloured by node
type. Only the two shared attention HEADS are labelled -- they are the surprise (everything
else in the shared top-20 is the input node and mid-layer MLPs, which the caption can say in
one clause); ringing all 18 was tried and read as clutter.

Axes are raw scores (different run lengths/harnesses, so the SCALES differ between axes;
the correlation the caption quotes, r=0.75, is scale-free). matplotlib, not plotnine: 18
labelled points in one quadrant need hand-tuned nudges and plotnine cannot express them.

Out: paper/figs/shared_circuit_scatter.pdf (~0.31\\textwidth slot, 1.45x1.5in like the row)
--pair selects the task pair; each supported pair carries its own hand-placed label layout in
PAIRS (label SELECTION is programmatic -- top-2 shared heads, top-2 shared MLPs, input -- but
POSITIONS are data-coordinate judgements no layout pass gets right at this panel size). The
arith pair's shared circuit is MLP-dominated with two heads; mcqa/arc_challenge is the
anatomical opposite: 8 of its 16 shared top-20 nodes are heads. Non-default pairs write
shared_circuit_scatter_<x>_<y>.pdf.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_task_corr_heatmap import names

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
SRC = Path("results/transfer_src/adam")

# per-pair display config: axis labels, zoom threshold, axis limits, and the hand-placed
# label layout {node: (tx, ty, ha)}. Nodes listed here that are not in the pair's shared
# top-20 raise, so a re-run after the scores change cannot silently label a stale node.
PAIRS = {
    ("arithmetic_subtraction", "addition"): dict(
        xlab="Score on Arith. (sub.)", ylab="Score on Addition",
        xlim=(0.8, 5.95), ylim=(1.9, 11.0),
        labels={"a15.h13": (3.30, 10.45, "right"), "m0": (4.90, 10.50, "left"),
                "a16.h21": (4.90, 9.70, "left"), "input": (4.90, 8.90, "left"),
                "m18": (4.90, 8.10, "left")}),
    ("mcqa", "arc_challenge"): dict(
        xlab="Score on MCQA", ylab="Score on ARC-C",
        xlim=(0.9, 7.55), ylim=(0.7, 6.75),
        labels={"a16.h22": (3.95, 6.45, "right"), "input": (5.75, 6.35, "left"),
                "m26": (5.75, 5.60, "left"), "a30.h26": (5.75, 4.85, "left"),
                "m14": (5.75, 4.10, "left")}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, default=["arithmetic_subtraction", "addition"],
                    metavar=("X_TASK", "Y_TASK"), help="task pair; must be a key of PAIRS")
    args = ap.parse_args()
    if tuple(args.pair) not in PAIRS:
        raise SystemExit(f"no label layout for pair {args.pair}; add it to PAIRS")
    cfg = PAIRS[tuple(args.pair)]
    nm = names()
    a = torch.load(SRC / f"{args.pair[0]}.pt").numpy()
    b = torch.load(SRC / f"{args.pair[1]}.pt").numpy()
    kind = np.array(["input" if n == "input" else "mlp" if n.startswith("m") else "head"
                     for n in nm])
    shared = set(np.argsort(-a)[:20]) & set(np.argsort(-b)[:20])
    r = np.corrcoef(a, b)[0, 1]   # ALL nodes, computed before the zoom discards the blob
    print(f"r={r:.3f}, top-20-both: {len(shared)}: {sorted(nm[i] for i in shared)}")
    hi = sorted(i for i in shared if nm[i].startswith("a"))   # the shared heads, and only them

    zoom = (a > 1.0) & (b > 1.0)   # the joint-top region
    print(f"zoom shows {zoom.sum()} nodes")
    fig, ax = plt.subplots(figsize=(1.45, 1.5))
    C = {"head": "#0072b2", "mlp": "#e69f00", "input": "#000000"}
    # Wong blue heads / orange MLPs / black input; black edge like the accauc scatters, so
    # coincident markers stay separable against the grid
    for k, lab, z in (("head", "Attn head", 2), ("mlp", "MLP", 1), ("input", "Input", 3)):
        m = zoom.numpy() & (kind == k) if hasattr(zoom, "numpy") else zoom & (kind == k)
        ax.scatter(a[m], b[m], s=11, c=C[k], alpha=1.0, linewidths=0.3,
                   edgecolors="#000000", label=lab, zorder=z)

    # house-style labels (leader line + white rounded bbox, series-coloured text): the top-2
    # shared heads, top-2 shared MLPs (by min of the two scores) and the input node, at the
    # pair's hand-placed positions. Most go into a right-margin label column (the XPAD
    # pattern of plot_mib_accauc_cpr_scatter: data range padded right to hold labels).
    ax.set_xlim(*cfg["xlim"]); ax.set_ylim(*cfg["ylim"])
    top_m = sorted((i for i in shared if nm[i].startswith("m")), key=lambda i: -min(a[i], b[i]))
    top_h = sorted(hi, key=lambda i: -min(a[i], b[i]))   # hi is index-sorted; strength here
    want = [nm[i] for i in top_h[:2]] + [nm[i] for i in top_m[:2]] + ["input"]
    missing = [n for n in want if n not in cfg["labels"]]
    stale = [n for n in cfg["labels"] if n not in want]
    if missing or stale:
        raise SystemExit(f"label layout out of date: place {missing}, drop {stale}")
    labels = [(nm.index(n), n, C["head" if n.startswith("a") else "input" if n == "input"
                                else "mlp"], (tx, ty), ha)
              for n, (tx, ty, ha) in cfg["labels"].items()]
    for i, lab, col, (tx, ty), ha in labels:
        ax.plot([a[i], tx - (0.05 if ha == "left" else -0.05)], [b[i], ty],
                lw=0.35, color="#888888", zorder=4)
        ax.annotate(lab, (tx, ty), fontsize=5, va="center", ha=ha, color=col, zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))

    ax.set_xlabel(cfg["xlab"], fontsize=6)
    ax.set_ylabel(cfg["ylab"], fontsize=6)
    ax.tick_params(labelsize=5, width=0.5, length=2)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.legend(fontsize=5, frameon=False, loc="lower right", handletextpad=0.1,
              borderaxespad=0.2, labelspacing=0.2)
    fig.tight_layout(pad=0.3)
    default = args.pair == ["arithmetic_subtraction", "addition"]
    fn = OUT / ("shared_circuit_scatter.pdf" if default
                else f"shared_circuit_scatter_{args.pair[0]}_{args.pair[1]}.pdf")
    fig.savefig(fn, dpi=300)
    print("wrote", fn)


if __name__ == "__main__":
    main()
