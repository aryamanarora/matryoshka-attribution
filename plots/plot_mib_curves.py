"""Appendix full-width per-cell curves over the MIB denoising sparsity sweep, four methods.

Two figures, same layout:
  - mib_accuracy_curves.pdf : decision accuracy (fraction of examples with metric>0) vs sparsity
  - mib_cpr_curves.pdf      : CPR / faithfulness (ablated normalised to [corrupted, clean])
x-axis = fraction of nodes kept, log scale (matches acc-AUC's log weighting). acc-AUC is the
log-x-weighted mean of the accuracy curve; CPR AUC is the linear-x area under the CPR curve.

STYLED TO MATCH plots/plot_sva_curves.py (figs/sva_curves_iia.pdf), which is the other
full-width per-cell curve appendix in this paper: raw matplotlib rather than plotnine, legend
on TOP, hairline curves with NO point markers, palette.furnish grid, and shared super-axis
labels. The two figures sit a few pages apart and answer the same question on different
benchmarks, so a reader should not have to re-learn the visual language between them.

*** THE POINT MARKERS AND THE X-JITTER ARE BOTH GONE, AND THEY WENT TOGETHER. *** The old
plotnine version drew a marker at each of the 10 sweep points and then multiplied x by a
per-method factor (+-0.12 decade at 9 series) so the markers would not stack. That jitter put
every series on a SLIGHTLY WRONG x -- tolerable when the marker was the thing being read,
indefensible once the marker is gone. Curves are now drawn at the true sweep fractions.

Y. Accuracy is a bounded fraction, so it shares one 0--1 axis across every panel and the
panels are directly comparable. CPR is unbounded and its ceiling is set by the cell (MCQA/Llama
reaches ~5 where IOI/Qwen reaches ~2), so it gets a per-panel y -- read shape, not height.

Each curve reads BOTH arrays from one self-consistent eval pkl:
  MAttr    -> results/<dir>/{task}_{model}_validation.pkl
  gradient -> MIB-circuit-track/results/<dir>/<sub>/{stask}_{model}_validation_abs-False.pkl
              (dir is *_accauc for the older runs, *_eval for ones evaluated after the
               `accuracies` array became standard -- see the METHODS comment)
The `eprun` loader branch is kept though no series uses it, so a mask-learning baseline can be
put back by adding one tuple to METHODS.

Run:  uv run python plots/plot_mib_curves.py
"""
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                   # noqa: E402

R = Path("results")
MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
OUT = Path("plots")
PCT = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

# task/model -> facet label (order = facet order)
COLUMNS = [
    ("ioi", "gpt2", "IOI (GPT-2)"), ("ioi", "qwen2.5", "IOI (Qwen)"),
    ("ioi", "gemma2", "IOI (Gemma)"), ("ioi", "llama3", "IOI (Llama)"),
    ("arithmetic_subtraction", "llama3", "Arithmetic (Llama)"),
    ("mcqa", "qwen2.5", "MCQA (Qwen)"), ("mcqa", "gemma2", "MCQA (Gemma)"),
    ("mcqa", "llama3", "MCQA (Llama)"),
    ("arc_easy", "gemma2", "ARC-E (Gemma)"), ("arc_easy", "llama3", "ARC-E (Llama)"),
    ("arc_challenge", "llama3", "ARC-C (Llama)"),
]
FACET_ORDER = [c[2] for c in COLUMNS]

# method -> (colour, linetype, loader-kind, dir/sub). Colours from plots/palette.py, the single
# source of truth shared with every other figure -- do not write hex codes here.
#
# FOUR SERIES, cut to these four on 2026-09-02 (requested). What was dropped and why it is not
# missing: the IG step ladder (5/10/30) made a point about integration budget that this figure
# is no longer about; "+hard" is a forward-pass ablation of MAttr; GIM, Node Pruning and DBM are
# baselines the tables and figs/mib_accauc_cpr_scatter.pdf still carry. Every one of them is
# still in tabs/mib_results.tex, so nothing is only visible here. Re-add by restoring a tuple.
#
# DASHES SURVIVE ON ONE SERIES ONLY, and for a different reason than before. They used to
# separate the three same-orange IG rungs and the two same-blue MAttr arms; both of those hue
# collisions are gone. But at their own optima the two MAttr arms score 1.879 (Adam) and 1.886
# (SGD) and their curves lie on top of each other in most panels, so solid-over-solid simply
# hides whichever is drawn first. SGD is dashed so the overlap reads AS overlap -- which is the
# result -- instead of as a missing series.
#
# *** MAttr+SGD IS NOW lr=1.0, ITS OWN SWEPT OPTIMUM, NOT Adam's 0.05. *** This series used to
# read softlog_sgd_lr_0.05 -- SGD pinned to Adam's LR as a single-knob contrast. That was
# defensible in a nine-series figure where the optimizer was one comparison among many; in a
# four-series figure whose whole content is Adam vs SGD it is not, because SGD at 20x off its
# optimum scores 1.413 against 1.886 at lr=1.0, and the curve would read as an optimizer result
# when it is an LR result. tabs/mib_results.tex has used the own-optimum dirs since 2026-08-24
# for exactly this reason (see make_mib_table.OUR_METHODS: "do NOT re-pin these to a shared
# LR"), so this also stops the figure and the table describing different runs of the same row.
METHODS = [
    ("MAttr (Adam)", P.color("MAttr"), "solid", "mattr", "topklog_lr_0.05"),
    ("MAttr (SGD)", P.color("MAttr (SGD)"), "dashed", "mattr", "softlog_sgd_lr_1.0"),
    # alpha ~ U(0,1) drawn per example instead of an m-point grid, at the same one-backward cost
    # as I x G. Seed 0, the headline dir every other consumer reads (make_mib_accauc_table,
    # make_mib_test_table, plot_mib_accauc_cpr_scatter); _s1/_s2 are 5/11 seed replicates.
    # Its own MIB subfolder name -- run_napig_mc.sh passes --method EAP-IG-inputs-mc precisely
    # so it cannot overwrite the fixed-grid NAP-IG pkls.
    ("Expected Gradients", P.color("Expected Gradients"), "solid", "base",
     ("napig_mc_eval", "EAP-IG-inputs-mc_patching_node")),
    # ig1_accauc, NOT ig1_eval: same run (area_under 0.478 vs 0.477 over 11 cells) but only the
    # _accauc dir carries the `accuracies` array the top figure needs.
    ("I×G", P.color("I×G"), "solid", "base", ("ig1_accauc", "EAP-IG-inputs_patching_node")),
]
METHOD_ORDER = [m[0] for m in METHODS]

# Typography and geometry lifted from plot_sva_curves.py so the two appendices match. 5.5in is
# iclr2027_conference.sty's \textwidth, so at width=\linewidth this is placed 1:1 and these are
# the sizes that reach the compiled PDF. ROW_H matches sva's 0.78in/row; HEAD is the legend band.
NCOL = 4
FIG_W, ROW_H, HEAD = 5.5, 0.86, 0.48
FS_TITLE, FS_TICK, FS_LAB, FS_LEG = 6.5, 5, 6.5, 5.5
LW = 1.3          # 4 series, not 9 -- thin hairlines were a density fix this cut removed
# plotnine linetype names -> matplotlib dash tuples. Kept as names in METHODS because that is
# what the three IG rungs are documented by; translated once, here.
# Dash lengths are in LINEWIDTHS, so they scale with LW: at 1.3pt the old (2.5, 1.5)
# reads as a dotted line rather than a dashed one.
DASH = {"solid": "solid", "dashed": (0, (3.2, 1.5)), "dotted": (0, (1, 1.3))}
XTICKS = [1e-3, 1e-2, 1e-1, 1e0]
XLAB = ["0.1%", "1%", "10%", "100%"]


def load(kind, loc, task, model):
    stask = task.replace("_", "-")
    if kind == "mattr":
        p = R / loc / f"{task}_{model}_validation.pkl"
    elif kind == "eprun":
        p = R / loc / "EdgePruning_patching_node" / f"{stask}_{model}_validation_abs-False.pkl"
    else:
        dirn, sub = loc
        p = MIB / dirn / sub / f"{stask}_{model}_validation_abs-False.pkl"
    if not p.exists():
        return None
    try:
        return pickle.load(open(p, "rb"))
    except Exception:
        return None


def build():
    """{(method, facet label): (x, acc, cpr)} plus the coverage report."""
    out = {}
    for mname, _, _, kind, loc in METHODS:
        n_found = 0
        for task, model, flabel in COLUMNS:
            d = load(kind, loc, task, model)
            if d is None:
                continue
            n_found += 1
            acc, faith = d.get("accuracies"), d.get("faithfulnesses")
            out[(mname, flabel)] = (np.asarray(PCT, float),
                                    np.asarray(acc, float) if acc else None,
                                    np.asarray(faith, float) if faith else None)
        # load() returns None for a path that does not exist, so a mistyped or not-yet-populated
        # dir makes the series vanish from the figure with no error -- which is exactly how GIM
        # was silently absent until the gim_accauc/gim_eval mixup was caught (see METHODS above).
        # Say it out loud instead: partial is expected while a sweep fills, absent is not.
        if n_found < len(COLUMNS):
            print(f"{'MISSING' if not n_found else 'partial'}: {mname} ({loc}) "
                  f"{n_found}/{len(COLUMNS)} cells")
    return out


def make(data, idx, ylab, out, hline=None, free_y=False):
    """One figure. `idx` is 1 for accuracy and 2 for CPR in build()'s value tuple."""
    style = {m[0]: (m[1], DASH[m[2]]) for m in METHODS}
    nrow = -(-len(COLUMNS) // NCOL)
    fh = ROW_H * nrow + HEAD
    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(nrow, NCOL, figsize=(FIG_W, fh), sharex=True,
                             sharey=not free_y, squeeze=False)

    drawn, npts = set(), 0
    for i, (_, _, flabel) in enumerate(COLUMNS):
        ax = axes[i // NCOL][i % NCOL]
        lo, hi = np.inf, -np.inf
        if hline is not None:
            ax.axhline(hline, color="#999999", lw=0.4, ls=(0, (2, 2)), zorder=1)
        for mname in METHOD_ORDER:
            v = data.get((mname, flabel))
            if v is None or v[idx] is None:
                continue
            x, y = v[0], v[idx]
            col, ls = style[mname]
            ax.plot(x, y, lw=LW, color=col, ls=ls, zorder=2, solid_capstyle="round")
            drawn.add(mname)
            npts += len(y)
            lo, hi = min(lo, y.min()), max(hi, y.max())
        ax.set_xscale("log")
        ax.set_xticks(XTICKS)
        ax.set_xticklabels(XLAB)
        ax.set_title(flabel, fontsize=FS_TITLE, pad=2.5)
        ax.tick_params(labelsize=FS_TICK, length=1.5, pad=1.5)
        P.furnish(ax)
        if free_y and np.isfinite(lo):
            # CPR: per PANEL, not per row -- the ceiling is set by the cell, and a shared axis
            # flattens IOI/Qwen (max ~2) against MCQA/Llama (max ~5).
            pad = 0.08 * (hi - lo or 1)
            ax.set_ylim(lo - pad, hi + pad)
    if not free_y:
        axes[0][0].set_ylim(0, 1)
    # 11 cells in a 3x4 grid leaves one empty slot. Hide it rather than letting an empty framed
    # panel read as a cell whose runs all failed.
    for j in range(len(COLUMNS), nrow * NCOL):
        axes[j // NCOL][j % NCOL].set_visible(False)
    # sharex suppresses tick labels on every panel that is not in the bottom ROW, but with 11
    # cells in a 3x4 grid the last column's bottom panel is in row 1 -- so that column ended up
    # with no x axis at all, its labels hidden by a neighbour that is not drawn. Re-enable them
    # on the lowest VISIBLE panel of each column.
    for c in range(NCOL):
        last = max((i for i in range(len(COLUMNS)) if i % NCOL == c), default=None)
        if last is not None:
            axes[last // NCOL][c].tick_params(labelbottom=True)

    fig.supxlabel("fraction of nodes kept (denoised)", fontsize=FS_LAB, y=0.012)
    fig.supylabel(ylab, fontsize=FS_LAB, x=0.005)
    # Legend band, as a LENGTH not a fraction: this figure is half plot_sva_curves.py's height,
    # so copying its rect top of 0.955 would reserve half as much ink-space for the same legend.
    # 0.20in is one row of 5.5pt keys; the four series fit on one row at ncol=5.
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.45, rect=(0.012, 0.022, 1, 1 - 0.20 / fh))
    handles = [Line2D([], [], color=style[m][0], lw=LW + 0.2, ls=style[m][1], label=m)
               for m in METHOD_ORDER if m in drawn]
    fig.legend(handles=handles, fontsize=FS_LEG, ncol=5, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=2.0,
               handletextpad=0.5, columnspacing=1.4)
    fig.savefig(OUT / out)
    fig.savefig(OUT / out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {OUT/out} ({npts} pts, {len(drawn)} methods)")


def main():
    data = build()
    make(data, 1, "decision accuracy (metric $>$ 0)", "mib_accuracy_curves.pdf")
    make(data, 2, "CPR (faithfulness)", "mib_cpr_curves.pdf", hline=1.0, free_y=True)


if __name__ == "__main__":
    main()
