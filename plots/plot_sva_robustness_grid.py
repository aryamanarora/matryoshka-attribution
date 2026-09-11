"""SVA+ robustness grid: every method x every evaluation setting, in one panel.

THE ARGUMENT THIS FIGURE MAKES, and the reason it is one figure rather than three. SVA+ varies
variable granularity (node / MLP neuron / neuron+head) and training loss (logit-diff / CE /
soft-acc), and the claim worth making is that those axes move the ANSWER as much as the choice
of method does -- so a method is only as good as its WORST cell, and most baselines have a bad
one. A scatter averaged over tasks (the parent figure, plots/plot_accauc_vs_faithauc.py) shows
where methods land; it cannot show that the ORDER changes from setting to setting, because each
panel is a separate pair of axes and the reader has to hold eleven positions in their head to
compare them. A grid can: colour carries the score, the printed digit carries the rank, and a
method's robustness is literally whether its row is one colour.

WHAT TO LOOK FOR, all of which is in the numbers and none of which needs the caption:
  * MAttr+SGD is the only row that never leaves the top band -- worst cell 0.517, worst rank
    3 of 8. IG is the best baseline and still drops to 0.454 / rank 4.
  * I x G and AttnLRP have a vertical stripe at the `ce` columns: they collapse under the
    cross-entropy objective (I x G to 0.071 at node, against a 0.027 random floor).
  * Node Pruning and DBM invert with GRANULARITY: competitive at node, near-floor on
    neurons. Read that with the budget caveat -- both run at
    s=0.9 OF THE SUBSTRATE, so their absolute circuit is ~1k units at node and ~2M at neuron
    scale, and the collapse is partly a budget mismatch rather than purely a method failure.
  * MAttr under Adam dips on the neuron substrates specifically (0.501 on MLP vs SGD's 0.596).
    That is the eps degeneracy (see scripts/sva/launch/submit_sva_eps.sh): at 2.29M mask logits the
    default eps=1e-8 sits below the typical |grad|, Adam's update becomes ~sign(g), and the
    score stops carrying magnitude.

RANK IS THE ANNOTATION, SCORE IS THE COLOUR, and that pairing is the point -- a cell can be
rank 1 and still be a bad circuit, which a rank-only grid would hide and a score-only grid
would make the reader compute. Ranks are over the 8 METHODS only; Random is a reference row,
printed with its score and no rank, because a control that scores 0.02 is not competing for
9th place. The floor row is not decoration: it is what makes "0.45 is a bad cell" legible, and
it is the thing whose absence would let the zero-ablation half (see SOURCES_PATCHED) be
misread.

WHY EIGHT METHODS. These are exactly the ones with complete coverage. Stepless IG (patched
-input SVA only) and MAttr+Adam at eps=1e-2 (logit-diff only) are deliberately absent rather
than drawn with holes -- a row with gaps, in a grid whose whole message is the uniformity of a
row, is unreadable in the exact way the figure is trying to exploit. Both belong in the text:
Stepless IG matches IG where it ran (0.675 vs 0.682 on the matched SVA subset), and the eps arm
is the fix for the MAttr+Adam dip, at 0.602 vs 0.501 on MLP.

Run:  uv run python plots/plot_sva_robustness_grid.py               -> patched (the paper figure)
      uv run python plots/plot_sva_robustness_grid.py --with-zero   -> + the zero-ablation half
Out:  plots/sva_robustness_grid[_withzero].pdf  (+ .png; plots/*.pdf is gitignored)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402
import plot_accauc_vs_faithauc as V                    # loaders + the task->model pin  # noqa: E402

# Row order: gradient baselines, then mask baselines, then ours. Within a block, worst-case
# ascending, so the grid builds downward toward the uniform row. NOT sorted by mean -- the
# figure is about the worst cell, and ordering by mean would put a method with one collapse
# above one with none.
ROWS = ["IxG", "AttnLRP", "IG", "eprun-s090", "sig_lr0.3_l16.0",
        "soft-log", "stopk-log", "softsgd-log"]
FLOOR = "Random"                                        # reference row, no rank
LOSSES = [("logit_diff", "LD"), ("ce", "CE"), ("acc", "Acc")]
# PATCHED ONLY BY DEFAULT, and this is the load-bearing choice in the file.
#
# The zero-ablation half has almost no dynamic range on IIA AUC. Its Random floor is 0.513, not
# the patched half's 0.027: zeroing every non-top-k unit destroys the model to near-constant
# logits, so the binary base-vs-source comparison the metric integrates becomes a coin flip and
# a random ranking already "wins" half the examples. Measured over its 78 cells per method, the
# BEST method clears that floor by +0.09 on average and I x G by +0.00-0.03, with 22-34 cells
# per method sitting AT OR BELOW their own floor. Drawn on one colour scale beside the patched
# half, those cells render as mid-bright and read as respectable scores.
#
# So the two halves cannot share a colour scale, a rank column or a worst-case column, and the
# honest single-figure version is the half that measures something. `--with-zero` draws both
# for the appendix; it prints the floor row per half, which is the only way the zero columns
# are readable at all. Do NOT quote a zero-ablation IIA AUC without its floor beside it.
SOURCES_PATCHED = [("results/sva_sweep", "Patched")]
SOURCES_BOTH = SOURCES_PATCHED + [("results/sva_zeroabl", "Zero-abl.")]

FIG_W = 5.5
FS_HEAD, FS_ROW, FS_CELL, FS_NOTE = 6.5, 6.5, 5, 5.5
CMAP = "cividis"          # perceptually uniform AND CVD-safe; the method palette is
                          # categorical and has no sequential ramp to borrow.


def grid(sources):
    """(rows x settings) score matrix, the column labels, and the Random floor per column."""
    cols, mat, floor = [], [], []
    per_source = {res: V.load(res) for res, _ in sources}
    for res, rlab in sources:
        raw = per_source[res]
        for sub, slab in V.SUBSTRATES:
            tasks = V.SVA + V.ARITH + (["arc_easy", "ioi"] if sub == "node" else [])
            for loss, llab in LOSSES:
                cols.append((rlab, slab, llab))
                col = []
                for m in ROWS:
                    vs = [raw[(m, loss, sub, t)][0] for t in tasks if (m, loss, sub, t) in raw]
                    if len(vs) != len(tasks):
                        raise SystemExit(
                            f"{V.METHODS[m][0]} covers {len(vs)}/{len(tasks)} of "
                            f"{rlab}/{slab}/{llab} -- this grid takes only 18/18 methods; "
                            "drop it from ROWS or wait for the sweep.")
                    col.append(float(np.mean(vs)))
                mat.append(col)
                # Random is loss-free: eval_sva stores it under the default logit_diff, so it
                # is read once per (ablation, substrate) and repeated across that block's three
                # loss columns rather than being absent from two of them.
                fv = [raw[(FLOOR, "logit_diff", sub, t)][0] for t in tasks
                      if (FLOOR, "logit_diff", sub, t) in raw]
                floor.append(float(np.mean(fv)) if len(fv) == len(tasks) else np.nan)
    return np.array(mat).T, cols, np.array(floor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--with-zero", action="store_true",
                    help="append the zero-ablation half; its IIA AUC sits on a 0.51 random "
                         "floor, so read it only against the floor row (see SOURCES_PATCHED)")
    a = ap.parse_args()
    sources = SOURCES_BOTH if a.with_zero else SOURCES_PATCHED
    out = a.out or ("plots/sva_robustness_grid%s.pdf" % ("_withzero" if a.with_zero else ""))

    M, cols, floor = grid(sources)                     # M[row, col]
    worst = M.min(axis=1)
    # Competition rank down each column, 1 = best. Ties share the smaller rank for the reason
    # given in plot_sva_percell_rank.ranks(); at this level of aggregation they are rare.
    rank = np.zeros_like(M, dtype=int)
    for j in range(M.shape[1]):
        order = np.argsort(-M[:, j])
        r, prev = {}, None
        for i, ri in enumerate(order):
            r[ri] = r[prev] if prev is not None and M[ri, j] == M[prev, j] else i + 1
            prev = ri
        for ri, v in r.items():
            rank[ri, j] = v

    plt.rcParams.update(P.RC)
    nrow, ncol = M.shape[0] + 1, M.shape[1] + 1        # +1 row for Random, +1 col for worst
    fig_h = 0.285 * nrow + 0.72
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    # Worst-case column is appended to the SAME image so it shares the colour scale -- it is a
    # score, in the same units, and giving it its own scale would let a bad worst-case render
    # as dark as a good one.
    full = np.hstack([M, worst[:, None]])
    full = np.vstack([full, np.append(floor, np.nanmin(floor))])
    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(full)))
    ax.imshow(full, cmap=CMAP, norm=norm, aspect="auto")

    cm = plt.get_cmap(CMAP)
    for i in range(nrow):
        for j in range(ncol):
            v = full[i, j]
            if np.isnan(v):
                continue
            # Text contrast against the cell, not a fixed colour: cividis runs dark navy ->
            # pale yellow, so a single ink colour is unreadable at one end or the other.
            lum = np.dot(cm(norm(v))[:3], (0.299, 0.587, 0.114))
            c = "#000000" if lum > 0.55 else "#ffffff"
            if i == nrow - 1 or j == ncol - 1:
                txt = f"{v:.2f}"                    # floor row and worst column: the score
            else:
                txt = str(rank[i, j])               # body: the rank
            ax.text(j, i, txt, ha="center", va="center", fontsize=FS_CELL, color=c)

    ax.set_yticks(range(nrow))
    ax.set_yticklabels([V.METHODS[m][0] for m in ROWS] + [V.METHODS[FLOOR][0]], fontsize=FS_ROW)
    # Bottom axis: the loss, repeated 6 times. The two coarser factors go above as spans, so
    # the header is three lines only where it has to be.
    ax.set_xticks(range(ncol))
    ax.set_xticklabels([c[2] for c in cols] + ["worst"], fontsize=FS_CELL)
    ax.tick_params(length=0, pad=2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # White gridlines between cells -- the standard heatmap separator; the shared
    # palette.furnish grid is for data axes and would draw on top of the image here.
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", lw=0.6)
    ax.tick_params(which="minor", length=0)

    # Two header bands: substrate (every 3 columns) and ablation (every 9).
    def band(step, idx, y, size, line):
        for s in range(0, M.shape[1], step):
            mid = s + step / 2 - 0.5
            ax.text(mid, y, cols[s][idx], ha="center", va="bottom", fontsize=size)
            if line:
                ax.plot([s - 0.45, s + step - 0.55], [y - 0.06] * 2, color="#000000",
                        lw=0.5, clip_on=False)
    band(3, 1, -0.85, FS_HEAD, True)
    if len(sources) > 1:
        band(9, 0, -1.75, FS_HEAD, True)
    # Same word top and bottom for the summary column: it was headed "min" above and
    # "worst" below, which reads as two different columns.
    ax.text(M.shape[1], -0.85, "worst", ha="center", va="bottom", fontsize=FS_HEAD)
    ax.set_ylim(nrow - 0.5, -2.0 if len(sources) > 1 else -1.35)

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=ax,
                      orientation="horizontal", fraction=0.05, pad=0.085, aspect=48)
    # Kept under ~95 characters: at FS_NOTE over FIG_W the label is not wrapped or
    # shrunk to fit, it is simply clipped at both ends, which is silent.
    cb.set_label("Compactness (↑).  Cell text = rank of 8 (1 = best);  Random row and "
                 "'worst' column show scores.", fontsize=FS_NOTE)
    cb.ax.tick_params(labelsize=FS_CELL, length=2)
    cb.outline.set_linewidth(0.5)

    fig.tight_layout(pad=0.3)
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print("wrote", out)

    w = np.argsort(-worst)
    print(f"\nworst-case IIA AUC over the {M.shape[1]} settings (the robustness number):")
    for i in w:
        print(f"  {V.METHODS[ROWS[i]][0]:<22} {worst[i]:.3f}   "
              f"(mean {M[i].mean():.3f}, mean rank {rank[i].mean():.2f})")
    print(f"  {'Random (floor)':<22} {np.nanmin(floor):.3f}")
    n_uniform = [(V.METHODS[ROWS[i]][0], int(rank[i].max())) for i in range(M.shape[0])]
    print("\nworst rank any setting gives each method (a robust method never leaves the top):")
    for name, r in sorted(n_uniform, key=lambda x: x[1]):
        print(f"  {name:<22} {r}")


if __name__ == "__main__":
    sys.exit(main())
