"""tabs/sparsity_sweep.tex as one figure: both metrics against each baseline's sparsity knob.

The sibling of plot_lr_sweep_summary.py, over the OTHER sweep table -- rows are the two metrics,
columns are the two blocks of make_lr_table.SPARSITY_METHODS (Node Pruning's target sparsity s,
DBM's L1 coefficient). Same import-the-table discipline: the blocks come from that module, so
"the blocks of the table" and "the panels of this figure" are one list and an unstyled block
RAISES rather than quietly vanishing.

WHAT IT SHOWS, and it is not what the CPR table alone suggests: THE TWO METRICS DISAGREE ABOUT
THE KNOB, in the same direction for both methods.

    Node Pruning   CPR AUC   peaks at s=0.5 (1.67), falls to 1.24 by s=0.99
                   acc-AUC   rises MONOTONICALLY across the whole grid, 0.09 -> 0.38
    DBM            CPR AUC   peaks at lambda=6 (1.50), falls at 20
                   acc-AUC   rises MONOTONICALLY, 0.19 -> 0.35

That is the cpr-auc-is-dense-end-dominated fact made visible: MIB's `area_under` is a linear
trapezoid over 0.001--1.0 sparsity, so ~90% of it comes from k>=20% where a denser circuit simply
scores better, while acc-AUC is log-normalised and reads the sparse end. Turning either knob up
trades one for the other. So "the best sparsity setting" is metric-dependent, and
make_mib_table.EPRUN_BEST_SPARSITY (s=0.5, picked by CPR) is SIXTH OF SEVEN on acc-AUC. That
caveat is already in that constant's comment; this figure is where it is legible.

THE RING RULE CARRIES THE POINT BY ITSELF. As in the LR figure, an argmax is ringed only when it
is INTERIOR to the swept grid. Both CPR panels ring; NEITHER acc-AUC panel does, because both
acc-AUC curves are still climbing at the sparsest setting we ran. An unringed maximum here is a
statement that the grid does not bracket the optimum -- these two sweeps do not establish a best
acc-AUC operating point, they establish that it is sparser than anything swept.

NO SHARED X AXIS, which is the deliberate difference from plot_lr_sweep_summary.py. There both
columns were learning rates and sharex was the whole point (the optima sit at different LRs on
one scale). Here the columns sweep different quantities in different units -- a target sparsity
in [0,1] against an unbounded L1 coefficient -- and putting them on one axis would invite reading
"s=0.5" against "lambda=0.5" as the same setting. The y axes ARE shared per row, so the vertical
comparison across columns stays honest: Node Pruning's knob moves CPR over a ~1.3 range and DBM's
over ~0.19, and that difference in leverage is real, not a scaling choice.

X SCALES DIFFER PER COLUMN for the same reason. s is plotted LINEARLY: it is a fraction, its
grid is not geometric, and the CPR turnover at 0.5 sits mid-axis where it reads. lambda is
plotted on a SYMLOG axis with linthresh below the smallest nonzero point, because its grid IS
geometric (0.2/0.6/2/6/20, x3 apart) and because lambda=0 is a real swept point -- the
unpenalised control -- that a log axis cannot place at all.

THE BOTTOM X AXIS IS A REQUEST, NOT AN OUTCOME, for the Node Pruning column, and the SECONDARY
TOP AXIS is what the request actually bought. s is the L0 anneal's TARGET, and the anneal does
not reach it -- the top axis is mined from the training logs (see achieved_sparsity below) and
reads

    target      0.1    0.25   0.5    0.8    0.9    0.95   0.99
    achieved    0.111  0.230  0.466  0.748  0.798  0.829  0.855

THE KNOB SATURATES AT ~0.86. The three sparsest settings in the grid span an achieved range of
0.798--0.855, i.e. they are very nearly the same circuit, so the acc-AUC rise across the top of
this axis is NOT a rise across genuinely sparser circuits. That is the anneal under-converging
(sparsity_warmup_frac=0.83 of 3000 steps leaves only ~510 steps at the final target -- see
node-pruning-l0-anneal-steps), and it is why s>1 is a meaningful setting to run: s enters ONLY
through the Lagrangian lambda_1(sp - s) + lambda_2(sp - s)^2, so an unreachable target simply
means unbounded, always-correctly-signed sparsity pressure. Do not read the bottom axis as
achieved sparsity, and do not read the top one as a knob -- it is a measurement.

The DBM column has NO top axis: learn_scores_sigmoid_mask never logs a final mask size (only
learn_scores_edge_pruning does, edge_pruning.py:174), so DBM's achieved density is not
recoverable from the logs, and it is not recoverable from the graph JSONs either -- those store
unthresholded logits for MIB's own top-k sweep to re-threshold, with `in_graph` False on every
maskable node. An empty top spine there is the honest rendering; do not fill it by thresholding
those logits (they are not probabilities; DBM's run to +-150).

COLOURS MATCH plot_lr_sweep_summary.py cell-for-cell (indigo Node Pruning, pink DBM, both from
palette.py), so the two sweep figures read as one pair. There is no legend: one series per panel,
already named by the column title.

Run:  uv run python plots/plot_sparsity_sweep_summary.py
Out:  plots/sparsity_sweep_summary.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
"""
import argparse
import collections
import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import palette as P                                     # noqa: E402
import plot_mib_accauc_cpr_scatter as S                 # RC only  # noqa: E402
import make_lr_table as M                               # SPARSITY_METHODS + cpr()  # noqa: E402
# The row-label parser is IMPORTED, not restated. Both sweep tables write their knob values as
# "<number>" or "<number> (<note>)" and both figures have to strip the note the same way; two
# copies of that regex is how the two figures would come to disagree about which rows are
# plottable. This is a pure function -- importing it cannot reshape the other figure the way
# importing a layout CONSTANT can (see plot_train_curves.py's ROW_H warning).
from plot_lr_sweep_summary import lr_of as knob_of      # noqa: E402

# block name (verbatim from make_lr_table.SPARSITY_METHODS) -> (column title, x label, colour,
# x-scale, achieved-sparsity log filter). Titles are shortened from the table's block names, which
# carry the fixed hyperparameter inline ("Node Pruning (logit-diff, LR $=$ 0.8)"); that belongs in
# a caption, not on an axis. The filter is (loss, lr, steps) as those three appear in the training
# log's "Training for %d steps (loss=%s, lr=%.3g, ...)" line, and it must pin the block EXACTLY --
# there are logit_diff Node Pruning runs at lr 0.1/0.3/1.5/3.0 on disk from the LR sweep, and
# folding those in would report an achieved sparsity averaged over a different LR than the curve.
# None means "no top axis" (see the docstring: the sigmoid gate never logs a mask size).
STYLE = {
    "Node Pruning (logit-diff, LR $=$ 0.8)":
        ("Node Pruning (logit-diff, lr 0.8)", "target sparsity $s$",
         P.METHOD["Node Pruning"], "linear", ("logit_diff", "0.8", "3000")),
    "DBM $+$ L1 (lr $=$ 0.3)":
        ("DBM (logit-diff, lr 0.3)", r"L1 coefficient $\lambda$",
         P.METHOD["DBM"], "symlog", None),
}
METRICS = [("area_under", "CPR AUC (↑)"), ("acc_auc", "IIA log-AUC (↑)")]
FIG_W, ROW_H = 5.4, 1.55
FS_LABEL, FS_TICK, FS_ANNOT = 7.5, 7, 6
# set_title draws ABOVE the axes rect, and on the top row so does the achieved-sparsity twiny,
# so the reserved header strip has to hold BOTH and the title needs a pad that clears the
# secondary axis's ticks + label rather than the default 3pt.
TITLE_H, TITLE_PAD = 0.58, 26
SYMLOG_LINTHRESH = 0.15  # below the smallest nonzero lambda (0.2), so only 0 sits in the linear leg


LOG_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "eprun_*.out")
RE_LEVEL = re.compile(r"(\w+)-level Edge Pruning: \d+ log-alpha parameters, "
                      r"target sparsity ([\d.]+)")
RE_TRAIN = re.compile(r"Training for (\d+) steps \(loss=(\w+), lr=([\d.]+)")
RE_FINAL = re.compile(r"Final deterministic mask keeps \d+/\d+ units \(sparsity ([\d.]+)\)")


def achieved_sparsity(loss, lr, steps):
    """{target: mean achieved sparsity} for one Node Pruning block, mined from logs/eprun_*.out.

    THIS IS THE ONLY PLACE ACHIEVED SPARSITY EXISTS. It is not in results/ at all: the graph
    JSONs hold unthresholded log-alphas because MIB's run_evaluation.py re-thresholds them at
    every k of its own sparsity sweep, and `in_graph` is False on all but one node. The single
    line that records the trained mask's size is edge_pruning.py:174, which goes to the SLURM
    log and nowhere else.

    The logs are keyed by job id, not by results dir, so a log is matched to a block by the
    hyperparameters it prints -- level, loss, lr, steps, target -- and never by filename. A
    target is reported only when exactly len(M.COLUMNS) logs match it, so a partially re-run
    budget cannot quietly report a mean over a different cell set than its neighbours; the
    caller prints what it dropped.
    """
    per = collections.defaultdict(list)
    for path in glob.glob(LOG_GLOB):
        with open(path, errors="ignore") as fh:
            txt = fh.read()
        lev, tr, fin = RE_LEVEL.search(txt), RE_TRAIN.search(txt), RE_FINAL.search(txt)
        if not (lev and tr and fin) or lev.group(1) != "node":
            continue
        if (tr.group(2), tr.group(3), tr.group(1)) != (loss, lr, steps):
            continue
        per[float(lev.group(2))].append(float(fin.group(1)))
    return {t: (float(np.mean(v)) if len(v) == len(M.COLUMNS) else None, len(v))
            for t, v in per.items()}


def load():
    """{block: [(knob, {metric: mean over the 11 cells or None})]}, plus the exclusion log.

    Per-metric completeness over make_lr_table.COLUMNS, the same rule and the same 11 MIB
    validation cells as plot_lr_sweep_summary.load(). As of this writing NOTHING is excluded --
    all 13 rows are 11/11 on both metrics -- but the check stays: a partially-landed row would
    otherwise contribute a mean over a different cell set than the point next to it.
    """
    out, skipped = {}, []
    for name, vals, *_ in M.SPARSITY_METHODS:
        if name not in STYLE:
            raise SystemExit(f"block {name!r} is in sparsity_sweep.tex but has no STYLE entry -- "
                             "add one (see the module docstring) rather than letting it vanish")
        pts = []
        for lab, dirn in vals:
            knob = knob_of(lab)
            if knob is None:
                skipped.append(f"  {name} row {lab!r} ({dirn}): not a knob row -- dropped")
                continue
            rec, miss = {}, []
            for key, _ in METRICS:
                got = [v for v in (M.cpr(dirn, t, m, key) for t, m, _ in M.COLUMNS)
                       if v is not None]
                rec[key] = float(np.mean(got)) if len(got) == len(M.COLUMNS) else None
                if rec[key] is None:
                    miss.append(f"{key} {len(got)}/{len(M.COLUMNS)}")
            if miss:
                skipped.append(f"  {name} {lab} ({dirn}): {', '.join(miss)} -- dropped from "
                               f"{'those panels' if any(rec.values()) else 'both panels'}")
            if any(v is not None for v in rec.values()):
                pts.append((knob, rec))
        out[name] = sorted(pts)
    return out, skipped


def draw(ax, pts, colr, xscale, metric):
    """One panel: the block's curve for one metric, argmax ringed only when interior."""
    xy = [(k, r[metric]) for k, r in pts if r[metric] is not None]
    if xy:
        x, y = [p[0] for p in xy], [p[1] for p in xy]
        ax.plot(x, y, "-", lw=0.9, color=colr, zorder=2)
        ax.plot(x, y, "s", ms=2.6, color=colr, mec="#000000", mew=0.35, ls="none", zorder=3)
        bi = int(np.argmax(y))
        if 0 < bi < len(y) - 1:
            ax.plot([x[bi]], [y[bi]], "o", ms=6.5, mfc="none", mec=colr, mew=0.8, zorder=4)
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH, linscale=0.4)
        # Tick the SWEPT VALUES, not the decades. Symlog's default locator labels 1e-1/1e0/1e1,
        # and not one of those three is a setting we ran -- a reader would take the axis for a
        # continuum that was sampled at those points. These ticks are exactly the six rows of the
        # table's DBM block, so every label has a square sitting on it and gaps in the grid are
        # gaps in the sweep. Minor ticks off for the same reason.
        ax.set_xticks([k for k, _ in pts])
        ax.set_xticklabels([f"{k:g}" for k, _ in pts])
        ax.set_xticks([], minor=True)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/sparsity_sweep_summary.pdf")
    a = ap.parse_args()

    data, skipped = load()
    blocks = [n for n, *_ in M.SPARSITY_METHODS]
    plt.rcParams.update(S.RC)
    nr, nc = len(METRICS), len(blocks)
    fh = ROW_H * nr + TITLE_H
    # sharey per ROW only. sharex is deliberately OFF -- see the docstring; the two columns are
    # different quantities and squeeze=False keeps the indexing uniform either way.
    fig, axes = plt.subplots(nr, nc, figsize=(FIG_W, fh), sharey="row", squeeze=False)
    ach_log = []
    for r, (metric, ylab) in enumerate(METRICS):
        for c, name in enumerate(blocks):
            title, xlab, colr, xscale, logfilt = STYLE[name]
            ax = axes[r][c]
            draw(ax, data[name], colr, xscale, metric)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=FS_LABEL)
            if r == nr - 1:
                ax.set_xlabel(xlab, fontsize=FS_LABEL)
            if r:
                continue
            # Top row only: the achieved-sparsity axis, then the title ABOVE it. Both columns
            # get a twiny so the two titles stay on one baseline even though only one column
            # has data to put under it -- an empty top spine on DBM is the honest rendering.
            tw = ax.twiny()
            tw.set_xscale(ax.get_xscale())
            tw.set_xlim(ax.get_xlim())            # twiny() shares y, NOT x -- copy it by hand
            tw.set_xticks([])
            for sp in tw.spines.values():
                sp.set_linewidth(0.5)
            if logfilt is not None:
                ach = achieved_sparsity(*logfilt)
                ticks = [(k, ach[k][0]) for k, _ in data[name]
                         if k in ach and ach[k][0] is not None]
                ach_log += [f"  {STYLE[name][0]}: target {k:g} -> achieved "
                            + (f"{ach[k][0]:.3f} (n={ach[k][1]})" if ach[k][0] is not None
                               else f"DROPPED, only {ach[k][1]}/{len(M.COLUMNS)} logs")
                            for k, _ in data[name] if k in ach]
                ach_log += [f"  {STYLE[name][0]}: target {k:g} -> NO matching training log"
                            for k, _ in data[name] if k not in ach]
                if ticks:
                    tw.set_xticks([t for t, _ in ticks])
                    # Rotated because the top three targets land 0.798/0.829/0.855 -- horizontal
                    # labels there print as "0.80.8386". The crowding IS the finding (the knob
                    # saturates), so the labels must stay legible rather than be thinned out.
                    tw.set_xticklabels([f"{v:.2f}" for _, v in ticks], fontsize=FS_ANNOT,
                                       rotation=45, ha="left", rotation_mode="anchor")
                    tw.set_xlabel("achieved sparsity", fontsize=FS_ANNOT, labelpad=3)
            tw.tick_params(labelsize=FS_ANNOT, length=2, pad=1)
            ax.set_title(title, fontsize=FS_LABEL, pad=TITLE_PAD)

    fig.tight_layout()
    fig.subplots_adjust(top=1.0 - TITLE_H / fh)
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)

    if skipped:
        print(f"\nexcluded ({len(M.COLUMNS)} MIB validation cells required per metric):")
        print("\n".join(skipped))
    else:
        print(f"\nno exclusions: every row is {len(M.COLUMNS)}/{len(M.COLUMNS)} on both metrics")
    if ach_log:
        print("\nachieved sparsity (mined from logs/eprun_*.out, edge_pruning.py:174):")
        print("\n".join(ach_log))
    print("\nbest knob per block:")
    for name in blocks:
        for metric, _ in METRICS:
            xy = [(k, r[metric]) for k, r in data[name] if r[metric] is not None]
            if not xy:
                print(f"  {STYLE[name][0]:<30} {metric:<10} -- no complete row")
                continue
            bi = int(np.argmax([p[1] for p in xy]))
            edge = "" if 0 < bi < len(xy) - 1 else "  (grid endpoint -- optimum NOT bracketed)"
            print(f"  {STYLE[name][0]:<30} {metric:<10} n={len(xy)}  best {xy[bi][1]:.3f} "
                  f"@ {xy[bi][0]:g}{edge}")


if __name__ == "__main__":
    sys.exit(main())
