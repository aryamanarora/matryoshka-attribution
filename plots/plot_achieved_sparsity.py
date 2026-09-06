"""What Node Pruning's sparsity target actually buys: achieved sparsity against requested.

s is NOT a sparsity. It is the target of an L0 Lagrangian --

    reg = lambda_1 * (sparsity - s) + lambda_2 * (sparsity - s)^2      (edge_pruning.py:150)

with both multipliers ASCENDED -- so the trained circuit lands wherever that constraint and the
task loss balance, and it lands short. This figure is the measurement, and it is the reason the
x axes of plot_sparsity_sweep_summary.py are captioned "target sparsity" and not "sparsity".

THE KNOB COMPRESSES, AND IT IS NOT A CEILING (revised 2026-08-25, when s=1.25 landed). On the
logit-diff series (MAttr's objective, the one tabs/sparsity_sweep.tex reports) achieved sparsity
runs

    target      0.1    0.25   0.5    0.8    0.9    0.95   0.99   1.25   2.0
    achieved    0.111  0.230  0.466  0.748  0.798  0.829  0.855  0.920  0.968

Everything up to s=0.99 flattens -- 0.798--0.855 across the top three, very nearly the same
circuit -- and until 1.25 existed that read as SATURATION, a hard ceiling near 0.86 that this
docstring asserted. It is not one. Pushing the target past 1 buys +0.065 at 1.25 and +0.113 by
2.0, so what looked like the anneal's limit was the grid running out below the point where an
unreachable target starts applying real pressure. NO CEILING IS VISIBLE YET even at s=2.0; the
curve is still climbing where the grid stops. The compression is still real and still the reason
the x axis says "target": any trend across 0.9/0.95/0.99 is a trend across nearly identical
circuits, and "s=0.99" is an ~0.86-sparse circuit (145 of 1056 nodes kept, not 11).

AND THAT MAKES A CLEAN TEST AVAILABLE THAT WAS NOT BEFORE. logit-diff at s=2.0 reaches 0.968,
just past KL at s=0.99 (0.962) -- so the two objectives can now be compared AT MATCHED ACHIEVED
DENSITY instead of at matched target. The "KL's worse CPR / better acc-AUC is density, not
objective" claim below is currently an inference from a confound; that pair of cells would
measure it directly. Nobody has run that comparison yet.

THE TWO LOSSES SATURATE IN DIFFERENT PLACES, which is the interesting part. KL (Edge Pruning's
own objective) tracks the request much further, reaching ~0.96 where logit-diff reaches ~0.86.
That is the mechanism behind the metric split in the sweep-summary figure: KL's worse CPR AUC and
better acc-AUC are what a sparser circuit scores, not what a different objective scores.

WHY IT UNDERSHOOTS, and what to do about it: sparsity_warmup_frac=0.83 spends 2490 of the 3000
steps ANNEALING the target and leaves ~510 at the final value, so the multipliers never finish
converging (node-pruning-l0-anneal-steps). Two levers, and only one of them is honest to use in a
table whose rows must be step-matched: raise the steps (changes the budget, breaks the match), or
raise s past 1. s>1 is not an error -- an unreachable target just makes the constraint permanently
violated, which is unbounded, correctly-signed sparsity pressure -- and the s=1.25 point above is
that lever working: same 3000 steps, same LR, so it stays step-matched with every other row while
reaching a sparsity no in-range target got to. The y=x line is drawn for exactly this reason: a
point on it got what it asked for, and nothing above s~0.5 does.

SOURCE. Achieved sparsity is NOT in results/ at all. The graph JSONs hold unthresholded
log-alphas because MIB's run_evaluation.py re-thresholds them at every k of its own sparsity
sweep, and `in_graph` is False on all but one node -- thresholding them here would invent a
number. The one line that records the trained mask's size is edge_pruning.py:174, which goes to
the SLURM log and nowhere else, so this figure mines logs/eprun_*.out. See achieved_sparsity()
for how a log is matched to a series and why the dedupe matters.

DBM IS ABSENT and cannot be added: learn_scores_sigmoid_mask never logs a mask size at all.

Run:  uv run python plots/plot_achieved_sparsity.py
Out:  plots/achieved_sparsity.pdf  (plots/*.pdf is gitignored -- regenerate, don't commit)
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                     # noqa: E402
import plot_mib_accauc_cpr_scatter as S                 # RC only  # noqa: E402
import make_lr_table as M                               # COLUMNS  # noqa: E402

# The KL series' results dirs, imported by plot_sparsity_sweep_summary.py so the two figures
# cannot disagree about which dirs are the KL sweep. Listed explicitly rather than derived from
# the logit-diff names because the layout is irregular: the s=0.9 KL runs sit in the UNSUFFIXED
# results/eprun_eval, since run_edge_pruning.sbatch only appends _s<S> when a sparsity is passed
# and 0.9 is its node default. Deriving "eprun_eval_s0.9" reads an empty dir and drops the point.
# (Same trap is documented in submit_node_pruning_sparsity.sh.)
#
# THE KL SERIES IS PERMANENTLY 3 POINTS, BY DECISION (2026-08-25). s=0.5 sits at 6/11 cells and
# s=0.8 at 9/11, so the 11-cell completeness gate drops both from every panel here and in
# plot_sparsity_sweep_summary.py. The seven jobs that would have filled them were CANCELLED --
# this is a closed gap, not a pending wave, so do NOT resubmit submit_node_pruning_sparsity.sh's
# KL arm on seeing "6/11, dropped" in the exclusion log. The three complete points (0.9/0.95/0.99
# -> 0.857/0.925/0.962) already carry the only claim the series is used for: KL tracks the target
# much further than logit-diff, so its worse-CPR/better-acc-AUC split is DENSITY, not objective.
KL_ROWS = [("0.5", "eprun_eval_s0.5"), ("0.8", "eprun_eval_s0.8"), ("0.9", "eprun_eval"),
           ("0.95", "eprun_eval_s0.95"), ("0.99", "eprun_eval_s0.99")]

DASH = (0, (3.2, 1.4))
# (label, dash, log filter). The filter is (loss, lr, steps) as those three appear in the training
# log's "Training for %d steps (loss=%s, lr=%.3g, ...)" line, and it must pin the series EXACTLY:
# there are logit_diff Node Pruning runs at lr 0.1/0.3/1.5/3.0 on disk from the LR sweep, and
# folding those in would average achieved sparsity over learning rates.
SERIES = [
    ("logit-diff loss (MAttr's)", "solid", ("logit_diff", "0.8", "3000")),
    ("KL loss (Edge Pruning's)", DASH, ("kl", "0.8", "3000")),
]
COLOUR = P.METHOD["Node Pruning"]
FIG_W, FIG_H = 2.75, 2.30
FS_LABEL, FS_TICK, FS_ANNOT = 7.5, 7, 6

LOG_GLOB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "eprun_*.out")
RE_LEVEL = re.compile(r"(\w+)-level Edge Pruning: \d+ log-alpha parameters, "
                      r"target sparsity ([\d.]+)")
RE_TRAIN = re.compile(r"Training for (\d+) steps \(loss=(\w+), lr=([\d.]+)")
RE_FINAL = re.compile(r"Final deterministic mask keeps \d+/(\d+) units \(sparsity ([\d.]+)\)")
RE_MODEL = re.compile(r"Loading ([\w\-./]+)\.\.\.")
RE_EXAMP = re.compile(r"Loaded (\d+) examples")


def achieved_sparsity(loss, lr, steps):
    """{target: (mean achieved sparsity or None, n distinct cells)} for one series.

    Logs are keyed by job id, not by results dir, so a log is matched to a series by the
    hyperparameters it prints -- level, loss, lr, steps, target -- and never by filename.

    THE DEDUPE IS LOAD-BEARING. Re-runs leave several logs for the same cell (KL s=0.9 has 14
    files for 11 cells) and averaging them would weight the re-run cells double. There is no task
    name anywhere in the log body -- it is only in the SLURM job name, which these files do not
    carry -- so a cell is identified by (model, unit count, example count), a triple verified
    unique across all 11 cells of every complete target. Latest file wins.

    A target is reported only at exactly len(M.COLUMNS) distinct cells, so a partially-trained
    budget cannot quietly contribute a mean over a different cell set than the point beside it.
    """
    per = collections.defaultdict(dict)
    for path in sorted(glob.glob(LOG_GLOB), key=os.path.getmtime):
        with open(path, errors="ignore") as fh:
            txt = fh.read()
        lev, tr, fin = RE_LEVEL.search(txt), RE_TRAIN.search(txt), RE_FINAL.search(txt)
        mod, exa = RE_MODEL.search(txt), RE_EXAMP.search(txt)
        if not (lev and tr and fin and mod and exa) or lev.group(1) != "node":
            continue
        if (tr.group(2), tr.group(3), tr.group(1)) != (loss, lr, steps):
            continue
        per[float(lev.group(2))][(mod.group(1), fin.group(1), exa.group(1))] = float(fin.group(2))
    return {t: (float(np.mean(list(v.values()))) if len(v) == len(M.COLUMNS) else None, len(v))
            for t, v in sorted(per.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="plots/achieved_sparsity.pdf")
    a = ap.parse_args()

    plt.rcParams.update(S.RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    log, xs = [], []
    for label, dash, filt in SERIES:
        ach = achieved_sparsity(*filt)
        for t, (v, n) in ach.items():
            log.append(f"  {label:<28} s={t:<5g} -> "
                       + (f"{v:.3f}  (n={n})" if v is not None
                          else f"DROPPED, {n}/{len(M.COLUMNS)} cells"))
        xy = [(t, v) for t, (v, _) in ach.items() if v is not None]
        if not xy:
            continue
        x, y = [p[0] for p in xy], [p[1] for p in xy]
        xs += x
        ax.plot(x, y, ls=dash, lw=0.9, color=COLOUR, zorder=2, label=label)
        ax.plot(x, y, "s", ms=2.6, color=COLOUR, mec="#000000", mew=0.35, ls="none", zorder=3)

    # y=x: a point on this line got the sparsity it asked for. Drawn across the swept range only,
    # so it never implies a target we did not run. LABELLED IN THE LEGEND rather than annotated
    # at its end -- the line ends in the top-right corner, exactly where an inline label has
    # nowhere to sit that the line does not run through.
    lo, hi = (min(xs), max(xs)) if xs else (0.0, 1.0)
    ax.plot([lo, hi], [lo, hi], "-", lw=0.6, color="#999999", zorder=1,
            label="achieved $=$ requested")

    ax.set_xlabel("target sparsity $s$ (requested)", fontsize=FS_LABEL)
    ax.set_ylabel("achieved sparsity", fontsize=FS_LABEL)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    ax.tick_params(labelsize=FS_TICK)
    ax.legend(fontsize=FS_ANNOT, loc="upper left", frameon=True, framealpha=0.9,
              borderpad=0.3, handlelength=2.4, handletextpad=0.5,
              labelspacing=0.25).get_frame().set_linewidth(0.4)

    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print("wrote", a.out)
    print(f"\nachieved sparsity (logs/eprun_*.out, edge_pruning.py:174; "
          f"{len(M.COLUMNS)} cells required):")
    print("\n".join(log))


if __name__ == "__main__":
    sys.exit(main())
