"""DBM's sparsity sweep on one MIB node-level task: does stitching the runs beat the best one?

DBM (the sigmoid-gate mask baseline, results/eprun_*_ld_sig_lr0.3_l1<L1>) is trained at five L1
penalties, and the penalty is the only knob that sets how sparse the learned mask ends up. Each
run is scored by MIB the same way every other method is: its SCORE VECTOR is swept over ten fixed
proportions of the node set and CPR is the area under that curve. So a run trained to be sparse is
still asked for a ranking at 50% density, and vice versa. The question this figure answers is
whether the five runs are complementary -- whether a ranking assembled from whichever run is best
at each proportion would beat any single run's CPR.

FOUR PANELS, two metrics x (scalar vs density | the curve the scalar is an area under). The
envelope is the pointwise maximum over runs -- an UPPER BOUND on stitching, not an achievable
ranking: taking the best run at each proportion gives sets that need not be nested, so no single
ordering is guaranteed to realise it. If the envelope barely clears the best single curve,
stitching cannot help.

*** THE TWO METRICS DISAGREE COMPLETELY ABOUT WHETHER STITCHING HELPS, AND THAT IS THE RESULT. ***
On CPR the envelope gains up to +21% (mcqa/llama3: 2.15 -> 2.60). On IIA it gains NOTHING -- 0.000
on four of the five cells checked and +0.001 on the fifth. The accuracy curves say why: each run is
at chance until its mask switches on and then jumps to 1.0 and stays, so the curves are nested step
functions and the sparsest run is pointwise >= every other run everywhere. There is no proportion
at which a denser run is better, so the envelope IS the sparsest run's curve.
The CPR gain therefore comes entirely from the one thing accuracy cannot see: logit-difference
MAGNITUDE. Faithfulness runs to 2.5 where accuracy saturates at 1.0, and which run over-recovers
the gap most changes with the proportion. Stitching on CPR is picking the best gap-padder at each
point, not assembling a better circuit -- the same property that motivates reporting IIA alongside
CPR in the first place.

FINAL OBSERVED SPARSITY comes from `k_log[-1]` in the training .pt -- the expected number of open
gates at the last step, which is the soft mask's own density, divided by the number of scored
nodes. It is NOT the `target_sparsity` arg (0.9 for every one of these runs; the sigmoid variant
ignores it and uses the L1 instead), and it is not read off the eval grid.

Colour is a LIGHTNESS RAMP OF DBM'S OWN HUE, built from palette.METHOD["DBM"] rather than five new
hexes: this is one method at five settings of one knob, and plots/palette.py reserves distinct
colour for distinct methods (see its note on the IG step ladder).

Run:  uv run python plots/plot_dbm_sparsity_stitch.py [--task ioi --model gpt2]
Out:  plots/dbm_sparsity_stitch.pdf (+ .png sibling for eyeballing)
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
import palette as P                                                   # noqa: E402

# MIB's ten evaluation proportions, and the trapezoid over them that defines CPR. Hardcoded
# rather than read from a pkl because the pkls store only the y values; verified against
# `area_under` for every run this figure draws (assert_cpr below), so a change to MIB's grid
# fails loudly here instead of silently rescaling the envelope.
PROPS = np.array([.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.])
# LADDER HISTORY, because the coefficients here are not a plan, they are what got run.
# The original sweep was 0.2--20 (submit_dbm_l1.sh's default) and CPR was still rising at its
# sparsest rung on 3 of 5 cells, i.e. the top end was open. It was extended on 2026-09-08 with
# 60/200/600/2000, and the FIRST cell of the 60 rung to land (ioi/gpt2) turned the curve over
# hard: density 0.168 -> 0.026 for a 3x penalty increase, CPR 1.78 -> 1.15. Past that cliff the
# L1 term stops acting as a sparsity knob and flattens the scores instead (score std 15.0 ->
# 4.6), so the RANKING degrades, not just the operating point. 600 and 2000 were cancelled on
# that evidence -- two and three rungs beyond a confirmed cliff -- and 40 was added to bracket
# the peak. 200 was launched, held, then cancelled once the fuller picture landed: averaged over
# the 11 cells the ladder PEAKS AT 6 (mean CPR 1.50 against 1.36 at 20 and 1.39 at 2), i.e. at
# the coefficient the paper already ships, and the "sparser is better" trend that motivated the
# extension turned out to be an IOI-cell property, not a general one. Two cells (arc_challenge
# and mcqa on llama3) are best at the DENSEST rung and fall monotonically from there. So the
# ladder ends at 60; anything sparser was answering a question the means had already closed.
# Coefficients whose runs are not on disk yet are SKIPPED with a printed line rather than
# raising, so this figure works while the wave lands; the printout says how many were drawn.
L1S = ["0.2", "0.6", "2.0", "6.0", "20.0", "40.0", "60.0"]
TRAIN = "results/eprun_node_ld_sig_lr0.3_l1{l1}/{task}_{model}_scores.pt"
EVAL = ("results/eprun_eval_ld_sig_lr0.3_l1{l1}/EdgePruning_patching_node/"
        "{task}_{model}_{split}_abs-False.pkl")
FIG_W, FIG_H = 5.5, 3.5
FS_AXIS, FS_TICK, FS_ANNOT = 7, 6, 6


LOGP = np.log(PROPS)


def cpr(faith):
    """MIB's CPR: LINEAR trapezoid of faithfulness over the kept proportion."""
    f = np.asarray(faith, float)
    return float(np.sum(0.5 * (PROPS[1:] - PROPS[:-1]) * (f[1:] + f[:-1])))


def iia(acc):
    """MIB's acc_auc: LOG-weighted trapezoid of accuracy, normalised by the log span.
    Mirrors MIB_circuit_track/evaluation.py:82-84."""
    a = np.asarray(acc, float)
    return float(np.sum((LOGP[1:] - LOGP[:-1]) * (a[1:] + a[:-1]) / 2) / (LOGP[-1] - LOGP[0]))


def load(task, model, split):
    """[(L1, density, CPR, faith curve, IIA, acc curve)] per run, ascending L1."""
    out = []
    for l1 in L1S:
        ev = Path(EVAL.format(l1=l1, task=task.replace("_", "-"), model=model, split=split))
        tr = Path(TRAIN.format(l1=l1, task=task, model=model))
        if not (ev.exists() and tr.exists()):
            print(f"  skip lambda={l1}: not on disk yet", file=sys.stderr)
            continue
        d = pickle.load(open(ev, "rb"))
        s = torch.load(tr, map_location="cpu")
        n = len(s["scores"])
        # k_log is the expected open-gate count per step; the last entry is where it converged.
        sparsity = float(s["k_log"][-1]) / n
        f = np.asarray(d["faithfulnesses"], float)
        acc = np.asarray(d["accuracies"], float)
        # The stored area_under is the number the paper's tables quote. Recomputing it from the
        # curve is what licenses computing the envelope's CPR the same way; if the two disagree,
        # the envelope number would be on a different scale than every CPR beside it.
        assert abs(cpr(f) - d["area_under"]) < 1e-6, (
            f"L1={l1}: recomputed CPR {cpr(f):.4f} != stored {d['area_under']:.4f} -- MIB's "
            "grid or trapezoid rule changed, update PROPS")
        assert abs(iia(acc) - d["acc_auc"]) < 1e-6, (
            f"L1={l1}: recomputed IIA {iia(acc):.4f} != stored {d['acc_auc']:.4f}")
        out.append((l1, sparsity, float(d["area_under"]), f, float(d["acc_auc"]), acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ioi")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--out", default="plots/dbm_sparsity_stitch.pdf")
    a = ap.parse_args()

    rows = load(a.task, a.model, a.split)
    env_f = np.max(np.stack([r[3] for r in rows]), axis=0)
    env_a = np.max(np.stack([r[5] for r in rows]), axis=0)
    # (scalar column index, curve column index, envelope curve, area fn, y label, curve y label)
    METRICS = [(2, 3, env_f, cpr, "CPR (↑)", "Faithfulness"),
               (4, 5, env_a, iia, "Compactness (↑)", "Accuracy")]

    # Lightness ramp of DBM's hue: lightest = weakest penalty (densest mask), full hue =
    # strongest. Interpolated toward white in RGB, which is enough for an ordered single-hue
    # ramp and keeps the darkest step exactly the palette's DBM colour.
    base = np.array([int(P.METHOD["DBM"][i:i + 2], 16) for i in (1, 3, 5)], float)
    def shade(i):
        t = 0.62 * (1 - i / max(len(rows) - 1, 1))      # 0 at the darkest, 0.62 at the lightest
        return tuple((base + t * (255 - base)) / 255)

    plt.rcParams.update(P.RC)
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, FIG_H))
    xs = [r[1] for r in rows]
    summary = []

    for r_i, (si, ci, env, area, ylab, curve_ylab) in enumerate(METRICS):
        axL, axR = axes[r_i]
        ys = [r[si] for r in rows]
        env_val = area(env)
        best = max(ys)
        summary.append((ylab, best, env_val))

        # left: converged density vs the scalar
        axL.plot(xs, ys, lw=0.7, color=P.METHOD["DBM"], alpha=0.55, zorder=1)
        for i, r in enumerate(rows):
            axL.scatter([r[1]], [r[si]], s=24, color=shade(i), edgecolor="#000000",
                        linewidth=0.4, zorder=3)
            axL.annotate(f"{r[0]}", (r[1], r[si]), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=FS_ANNOT - 0.5, color="#000000")
        axL.axhline(env_val, ls=(0, (4, 2)), lw=0.7, color="#000000", zorder=2)
        axL.annotate(f"envelope {env_val:.2f}", (0.97, env_val),
                     xycoords=("axes fraction", "data"), xytext=(0, 3),
                     textcoords="offset points", ha="right", va="bottom", fontsize=FS_ANNOT)
        axL.set_xlabel("Converged mask density", fontsize=FS_AXIS)
        axL.set_ylabel(ylab, fontsize=FS_AXIS)
        span = max(max(ys), env_val) - min(ys)
        axL.set_ylim(min(ys) - 0.10 * max(span, 0.1), max(max(ys), env_val) + 0.30 * max(span, 0.1))
        pad = 0.08 * (max(xs) - min(xs))
        axL.set_xlim(min(xs) - pad, max(xs) + pad)

        # right: the curve the scalar is an area under, plus the pointwise max
        for i, r in enumerate(rows):
            axR.plot(PROPS, r[ci], lw=0.8, color=shade(i), marker="o", ms=1.6,
                     markeredgewidth=0, label=f"$\\lambda$={r[0]}", zorder=2)
        axR.plot(PROPS, env, lw=0.9, color="#000000", ls=(0, (4, 2)), zorder=3,
                 label="Envelope")
        # Each run's own operating point on the same x units: this is what shows the runs are
        # NOT spread across the grid -- they bunch in the dense half of it.
        # BLENDED transform, not a y from get_ylim(): the limits are still autoscaling at this
        # point, so a data-space y put the markers adrift in the middle of the panel. x stays in
        # data units (the same proportion axis), y is pinned to the bottom of the axes.
        tick_tf = blended_transform_factory(axR.transData, axR.transAxes)
        for i, r in enumerate(rows):
            axR.plot([r[1]], [0], marker="^", ms=2.4, color=shade(i), transform=tick_tf,
                     clip_on=False, markeredgecolor="#000000", markeredgewidth=0.3, zorder=4)
        axR.set_xscale("log")
        axR.set_xlabel("Kept proportion of nodes", fontsize=FS_AXIS)
        axR.set_ylabel(curve_ylab, fontsize=FS_AXIS)
        axR.set_xticks([0.001, 0.01, 0.1, 1])
        axR.set_xticklabels(["10⁻³", "10⁻²", "10⁻¹", "10⁰"])
        if r_i == 0:
            axR.legend(fontsize=FS_ANNOT - 1, frameon=False, loc="upper left", ncol=2,
                       handlelength=1.0, handletextpad=0.4, labelspacing=0.15,
                       columnspacing=0.8, borderaxespad=0.3)

    for ax in axes.ravel():
        P.furnish(ax)
        ax.tick_params(labelsize=FS_TICK)
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=0.9)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {a.out}  ({a.task}/{a.model}, {a.split}, {len(rows)}/{len(L1S)} coefficients)")
    print(f"{'lambda':>7} {'density':>9} {'CPR':>7} {'IIA':>7}")
    for r in rows:
        print(f"{r[0]:>7} {r[1]:9.3f} {r[2]:7.3f} {r[4]:7.3f}")
    for ylab, best, env_val in summary:
        gain = env_val - best
        pct = 100 * gain / best if best else float("nan")
        print(f"{ylab:>16}: best single {best:.3f}  envelope {env_val:.3f}  "
              f"gain {gain:+.3f} ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
