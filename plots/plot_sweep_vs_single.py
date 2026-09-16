"""Multi-sparsity ladders against the best single reported config, per MIB cell: NP and DBM.

    uv run python plots/plot_sweep_vs_single.py [--split test]
    -> plots/sweep_vs_single.pdf (+ .png); copied to figs/ next to the sparsity sweep figure.

FOUR BARS PER CELL. For each of the two mask learners, the single configuration the test table
reports (make_mib_test_table.NODE_PRUNING -- Node Pruning at make_mib_table.EPRUN_BEST_SPARSITY,
s=0.5 logit-diff -- and MASK_NODE_BASELINES' DBM at lr 0.3, lambda 6) beside the same method's
sparsity LADDER read as a frontier (scripts/mib/eval_dbm_multisparsity.py: every rung evaluated
at the size of the mask it emits, scored on MIB's grid by dbm_multisparsity.cell, the reader the
test table and the bar charts use). Same pkl keys as everywhere: CPR = `area_under`, compactness
= `acc_auc`; the ladder's two scalars are the reader's recomputation on MIB's ten proportions.

The ladder bars are hatched in the method's colour: same method, more training runs (9 for Node
Pruning, 8 for DBM, against one -- make_mib_table.NP_MULTI_COST / DBM_MULTI_COST), which is the
caveat the caption has to carry. A 13th "Avg" group after a rule averages over the cells all
four bars have; a cell whose ladder has not landed is drawn without that bar and left out of
the average, and named on stdout.

TEST split by default: the ladders were evaluated on test (validation was declined); the single
configs have both splits. Full width, two stacked panels (CPR AUC unbounded, ~1 at chance;
acc-AUC in [0, 1]) -- not to be compared by bar height across panels.
"""
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                    # noqa: E402
import make_mib_table as M                             # COLUMNS, EPRUN_NAME  # noqa: E402
import make_mib_test_table as T                        # NODE_PRUNING, MASK_NODE_BASELINES  # noqa: E402
import dbm_multisparsity as _DBMMS                     # cell(), RESULTS, NP_RESULTS  # noqa: E402
from plot_objective_ablation import TASK_SHORT, MODEL_SHORT   # noqa: E402

# (label, colour, hatch, loader). Loaders return (cpr, acc) or None.
DBM_SINGLE = next(d for name, d, _ in T.MASK_NODE_BASELINES if name == "DBM")
NP_SINGLE = T.NODE_PRUNING[1]
COLOUR = {"np": P.METHOD["Node Pruning"], "dbm": P.METHOD["DBM"]}
METRICS = [(0, "CPR AUC (↑)"), (1, "Compactness (↑)")]
FIG_W, PANEL_H, FOOT, HEAD = 5.5, 1.05, 0.42, 0.22
FS_AXIS, FS_TICK, FS_ANNOT, FS_LEG = 6.5, 5.5, 4.0, 6.0
BAR_W = 0.2


def single(dirn, task, model, split):
    p = (T.RESULTS_BASE / dirn / "EdgePruning_patching_node"
         / f"{task.replace('_', '-')}_{model}_{split}_abs-False.pkl")
    if not p.exists():
        return None
    with open(p, "rb") as f:
        r = pickle.load(f)
    return r.get("area_under"), r.get("acc_auc")


def ladder(base, task, model, split):
    got = _DBMMS.cell(task, model, split, base=base)
    return None if got is None else (got[0], got[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--out", default="plots/sweep_vs_single.pdf")
    a = ap.parse_args()
    series = [
        ("np", "Node Pruning", False, lambda t, m: single(NP_SINGLE, t, m, a.split)),
        ("np", "Node Pruning (sweep)", True, lambda t, m: ladder(_DBMMS.NP_RESULTS, t, m, a.split)),
        ("dbm", "DBM", False, lambda t, m: single(DBM_SINGLE, t, m, a.split)),
        ("dbm", "DBM (sweep)", True, lambda t, m: ladder(_DBMMS.RESULTS, t, m, a.split)),
    ]
    cols = [(t, m) for t, m, _ in M.COLUMNS]
    data = {lab: {c: f(*c) for c in cols} for _, lab, _, f in series}
    shared = [c for c in cols if all(data[lab][c] is not None for _, lab, _, _ in series)]
    for _, lab, _, _ in series:
        miss = [f"{t}/{m}" for t, m in cols if data[lab][(t, m)] is None]
        if miss:
            print(f"  NOTE {lab} ({a.split}): missing {miss}")
    print(f"  Avg over {len(shared)}/{len(cols)} cells present in all four series")

    plt.rcParams.update(P.RC)
    fh = HEAD + len(METRICS) * PANEL_H + FOOT
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(FIG_W, fh), sharex=True)
    groups = cols + ["avg"]
    xs = list(range(len(groups)))
    for ax, (key, ylab) in zip(axes, METRICS):
        for j, (fam, lab, hatched, _) in enumerate(series):
            vals = []
            for g in groups:
                if g == "avg":
                    v = [data[lab][c][key] for c in shared if data[lab][c][key] is not None]
                    vals.append(sum(v) / len(v) if v else None)
                else:
                    r = data[lab][g]
                    vals.append(None if r is None or r[key] is None else r[key])
            off = (j - (len(series) - 1) / 2) * BAR_W
            ax.bar([x + off for x in xs], [0 if v is None else v for v in vals], width=BAR_W,
                   color=COLOUR[fam], hatch="////" if hatched else None,
                   edgecolor="white" if hatched else COLOUR[fam], lw=0, zorder=2, label=lab)
            for x, v in zip(xs, vals):
                if v is not None:
                    ax.annotate(f"{v:.2f}", (x + off, v), textcoords="offset points",
                                xytext=(0, 1.2), ha="center", va="bottom", fontsize=FS_ANNOT,
                                rotation=90, zorder=6)
        ax.axvline(len(cols) - 0.5, color="#999999", lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        top = max(v[key] for lab in data for v in data[lab].values() if v is not None and v[key] is not None)
        ax.set_ylim(0, top * 1.28)
        ax.grid(True, lw=0.25, color="#dddddd"); ax.set_axisbelow(True); ax.grid(False, axis="x")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        ax.tick_params(axis="y", labelsize=FS_TICK)
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xticks(xs)
    axes[-1].set_xticklabels([f"{TASK_SHORT[t]}\n{MODEL_SHORT[m]}" for t, m in cols] + ["Avg"],
                             fontsize=FS_TICK)
    axes[-1].set_xlim(-0.5 - BAR_W, len(groups) - 0.5 + BAR_W)
    fig.tight_layout(pad=0.3, h_pad=0.5)
    fig.subplots_adjust(top=1.0 - HEAD / fh)
    plt.rcParams["hatch.linewidth"] = 0.5
    fig.legend(handles=[Patch(facecolor=COLOUR[fam], hatch="////" if h else None,
                              edgecolor="white" if h else COLOUR[fam], label=lab)
                        for fam, lab, h, _ in series],
               fontsize=FS_LEG, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               frameon=False, handlelength=1.4, handleheight=1.0, handletextpad=0.4,
               columnspacing=1.4)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {a.out} ({a.split}, {len(shared)}/{len(cols)} complete cells)")


if __name__ == "__main__":
    main()
