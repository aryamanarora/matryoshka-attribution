"""Objective ablation of the headline \\ourmethod{} per MIB cell: iso vs cause vs joint.

    uv run python plots/plot_objective_ablation.py [--split validation]
    -> plots/objective_ablation.pdf (+ .png); the paper copies it to figs/ for its own appendix
       section (sections/objective-ablation.tex).

THE THREE RUNS are the headline node-level recipe (soft top-k forward, uniform k, Adam, lr 0.05,
500 steps -- mattr_variants.HEADLINE) trained under three objectives and otherwise identical:

    iso     top-k CLEAN, complement corrupted (denoising; = `sufficient`). What every MIB row of
            ours is trained with, and what MIB's CPR measures.
    cause   top-k CORRUPTED, complement clean (noising; = `necessary`).
    joint   a fair coin per step between the two (losses.resolve_direction).

Dirs are summarize_mode_ablation.DIRS; the cells are make_mib_table.COLUMNS (12). Both metrics
are read from the same eval pkl as the tables (CPR = `area_under`, compactness = `acc_auc`).

FULL WIDTH, ONE GROUP PER CELL, and a 13th "Avg" group after a rule: the question is WHERE the
objective matters, not just how much on average. Cause collapses to the 0.25 CPR floor on every
IOI cell and both llama3 ARC cells while staying near 1.0 on arithmetic and MCQA; the average
alone (0.59 vs 2.06) cannot say that.

Two panels stacked, sharing x, separate y (CPR AUC is unbounded and ~1 at chance, acc-AUC is in
[0, 1]); they are not to be compared by bar height across panels.

PANELS 3 AND 4 (2026-09-16) are the same three runs under the FLIPPED intervention
(scripts/mib/eval_mib_noising.py -> <dir>/<task>_<model>_<split>_noising.pkl: the top-k are
corrupted, the rest clean -- the direction cause and joint were trained for): `area_from_1`
(area between 1 and the faithfulness curve; the top-k break the behaviour sooner -> higher) and
`flip_acc_auc` (log-weighted fraction of examples whose answer flips to the counterfactual --
the reverse IIA). Each panel averages over the cells it has for all three runs and draws no bar
where a cell is missing, so a partial noising wave shows as gaps, not zeros; stdout names them.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mib"))
import palette as P                                    # noqa: E402
import make_mib_table as M                             # COLUMNS  # noqa: E402
import summarize_mode_ablation as A                    # DIRS, read()  # noqa: E402

# iso is drawn in the method's own colour (it IS the headline); cause and joint in two Wong
# hues the MIB figures do not otherwise assign to a method, so neither reads as a baseline.
COLOUR = {"iso": P.METHOD["MAttr"], "cause": "#e69f00", "joint": "#009e73"}
LABEL = {"iso": "iso (headline)", "cause": "cause", "joint": "joint"}
TASK_SHORT = {"ioi": "IOI", "arithmetic_addition": "Arith ($+$)", "arithmetic_subtraction": "Arith ($-$)",
              "mcqa": "MCQA", "arc_easy": "ARC-E", "arc_challenge": "ARC-C"}
MODEL_SHORT = {"gpt2": "GPT-2", "qwen2.5": "Qwen", "gemma2": "Gemma", "llama3": "Llama"}
# (reader name, index into the reader's tuple, y label), top to bottom. "den" is
# summarize_mode_ablation.read -> (area_under, acc_auc); "noi" is read_noising below ->
# (area_from_1, flip_acc_auc).
PANELS = [("den", 0, "CPR AUC (↑)"), ("den", 1, "Compactness (↑)"),
          ("noi", 0, "Noising: area from 1 (↑)"), ("noi", 1, "Noising: flip acc-AUC (↑)")]


def read_noising(d, task, model, split):
    """(area_from_1, flip_acc_auc) from eval_mib_noising.py's pkl, or None if absent."""
    import pickle
    p = M.RESULTS_BASE / d / f"{task}_{model}_{split}_noising.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        r = pickle.load(f)
    return r.get("area_from_1"), r.get("flip_acc_auc")

FIG_W, PANEL_H, FOOT, HEAD = 5.5, 0.95, 0.42, 0.22
FS_AXIS, FS_TICK, FS_ANNOT, FS_LEG = 6.5, 5.5, 4.0, 6.0
BAR_W = 0.26


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--out", default="plots/objective_ablation.pdf")
    a = ap.parse_args()

    cols = [(t, m) for t, m, _ in M.COLUMNS]
    objs = [(name.split(" ")[0], d) for name, d in A.DIRS[a.split]]      # "iso (headline)" -> iso
    readers = {"den": A.read, "noi": read_noising}
    # data[reader][objective][cell] -> tuple or None; shared[reader] = cells every objective has
    data = {r: {o: {c: f(d, *c, a.split) for c in cols} for o, d in objs} for r, f in readers.items()}
    shared = {r: [c for c in cols if all(data[r][o][c] is not None for o in data[r])] for r in readers}
    for r in readers:
        missing = [f"{t}/{m}" for t, m in cols if (t, m) not in shared[r]]
        if missing:
            print(f"  NOTE {r} ({a.split}): {len(shared[r])}/{len(cols)} cells in every objective; "
                  f"missing {missing}")

    plt.rcParams.update(P.RC)
    fh = HEAD + len(PANELS) * PANEL_H + FOOT
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(FIG_W, fh), sharex=True)
    groups = cols + ["avg"]
    xs = list(range(len(groups)))
    for ax, (rd, key, ylab) in zip(axes, PANELS):
        dd, sh = data[rd], shared[rd]
        for j, (o, _) in enumerate(objs):
            vals = []
            for g in groups:
                if g == "avg":
                    v = [dd[o][c][key] for c in sh if dd[o][c][key] is not None]
                    vals.append(sum(v) / len(v) if v else None)
                else:
                    r = dd[o][g]
                    vals.append(None if r is None or r[key] is None else r[key])
            off = (j - (len(objs) - 1) / 2) * BAR_W
            ax.bar([x + off for x in xs], [0 if v is None else v for v in vals], width=BAR_W,
                   color=COLOUR[o], lw=0, zorder=2, label=LABEL[o])
            for x, v in zip(xs, vals):
                if v is not None:
                    ax.annotate(f"{v:.2f}", (x + off, v), textcoords="offset points",
                                xytext=(0, 1.2), ha="center", va="bottom", fontsize=FS_ANNOT,
                                rotation=90, zorder=6)
        ax.axvline(len(cols) - 0.5, color="#999999", lw=0.5, ls=(0, (2, 2)), zorder=1)
        ax.set_ylabel(ylab, fontsize=FS_AXIS)
        top = max([v[key] for o in dd for v in dd[o].values() if v is not None and v[key] is not None] or [1.0])
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
    fig.legend(handles=[Patch(facecolor=COLOUR[o], label=LABEL[o]) for o, _ in objs],
               fontsize=FS_LEG, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.998),
               frameon=False, handlelength=1.2, handleheight=1.0, handletextpad=0.4,
               columnspacing=1.4)
    fig.savefig(a.out)
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {a.out} ({a.split}; complete cells: denoising {len(shared['den'])}/{len(cols)}, "
          f"noising {len(shared['noi'])}/{len(cols)})")


if __name__ == "__main__":
    main()
