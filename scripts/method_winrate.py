"""Head-to-head win rates between circuit-discovery methods on the SVA+ sweep.

Why win rates rather than mean AUCs. The two ablation settings put faith-AUC on different
scales (its (F_clean - F_patch) denominator shrinks ~1.9x under zeroing), faith-AUC is unbounded
and gap-paddable (a logit-diff-trained circuit can inflate it by widening the logit gap without
recovering the decision), and coverage across cells is ragged while the sweep is still running.
A mean over cells is sensitive to all three. A win rate is invariant to any monotone rescaling
of the metric, so it survives the first two outright, and the pairwise-complete rule below
handles the third.

THE UNIT is a cell = (setting, substrate, task, loss). Methods are compared only inside a cell,
i.e. at MATCHED training loss on matched data. Matched-loss is the honest default: letting each
method pick its best loss per cell is an oracle that rewards whichever method has the most
variants on disk. `--best-loss` reports that oracle view too, clearly labelled.

PAIRWISE-COMPLETE: every pair of methods is scored only over the cells where BOTH ran. Method
A's headline number is its wins over all its own head-to-heads, so a method that has run on
fewer cells is not penalised for the cells it is missing -- but it is also not credited for
them, and `n` is printed so a number resting on few comparisons is visible as such.

SETTINGS ARE NEVER POOLED. Patched and zero ablation are different experiments (MAttr and the
mask baselines retrain through whichever intervention they are scored under; the gradient
baselines change estimator), so a cell in one is not comparable to a cell in the other.

Metrics: both `acc` (accuracy AUC) and `faith` (faithfulness AUC) are used exactly as stored.
Prefer `acc` when the two disagree: it is bounded and immune to gap-padding. Under zeroing its
baseline sits near 0.5 rather than patching's 0.0, which makes the two settings' absolute values
incomparable -- but win rates only ever compare inside a cell, where that baseline is common to
every competitor, so it cannot affect a single number here.

Run:  uv run python scripts/method_winrate.py
      uv run python scripts/method_winrate.py --metric faith --by-group
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "plots"))
# Reuse the figure's own tag parser and method registry rather than restating them. These have
# a history of silently folding the headline `sufficient_topk_` runs and `attnlrp` into the IG
# series via a catch-all `else: return "IG"`, so a second copy here is a real hazard.
from plot_accauc_vs_faithauc import (  # noqa: E402
    METHODS, SVA, ARITH, parse_method, on_model)

SOURCES = [("results/sva_sweep", "Patched", "−input"),
           ("results/sva_sweep_input", "Patched", "+input"),
           ("results/sva_zeroabl", "Zero-abl.", "−input")]
GROUP_OF = {t: "SVA" for t in SVA} | {t: "Arith" for t in ARITH}
GROUP_OF |= {"arc_easy": "ARC-E", "ioi": "IOI"}
# (nodes, inp, label) for --by-substrate, in the paper figure's column order. `mlp` and
# `mlp+attn_head` are per-POSITION layouts and exist only in the -input dirs; `node` is the only
# substrate that can score the input embedding, hence the +input column.
COLUMNS = [("mlp", "−input", "MLP"),
           ("mlp+attn_head", "−input", "MLP+attn"),
           ("node", "−input", "Node"),
           ("node", "+input", "Node, +input")]


LOSS_LABEL = {"acc": "acc", "ce": "CE", "logit_diff": "logit-diff"}


def load(split_loss=True):
    """cell key -> {competitor: (acc_corr, faith)}.

    `split_loss=True` (the default) makes the training loss part of the COMPETITOR identity,
    so MAttr-CE and MAttr-logit-diff are two entrants racing inside one (setting, substrate,
    task) cell. That is the right unit when the question is "which recipe wins", because the
    loss is a choice the practitioner makes, not a nuisance parameter -- and the SVA+ figure
    already treats (method, loss) as its plotted point.

    `split_loss=False` instead pins the loss as part of the CELL, so only same-loss pairs ever
    meet. Use that to ask "which mask/attribution machinery is better, holding the objective
    fixed" -- a narrower question that cannot be answered by the split view, since there a
    method can win purely by having one loss that suits the metric.
    """
    cells = defaultdict(dict)
    for res, setting, inp in SOURCES:
        for f in glob.glob(str(ROOT / res / "*.json")):
            d = json.load(open(f))
            m = parse_method(os.path.basename(f), d)
            if m is None or m not in METHODS:
                continue
            # Same pin as the figure: results/sva_sweep carries a llama3 IOI wave, and the cell key
            # below has no model in it, so both files would land on one key and glob order would
            # decide which model the IOI column reports. Imported, not restated -- see on_model.
            if not on_model(d):
                continue
            # acc-AUC as stored. There is no chance correction here (see `plot_accauc_vs_faithauc
            # .load` for the full autopsy): every method inside a cell shares that cell's task,
            # substrate and setting, so any floor shared across them is an affine map that leaves
            # the ordering -- and therefore every win rate -- untouched. The only correction that
            # would change a win rate is a per-RUN one, and that is precisely the broken kind.
            corr = d["acc_auc"]
            loss = d["loss"]
            if split_loss:
                cells[(setting, d["nodes"], inp, d["task"])][(m, loss)] = (corr, d["faith_auc"])
            else:
                cells[(setting, d["nodes"], inp, d["task"], loss)][(m, None)] = (corr,
                                                                                d["faith_auc"])
    return cells


def clabel(c):
    """Display label for a competitor key (method, loss|None)."""
    m, loss = c
    return METHODS[m][0] + (f" · {LOSS_LABEL[loss]}" if loss else "")


def collapse_best_loss(cells, idx):
    """Oracle view: keep each method's best loss per cell, collapsing the loss axis.

    Not the default. It answers "how good can this method be if you tune the loss per cell",
    which flatters whichever method has more losses on disk and is not how the paper reports.
    """
    best = defaultdict(dict)
    for key, bym in cells.items():
        k = key[:4]
        for (m, _loss), v in bym.items():
            cur = best[k].get((m, None))
            if cur is None or (np.isfinite(v[idx]) and v[idx] > cur[idx]):
                best[k][(m, None)] = v
    return best


def winrates(cells, idx, keyfilter=None):
    """Pairwise wins/losses per competitor over cells where both members of the pair ran."""
    pair = defaultdict(lambda: [0, 0, 0])          # (a,b) -> [a_wins, b_wins, ties]
    ncell = defaultdict(int)
    order = {c: i for i, c in enumerate(
        (m, l) for m in METHODS for l in (None, "acc", "ce", "logit_diff"))}
    for key, bym in cells.items():
        if keyfilter and not keyfilter(key):
            continue
        present = sorted((c for c in bym if np.isfinite(bym[c][idx])), key=order.get)
        for c in present:
            ncell[c] += 1
        for a, b in combinations(present, 2):
            va, vb = bym[a][idx], bym[b][idx]
            rec = pair[(a, b)]
            rec[0 if va > vb else 1 if vb > va else 2] += 1
    tally = defaultdict(lambda: [0, 0, 0])         # competitor -> [wins, losses, ties]
    for (a, b), (aw, bw, ti) in pair.items():
        tally[a][0] += aw; tally[a][1] += bw; tally[a][2] += ti
        tally[b][0] += bw; tally[b][1] += aw; tally[b][2] += ti
    return tally, pair, ncell


def ranked(tally, ncell):
    """[(winrate, competitor, W, L, T, h2h, cells)] sorted best-first."""
    rows = []
    for c, (w, l, t) in tally.items():
        n = w + l + t
        rows.append((w / n if n else np.nan, c, w, l, t, n, ncell[c]))
    return sorted(rows, key=lambda r: (-r[0], clabel(r[1])))


def fmt(tally, ncell, title):
    print(f"\n{title}")
    rows = ranked(tally, ncell)
    if not rows:
        print("  (no cells)")
        return
    w = max(len(clabel(r[1])) for r in rows)
    print(f"  {'competitor':{w}s} {'winrate':>8s} {'W':>5s} {'L':>5s} {'T':>4s} "
          f"{'h2h':>5s} {'cells':>6s}")
    for wr, c, ww, l, t, n, nc in rows:
        print(f"  {clabel(c):{w}s} {wr:8.3f} {ww:5d} {l:5d} {t:4d} {n:5d} {nc:6d}")


def render(panels, metric_label, unit_label, cell_label, out, ncols=None):
    """Draw the ranked tables as an image, laid out in a `ncols`-wide grid.

    Hand-drawn rather than matplotlib's table(): the win-rate bar behind each row is the point
    (rank order is read from bar length at a glance, the digits are for checking), and table()
    gives no way to put a bar under the text.

    `panels` is a flat list of (title, rows) in row-major grid order; a panel with no rows is
    drawn as an explicit "(no cells)" so the grid stays aligned and a missing combination reads
    as missing rather than shifting its neighbours left.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    ncols = ncols or len(panels)
    nrows = -(-len(panels) // ncols)
    # One shared row count so every panel has the same vertical scale: a competitor's bar is
    # then comparable across panels by eye, which is the whole reason for a grid.
    nrow = max(len(rows) for _, rows in panels)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(5.6 * ncols, nrows * (0.34 * nrow + 1.5)))
    flat = [ax for row in axes for ax in row]
    for ax in flat[len(panels):]:
        ax.axis("off")
    for ax, (title, rows) in zip(flat, panels):
        ax.set_xlim(0, 1); ax.set_ylim(-nrow - 0.5, 1.6); ax.axis("off")
        ax.text(0, 1.15, title, fontsize=12, fontweight="bold", va="bottom")
        for x, lab in ((0.035, "#"), (0.075, "competitor"), (0.60, "win rate"),
                       (0.74, "W–L–T"), (0.99, "h2h")):
            ax.text(x, 0.35, lab, fontsize=8, color="#555",
                    ha="right" if lab in ("#", "h2h") else "left")
        ax.plot([0, 1], [0.05, 0.05], color="#999", lw=0.8)
        if not rows:
            ax.text(0.075, -0.65, "(no cells)", fontsize=9, color="#999", style="italic")
        for i, (wr, c, w, l, t, n, _nc) in enumerate(rows):
            y = -i - 0.55
            # Bar is drawn first so it reads as a row background, not a separate column, and is
            # scaled to the win-rate column's left edge so a full-width bar never runs under the
            # digits. alpha keeps the text on top legible.
            ax.add_patch(Rectangle((0.075, y - 0.3), 0.52 * wr, 0.6,
                                   color=METHODS[c[0]][1], alpha=0.28, lw=0))
            ax.text(0.035, y - 0.1, f"{i + 1}", fontsize=8, color="#777", ha="right")
            ax.add_patch(Rectangle((0.05, y - 0.16), 0.016, 0.32,
                                   color=METHODS[c[0]][1], lw=0))
            ax.text(0.075, y - 0.1, clabel(c), fontsize=9)
            ax.text(0.60, y - 0.1, f"{wr:.3f}", fontsize=9, fontweight="bold")
            ax.text(0.74, y - 0.1, f"{w}–{l}" + (f"–{t}" if t else ""), fontsize=8, color="#444")
            ax.text(0.99, y - 0.1, f"{n}", fontsize=8, color="#777", ha="right")
    fig.suptitle(f"Head-to-head win rate · {metric_label} · {unit_label}", fontsize=11, y=0.997)
    fig.text(0.5, 0.008,
             "Pairwise-complete: each pair scored only over cells where both ran. "
             f"Cell = {cell_label}. Settings never pooled.",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(out, dpi=170)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="acc", choices=["acc", "faith"],
                    help="acc = accuracy AUC (default, bounded); faith = faith-AUC")
    ap.add_argument("--best-loss", action="store_true",
                    help="oracle: collapse to each method's best loss per cell")
    ap.add_argument("--by-group", action="store_true", help="also break down by task group")
    ap.add_argument("--by-substrate", action="store_true",
                    help="split every panel by granularity too, so the grid is "
                         "setting x substrate and the cell is just a task")
    ap.add_argument("--matched-loss", action="store_true",
                    help="pin the loss as part of the CELL, so only same-loss pairs meet "
                         "(default: the loss is part of the competitor's identity)")
    ap.add_argument("--head-to-head", default=None,
                    help="print this method's record against each opponent (key or label)")
    ap.add_argument("--image", nargs="?", const="plots/method_winrate.png", default=None,
                    help="also render the ranked tables to this PNG")
    a = ap.parse_args()

    idx = 0 if a.metric == "acc" else 1
    label = ("acc-AUC" if a.metric == "acc" else "faith-AUC")
    cells = load(split_loss=not a.matched_loss)
    if a.best_loss:
        cells = collapse_best_loss(cells, idx)
    unit = ("best loss per cell (ORACLE)" if a.best_loss else
            "matched loss" if a.matched_loss else "loss counted as part of the method")

    print(f"metric: {label}   [{unit}]")
    print(f"cells loaded: {len(cells)}")

    panels = []
    for setting in ("Patched", "Zero-abl."):
        tally, pair, ncell = winrates(cells, idx, lambda k, s=setting: k[0] == s)
        fmt(tally, ncell, f"=== {setting} ===")
        if a.by_substrate:
            # One panel per (setting, substrate), in the SAME column order as the paper figure's
            # facet_grid, and every column emitted even when empty -- a granularity where a
            # method never ran has to be visible as a hole, not silently closed up.
            for nodes, inp, slabel in COLUMNS:
                t2, _, n2 = winrates(
                    cells, idx,
                    lambda k, s=setting, n=nodes, i=inp: k[0] == s and (k[1], k[2]) == (n, i))
                fmt(t2, n2, f"--- {setting} / {slabel} ---")
                panels.append((f"{setting} · {slabel}", ranked(t2, n2)))
        elif tally:
            panels.append((setting, ranked(tally, ncell)))
        if a.by_group:
            for g in ("SVA", "Arith", "ARC-E", "IOI"):
                t2, _, n2 = winrates(
                    cells, idx, lambda k, s=setting, g=g: k[0] == s and GROUP_OF[k[3]] == g)
                if t2:
                    fmt(t2, n2, f"--- {setting} / {g} ---")
        if a.head_to_head:
            # Competitors are (method, loss) pairs now, so a bare method name selects EVERY
            # loss variant of that method rather than one row -- which is what you want when
            # asking "how does MAttr do against everyone", and the loss is spelled out in the
            # printed label so the several MAttr rows stay distinguishable.
            want = [c for c in {k for kk in pair for k in kk}
                    if a.head_to_head in (c[0], METHODS[c[0]][0], clabel(c))]
            if not want:
                raise SystemExit(f"unknown method {a.head_to_head!r}; "
                                 f"pick from {sorted({METHODS[m][0] for m in METHODS})}")
            for c in sorted(want, key=clabel):
                print(f"\n  {clabel(c)} head-to-head ({setting}):")
                for (x, y), (xw, yw, ti) in sorted(pair.items(), key=lambda kv: clabel(kv[0][0])):
                    if c not in (x, y):
                        continue
                    opp = y if x == c else x
                    w, l = (xw, yw) if x == c else (yw, xw)
                    n = w + l + ti
                    print(f"    vs {clabel(opp):24s} {w:3d}-{l:3d}"
                          f"{f'-{ti}' if ti else '   '}  ({w / n:.3f} of {n})")

    if a.image and any(rows for _, rows in panels):
        render(panels, label, unit,
               "(task)" if a.by_substrate else "(substrate, task)",
               str(ROOT / a.image), ncols=len(COLUMNS) if a.by_substrate else None)

    print("\nCoverage is still filling in -- these numbers will move. `cells` is how many cells a"
          "\nmethod ran in; `h2h` is how many head-to-head comparisons back its win rate.")


if __name__ == "__main__":
    main()
