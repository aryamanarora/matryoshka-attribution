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

Metrics: `acc` is the chance-corrected accuracy AUC -- under zeroing acc_base sits at a 0.5
floor rather than patching's 0.0, so the raw column reads ~0.5 for a circuit with no signal;
the correction is the identity when acc[0] = 0 and so leaves patched values untouched. `faith`
is faith-AUC as stored. Prefer `acc` when the two disagree: it is bounded and immune to padding.

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
    METHODS, SVA, ARITH, parse_method, _auc_of)

SOURCES = [("results/sva_sweep", "Patched", "−input"),
           ("results/sva_sweep_input", "Patched", "+input"),
           ("results/sva_zeroabl", "Zero-abl.", "−input")]
GROUP_OF = {t: "SVA" for t in SVA} | {t: "Arith" for t in ARITH}
GROUP_OF |= {"arc_easy": "ARC-E", "ioi": "IOI"}


def load():
    """(setting, substrate, input, task, loss) -> {method: (acc_corr, faith)}."""
    cells = defaultdict(dict)
    for res, setting, inp in SOURCES:
        for f in glob.glob(str(ROOT / res / "*.json")):
            d = json.load(open(f))
            m = parse_method(os.path.basename(f), d)
            if m is None or m not in METHODS:
                continue
            acc = d["iso_metrics"]["acc_base"]
            a0 = acc[0]
            # Identity when a0 == 0 (the patched case), so both settings share one column.
            corr = (np.nan if a0 == 1 else
                    _auc_of(d["n_nodes"], [(x - a0) / (1 - a0) for x in acc]))
            key = (setting, d["nodes"], inp, d["task"], d["loss"])
            cells[key][m] = (corr, d["faith_auc"])
    return cells


def collapse_best_loss(cells, idx):
    """Oracle view: keep each method's best loss per (setting, substrate, input, task).

    Not the default. It answers "how good can this method be if you tune the loss per cell",
    which flatters whichever method has more losses on disk and is not how the paper reports.
    """
    best = defaultdict(dict)
    for (setting, sub, inp, task, _loss), bym in cells.items():
        for m, v in bym.items():
            k = (setting, sub, inp, task, "best")
            cur = best[k].get(m)
            if cur is None or (np.isfinite(v[idx]) and v[idx] > cur[idx]):
                best[k][m] = v
    return best


def winrates(cells, idx, keyfilter=None):
    """Pairwise wins/losses per method over cells where both members of the pair ran."""
    pair = defaultdict(lambda: [0, 0, 0])          # (a,b) -> [a_wins, b_wins, ties]
    ncell = defaultdict(int)
    for key, bym in cells.items():
        if keyfilter and not keyfilter(key):
            continue
        present = [m for m in METHODS if m in bym and np.isfinite(bym[m][idx])]
        for m in present:
            ncell[m] += 1
        for a, b in combinations(present, 2):
            va, vb = bym[a][idx], bym[b][idx]
            rec = pair[(a, b)]
            rec[0 if va > vb else 1 if vb > va else 2] += 1
    tally = defaultdict(lambda: [0, 0, 0])         # method -> [wins, losses, ties]
    for (a, b), (aw, bw, ti) in pair.items():
        tally[a][0] += aw; tally[a][1] += bw; tally[a][2] += ti
        tally[b][0] += bw; tally[b][1] += aw; tally[b][2] += ti
    return tally, pair, ncell


def fmt(tally, ncell, title):
    print(f"\n{title}")
    rows = []
    for m, (w, l, t) in tally.items():
        n = w + l + t
        rows.append((w / n if n else np.nan, m, w, l, t, n, ncell[m]))
    if not rows:
        print("  (no cells)")
        return
    print(f"  {'method':14s} {'winrate':>8s} {'W':>5s} {'L':>5s} {'T':>4s} {'h2h':>5s} {'cells':>6s}")
    for wr, m, w, l, t, n, nc in sorted(rows, reverse=True):
        print(f"  {METHODS[m][0]:14s} {wr:8.3f} {w:5d} {l:5d} {t:4d} {n:5d} {nc:6d}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", default="acc", choices=["acc", "faith"],
                    help="acc = chance-corrected accuracy AUC (default, bounded); faith = faith-AUC")
    ap.add_argument("--best-loss", action="store_true",
                    help="oracle: collapse to each method's best loss per cell")
    ap.add_argument("--by-group", action="store_true", help="also break down by task group")
    ap.add_argument("--head-to-head", default=None,
                    help="print this method's record against each opponent (key or label)")
    a = ap.parse_args()

    idx = 0 if a.metric == "acc" else 1
    label = ("chance-corrected acc-AUC" if a.metric == "acc" else "faith-AUC")
    cells = load()
    if a.best_loss:
        cells = collapse_best_loss(cells, idx)

    print(f"metric: {label}"
          f"{'   [ORACLE: best loss per cell]' if a.best_loss else '   [matched loss]'}")
    print(f"cells loaded: {len(cells)}")

    for setting in ("Patched", "Zero-abl."):
        tally, pair, ncell = winrates(cells, idx, lambda k, s=setting: k[0] == s)
        fmt(tally, ncell, f"=== {setting} ===")
        if a.by_group:
            for g in ("SVA", "Arith", "ARC-E", "IOI"):
                t2, _, n2 = winrates(
                    cells, idx, lambda k, s=setting, g=g: k[0] == s and GROUP_OF[k[3]] == g)
                if t2:
                    fmt(t2, n2, f"--- {setting} / {g} ---")
        if a.head_to_head:
            want = next((k for k in METHODS
                         if k == a.head_to_head or METHODS[k][0] == a.head_to_head), None)
            if want is None:
                raise SystemExit(f"unknown method {a.head_to_head!r}; pick from {list(METHODS)}")
            print(f"\n  {METHODS[want][0]} head-to-head ({setting}):")
            for (x, y), (xw, yw, ti) in sorted(pair.items()):
                if want not in (x, y):
                    continue
                opp = y if x == want else x
                w, l = (xw, yw) if x == want else (yw, xw)
                n = w + l + ti
                print(f"    vs {METHODS[opp][0]:14s} {w:3d}-{l:3d}"
                      f"{f'-{ti}' if ti else '   '}  ({w / n:.3f} of {n})")

    print("\nCoverage is still filling in -- these numbers will move. `cells` is how many cells a"
          "\nmethod ran in; `h2h` is how many head-to-head comparisons back its win rate.")


if __name__ == "__main__":
    main()
