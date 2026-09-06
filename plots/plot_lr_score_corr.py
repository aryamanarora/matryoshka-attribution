"""Spearman rho between the SCORE VECTORS of every pair of learning rates, per optimizer arm.

This answers a question the acc-AUC-vs-LR table cannot: when a sweep returns a flat metric
across four orders of magnitude of LR, is that because the LR genuinely does not matter (the
runs are finding the SAME ranking) or because the metric is too blunt to see runs that are in
fact finding DIFFERENT rankings that happen to score alike? Those have opposite implications --
the first says the hyperparameter is settled, the second says the metric is.

The unit is the raw per-unit score vector in `<run>/*.scores.pt`, ranked and correlated
pairwise. Spearman rather than Pearson because everything downstream of the score is a top-k
SELECTION: only the ordering is read, and two runs whose scores differ by a positive monotone
transform produce byte-identical circuits. That also makes the diagonal structure meaningful --
an arm whose backward reads no score magnitude is exactly rank-invariant in the LR, so its
whole matrix should be 1.000.

Data: results/sva_mlp_lr/<variant>_<opt>/lr_<lr>/ (scripts/sva/launch/submit_sva_mlp_lr.sh). ONE cell --
addition / llama3 / --nodes mlp / logit_diff, 2.29M mask logits -- so read this as the anatomy
of a single well-chosen failure case, not a population result. The three arms complete the
optimizer x gate cross that submit_sva_sweep.sh never runs.

THE id-STE PANEL IS A NOISE FLOOR, NOT A RESULT -- read the other two against it. That arm is
provably rank-invariant in the LR: the hard forward reads only the score RANKING and the
identity backward reads no score magnitude, so the whole trajectory is `lr x (a fixed vector)`
and the ordering cannot depend on lr. Observed, it is +1.0000 at exactly the two power-of-two
ratios in the grid (0.005/0.01 and 0.05/0.1) and +0.41..+0.60 everywhere else. That split is
arithmetic, not learning: eval_sva loads the model in bfloat16, and rescaling by a power of two
is exact in any binary float while x3, x5, x10 are not. So ~+0.45 is what TWO RUNS OF THE SAME
SOLUTION correlate at once rounding differs, at 2.29M near-degenerate mask logits.

Consequences, both of which cut against the obvious readings:
  - The soft top-k arms sit BELOW that floor (mean +0.20 Adam, +0.16 SGD). Their flat acc-AUC
    across four orders of LR is therefore NOT "the LR does not matter, same solution found" --
    the solutions genuinely differ and the metric cannot see it. It is a metric null.
  - Any rank correlation computed on a neuron substrate at this scale needs the floor quoted
    beside it. A bare rho of 0.5 between two methods here is not evidence that they agree; it
    is indistinguishable from one method compared against itself.
Both statements are calibrated on TWO power-of-two pairs in ONE cell. Widening the grid to
0.05/0.1/0.2/0.4 would test the floor properly and has not been run.

The colour ramp is deliberately NOT from palette.py: that module is a METHOD palette (a
qualitative scale keyed by attribution method), and rho is a continuous [-1, 1] quantity. A
diverging ramp centred at 0 is the honest encoding and borrowing method hexes for it would
imply a correspondence that does not exist.

Run:  uv run python plots/plot_lr_score_corr.py
      uv run python plots/plot_lr_score_corr.py --res results/sva_mlp_lr --out plots/lr_corr.pdf
"""
import argparse
import glob
import itertools
import os

import numpy as np
import pandas as pd
import torch
from plotnine import (aes, element_blank, element_text, facet_wrap, geom_text, geom_tile,
                      ggplot, labs, scale_fill_gradient2, theme, theme_bw)

RES = "results/sva_mlp_lr"
# Display names for the arm directories, in the order submit_sva_mlp_lr.sh documents them.
ARMS = [("topk_adam", "soft top-$k$ / Adam"),
        ("topk_sgd", "soft top-$k$ / SGD"),
        ("hard_topk_identity_adam", "id-STE / Adam")]


def scores(run_dir):
    """Flat float score vector for a run, or None if the run has no scores.pt."""
    g = glob.glob(os.path.join(run_dir, "*.scores.pt"))
    if not g:
        return None
    x = torch.load(g[0], map_location="cpu", weights_only=False)
    if isinstance(x, dict):
        for k in ("scores", "score", "importances"):
            if k in x:
                x = x[k]
                break
        else:
            x = next(iter(x.values()))
    return np.asarray(x.float().flatten().numpy(), dtype=np.float64)


def topk_sets(vecs, k):
    """{label: frozenset of the k highest-scoring unit indices}. Higher score = selected."""
    return {lab: set(np.argpartition(v, -k)[-k:].tolist()) for lab, v in vecs.items()}


def overlap_matrix(vecs, k):
    """Pairwise |A n B| / k over top-k sets. Equal set sizes, so this is symmetric and is both
    the overlap coefficient and (up to the 2-x/(1+x) reparametrisation) the Jaccard.

    CHANCE IS NOT ZERO and is the whole point of reading this instead of a rank correlation:
    two INDEPENDENT top-k sets share k/n of their members in expectation. At k=1000 of 2.29M
    that is 0.0004, i.e. any visible overlap is real; at k=100000 it is 0.044. The caller
    prints the chance line next to every k.
    """
    S = topk_sets(vecs, k)
    out = {}
    for a, b in itertools.combinations_with_replacement(S, 2):
        v = 1.0 if a == b else len(S[a] & S[b]) / k
        out[(a, b)] = out[(b, a)] = v
    return out


def spearman_matrix(vecs):
    """Pairwise Spearman over {label: vector}. Ranks ONCE per vector, then Pearson on ranks.

    scipy.stats.spearmanr would re-rank both inputs on every call, which at 2.29M units and
    n(n-1)/2 pairs is the dominant cost. Ranking up front is the same statistic.
    """
    import scipy.stats as st
    r = {k: st.rankdata(v) for k, v in vecs.items()}
    out = {}
    for a, b in itertools.combinations_with_replacement(r, 2):
        rho = 1.0 if a == b else float(np.corrcoef(r[a], r[b])[0, 1])
        out[(a, b)] = out[(b, a)] = rho
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=None)
    # Top-k overlap instead of Spearman. Spearman weights all 2.29M units equally, but nothing
    # downstream ever looks below the top few thousand -- two runs can disagree completely in
    # the tail and still produce the same circuit. `--k` asks the question the circuit actually
    # poses: do these runs SELECT the same units?
    ap.add_argument("--k", type=int, default=None,
                    help="if set, plot |A n B|/k over top-k sets instead of Spearman rho")
    # Printed alongside every run: overlap depends strongly on k, and a single k invites
    # cherry-picking. Chance (k/n) is printed with each row.
    ap.add_argument("--k-ladder", default="100,1000,10000,100000",
                    help="comma-separated k values summarised in the stdout table")
    a = ap.parse_args()
    a.out = a.out or (f"plots/lr_topk_overlap_k{a.k}.pdf" if a.k else "plots/lr_score_corr.pdf")

    rows, notes, ladder = [], [], []
    for arm, alabel in ARMS:
        runs = {}
        for d in sorted(glob.glob(os.path.join(a.res, arm, "lr_*"))):
            lr = float(os.path.basename(d)[3:])
            v = scores(d)
            if v is not None:
                runs[lr] = v
        if len(runs) < 2:
            notes.append((alabel, len(runs), None))
            continue
        # Every arm must be correlated on a COMMON unit set. They are all the same cell so the
        # lengths agree, but assert rather than silently truncate -- a length mismatch would
        # mean two different substrates got mixed into one matrix.
        lens = {len(v) for v in runs.values()}
        assert len(lens) == 1, f"{arm}: score vectors differ in length {sorted(lens)}"
        n_units = lens.pop()
        M = overlap_matrix(runs, a.k) if a.k else spearman_matrix(runs)
        lrs = sorted(runs)
        off = [M[(x, y)] for x, y in itertools.combinations(lrs, 2)]
        notes.append((alabel, len(runs), (min(off), float(np.mean(off)))))
        for kk in [int(s) for s in a.k_ladder.split(",")]:
            if kk >= n_units:
                continue
            O = overlap_matrix(runs, kk)
            o = [O[(x, y)] for x, y in itertools.combinations(lrs, 2)]
            ladder.append((alabel, kk, kk / n_units, min(o), float(np.mean(o)), max(o)))
        for x in lrs:
            for y in lrs:
                rows.append(dict(arm=alabel, x=f"{x:g}", y=f"{y:g}", rho=M[(x, y)],
                                 label=f"{M[(x, y)]:.2f}".lstrip("0") or "0"))
    if not rows:
        print("no scores.pt found under", a.res)
        return

    df = pd.DataFrame(rows)
    df["arm"] = pd.Categorical(df["arm"], [lb for _, lb in ARMS if lb in set(df["arm"])])
    # Axis order is NUMERIC lr, not the string sort that geom_tile would otherwise impose
    # ("10" < "3" lexically). Built per-arm because the three arms sweep different grids.
    order = sorted({float(v) for v in df["x"]})
    lab = [f"{v:g}" for v in order]
    df["x"] = pd.Categorical(df["x"], lab)
    df["y"] = pd.Categorical(df["y"], lab[::-1])

    p = (
        ggplot(df, aes("x", "y", fill="rho"))
        + geom_tile(color="white", size=0.3)
        # rho printed in every cell: the ramp carries the gestalt, the number carries the claim.
        # A reader checking "is this arm exactly rank-invariant" needs 1.00 vs 0.98, which no
        # colour scale resolves.
        + geom_text(aes(label="label"), size=4.5, color="#222222")
        + facet_wrap("arm", scales="free")
        # Diverging and centred at 0, with the full [-1, 1] domain pinned via `limits` so the
        # ramp means the same thing in all three panels. Letting it auto-scale would make a
        # panel whose values are all >0.99 look as varied as one spanning -0.1 to 1.
        # Overlap is bounded [0, 1] and its neutral point is chance (k/n ~ 0), not 0.5, so the
        # ramp is pinned to [0, 1] with the midpoint left at 0 -- i.e. it reads as sequential.
        + (scale_fill_gradient2(low="#f7f7f7", mid="#f7f7f7", high="#2166ac",
                                midpoint=0.0, limits=(0.0, 1.0),
                                name=f"overlap@$k$={a.k:,}") if a.k
           else scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                                     midpoint=0.0, limits=(-1.0, 1.0), name=r"Spearman $\rho$"))
        + labs(x="learning rate", y="learning rate")
        + theme_bw()
        + theme(figure_size=(9.0, 3.2), dpi=300, text=element_text(size=7),
                axis_text_x=element_text(size=6), axis_text_y=element_text(size=6),
                strip_background=element_blank(), panel_grid=element_blank())
    )
    p.save(a.out, verbose=False)
    p.save(a.out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print("wrote", a.out)
    what = f"overlap@k={a.k}" if a.k else "rho"
    print(f"\noff-diagonal {what} per arm (min / mean) -- 1.000 = identical selection:")
    for alabel, n, stats in notes:
        if stats is None:
            print(f"  {alabel:22s} n={n}  (need >=2 runs with scores.pt)")
        else:
            print(f"  {alabel:22s} n={n}  min {stats[0]:+.4f}   mean {stats[1]:+.4f}")
    if ladder:
        print("\ntop-k overlap vs k. `chance` is k/n, what two INDEPENDENT selections would")
        print("share; `x chance` is the honest effect size (mean / chance).")
        print(f"  {'arm':22s}{'k':>8s}{'chance':>9s}{'min':>8s}{'mean':>8s}{'max':>8s}{'x chance':>10s}")
        for alabel, kk, ch, lo, mu, hi in ladder:
            print(f"  {alabel:22s}{kk:8d}{ch:9.4f}{lo:8.3f}{mu:8.3f}{hi:8.3f}{mu / ch:10.1f}")


if __name__ == "__main__":
    main()
