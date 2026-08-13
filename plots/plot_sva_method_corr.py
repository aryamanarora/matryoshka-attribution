"""Spearman rank-correlation between every pair of (attribution method x training loss)
combinations, faceted into the four SVA+ subtasks.

One panel per SVA subtask (Simple / Noun PP / RC / Within RC), each an N x N matrix over the
METHOD x LOSS grid that scripts/make_sva_neuron_table.py already tabulates: 6 methods x 3 losses
= 18 series, or 24 with --wide. Every entry is Spearman's rho between the two runs' per-unit
score vectors over the WHOLE substrate, not a top-k overlap -- this asks whether two runs order
the units the same way, which is the property every faithfulness curve in the paper is a
function of.

WHY SUBTASK IS THE FACET AND NOT A ROW. Two score vectors are only comparable if they index the
same units, and at the positional substrates they do not across subtasks: seq_len is 3/6/7/6 for
simple/nounpp/rc/within_rc, so `mlp` vectors are 1376256/2752512/3211264/2752512 long. There is
no meaningful "correlation between Simple and RC" to compute at all. Faceting by subtask is what
keeps every number in this figure well defined. (`node` happens to be 1056 for all four, since
32 layers x 32 heads + 32 MLP blocks does not depend on sequence length -- but faceting the same
way there keeps the four figures readable against each other.)

ORDERING is method-major, loss-minor, so the 3x3 blocks ON the diagonal are "the same method
trained against the three different objectives". That block is the question the figure exists to
answer: if it is dark, the method's ranking is a property of the method and the loss is a
detail; if it is pale, the target metric moves the circuit as much as the attribution method
does. Rules are drawn at the method boundaries so the blocks are findable without counting.

SIGNED SCORES, not |score|. Same convention as plot_method_corr_heatmap.py and as
evaluate.sparsity_sweep's `flat.argsort(descending=True)`: these are the rankings our own
numbers were computed from. Ranking by magnitude would correlate a DIFFERENT set of orderings
than the ones the paper reports, for IG/IxG especially, whose scores are signed effects.

Run:  uv run python plots/plot_sva_method_corr.py [--substrate node] [--include-input] [--wide]
      -> paper/figs/sva_method_corr_<substrate stem>.pdf
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from plotnine import (ggplot, aes, geom_tile, geom_point, geom_blank, geom_vline, geom_hline, labs,
                      facet_wrap, scale_fill_gradient2, scale_x_discrete, scale_y_discrete,
                      theme_bw, theme_set, theme, element_text, element_line, element_blank)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
# One source of truth for the grid: this figure and tabs/sva_top_*.tex must describe the same
# runs, so the substrate map, method templates, loss fragments and subtask list are IMPORTED
# rather than restated. A method added to that table appears here on the next run.
from make_sva_neuron_table import (  # noqa: E402
    SUBSTRATES, METHODS, LOSSES, TASKS, load_run, layout)

OUT = Path("paper/figs")

# Axis labels. make_sva_neuron_table.LABELS is LaTeX (`\ourmethod{}`, `I$\times$G`) and cannot go
# through matplotlib, and at 18-24 rows the long names do not fit anyway, so these are the short
# forms -- matched to plot_method_corr_heatmap.py's DISPLAY dict where the method exists in both.
SHORT = {"IG": "IG", "IxG": "IxG", "eprun-s090": "NodePrun", "sig_lr0.3_l16.0": "DBM",
         "stopk-log": "MAttr", "stopk-unif": "MAttr+u",
         "attnlrp": "AttnLRP", "conductance": "Conduct"}
LOSS_SHORT = {"logit_diff": "LD", "ce": "CE", "acc": "Acc"}
# --wide adds the two baselines that are swept at every SVA substrate but are not columns of
# tabs/sva_top_*.tex. They are kept OFF by default so the default figure and the tables cover
# the same method set; 24 rows also pushes the tick labels below legibility at this width.
WIDE = [("attnlrp", "attnlrp%s"), ("conductance", "conductance%s")]

# A run that never learned a usable circuit still HAS a score vector, and it still correlates
# with everything else -- usually near 0, sometimes strongly negative. Read naively those cells
# say "this method orders units differently", when what they actually say is "this run failed".
# Both failure modes present in this sweep are real and already known: `ce` cripples IxG (see the
# sva-sweep notes), and DBM under the accuracy loss collapses at the `node` substrate on 3 of 4
# subtasks. So collapsed runs are MARKED (a dot on their diagonal cell in the panel where they
# collapsed) rather than dropped -- dropping them would hide a genuine result about the loss.
#
# The cut is on the run's own acc_auc, which is normalised, and 0.10 is a gap in this sweep, not
# a round number: the collapsed runs sit at 0.021-0.065 and the worst healthy run at 0.153.
COLLAPSE_ACC_AUC = 0.10


def spearman_matrix(vectors):
    """(n, n) Spearman rho for n equal-length score vectors.

    Ranked ONCE per vector and then correlated as a Gram matrix, rather than calling
    scipy.stats.spearmanr on each of the n(n-1)/2 pairs. That is not just a speed-up: pairwise
    would re-rank each 3.2M-element vector n-1 times, ~400 sorts instead of 24 for the same
    numbers. rankdata's default 'average' tie handling is what makes this exactly Spearman
    (as opposed to a Pearson-on-ordinals approximation), and ties are not a corner case here --
    IG and IxG score positions 0 and 1 at exactly 0.0 across the whole layer stack, and a
    converged mask saturates, so tied blocks are a large fraction of some vectors.
    """
    R = np.empty((len(vectors), vectors[0].size), dtype=np.float64)
    for i, v in enumerate(vectors):
        R[i] = rankdata(v)
    R -= R.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(R, axis=1, keepdims=True)
    # A run whose scores are all identical has no rank order, so rho against it is undefined
    # rather than 0. Guard the divide and propagate NaN instead of emitting a confident zero.
    dead = norm[:, 0] == 0
    norm[dead] = 1.0
    R /= norm
    M = np.clip(R @ R.T, -1.0, 1.0)     # clip absorbs float error at the diagonal
    M[dead, :] = np.nan
    M[:, dead] = np.nan
    return M, dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substrate", default="mlp", choices=sorted({s for s, _ in SUBSTRATES}))
    ap.add_argument("--include-input", action="store_true")
    ap.add_argument("--wide", action="store_true",
                    help="add AttnLRP and Conductance (24 series instead of 18)")
    args = ap.parse_args()
    sub, inp = args.substrate, args.include_input
    if (sub, inp) not in SUBSTRATES:
        ap.error(f"--include-input was only swept at --substrate node, not {sub}")
    cfg = SUBSTRATES[(sub, inp)]
    res = Path(cfg["res"])
    methods = list(METHODS) + (WIDE if args.wide else [])

    # Series order: method-major, loss-minor. `series` is the y/x axis order, top-to-bottom.
    series = [(mk, lk) for mk, _ in methods for lk, _, _ in LOSSES]
    names = [f"{SHORT[mk]} ({LOSS_SHORT[lk]})" for mk, lk in series]
    assert len(set(names)) == len(names), "duplicate axis label"

    frames, marks, missing = [], [], []
    for task, tlabel in TASKS:
        vecs, total = [], None
        for mk, tmpl in methods:
            for lk, lsuf, _ in LOSSES:
                scores, meta = load_run(res, task, cfg["tag"], tmpl % lsuf)
                if scores is None:
                    missing.append(f"{task}/{tmpl % lsuf}")
                    vecs.append(None)
                    continue
                assert meta["_method"] == mk and meta.get("loss") == lk, \
                    f"{task}/{tmpl % lsuf} is {meta['_method']}/{meta.get('loss')}, not {mk}/{lk}"
                assert meta.get("nodes") == sub and meta["total"] == scores.numel()
                layout(meta, sub, inp)          # pins the flat layout, as in the table script
                # Equal length WITHIN a subtask is the precondition for the whole panel; a
                # mismatch here would mean two runs at the same subtask indexed different units.
                total = total if total is not None else scores.numel()
                assert scores.numel() == total, \
                    f"{task}: {tmpl % lsuf} has {scores.numel()} units, expected {total}"
                vecs.append(scores.to(torch.float64).numpy())
                aa = meta.get("acc_auc")
                if aa is not None and aa < COLLAPSE_ACC_AUC:
                    marks.append(dict(task=tlabel,
                                      row=f"{SHORT[mk]} ({LOSS_SHORT[lk]})",
                                      col=f"{SHORT[mk]} ({LOSS_SHORT[lk]})", acc_auc=aa))
        present = [i for i, v in enumerate(vecs) if v is not None]
        if len(present) < 2:
            continue
        M, dead = spearman_matrix([vecs[i] for i in present])
        for a, i in enumerate(present):
            for b, j in enumerate(present):
                frames.append(dict(task=tlabel, row=names[i], col=names[j], rho=M[a, b]))
        if dead.any():
            print(f"  ! {tlabel}: constant score vector in "
                  + ", ".join(names[present[a]] for a in np.flatnonzero(dead)), file=sys.stderr)

    df = pd.DataFrame(frames)
    df["task"] = pd.Categorical(df["task"], [t for _, t in TASKS])
    df["row"] = pd.Categorical(df["row"], names[::-1])   # first series at the TOP
    df["col"] = pd.Categorical(df["col"], names)
    mk_df = pd.DataFrame(marks, columns=["task", "row", "col", "acc_auc"])
    if len(mk_df):
        mk_df["task"] = pd.Categorical(mk_df["task"], [t for _, t in TASKS])
        mk_df["row"] = pd.Categorical(mk_df["row"], names[::-1])
        mk_df["col"] = pd.Categorical(mk_df["col"], names)
        print(f"  collapsed (acc-AUC < {COLLAPSE_ACC_AUC}), marked on the diagonal: "
              + ", ".join(f"{r.task}/{r.row} {r.acc_auc:.3f}" for r in mk_df.itertuples()))

    # What the figure is for, in numbers, so the claim in the caption is not read off the colours:
    # same method across the three losses (the on-diagonal 3x3 blocks, off-diagonal entries only)
    # vs different methods at the same loss.
    off = df[df["row"] != df["col"]].copy()
    off["m_r"] = off["row"].str.replace(r" \(.*", "", regex=True)
    off["m_c"] = off["col"].str.replace(r" \(.*", "", regex=True)
    off["l_r"] = off["row"].str.extract(r"\((.*)\)")
    off["l_c"] = off["col"].str.extract(r"\((.*)\)")
    same_m = off[off.m_r == off.m_c]
    same_l = off[(off.m_r != off.m_c) & (off.l_r == off.l_c)]
    print(f"[{cfg['out']}] {len(series)} series x {len(TASKS)} subtasks")
    print(f"  same method, different loss : mean rho {same_m.rho.mean():+.3f} "
          f"(median {same_m.rho.median():+.3f}, n={len(same_m)})")
    print(f"  different method, same loss : mean rho {same_l.rho.mean():+.3f} "
          f"(median {same_l.rho.median():+.3f}, n={len(same_l)})")
    for m in [SHORT[mk] for mk, _ in methods]:
        s = same_m[same_m.m_r == m]
        print(f"    {m:9s} across losses: {s.rho.mean():+.3f}")

    # Rules at the method boundaries. Continuous positions on a discrete scale sit BETWEEN
    # categories at the half-integers, and the y axis is reversed, so the two are mirrored.
    n, nl = len(series), len(LOSSES)
    cuts = [i + 0.5 for i in range(nl, n, nl)]
    p = (ggplot(df, aes("col", "row", fill="rho"))
         + geom_tile()
         + geom_vline(xintercept=cuts, size=0.25, color="#666666")
         + geom_hline(yintercept=[n - c for c in cuts], size=0.25, color="#666666")
         # Collapsed runs, on their own diagonal cell (always rho = 1, so nothing is hidden).
         + (geom_point(mk_df, aes("col", "row"), size=0.7, color="#000000", inherit_aes=False)
            if len(mk_df) else geom_blank())
         + facet_wrap("~task", ncol=2)
         + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                                midpoint=0, limits=[-1, 1], na_value="#eeeeee",
                                name="Spearman $\\rho$")
         + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
         + labs(x="", y="")
         + theme(figure_size=(6.0, 6.4) if len(series) <= 18 else (7.0, 7.4),
                 axis_text_x=element_text(size=5, rotation=90, hjust=1, vjust=0.5),
                 axis_text_y=element_text(size=5),
                 legend_position="right"))

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"sva_method_corr_{cfg['out'].replace('sva_top_', '')}" + ("_wide" if args.wide else "")
    p.save(OUT / f"{stem}.pdf", dpi=300, verbose=False)
    p.save(OUT / f"{stem}.png", dpi=150, verbose=False)
    print(f"wrote {OUT / stem}.pdf")
    if missing:
        print("MISSING runs:", ", ".join(missing))


theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        axis_title=element_text(size=7),
        legend_text=element_text(size=5.5),
        legend_title=element_text(size=6),
        legend_key_size=8,
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7, face="plain"),
    )
)

if __name__ == "__main__":
    main()
