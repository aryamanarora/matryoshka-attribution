"""Mean rank per method over the SVA+ sweep -- the compact replacement for the four
`tabs/fingerprint_*` tables.

WHY RANKS. The fingerprint tables are 4 tables x 42 rows x 24--27 columns ~ 4000 heat-coloured
numbers, and the colour (not the digits) is what a reader actually uses. Collapsing them needs a
statistic that survives this data's three pathologies, which a mean over cells does not:

  1. faith-AUC is UNBOUNDED and gap-paddable -- a logit-diff-trained circuit can inflate it by
     widening the clean-corrupt logit gap without recovering the decision (values >1 are common
     in the mlp tables). A mean is dragged by those; a rank is not.
  2. k* is a percentage that SATURATES at 100 whenever a method never reaches the 0.5 threshold,
     so its mean mixes a measurement with a censoring indicator.
  3. Coverage is RAGGED while the sweep runs, and the substrates carry different task sets
     (ARC-E/IOI are node-only -- structural, see mib-tasks-node-substrate-only).

Ranks are invariant to any monotone rescaling of a metric, which disposes of (1) and (2)
outright. (3) is handled by COMPLETE-CASE ranking, below.

THE UNIT is a cell = (substrate, task, loss), and methods are ranked WITHIN a cell -- i.e. at
matched training loss on matched data. Matched-loss is the honest default; letting each method
take its best loss per cell is an oracle that rewards whichever method has the most variants on
disk (the same rule scripts/method_winrate.py argues for at length).

COMPLETE-CASE: a cell contributes only if EVERY plotted arm ran in it. Ranking over whichever
arms happen to be on disk would silently reward a method for the cells it is missing -- drop a
method from a cell it would have lost and every survivor's rank improves. Dropped cells are
printed, and the retained count per panel is drawn on the figure, so a thin panel is visible as
such rather than reading like a complete result.

SETTINGS ARE NEVER POOLED: this figure is the patched, -input sweep only (results/sva_sweep).
Zero-ablation is a different experiment -- MAttr and the mask baselines retrain through the
intervention they are scored under, and the gradient baselines change estimator -- so a cell
there is not comparable to a cell here (Spearman ~0.44 on matched cells).

CIs are a nonparametric bootstrap over CELLS (the resampling unit is the cell, not the
individual rank, so the correlation between the arms inside a cell is preserved). Overlapping
intervals are NOT a significance test between two arms -- they bound each arm's own mean rank.
Read "these two are not separated" from overlap, never "these two differ" from non-overlap.

THREE FACETINGS, one ranking. None of them changes how a rank is COMPUTED -- the loss is part
of the cell in every mode, so an arm is only ever compared against arms trained on the SAME
loss. They differ only in what happens after:

  `--by armloss` (DEFAULT) -- substrate x metric, competitor = (method, loss), 39 rows. Nothing
      is averaged across losses; each loss keeps its own row.
  `--by loss`              -- substrate x loss for ONE metric. Nothing is averaged across
      losses; the loss is the facet.
  `--by metric`            -- substrate x metric, averaging each arm's ranks over all three
      losses. DO NOT USE THIS FOR THE PAPER. The loss is not a nuisance dimension here: the
      three losses induce genuinely different circuits (see the ARITH split, where MAttr+SGD's
      win over Adam at the neuron substrates is +0.22 acc-AUC under CE and +0.07 under acc, and
      where it beats IG only under logit-diff and loses 0/4 in every CE and acc cell). Averaging
      those into one number reports a method that nobody runs. Kept only because the pooled
      view is occasionally useful as a sanity check that a conclusion is not loss-specific --
      it prints a warning and writes to `method_ranks_pooled.pdf`, never to paper/figs/.

Three losses x three metrics is 27 panels, which is why `--by loss` is a separate figure rather
than a third facet dimension.

Run:  uv run python plots/plot_method_ranks.py
      uv run python plots/plot_method_ranks.py --by loss                 # acc-AUC by default
      uv run python plots/plot_method_ranks.py --by loss --metric kstar_pct
      uv run python plots/plot_method_ranks.py --out paper/figs/method_ranks.pdf
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from plotnine import (aes, element_blank, element_text, facet_grid, facet_wrap, geom_errorbarh,
                      geom_point, geom_text, ggplot, labs, scale_color_manual,
                      scale_x_continuous, scale_y_discrete, theme, theme_bw)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                   # noqa: E402
from plot_accauc_vs_faithauc import parse_method, on_model  # noqa: E402  (one tag parser, not two)

RES = "results/sva_sweep"
# (metric key, panel label, higher_is_better). k* is stored as an absolute unit count; the
# tables normalise it by `total`, so do the same here or the node and mlp substrates would be
# ranked on different scales.
METRICS = [("acc_auc", "acc-AUC (↑)", True),
           ("faith_auc", "faith-AUC (↑)", True),
           ("kstar_pct", "$k^\\star$% (↓)", False)]
SUBSTRATES = [("node", "Node"), ("mlp", "MLP"), ("mlp+attn_head", "MLP+Attn")]
# Training loss, in the sweep's own sharpness order (softest signal -> hardest), matching
# plot_accauc_vs_faithauc.LOSS_PATH so the two figures read left-to-right the same way.
LOSSES = [("ce", "CE"), ("acc", "acc"), ("logit_diff", "logit-diff")]
# Arms, top to bottom. Same keys and same order as make_fingerprint_tables.SECTIONS flattened,
# so this figure and the table it replaces describe exactly the same runs.
ARMS = [("stopk-log", r"MAttr"), ("stopk-unif", r"MAttr $+$ unif $k$"),
        ("softsgd-log", r"MAttr $+$ SGD"), ("softsgd-unif", r"MAttr $+$ SGD, unif $k$"),
        ("soft-log", r"$+$ hard"), ("soft-unif", r"$+$ hard, unif $k$"),
        ("idSTE-log", r"$+$ id-STE"), ("idSTE-unif", r"$+$ id-STE, unif $k$"),
        ("IG", "IG"), ("IxG", "I×G"), ("AttnLRP", "AttnLRP"),
        ("eprun-s090", "Node Pruning"), ("sig_lr0.3_l16.0", "DBM")]
# Colour = method FAMILY, not arm: the k-schedule and optimizer variants of one method share a
# hue so the eye groups them, exactly as palette.py already does for MAttr vs MAttr (SGD).
FAMILY = {"stopk-log": "MAttr", "stopk-unif": "MAttr",
          "softsgd-log": "MAttr", "softsgd-unif": "MAttr",
          "soft-log": "+hard", "soft-unif": "+hard",
          "idSTE-log": "+hard", "idSTE-unif": "+hard",
          "IG": "IG", "IxG": "I×G", "AttnLRP": "AttnLRP",
          "eprun-s090": "Node Pruning", "sig_lr0.3_l16.0": "DBM"}
N_BOOT = 2000
SEED = 0
# Separator for the reorder_within trick (see --sort within); must never occur in an arm label.
SEP = "\x00"


def load(res=RES):
    """(substrate, task, loss) -> {arm: {metric: value}}, over every arm in ARMS."""
    cells = {}
    keys = {k for k, _ in ARMS}
    for f in glob.glob(res + "/*.json"):
        d = json.load(open(f))
        m = parse_method(os.path.basename(f), d)
        if m not in keys:
            continue
        # The cell key below has no model in it, and results/sva_sweep holds a llama3 IOI wave
        # alongside the canonical qwen2.5 one -- without this the IOI cells' ranks would depend on
        # glob order. Ranking is WITHIN a cell, so a mixed cell silently ranks two models' runs.
        if not on_model(d):
            continue
        ks = d.get("kstar_50")
        cells.setdefault((d["nodes"], d["task"], d["loss"]), {})[m] = {
            "acc_auc": d["acc_auc"], "faith_auc": d["faith_auc"],
            # k* censors at `total` when the method never reaches the 0.5 threshold. Kept as the
            # tables keep it -- the censoring is real information (the circuit never recovers the
            # decision), and under a RANK it degrades gracefully to "last", which is the right
            # answer, instead of dragging a mean the way it does in the tables.
            "kstar_pct": 100.0 * (ks if ks is not None else d["total"]) / d["total"]}
    return cells


def by_task(cells):
    """Regroup (substrate, task, loss) -> {arm: ...} into (substrate, task) -> {(arm, loss): ...}.

    This is the `--by armloss` view: the TRAINING LOSS is folded into the competitor identity,
    so a cell is a (substrate, task) and 13 arms x 3 losses = 39 competitors are ranked against
    each other in it. That deliberately breaks the matched-loss discipline the module docstring
    argues for, and it answers a different question. Matched-loss asks "given this loss, which
    method?"; this asks "which (method, loss) pair should I run?", which is the question a
    practitioner actually faces and the one the fingerprint tables let you answer by eye, since
    they print raw values side by side across losses. Both are honest; only the second lets a
    method be beaten by a rival trained differently, so it is the harsher of the two.
    """
    out = {}
    for (sub, task, loss), per_arm in cells.items():
        for arm, vals in per_arm.items():
            out.setdefault((sub, task), {})[(arm, loss)] = vals
    return out


def ranks(cells, sub, metric, higher, arms):
    """Per-cell ranks (1 = best) over the complete-case cells of one substrate.

    Returns (arm-ordered rank matrix, kept cell keys, dropped cell keys). Ties get the average
    rank (scipy-free: rankdata's 'average' method), so two arms that produce the identical
    circuit share the rank rather than one arbitrarily winning.
    """
    kept, dropped, rows = [], [], []
    for key, per_arm in sorted(cells.items()):
        if key[0] != sub:
            continue
        if not all(a in per_arm for a in arms):
            dropped.append((key, [a for a in arms if a not in per_arm]))
            continue
        v = np.array([per_arm[a][metric] for a in arms], float)
        order = -v if higher else v            # ascending sort -> rank 1 is best
        # average ranks for ties
        idx = np.argsort(order, kind="stable")
        r = np.empty(len(v))
        r[idx] = np.arange(1, len(v) + 1)
        for val in np.unique(order):
            tie = order == val
            if tie.sum() > 1:
                r[tie] = r[tie].mean()
        rows.append(r)
        kept.append(key)
    return (np.array(rows) if rows else np.zeros((0, len(arms)))), kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=RES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--by", choices=["metric", "loss", "armloss"], default="armloss",
                    help="'armloss' = substrate x metric with (method, loss) as the competitor, "
                         "39 rows; 'loss' = substrate x loss for one --metric; 'metric' = "
                         "substrate x metric with ranks AVERAGED OVER LOSSES (sanity check "
                         "only, not for the paper -- see module docstring)")
    ap.add_argument("--metric", choices=[m for m, _, _ in METRICS], default="acc_auc",
                    help="only used with --by loss; acc-AUC is the default because it is the "
                         "bounded one -- faith-AUC is gap-paddable and so is itself "
                         "loss-dependent, which would confound the very axis being drawn")
    # k* is NEARLY REDUNDANT with acc-AUC here: per-cell Spearman between the two rankings is
    # +0.90 (node) / +0.96 (mlp) / +0.96 (mlp+attn), so its column costs a third of the figure
    # to move points by an rms of 1.5 ranks out of 39. Dropping it is a flag, not an edit,
    # because the residual is SYSTEMATIC rather than noise and runs against us: IG sits ~2-3
    # ranks better on k* than on acc-AUC and the MAttr variants ~2-4 ranks worse (max 4.6),
    # i.e. MAttr's lead lives in the whole recovery curve and IG partly closes it at the
    # specific 50%-recovery threshold. Keep the column for any claim pinned to that threshold.
    ap.add_argument("--metrics", default=",".join(m for m, _, _ in METRICS),
                    help="comma-separated metric columns; use acc_auc,faith_auc to drop k*")
    # Per-panel y ordering. The shared order (default) makes the y axis a fixed reference, so a
    # method's vertical position is the same everywhere and the eye reads DIFFERENCES between
    # panels -- e.g. that MAttr + unif k sits high at Node and low at MLP. Sorting within a panel
    # instead makes each panel a clean leaderboard, at the cost of that cross-panel comparison
    # and of one label column per panel. Both are useful; neither is right in general.
    ap.add_argument("--sort", choices=["global", "within"], default="global",
                    help="'within' sorts each panel by its own mean rank (best on top)")
    a = ap.parse_args()
    out = a.out or {"metric": "plots/method_ranks_pooled.pdf",
                    "loss": f"plots/method_ranks_by_loss_{a.metric}.pdf",
                    "armloss": "plots/method_ranks.pdf"}[a.by]
    if a.by == "metric":
        print("WARNING: --by metric averages each arm's ranks ACROSS the three training losses.\n"
              "         The losses induce different circuits and the method ordering is not\n"
              "         stable across them, so a pooled row describes a run nobody performs.\n"
              "         Sanity check only -- use --by armloss or --by loss for the paper.")
    want = a.metrics.split(",")
    metrics = ([m for m in METRICS if m[0] in want] if a.by in ("metric", "armloss")
               else [m for m in METRICS if m[0] == a.metric])

    cells = load(a.res)
    rng = np.random.default_rng(SEED)
    llabel = dict(LOSSES)
    if a.by == "armloss":
        # Competitor = (method, loss). Ordered method-major so a method's three losses sit in
        # one contiguous band and the within-method loss spread is read vertically, while the
        # between-method comparison is still the coarse top-to-bottom sweep.
        cells = by_task(cells)
        arms = [(k, ls) for k, _ in ARMS for ls, _ in LOSSES]
        labels = {(k, ls): f"{lb}  ({llabel[ls]})" for k, lb in ARMS for ls, _ in LOSSES}
        family = {(k, ls): FAMILY[k] for k, _ in ARMS for ls, _ in LOSSES}
    else:
        arms = [k for k, _ in ARMS]
        labels = dict(ARMS)
        family = FAMILY
    rows, notes, drop_log = [], [], []
    for sub, slabel in SUBSTRATES:
        for metric, mlabel, higher in metrics:
            R, kept, dropped = ranks(cells, sub, metric, higher, arms)
            drop_log += [(sub, metric, k, miss) for k, miss in dropped]
            # Ranks are computed over ALL kept cells above and only SPLIT here -- the loss is
            # part of the cell, so an arm was never compared against a differently-trained one
            # in either mode. `--by loss` stops pooling at the aggregation step, nothing more.
            if a.by in ("metric", "armloss"):
                groups = [(mlabel, np.arange(len(kept)))]
            else:
                groups = [(llabel, np.array([i for i, k in enumerate(kept) if k[2] == loss]))
                          for loss, llabel in LOSSES]
            for col, idx in groups:
                if len(idx) == 0:
                    notes.append((slabel, col, 0))
                    continue
                Rg = R[idx]
                # Bootstrap over CELLS (rows of Rg), which keeps the within-cell correlation
                # between arms -- resampling individual ranks would break it and give intervals
                # that are far too narrow.
                boot = Rg[rng.integers(0, len(Rg), size=(N_BOOT, len(Rg)))].mean(axis=1)
                lo, hi = np.percentile(boot, [(100 - a.ci) / 2, 100 - (100 - a.ci) / 2], axis=0)
                for i, arm in enumerate(arms):
                    rows.append(dict(arm=labels[arm], family=family[arm], substrate=slabel,
                                     metric=col, mean=Rg[:, i].mean(), lo=lo[i], hi=hi[i]))
                notes.append((slabel, col, len(idx)))
    if not rows:
        print("no complete-case cells -- nothing to draw")
        return
    cols = ([m for _, m, _ in metrics] if a.by in ("metric", "armloss")
            else [lb for _, lb in LOSSES])
    df = pd.DataFrame(rows)
    df["arm"] = pd.Categorical(df["arm"], [labels[k] for k in arms][::-1])   # first arm on top
    df["substrate"] = pd.Categorical(df["substrate"], [s for _, s in SUBSTRATES])
    df["metric"] = pd.Categorical(df["metric"], cols)

    if a.sort == "within":
        # tidytext's reorder_within, done by hand: give each (panel, arm) its own y level, order
        # the levels panel-major then by mean rank, and strip the disambiguating suffix at the
        # tick labeller. facet_GRID cannot do this -- its free scales are per ROW, so the three
        # metric columns of one substrate would still share an order -- hence the switch to
        # facet_wrap, whose free scales are genuinely per panel. SEP must be a character that
        # cannot occur in an arm label; \x00 is the safe choice since the labels are LaTeX.
        df["panel"] = df["substrate"].astype(str) + ", " + df["metric"].astype(str)
        panels = [f"{s}, {m}" for s, _ in [(s, 0) for _, s in SUBSTRATES] for m in cols]
        df["panel"] = pd.Categorical(df["panel"], [p for p in panels if p in set(df["panel"])])
        df["arm"] = df["arm"].astype(str) + SEP + df["panel"].astype(str)
        # descending by mean within panel: plotnine puts the FIRST category at the bottom, so
        # the largest (worst) mean rank must come first for rank 1 to land on top.
        lv = (df.sort_values(["panel", "mean"], ascending=[True, False])["arm"]).tolist()
        df["arm"] = pd.Categorical(df["arm"], lv)

    # n per panel, drawn in the corner. A panel resting on three cells must not look like one
    # resting on thirty-six.
    nlab = pd.DataFrame([dict(substrate=s, metric=m, n=n) for s, m, n in notes if n]).astype(
        {"substrate": df["substrate"].dtype, "metric": df["metric"].dtype})
    # Anchored to the TOP row, not the bottom: the bottom arm (DBM) sits at the far right of
    # every panel, which is exactly where a right-aligned corner label lands, and the two
    # collided. The top arm is a MAttr variant and is always at the left, so the corner is free.
    nlab["mean"] = len(arms)
    nlab["label"] = "n=" + nlab["n"].astype(str)
    if a.sort == "within":
        # Each panel has its own levels now, so "the top row" is panel-specific: take the last
        # level belonging to that panel. Sorted panels put the BEST arm on top, and the best arm
        # is at the far left, so the right-hand corner stays free here too.
        nlab["panel"] = nlab["substrate"].astype(str) + ", " + nlab["metric"].astype(str)
        nlab["panel"] = pd.Categorical(nlab["panel"], df["panel"].cat.categories)
        top = {p: [c for c in df["arm"].cat.categories if c.endswith(SEP + p)][-1]
               for p in df["panel"].cat.categories}
        nlab["arm"] = pd.Categorical(nlab["panel"].map(top), df["arm"].cat.categories)
    else:
        nlab["arm"] = df["arm"].cat.categories[-1]

    colors = {f: P.color(f) for f in df["family"].unique()}
    p = (
        ggplot(df, aes("mean", "arm", color="family"))
        + geom_errorbarh(aes(xmin="lo", xmax="hi"), height=0, size=0.5)
        + geom_point(size=1.6)
        # va="top": anchored to the topmost arm and drawn DOWNWARD. "bottom" put it above that
        # row, i.e. outside the panel, where the facet strip clipped it.
        + geom_text(aes("mean", "arm", label="label"), data=nlab, color="#666666",
                    size=5, ha="right", va="top", inherit_aes=False)
        + (facet_wrap("panel", ncol=len(cols), scales="free") if a.sort == "within"
           else facet_grid("substrate ~ metric", scales="free_x"))
        + scale_y_discrete(labels=lambda bs: [b.split(SEP)[0] for b in bs])
        # Spell out what is being averaged over: `--by loss` panels average over TASKS only, so
        # they rest on a third of the cells and their intervals are correspondingly wider --
        # that widening is the split, not a change in the data.
        + scale_x_continuous(name={
            "metric": f"mean rank over cells (1 = best of {len(arms)}), "
                      f"{a.ci:.0f}% bootstrap CI",
            "armloss": f"mean rank over tasks (1 = best of {len(arms)} method$\\times$loss "
                       f"pairs), {a.ci:.0f}% bootstrap CI",
            "loss": f"{dict((m, lb) for m, lb, _ in METRICS)[a.metric]}: mean rank over tasks "
                    f"(1 = best of {len(arms)}), {a.ci:.0f}% bootstrap CI"}[a.by])
        + scale_color_manual(values=colors, guide=None)   # arm names are the y axis already
        + labs(y="")
        + theme_bw()
        # 39 rows need the height; 13 do not. `--sort within` also needs WIDTH: per-panel
        # ordering means per-panel tick labels, so every column carries its own ~30-character
        # label block instead of one shared block down the left edge.
        + theme(figure_size=(7.0 + (3.1 * len(cols) if a.sort == "within" else 0.0),
                             10.5 if a.by == "armloss" else 5.2), dpi=300,
                text=element_text(size=7),
                axis_text_y=element_text(size=5 if a.by == "armloss" else 6),
                strip_background=element_blank(),
                strip_text=element_text(size=7), panel_grid_minor=element_blank())
    )
    p.save(out, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print("wrote", out)

    print(f"\ncomplete-case cells per panel (substrate x {a.by}):")
    for slabel, mlabel, n in notes:
        print(f"  {slabel:<10} {mlabel:<16} n={n}")
    if drop_log:
        # The to-run list. A cell is dropped for ONE missing arm, so this is usually a coverage
        # hole in a single sweep rather than a broken cell -- print which arm so it is actionable.
        miss = {}
        for sub, metric, key, arms_missing in drop_log:
            for m in arms_missing:
                miss.setdefault(m if isinstance(m, str) else "/".join(m), set()).add(key)
        print(f"\nDROPPED cells by the arm that is missing (complete-case rule):")
        for m, ks in sorted(miss.items(), key=lambda kv: -len(kv[1])):
            ex = sorted(ks)[:3]
            print(f"  {m:<28} {len(ks):3d} cells   e.g. " +
                  ", ".join("/".join(map(str, k)) for k in ex))


if __name__ == "__main__":
    main()
