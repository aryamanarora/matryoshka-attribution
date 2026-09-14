"""MAttr on the Bilodeau et al. (2022) impossibility benchmark: ROC panel + budget curve.

Reads `results/impossibility/*.json` (from `scripts/impossibility/impossibility_bench.py`) and writes

  paper/figs/impossibility_roc.pdf      ROC per end-task x READOUT, pooled over models and
                                        examples and vertically averaged over the five UCI
                                        datasets. The readout row is the point of the figure:
                                        under the paper's own per-example min-max normalization
                                        the spurious panel collapses to the diagonal for EVERY
                                        method including brute force, which estimates the
                                        oracle's own statistic -- the spurious oracle is a
                                        GLOBAL variance threshold, so a per-example rescaling
                                        removes the information the test asks about.
  paper/figs/impossibility_budget.pdf   ROC AUC vs forward-passed rows, global readout only
                                        (under theirs there is nothing to trace -- every curve
                                        sits on the floor).

COLOUR. This figure has more series than `plots/palette.py` carries names for, and the palette
module's whole point is that new competing hues go through its CVD check rather than getting
invented per figure. So nothing new is defined here: the series reuse existing METHOD hexes,
chosen so the FAMILY reads off the colour temperature the way it does everywhere else in the
paper -- cool/neutral for the mask-search methods (ours + the brute-force reference), warm for
the paper's gradient and complete-and-linear baselines, grey for the random floor. The specific
aliasing (SHAP -> the I×G wine, Gradient -> the AttnLRP vermillion, ...) is a within-figure
label mapping, not a claim that those are the same method, and every series is legend-labelled.

    uv run python plots/plot_impossibility.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (aes, element_blank, element_line, element_text, facet_grid,
                      facet_wrap, geom_abline,
                      geom_hline, geom_line, geom_point, ggplot, labs, scale_color_manual,
                      scale_x_log10, scale_y_continuous, theme, theme_bw, theme_set)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "impossibility"))
import palette                                              # noqa: E402
from impossibility_report import TASKS, auc, collect, rung  # noqa: E402

RESULTS = Path("results/impossibility")
EPS_RESULTS = Path("results/impossibility_eps")
FIGS = Path("paper/figs")
# arms that only exist in the eps-sweep dir
EPS_ARMS = {"mattr_adam_eps0.1"}

TASK_LABEL = {"recourse": "Recourse", "spurious": "Spurious features"}
READOUT_LABEL = {"paper": "Per-example min-max (theirs)", "raw": "Global scale"}
READOUT_ORDER = [READOUT_LABEL["paper"], READOUT_LABEL["raw"]]

# display name -> (palette key, arm key in the results). See the COLOUR note above.
# `mattr_adam` in RESULTS is Adam at the LIBRARY-DEFAULT eps=1e-8; the tuned arm lives in
# EPS_RESULTS (scripts/impossibility/impossibility_bench.py --eps-sweep). The paper's encoding already
# distinguishes those two -- blue is the Adam configuration we recommend, violet the default-
# eps one -- so both are drawn rather than silently quoting whichever is on hand.
SERIES = [
    ("MAttr (SGD)",   "MAttr (SGD)",   "mattr_sgd"),
    ("MAttr (Adam)",  "MAttr",         "mattr_adam_eps0.1"),
    ("MAttr (Adam, def. eps)", "MAttr (Adam, default eps)", "mattr_adam"),
    ("Brute force",   "Node Pruning",  "bruteforce"),
    ("SHAP",          "I×G",           "shap"),
    ("IG (min)",      "IG",            "ig_min"),
    ("Gradient",      "AttnLRP",       "grad"),
    ("SmoothGrad",    "Expected Gradients",   "smoothgrad"),
    ("Random",        "Random",        "random"),
]
COLORS = {name: palette.METHOD[key] for name, key, _ in SERIES}
ORDER = [n for n, _, _ in SERIES]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.4, 2.0),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

FPR_GRID = np.linspace(0, 1, 201)
_SUP = str.maketrans("0123456789-", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b")


def log_label(breaks):
    """10^n with Unicode superscripts -- LaTeX in a tick label would swap the font."""
    out = []
    for b in breaks:
        if b is None or not np.isfinite(b):
            out.append("")
        else:
            e = int(round(np.log10(b)))
            out.append("1" if e == 0 else "10" + str(e).translate(_SUP))
    return out


def roc(scores, labels):
    """(fpr, tpr) of the threshold sweep, ties handled by sorting on the score."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    o = np.argsort(-s, kind="mergesort")
    tp = np.cumsum(y[o])
    fp = np.cumsum(1 - y[o])
    return (np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)])


def load(at_budget, readout):
    """-> (roc_df, budget_df). `at_budget` picks which MAttr/brute-force rung the ROC panel
    shows; the budget panel shows every rung."""
    arms = {}
    for name, _, key in SERIES:
        arms[name] = key
    roc_rows, bud_rows = [], []
    for path in sorted(RESULTS.glob("*.json")):
        d, acc, bud = collect(path, readout)
        eps_path = EPS_RESULTS / path.name
        if eps_path.exists():
            _, eacc, ebud = collect(eps_path, readout)
            for task in TASKS:
                for k in EPS_ARMS:
                    for kk in [z for z in eacc[task] if z.split("@")[0] == k]:
                        acc[task][kk] = eacc[task][kk]
                        bud[kk] = ebud[kk]
        for task in TASKS:
            per = acc[task]
            for disp, key in arms.items():
                # fixed-budget baselines carry no "@"; MAttr / brute force are rungs
                cands = ([key] if key in per else
                         sorted((k for k in per if k.startswith(key + "@")),
                                key=lambda k: int(k.split("@")[1])))
                for k in cands:
                    s = sum((per[k][m][0] for m in per[k]), [])
                    l = sum((per[k][m][1] for m in per[k]), [])
                    a = auc(s, l)
                    if np.isnan(a):
                        continue
                    b = bud[k]
                    bud_rows.append({"dataset": path.stem, "task": TASK_LABEL[task],
                                     "readout": READOUT_LABEL[readout], "method": disp,
                                     "rung": rung(k, b, d["p"]), "budget": max(b, 0.5),
                                     "auc": a})
                    # ROC only for the quoted rung. Brute force's rungs are n*p, so they
                    # never land exactly on a power of two -- take each family's CLOSEST
                    # rung rather than requiring equality, or it drops out of the panel.
                    if len(cands) > 1:
                        near = min(cands, key=lambda z: abs(int(z.split("@")[1]) - at_budget))
                        if k != near:
                            continue
                    f, t = roc(s, l)
                    roc_rows.append(pd.DataFrame({
                        "dataset": path.stem, "task": TASK_LABEL[task],
                        "readout": READOUT_LABEL[readout], "method": disp,
                        "fpr": FPR_GRID, "tpr": np.interp(FPR_GRID, f, t)}))
    return pd.concat(roc_rows), pd.DataFrame(bud_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS))
    ap.add_argument("--eps-results", default=str(EPS_RESULTS))
    ap.add_argument("--figs", default=str(FIGS))
    ap.add_argument("--at-budget", type=int, default=1024)
    args = ap.parse_args()
    globals()["RESULTS"] = Path(args.results)
    globals()["EPS_RESULTS"] = Path(args.eps_results)
    globals()["FIGS"] = Path(args.figs)
    both = [load(args.at_budget, ro) for ro in ("paper", "raw")]
    roc_df = pd.concat([x[0] for x in both])
    bud_df = pd.concat([x[1] for x in both])
    FIGS.mkdir(parents=True, exist_ok=True)

    # ROC: vertical average of the per-dataset curves on a common FPR grid
    r = roc_df.groupby(["readout", "task", "method", "fpr"], as_index=False)["tpr"].mean()
    r["method"] = pd.Categorical(r["method"], categories=ORDER, ordered=True)
    r["readout"] = pd.Categorical(r["readout"], categories=READOUT_ORDER, ordered=True)
    r = r.sort_values("method", ascending=False)      # ours drawn last, i.e. on top
    p = (ggplot(r, aes("fpr", "tpr", color="method"))
         + geom_abline(intercept=0, slope=1, color="#999999", size=0.3, linetype="dashed")
         + geom_line(size=0.6)
         + facet_grid("readout~task")
         + scale_color_manual(values=COLORS)
         + scale_y_continuous(limits=(0, 1))
         + theme(figure_size=(5.4, 4.0), strip_text_y=element_text(size=6.5))
         + labs(x="False positive rate", y="True positive rate"))
    p.save(FIGS / "impossibility_roc.pdf", verbose=False)

    # budget: mean over datasets, AUC vs measured forward-passed rows
    b = bud_df[bud_df["readout"] == READOUT_LABEL["raw"]]
    # average over datasets at matched RUNGS, then place the point at the mean row count
    b = b.groupby(["task", "method", "rung"], as_index=False).agg(
        auc=("auc", "mean"), budget=("budget", "mean"))
    b["method"] = pd.Categorical(b["method"], categories=ORDER, ordered=True)
    LINES = ["MAttr (SGD)", "MAttr (Adam)", "MAttr (Adam, def. eps)", "Brute force"]
    curves = b[b["method"].isin(LINES)]
    points = b[~b["method"].isin(LINES + ["Random"])]
    rnd = b[b["method"] == "Random"]
    p2 = (ggplot(mapping=aes("budget", "auc", color="method"))
          + geom_hline(rnd, aes(yintercept="auc"), color=COLORS["Random"],
                       size=0.4, linetype="dashed")
          + geom_line(curves, size=0.6)
          + geom_point(curves, size=0.8)
          + geom_point(points, size=1.8, stroke=0.5)
          + facet_wrap("~task")
          + scale_color_manual(values=COLORS)
          + scale_x_log10(labels=log_label)
          + labs(x="Model evaluations (rows)", y="ROC AUC"))
    p2.save(FIGS / "impossibility_budget.pdf", verbose=False)
    print(f"wrote {FIGS/'impossibility_roc.pdf'} and {FIGS/'impossibility_budget.pdf'}")


if __name__ == "__main__":
    main()
