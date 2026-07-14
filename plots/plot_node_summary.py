"""Concise node-granularity summaries: method x loss x metric x task-group.

Two encodings on the no-input node data (results/sva_sweep):
  A) fingerprint heatmap: rows = method x loss, cols = metric, facet = task-group; colour =
     per-metric "goodness" (higher=better, k* flipped), cell text = raw value.
  B) insight scatter: acc-AUC vs faith-AUC, colour=method, shape=loss, facet=task-group;
     gap-padding shows as logit_diff points riding high on faith at equal acc.

Tasks collapsed to 3 groups: SVA (mean of nounpp/rc/simple/within_rc), ARC-E, IOI.
Run:  uv run python plots/plot_node_summary.py
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, geom_text, geom_point, geom_hline, facet_grid, facet_wrap,
    labs, theme, theme_set, theme_bw, element_text, element_line, element_blank,
    scale_fill_gradientn, scale_color_brewer, scale_shape_manual,
)

theme_set(theme_bw(base_size=8) + theme(
    text=element_text(color="#000", family="Inter"), figure_size=(2.4, 1.7),
    axis_title=element_text(size=7), axis_text=element_text(size=6),
    panel_grid_major=element_line(size=0.25, color="#dddddd"), panel_grid_minor=element_blank(),
    panel_spacing_x=0.03, panel_spacing_y=0.03, strip_background=element_blank(),
    strip_text=element_text(size=7), legend_title=element_text(size=7),
    legend_text=element_text(size=6), legend_key_size=6, legend_position="top"))

RES = "results/sva_sweep"
METHODS = ["IG", "IxG", "soft-log", "soft-unif", "soft-fixed",
           "idSTE-log", "idSTE-unif", "idSTE-fixed"]
LOSSES = ["ce", "acc", "logit_diff"]
METRICS = [("acc_auc", "acc-AUC", True), ("faith_auc", "faith-AUC", True),
           ("kstar_50", "k*", False)]   # (key, label, higher_is_better)
SVA = {"nounpp", "rc", "simple", "within_rc"}


def parse_method(fname, d):
    tag = fname.split("_" + d["nodes"].replace("+", "-") + "_", 1)[1].rsplit(".json", 1)[0]
    if tag.startswith(("random", "conductance")):
        return None
    if "hard_topk" in tag:
        if re.search(r"_ig\d+", tag):
            return None
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
        return f"{fam}-{ks}"
    return "IxG" if tag.startswith("ixg") else "IG"


def taskgroup(d):
    return "SVA" if d["task"] in SVA else ("ARC-E" if d["task"] == "arc_easy" else "IOI")


def load():
    rows = []
    for f in glob.glob(RES + "/*.json"):
        d = json.load(open(f))
        if d["nodes"] != "node":
            continue
        m = parse_method(os.path.basename(f), d)
        if m not in METHODS:
            continue
        rows.append({"method": m, "loss": d["loss"], "grp": taskgroup(d),
                     "acc_auc": d["acc_auc"], "faith_auc": d["faith_auc"],
                     "kstar_50": d.get("kstar_50")})
    df = pd.DataFrame(rows)
    # collapse SVA (mean over its 4 tasks); kstar mean ignores None
    g = df.groupby(["method", "loss", "grp"], as_index=False).agg(
        acc_auc=("acc_auc", "mean"), faith_auc=("faith_auc", "mean"),
        kstar_50=("kstar_50", lambda s: np.nanmean([x if x is not None else np.nan for x in s])))
    return g


def goodness(df):
    """Per-metric min-max goodness in [0,1] (k* flipped, log-scaled)."""
    out = []
    for key, label, hib in METRICS:
        v = df[key].astype(float).copy()
        x = np.log10(v.clip(lower=1)) if key.startswith("kstar") else v
        lo, hi = np.nanmin(x), np.nanmax(x)
        gd = (x - lo) / (hi - lo + 1e-9)
        if not hib:
            gd = 1 - gd
        gd = gd.fillna(0.0)          # None/collapsed k* -> worst
        raw = [("%.2f" % r if not key.startswith("kstar") else
                ("%.0f" % r if np.isfinite(r) else "—")) for r in v]
        out.append(pd.DataFrame({"method": df["method"], "loss": df["loss"], "grp": df["grp"],
                                 "metric": label, "goodness": gd.values, "raw": raw}))
    return pd.concat(out, ignore_index=True)


def heatmap(g):
    h = goodness(g)
    ml = [f"{m} / {l}" for m in METHODS for l in LOSSES]
    h["ml"] = h["method"] + " / " + h["loss"]
    h["ml"] = pd.Categorical(h["ml"], categories=list(reversed(ml)), ordered=True)
    h["metric"] = pd.Categorical(h["metric"], categories=[l for _, l, _ in METRICS], ordered=True)
    h["grp"] = pd.Categorical(h["grp"], categories=["SVA", "ARC-E", "IOI"], ordered=True)
    p = (ggplot(h, aes("metric", "ml", fill="goodness"))
         + geom_tile(color="white", size=0.4)
         + geom_text(aes(label="raw"), size=5)
         + facet_grid(". ~ grp")
         + scale_fill_gradientn(colors=["#d73027", "#fee08b", "#1a9850"], limits=[0, 1])
         + labs(x="", y="", title="")
         + theme(figure_size=(6.0, 5.2), axis_text_x=element_text(size=6),
                 legend_position="none"))
    p.save(RES + "/node_fingerprint.pdf", verbose=False)
    p.save("/tmp/node_fingerprint.png", dpi=140, verbose=False)
    print("wrote", RES + "/node_fingerprint.pdf")


def scatter(g):
    meth = ["IG", "IxG", "soft-log", "idSTE-log"]   # drop unif (and fixed-k: interaction figure)
    llab = {"ce": "CE", "acc": "acc", "logit_diff": "logit-diff"}
    d = g[g["method"].isin(meth)].copy()
    d["loss"] = pd.Categorical(d["loss"].map(llab), categories=list(llab.values()), ordered=True)
    d["method"] = pd.Categorical(d["method"], categories=meth, ordered=True)
    d["grp"] = pd.Categorical(d["grp"], categories=["SVA", "ARC-E", "IOI"], ordered=True)
    p = (ggplot(d, aes("acc_auc", "faith_auc", color="loss", shape="method"))
         + geom_hline(yintercept=1.0, linetype="dashed", color="#888888", size=0.3)  # overshoot
         + geom_point(size=2.0, alpha=0.9)
         + facet_wrap("~ grp")
         + scale_color_brewer(type="qual", palette="Dark2")
         + scale_shape_manual(values=["o", "^", "s", "D"])
         + labs(x="Accuracy AUC ($\\uparrow$)", y="Faithfulness AUC",
                color="Loss", shape="Method")
         + theme(figure_size=(5.5, 2.2)))
    p.save(RES + "/node_scatter.pdf", verbose=False)
    p.save("/tmp/node_scatter.png", dpi=150, verbose=False)
    print("wrote", RES + "/node_scatter.pdf")


if __name__ == "__main__":
    g = load()
    print(f"loaded {len(g)} (method,loss,group) cells")
    heatmap(g)
    scatter(g)
