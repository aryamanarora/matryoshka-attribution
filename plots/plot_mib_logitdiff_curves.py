"""Appendix full-width per-task curves of RAW logit diff (un-normalised faithfulness) over the
MIB denoising sparsity sweep, for the 5 key node methods, faceted by task/model.

Raw logit diff is NOT stored in the result pkls (only the normalised faithfulness is). But
faithfulness = (raw - C) / (B - C) with method-independent anchors B=baseline (clean) and
C=corrupted (all-ablated). So we invert the stored faithfulness for every method:
    raw(pct) = C + faithfulness(pct) * (B - C)
using anchors computed once per task/model by scripts/eval_mib_anchors.py -> results/anchors/.

Dashed line = clean baseline B (raw diff of the full model); dotted line = corrupted floor C.
Curves above B are absolute-terms gap-padding (ablated circuit beats the clean model).
Run:  uv run python plots/plot_mib_logitdiff_curves.py  ->  plots/mib_logitdiff_curves.pdf
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import palette as P
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_wrap, labs, theme,
    theme_set, theme_bw, element_text, element_line, element_blank,
    scale_color_manual, scale_x_log10,
)

R = Path("results")
MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
ANCH = R / "anchors"
OUT = Path("plots")
PCT = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

COLUMNS = [
    ("ioi", "gpt2", "IOI (GPT-2)"), ("ioi", "qwen2.5", "IOI (Qwen)"),
    ("ioi", "gemma2", "IOI (Gemma)"), ("ioi", "llama3", "IOI (Llama)"),
    ("arithmetic_subtraction", "llama3", "Arithmetic (Llama)"),
    ("mcqa", "qwen2.5", "MCQA (Qwen)"), ("mcqa", "gemma2", "MCQA (Gemma)"),
    ("mcqa", "llama3", "MCQA (Llama)"),
    ("arc_easy", "gemma2", "ARC-E (Gemma)"), ("arc_easy", "llama3", "ARC-E (Llama)"),
    ("arc_challenge", "llama3", "ARC-C (Llama)"),
]
FACET_ORDER = [c[2] for c in COLUMNS]

METHODS = [
    ("MAttr", P.color("MAttr"), "mattr", "topklog_lr_0.05"),
    ("+hard", P.color("+hard"), "mattr", "htklog_lr_0.05"),
    ("IG",    P.color("IG"), "base",  ("napig_ref_accauc", "EAP-IG-inputs_patching_node")),
    ("I×G",   P.color("I×G"), "base",  ("ig1_accauc",       "EAP-IG-inputs_patching_node")),
    ("GIM",   P.color("GIM"), "base",  ("gim_eval",         "GIM_patching_node")),
]
METHOD_ORDER = [m[0] for m in METHODS]
JIT = {m: 10 ** off for m, off in
       zip(METHOD_ORDER, np.linspace(-0.06, 0.06, len(METHOD_ORDER)))}

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.6),
        axis_title=element_text(size=8),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03, panel_spacing_y=0.04,
        strip_background=element_blank(), strip_text=element_text(size=7),
        legend_title=element_text(size=7), legend_text=element_text(size=7),
        legend_key_size=10, legend_position="bottom", legend_direction="horizontal",
    )
)


def faith(kind, loc, task, model):
    if kind == "mattr":
        p = R / loc / f"{task}_{model}_validation.pkl"
    else:
        dirn, sub = loc
        p = MIB / dirn / sub / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl"
    if not p.exists():
        return None
    try:
        return pickle.load(open(p, "rb")).get("faithfulnesses")
    except Exception:
        return None


def anchors():
    a = {}
    for task, model, _ in COLUMNS:
        p = ANCH / f"{task}_{model}.json"
        if p.exists():
            d = json.load(open(p))
            a[(task, model)] = (d["baseline"], d["corrupted"])
    return a


def main():
    A = anchors()
    missing = [f"{t}/{m}" for t, m, _ in COLUMNS if (t, m) not in A]
    if missing:
        print(f"WARNING: missing anchors for {missing} -> those facets will be empty")
    rows, refs = [], []
    for task, model, flabel in COLUMNS:
        if (task, model) not in A:
            continue
        B, C = A[(task, model)]
        refs.append(dict(facet=flabel, B=B, C=C))
        for mname, _, kind, loc in METHODS:
            f = faith(kind, loc, task, model)
            if f is None:
                continue
            for i, pct in enumerate(PCT):
                rows.append(dict(method=mname, facet=flabel, x=pct * JIT[mname],
                                 raw=C + f[i] * (B - C)))
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER)
    df["facet"] = pd.Categorical(df["facet"], FACET_ORDER)
    rdf = pd.DataFrame(refs)
    rdf["facet"] = pd.Categorical(rdf["facet"], FACET_ORDER)

    colors = {m[0]: m[1] for m in METHODS}
    p = (
        ggplot(df, aes("x", "raw", color="method"))
        + geom_hline(yintercept=0, color="#000000", size=0.4)  # decision boundary (metric=0)
        + geom_hline(rdf, aes(yintercept="B"), linetype="dashed", color="#666666", size=0.3)
        + geom_hline(rdf, aes(yintercept="C"), linetype="dotted", color="#999999", size=0.3)
        + geom_line(size=0.5) + geom_point(size=1.0)
        + facet_wrap("facet", ncol=4, scales="free_y")
        + scale_x_log10(breaks=[.001, .01, .1, 1], labels=["0.1%", "1%", "10%", "100%"])
        + scale_color_manual(values=colors, name="Method", limits=METHOD_ORDER)
        + labs(x="fraction of nodes kept (denoised)",
               y="raw logit diff (solid = 0, dashed = clean, dotted = corrupted)")
    )
    p.save(OUT / "mib_logitdiff_curves.pdf", dpi=300, verbose=False)
    print(f"wrote {OUT}/mib_logitdiff_curves.pdf ({len(df)} pts, {df['method'].nunique()} methods, "
          f"{df['facet'].nunique()} facets)")


if __name__ == "__main__":
    main()
