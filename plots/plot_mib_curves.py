"""Appendix full-width per-task curves over the MIB denoising sparsity sweep, for the 5
key node methods (MAttr, +hard, IG, I×G, GIM), faceted by task/model.

Two figures, same layout:
  - mib_accuracy_curves.pdf : decision accuracy (fraction of examples with metric>0) vs sparsity
  - mib_cpr_curves.pdf      : CPR / faithfulness (ablated normalised to [corrupted, clean]) vs sparsity
x-axis = fraction of nodes kept, log scale (matches acc-AUC's log weighting). acc-AUC is the
log-x-weighted mean of the accuracy curve; CPR AUC is the linear-x area under the faithfulness curve.

Each curve reads BOTH arrays from one self-consistent eval pkl:
  MAttr/+hard -> results/{topklog,htklog}_lr_0.05/{task}_{model}_validation.pkl
  baselines   -> MIB-circuit-track/results/*_accauc/<sub>/{stask}_{model}_validation_abs-False.pkl
Run:  uv run python plots/plot_mib_curves.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_wrap, labs, theme,
    theme_set, theme_bw, element_text, element_line, element_blank,
    scale_color_manual, scale_x_log10,
)

R = Path("results")
MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")
OUT = Path("plots")
PCT = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)

# task/model -> facet label (order = facet order)
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

# method -> (colour, loader-kind, dir/sub). colours match accauc_vs_faithauc.pdf; GIM = orange.
METHODS = [
    ("MAttr", "#1f77b4", "mattr", "topklog_lr_0.05"),
    ("+hard", "#2ca02c", "mattr", "htklog_lr_0.05"),
    ("IG",    "#8c564b", "base",  ("napig_ref_accauc", "EAP-IG-inputs_patching_node")),
    ("I×G",   "#e377c2", "base",  ("ig1_accauc",       "EAP-IG-inputs_patching_node")),
    ("GIM",   "#ff7f0e", "base",  ("gim_accauc",       "GIM_patching_node")),
]
METHOD_ORDER = [m[0] for m in METHODS]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.6),
        axis_title=element_text(size=8),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.04,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=7),
        legend_key_size=10,
        legend_position="bottom",
        legend_direction="horizontal",
    )
)


def load(kind, loc, task, model):
    if kind == "mattr":
        p = R / loc / f"{task}_{model}_validation.pkl"
    else:
        dirn, sub = loc
        p = MIB / dirn / sub / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl"
    if not p.exists():
        return None
    try:
        return pickle.load(open(p, "rb"))
    except Exception:
        return None


# small per-method multiplicative x-offset (evenly spread in log space) so the 5 curves'
# markers at each sweep point don't sit exactly on top of each other. ±0.06 decade ≈ ±15%.
JIT = {m: 10 ** off for m, off in
       zip(METHOD_ORDER, np.linspace(-0.06, 0.06, len(METHOD_ORDER)))}


def build():
    rows = []
    for mname, _, kind, loc in METHODS:
        for task, model, flabel in COLUMNS:
            d = load(kind, loc, task, model)
            if d is None:
                continue
            acc, faith = d.get("accuracies"), d.get("faithfulnesses")
            for i, pct in enumerate(PCT):
                rows.append(dict(method=mname, facet=flabel,
                                 pct=pct, x=pct * JIT[mname],
                                 acc=acc[i] if acc else None,
                                 cpr=faith[i] if faith else None))
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], METHOD_ORDER)
    df["facet"] = pd.Categorical(df["facet"], FACET_ORDER)
    return df


def make(df, ycol, ylab, out, hline=None, free_y=False):
    colors = {m[0]: m[1] for m in METHODS}
    sub = df[df[ycol].notna()]
    p = (
        ggplot(sub, aes("x", ycol, color="method"))
        + (geom_hline(yintercept=hline, linetype="dashed", color="#999999", size=0.3)
           if hline is not None else geom_blank())
        + geom_line(size=0.5)
        + geom_point(size=1.0)
        + facet_wrap("facet", ncol=4, scales="free_y" if free_y else "fixed")
        + scale_x_log10(breaks=[.001, .01, .1, 1], labels=["0.1%", "1%", "10%", "100%"])
        + scale_color_manual(values=colors, name="Method", limits=METHOD_ORDER)
        + labs(x="fraction of nodes kept (denoised)", y=ylab)
    )
    p.save(OUT / out, dpi=300, verbose=False)
    print(f"wrote {OUT/out} ({len(sub)} pts, {sub['method'].nunique()} methods)")


# geom_blank shim (plotnine has it, import lazily to keep the top clean)
from plotnine import geom_blank  # noqa: E402


def main():
    df = build()
    make(df, "acc", "decision accuracy (metric > 0)", "mib_accuracy_curves.pdf")
    make(df, "cpr", "CPR (faithfulness)", "mib_cpr_curves.pdf", hline=1.0, free_y=True)


if __name__ == "__main__":
    main()
