"""Appendix full-width per-task curves over the MIB denoising sparsity sweep, for the 7
key node methods (MAttr, +hard, IG, I×G, GIM, Node Pruning, DBM), faceted by task/model.

Two figures, same layout:
  - mib_accuracy_curves.pdf : decision accuracy (fraction of examples with metric>0) vs sparsity
  - mib_cpr_curves.pdf      : CPR / faithfulness (ablated normalised to [corrupted, clean]) vs sparsity
x-axis = fraction of nodes kept, log scale (matches acc-AUC's log weighting). acc-AUC is the
log-x-weighted mean of the accuracy curve; CPR AUC is the linear-x area under the faithfulness curve.

Each curve reads BOTH arrays from one self-consistent eval pkl:
  MAttr/+hard -> results/{topklog,htklog}_lr_0.05/{task}_{model}_validation.pkl
  gradient    -> MIB-circuit-track/results/*_accauc/<sub>/{stask}_{model}_validation_abs-False.pkl
  mask-learn  -> results/eprun_eval_*/EdgePruning_patching_node/{stask}_{model}_validation_abs-False.pkl

The two mask-learning baselines sweep sparsity the same way everything else does: their
learned per-node mask logits are a ranking, and MIB's eval thresholds that ranking at each
sweep point. The mask's own converged density is one point on that x-axis, not the curve.
Run:  uv run python plots/plot_mib_curves.py
"""
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

# method -> (colour, loader-kind, dir/sub). Colours from plots/palette.py, the single
# source of truth shared with every other figure -- do not write hex codes here.
METHODS = [
    ("MAttr", P.color("MAttr"), "mattr", "topklog_lr_0.05"),
    ("+hard", P.color("+hard"), "mattr", "htklog_lr_0.05"),
    ("IG",    P.color("IG"), "base",  ("napig_ref_accauc", "EAP-IG-inputs_patching_node")),
    ("I×G",   P.color("I×G"), "base",  ("ig1_accauc",       "EAP-IG-inputs_patching_node")),
    ("GIM",   P.color("GIM"), "base",  ("gim_accauc",       "GIM_patching_node")),
    # The two mask-learning baselines at their best swept setting -- Node Pruning s=0.5 with
    # the logit-diff objective, DBM at lr 0.3 with lambda_L1 6.0. Both are the same dirs the
    # test table and the correlation heatmap read, so a reader comparing figures is looking
    # at one run per method rather than three different budgets of it.
    ("Node Pruning", P.color("Node Pruning"), "eprun", "eprun_eval_s0.5_ld"),
    ("DBM",          P.color("DBM"),          "eprun", "eprun_eval_ld_sig_lr0.3_l16.0"),
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
    stask = task.replace("_", "-")
    if kind == "mattr":
        p = R / loc / f"{task}_{model}_validation.pkl"
    elif kind == "eprun":
        p = R / loc / "EdgePruning_patching_node" / f"{stask}_{model}_validation_abs-False.pkl"
    else:
        dirn, sub = loc
        p = MIB / dirn / sub / f"{stask}_{model}_validation_abs-False.pkl"
    if not p.exists():
        return None
    try:
        return pickle.load(open(p, "rb"))
    except Exception:
        return None


# small per-method multiplicative x-offset (evenly spread in log space) so the 7 curves'
# markers at each sweep point don't sit exactly on top of each other. ±0.09 decade ≈ ±23%.
# The half-width grew with the series count (was ±0.06 for 5) to hold the SPACING between
# adjacent series fixed at 0.03 decade -- about one marker width. Shrinking back would stack
# the markers; the total spread is still 6% of a 3-decade axis, well inside one sweep step.
JIT = {m: 10 ** off for m, off in
       zip(METHOD_ORDER, np.linspace(-0.09, 0.09, len(METHOD_ORDER)))}


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
