"""Task x task Spearman rank-correlation of ONE method's per-node attribution scores.

The sibling figure, plots/plot_method_corr_heatmap.py, holds the TASK fixed and asks whether two
methods rank the same nodes. This one holds the METHOD fixed and asks the transposed question:
given one attribution method, how much does the circuit it finds depend on the task?

Everything here is llama3-8B, so the model is held fixed too and the only thing varying between
two cells is the task. That is what makes the comparison mean anything: node names are positions
in a fixed architecture (1 input + 32 layers x 32 heads + 32 MLP blocks = 1057 nodes), so
`a17.h5` denotes the SAME head in every panel. Correlating across MODELS would be comparing
different objects that happen to share a naming scheme, which is why this figure is llama3-only
rather than the 11 cells the method heatmap averages over.

Method: MAttr, soft top-k forward, log-k schedule, SGD lr=1.0 -- the headline `\\ourmethod{}` row
(see CLAUDE.md's results-dir table). Single method on purpose: the question is about tasks, and
overlaying methods would fold the method-disagreement this repo already has a figure for back
into a panel that is trying to isolate task-disagreement.

TWO HARNESSES, ONE INDEX SPACE. The five MIB cells come from `eval_mib.py` (results/<METHOD_DIR>/
{task}_{model}_importances.json, node names in the json). The four SVA and four
arithmetic-in-the-wild cells come from `eval_sva.py --nodes node --include-input`
(results/<SVA_DIR>/{task}_{model}_node_{tag}.json plus a sibling `.scores.pt` holding the raw
score TENSOR -- no node names anywhere in the json). Searching for `*_importances.json` finds
zero SVA runs even though they exist; that is a filename convention, not an absence of data.

The tensor layout is fixed by LlamaAttributionHooks (src/.../models/llama.py:76, 391, 441) and is
what eval_mib.py:346-372 itself decodes: index 0 is the input node (present iff --include-input),
then num_layers x num_heads attention heads in (layer, head) order, then num_layers MLP blocks.
NAMES() below is that layout. It is verified, not assumed: arc_easy is the one task run through
BOTH harnesses, and the two score vectors agree at rho 0.77 overall / 0.98 on MLPs (printed at
the bottom of every run). A wrong index map would put that at ~0.

That same 0.77 is the calibration constant for this whole figure: it is what "the same task,
scored twice" looks like across a harness difference (eval_mib 500 steps + MIB's dataloader vs
eval_sva 2000 steps + ours), so no off-diagonal cell should be read as "these tasks share a
circuit" unless it approaches it. The MIB<->SVA block of the heatmap carries that difference on
top of the task difference; the within-MIB and within-SVA blocks do not.

Out: paper/figs/task_corr_heatmap.pdf
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from scipy.stats import spearmanr
from plotnine import (ggplot, aes, geom_tile, geom_vline, geom_hline, labs,
                      facet_wrap, facet_grid,
                      scale_fill_gradient2, theme_bw, theme_set, theme,
                      element_text, element_line, element_blank)

from learning_to_attribute.deps import mib_results_dir

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

# The headline MAttr dirs, one per harness. MIB: verified against
# scripts/mib/launch/submit_softlog_sgd_lr.sh:94-96 (`--k-schedule log --masking topk --optimizer sgd
# --include-input`, node level, 500 steps). SVA: the `_sufficient_topk_sgd_bs1` tag is the same
# recipe (variant=topk soft forward, k_schedule=log, optimizer=sgd, lr=1.0, loss=logit_diff,
# mode=sufficient) at 2000 steps; the `_acc` / `_ce` siblings are loss ablations and `_uniformk`
# is the uniform-k one, so the bare tag is the row we ship. assert_recipe() re-checks all of that
# from each json's config rather than trusting the filename.
METHOD_DIR = "softlog_sgd_lr_1.0"
SVA_DIR = "sva_sweep_input"          # the --include-input twin of results/sva_sweep
SVA_TAG = "node_sufficient_topk_sgd_bs1"
MODEL = "llama3"
NL, NH = 32, 32                      # llama3-8B: 32 layers, 32 heads

# The second figure's comparison method. Expected Gradients (alpha ~ U(0,1) per example, m=1) is the
# gradient baseline that ties 30-step IG on MIB at I x G's cost, so it is the strongest
# gradient-side answer to "is this task structure a property of MAttr or of the tasks?".
#
# Its MIB cells come from MIB's own run_evaluation output, not ours -- a nested
# <sub>/{task-with-dashes}_{model}/importances.json, same schema as plot_method_corr_heatmap.py's
# "nested" layout. Its SVA cells are in results/sva_sweep (NOT sva_sweep_input): they were
# submitted WITHOUT --include-input, so the tensor is 1056 long and has no input node. That is
# what forces METHOD_TASKS' node space to drop `input` -- see main().
R_MIB = mib_results_dir()
SIG_MIB_DIR = "napig_mc/EAP-IG-inputs-mc_patching_node"
SIG_SVA_DIR = "sva_sweep"
SIG_SVA_TAG = "node_mc_ig_m1_s42"

# (task, display name, harness, group). Three groups:
# MIB, subject-verb agreement, and arithmetic in the wild. Ordered within each group as the paper
# already orders them (make_mib_test_table.COLUMNS; tabs/sva_top_mlp_attn.tex's Simple / Noun PP /
# RC / Within RC and Addition / Months / Weekdays / Hours headings). GROUP is a display grouping
# only -- SVA and Arith. share the eval_sva harness, so the group boundary between them is not a
# provenance boundary the way the MIB one is.
#
# arithmetic_subtraction is labelled "Arith. (sub.)" rather than the tables' bare "Arith.": with
# Feucht's "Addition" four columns away, an unqualified "Arith." reads as the same benchmark when
# it is a different one (MIB's arithmetic split vs arithmetic in the wild).
#
# arc_easy is taken from the MIB harness so that all five MIB cells share a provenance; its SVA
# twin is not plotted but IS loaded, as the cross-harness check at the bottom of main().
TASKS = [
    ("ioi",                    "IOI",           "mib", "MIB"),
    ("arithmetic_subtraction", "Arith. (sub.)", "mib", "MIB"),
    ("mcqa",                   "MCQA",          "mib", "MIB"),
    ("arc_easy",               "ARC-E",         "mib", "MIB"),
    ("arc_challenge",          "ARC-C",         "mib", "MIB"),
    ("simple",                 "Simple",        "sva", "SVA"),
    ("nounpp",                 "Noun PP",       "sva", "SVA"),
    ("rc",                     "RC",            "sva", "SVA"),
    ("within_rc",              "Within RC",     "sva", "SVA"),
    ("addition",               "Addition",      "sva", "Arith."),
    ("months",                 "Months",        "sva", "Arith."),
    ("weekdays",               "Weekdays",      "sva", "Arith."),
    ("hours",                  "Hours",         "sva", "Arith."),
]

# Same three views as the main-text method heatmap, and the same regexes, so "Attention heads"
# means the same set of nodes in both figures. The split earns its space here: attention heads
# outnumber MLP blocks 32:1, so an "all nodes" rho is an attention rho with a rounding error
# attached, and the MLP facet is the only place a 32-node effect is visible at all.
KEEP = {
    "All nodes":       lambda n: True,
    "Attention heads": lambda n: bool(re.fullmatch(r"a\d+\.h\d+", n)),
    "MLPs":            lambda n: bool(re.fullmatch(r"m\d+", n)),
}

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        legend_text=element_text(size=5.5),
        legend_title=element_text(size=6),
        legend_key_size=8,
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=7, face="plain"),
    )
)


def names():
    """Node names in `.scores.pt` tensor order, for --nodes node --include-input.

    Mirrors LlamaAttributionHooks: index 0 is the input embedding node, then attention heads at
    offset + li * num_heads + h, then MLP blocks at offset + num_layers * num_heads + li. The MIB
    harness writes these names itself, so this function exists only for the SVA side -- and the
    arc_easy cross-harness check in main() is what proves the two agree.
    """
    return (["input"]
            + [f"a{l}.h{h}" for l in range(NL) for h in range(NH)]
            + [f"m{l}" for l in range(NL)])


def assert_recipe(cfg, path):
    """The SVA filename encodes the recipe; the config is what actually ran. Check the config, so
    a re-used tag or a renamed file cannot quietly swap in a different method."""
    # eval_sva.py calls the forward variant `variant` where eval_mib.py's flag is `--masking`;
    # same knob, different arg name, so this dict follows the SVA config's spelling.
    want = {"nodes": "node", "variant": "topk", "k_schedule": "log", "optimizer": "sgd",
            "mode": "sufficient", "ablation": "patch", "method": "mattr",
            "loss": "logit_diff", "lr": 1.0, "include_input": True, "model": MODEL}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(f"{path}: config is not the headline MAttr recipe -- "
                         + ", ".join(f"{k}={got!r} (want {exp!r})" for k, (got, exp) in bad.items()))


def load_mib(task):
    """{node name: score} from eval_mib.py's importances.json. `logits` is dropped: it is a sink
    in MIB's graph, not a variable anything attributes to, and the method heatmap drops it too."""
    p = R / METHOD_DIR / f"{task}_{MODEL}_importances.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- is {METHOD_DIR} still the headline dir? (CLAUDE.md)")
    nodes = json.load(open(p)).get("nodes", {})
    return {n: i["score"] for n, i in nodes.items() if n != "logits" and "score" in i}


def load_sva(task):
    """{node name: score} from eval_sva.py's tensor + this file's NAMES() layout."""
    p = R / SVA_DIR / f"{task}_{MODEL}_{SVA_TAG}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- eval_sva.py writes {{task}}_{{model}}_{{nodes}}_{{tag}}"
                         ".json, NOT importances.json; check results/ for the right tag")
    assert_recipe(json.load(open(p)).get("config", {}), p)
    vec = torch.load(p.with_suffix(".scores.pt")).float().tolist()
    nm = names()
    if len(vec) != len(nm):
        raise SystemExit(f"{p}: {len(vec)} scores but the node layout has {len(nm)} -- "
                         "a different substrate or --include-input was off")
    return dict(zip(nm, vec))


def load_sig_mib(task):
    """{node name: score} for Expected Gradients from MIB's nested run_evaluation output."""
    p = R_MIB / SIG_MIB_DIR / f"{task.replace('_', '-')}_{MODEL}" / "importances.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- Expected Gradients has no {task}/{MODEL} cell")
    d = json.load(open(p)); nodes = d.get("nodes", d)
    return {n: i["score"] for n, i in nodes.items() if n != "logits" and "score" in i}


def load_sig_sva(task):
    """{node name: score} for Expected Gradients from eval_sva.py's tensor. These runs predate the
    --include-input twin dir, so the layout is names() MINUS the input node."""
    p = R / SIG_SVA_DIR / f"{task}_{MODEL}_{SIG_SVA_TAG}.json"
    if not p.exists():
        raise SystemExit(f"missing {p} -- no Expected Gradients run for {task}; the arithmetic-wild "
                         "tasks have `_node_ig` (10-step) but not `_node_mc_ig_m1_s42`")
    cfg = json.load(open(p)).get("config", {})
    want = {"nodes": "node", "method": "mc_ig", "ig_steps": 1, "mode": "sufficient",
            "ablation": "patch", "loss": "logit_diff", "include_input": False, "model": MODEL}
    bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
    if bad:
        raise SystemExit(f"{p}: not the Expected Gradients recipe -- "
                         + ", ".join(f"{k}={g!r} (want {e!r})" for k, (g, e) in bad.items()))
    vec = torch.load(p.with_suffix(".scores.pt")).float().tolist()
    nm = names()[1:]                     # include_input=False -> no index-0 input node
    if len(vec) != len(nm):
        raise SystemExit(f"{p}: {len(vec)} scores, layout has {len(nm)}")
    return dict(zip(nm, vec))


LOADERS = {"mib": load_mib, "sva": load_sva,
           "sig_mib": load_sig_mib, "sig_sva": load_sig_sva}

# Second figure: method x subset, over the tasks BOTH methods cover. Expected Gradients has no run for
# the four arithmetic-in-the-wild tasks (results/sva_sweep has `_node_ig` at 10 steps for them,
# not `_node_mc_ig_m1_s42`), so this drops the Arith. group and keeps MIB + SVA = 9 tasks.
METHOD_TASKS = [t for t in TASKS if t[3] != "Arith."]
# Strip labels are plain text, not LaTeX: plotnine hands them to matplotlib, which knows
# mathtext but not \ourmethod{}. plot_mib_test_avg.py expands the macro for the same reason.
METHODS = [("MAttr", {"mib": "mib", "sva": "sva"}),
           ("Expected Gradients", {"mib": "sig_mib", "sva": "sig_sva"})]


SEP = dict(color="#555555", size=0.35)
FILL = dict(low="#b2182b", mid="#f7f7f7", high="#2166ac",
            midpoint=0, limits=[-1, 1], na_value="#eeeeee")


def check_namespace(scores, keys):
    """The index space is identical across tasks by construction (same model, same substrate,
    same --include-input) even though it arrives via two different writers. Assert it rather than
    silently intersecting: a task with a different node set would mean a run at the wrong
    granularity, and quietly correlating the overlap would hide that behind a plausible number."""
    for t, sc in scores.items():
        if set(sc) != keys:
            raise SystemExit(f"{t}: node namespace differs from the {len(keys)}-node layout "
                             f"({len(sc)} nodes) -- wrong substrate?")


def corr_frame(tasks, scores, keys, extra=None):
    """Long frame of every (task_a, task_b, subset) Spearman rho, one row per cell."""
    rows = []
    for subname, keep in KEEP.items():
        common = sorted(n for n in keys if keep(n))
        for (ta, la, _h, _g) in tasks:
            for (tb, lb, _h2, _g2) in tasks:
                r = spearmanr([scores[ta][n] for n in common],
                              [scores[tb][n] for n in common])[0]
                rows.append(dict(a=la, b=lb, rho=r, sub=f"{subname} ({len(common)})",
                                 **(extra or {})))
    return pd.DataFrame(rows)


def categorise(df, tasks):
    """Fixed task order, NOT clustered. The method heatmap clusters because "which methods form a
    family" is one of its findings; here the axis is tasks the paper already groups, and
    reordering them per-facet would stop the panels from being readable against each other. y is
    reversed so the diagonal runs top-left to bottom-right."""
    labels = [lab for _, lab, _h, _g in tasks]
    df["a"] = pd.Categorical(df["a"], categories=labels, ordered=True)
    df["b"] = pd.Categorical(df["b"], categories=labels[::-1], ordered=True)
    df["sub"] = pd.Categorical(df["sub"], categories=list(dict.fromkeys(df["sub"])), ordered=True)
    return df


def separators(tasks):
    """Group-boundary line positions. The blocks are already visible as blocks, but the
    boundaries are what the reader has to trust to read them, so draw them rather than leaving it
    to the eye. Continuous positions on a discrete axis: a boundary after the n-th category sits
    at n + 0.5, and y is reversed so it mirrors to len - n + 0.5."""
    cuts = [i + 0.5 for i in range(1, len(tasks)) if tasks[i][3] != tasks[i - 1][3]]
    return cuts, [len(tasks) - c + 1 for c in cuts]


def report(df, tasks, by=None):
    """Off-diagonal block means, printed because they are the numbers a caption would quote and
    they drift with every re-evaluation. Split by task group so the MIB<->SVA harness difference
    (which only the cross blocks carry) is not averaged into the within-group numbers."""
    grp = {lab: g for _, lab, _h, g in tasks}
    seen = list(dict.fromkeys(g for _, _l, _h, g in tasks))
    pairs = ([(g, g) for g in seen]
             + [(a, b) for i, a in enumerate(seen) for b in seen[i + 1:]])
    for key, sec in (df.groupby(by, observed=True) if by else [(None, df)]):
        for sub, g in sec.groupby("sub", observed=True):
            off = g[g.a.astype(str) != g.b.astype(str)]
            print(f"  {(str(key) + ' / ') if by else ''}{sub}")
            for ga, gb in pairs:
                v = off[off.apply(lambda r: {grp[r.a], grp[r.b]} == {ga, gb}, axis=1)]["rho"]
                tag = ga if ga == gb else f"{ga} x {gb}"
                print(f"      {tag:>16}: {v.mean():+.3f} [{v.min():+.3f}, {v.max():+.3f}]")


def figure_tasks():
    """Figure 1: MAttr alone, all 13 llama3 cells, at the full node+input granularity."""
    scores = {t: LOADERS[h](t) for t, _l, h, _g in TASKS}
    keys = set(names())
    check_namespace(scores, keys)
    print(f"{len(keys)} nodes x {len(TASKS)} tasks, model={MODEL}, "
          f"dirs={METHOD_DIR} (MIB) + {SVA_DIR} (SVA)")

    df = categorise(corr_frame(TASKS, scores, keys), TASKS)
    vcuts, hcuts = separators(TASKS)
    # No in-cell numbers, unlike the method heatmap. 13 tasks in a 1.6in panel leaves ~9pt per
    # cell; "0.26" does not fit in 9pt at any size a reader can see, and shrinking it until it
    # does produces a figure that LOOKS quantitative and is not. The colour bar carries the
    # magnitude and the block means are in the caption.
    p = (ggplot(df, aes("a", "b", fill="rho")) + geom_tile(color="white", size=0.2)
         + geom_vline(xintercept=vcuts, **SEP) + geom_hline(yintercept=hcuts, **SEP)
         + facet_wrap("~sub", nrow=1)
         # Same diverging scale and the same [-1, 1] limits as the method heatmap: the figures
         # share a colour vocabulary, so a tile of a given blue means the same rho in all of
         # them. Fixing the limits (rather than letting them float to the data) is what makes
         # "these are all pale" a readable statement instead of a rescaled one.
         + scale_fill_gradient2(**FILL) + labs(x="", y="", fill="ρ")
         + theme(figure_size=(5.5, 2.35), panel_grid=element_blank(),
                 axis_text_x=element_text(rotation=45, ha="right", size=6),
                 axis_text_y=element_text(size=6)))
    p.save(OUT / "task_corr_heatmap.pdf", dpi=300, verbose=False)
    print("wrote", OUT / "task_corr_heatmap.pdf")
    report(df, TASKS)
    return scores, keys


def figure_methods():
    """Figure 2: does the task structure belong to the method or to the tasks? Same matrix for
    MAttr and for Expected Gradients, rows = method, columns = subset.

    The node space here EXCLUDES the input node, because Expected Gradients's SVA runs were submitted
    without --include-input. Dropping it costs nothing measurable -- it is 1 of 1057 nodes, and
    the MAttr block means move by <=0.002 against figure 1 -- but it is a real difference between
    the two figures, so it is asserted (1056) rather than assumed.
    """
    keys = set(names()[1:])
    frames = []
    for label, route in METHODS:
        scores = {t: {k: v for k, v in LOADERS[route[h]](t).items() if k != "input"}
                  for t, _l, h, _g in METHOD_TASKS}
        check_namespace(scores, keys)
        frames.append(corr_frame(METHOD_TASKS, scores, keys, extra={"method": label}))
    df = categorise(pd.concat(frames, ignore_index=True), METHOD_TASKS)
    df["method"] = pd.Categorical(df["method"], categories=[m for m, _ in METHODS], ordered=True)
    print(f"{len(keys)} nodes (input dropped) x {len(METHOD_TASKS)} tasks x {len(METHODS)} methods")

    vcuts, hcuts = separators(METHOD_TASKS)
    p = (ggplot(df, aes("a", "b", fill="rho")) + geom_tile(color="white", size=0.2)
         + geom_vline(xintercept=vcuts, **SEP) + geom_hline(yintercept=hcuts, **SEP)
         # facet_grid, not facet_wrap: the point of this figure is reading DOWN a column (same
         # subset, two methods), which only a grid guarantees is the same pair of axes.
         + facet_grid("method ~ sub")
         + scale_fill_gradient2(**FILL) + labs(x="", y="", fill="ρ")
         + theme(figure_size=(5.5, 3.6), panel_grid=element_blank(),
                 axis_text_x=element_text(rotation=45, ha="right", size=6),
                 axis_text_y=element_text(size=6),
                 strip_text_y=element_text(size=6.5, angle=-90)))
    p.save(OUT / "task_corr_heatmap_methods.pdf", dpi=300, verbose=False)
    print("wrote", OUT / "task_corr_heatmap_methods.pdf")
    report(df, METHOD_TASKS, by="method")


def main():
    scores, keys = figure_tasks()

    # Cross-harness calibration: arc_easy is the only task both harnesses ran. This is the
    # verification that names() decodes the SVA tensor correctly (a wrong map gives ~0) AND the
    # ceiling against which the cross-family blocks above should be read.
    twin = load_sva("arc_easy")
    for subname, keep in KEEP.items():
        common = sorted(n for n in keys if keep(n))
        r = spearmanr([scores["arc_easy"][n] for n in common], [twin[n] for n in common])[0]
        print(f"  arc_easy MIB-vs-SVA harness ({subname}): rho {r:+.3f}")

    figure_methods()


if __name__ == "__main__":
    sys.exit(main())
