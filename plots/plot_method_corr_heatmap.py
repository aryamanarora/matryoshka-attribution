"""Heatmaps of Spearman rank-correlation between every pair of node-attribution
methods' per-node scores: (1) averaged over the 11 task/model pairs, and
(2) faceted by task. Includes all MAttr ablations + gradient-attribution baselines.
Run on sc."""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from plotnine import (ggplot, aes, geom_tile, geom_text, labs, facet_wrap,
                      scale_fill_gradient2, scale_x_discrete, scale_y_discrete,
                      scale_color_identity, guides,
                      theme_bw, theme_set, theme,
                      element_text, element_line, element_blank, element_rect)

R = Path("results")                                             # l2a: flat MAttr importances
R_MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")  # nested gradient baselines
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(
    theme_bw(base_size=8)
    + theme(
        # Black panel frame, matching palette.SPINE_COLOR and figs/baseline_strongreject.
        # theme_bw's own panel_border is grey20 and read lighter than the matplotlib figures
        # beside it on the same page.
        panel_border=element_rect(color="#000000", fill=None, size=0.5),
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
        plot_title=element_text(size=8, face="plain"),
    )
)

# Node Pruning learns ONE mask per (objective, target sparsity) and those runs do NOT agree with
# each other, so picking one is a real choice, not a formality -- and it moves this figure a lot.
# We show the run that wins the metric the paper leads with, CPR AUC; per mib_results.tex that is
# the logit-diff s=0.5 row (1.67 avg) and NOT the KL s=0.9 row (1.00) this used to point at.
# rho vs MAttr, averaged over the 11 cells:
#     eprun_node        (KL,  s=0.9)  all +0.255  attn +0.191  MLPs +0.240
#     eprun_node_s0.5_ld (ld, s=0.5)  all +0.567  attn +0.546  MLPs +0.386
#     eprun_node_s0.8_ld (ld, s=0.8)  all +0.493  attn +0.451  MLPs +0.481
#     eprun_node_s0.99_ld(ld, s=0.99) all +0.390  attn +0.329  MLPs +0.481
# So most of the old "Node Pruning ranks nodes unlike MAttr" signal was the OBJECTIVE mismatch
# (KL vs logit-diff), not the mask parameterization -- the honest comparison holds the loss fixed.
# Change the line below and re-run if the headline metric or the winning budget changes.
EPRUN_BEST = ("eprun_node_s0.5_ld", "0.5")   # (results dir, target sparsity); see EPRUN_SPARSITIES

# DBM (pyvene's sigmoid mask) has the same "which run?" problem and gets the same answer: the
# setting that wins CPR AUC on validation, lr=0.3 with L1 6.0 (Avg 1.50 vs 1.31 unpenalised,
# and both sweeps peak in the interior of their grids). This is the exact run the main-text
# scatter plots and the test table reports, so all three artifacts describe one circuit.
# The gate is sigmoid(mask/temperature), monotone in the stored logits, so ranking these raw
# `score` values is the same ranking as the gates themselves -- nothing to rescale for Spearman.
DBM_BEST = "eprun_node_ld_sig_lr0.3_l16.0"

# (label, dir/subfolder, layout). flat  = {task}_{model}_importances.json
#                                 nested = <sub>/{stask}_{model}/importances.json
#                                 graph  = graph_{task}_{model}.json  (Node Pruning mask logits)
# hard (REINFORCE) and log-k MAttr ablations are dropped to declutter.
METHODS = [
    ("+hard (unif)*",     "htk_lr_0.05",                                   "flat"),   # hard-STE uniform-k (lr=0.05, best from sweep)
    ("+hard (log)*",      "htklog_lr_0.05",                                "flat"),   # hard-STE log-k (lr=0.05, best from sweep)
    ("+Gumbel",           "mib_node_hard_topk_gumbel",                     "flat"),
    ("MAttr (unif)",      "final_node",                                    "flat"),   # soft-fwd uniform-k
    ("MAttr (log)*",      "topklog_lr_0.05",                               "flat"),   # HEADLINE: soft-fwd log-k (lr=0.05, best from sweep)
    # The optimizer ablation, each arm at ITS OWN best LR -- the same dirs make_mib_table's two
    # \ourmethod{}-SGD rows point at (log-k peaks at lr=1.0, uniform-k at 3.0; see OUR_METHODS).
    # Matching LRs instead would make these rows a statement about SGD being 20x off its
    # optimum rather than about the optimizer, which is exactly the reading the table repoint
    # was made to avoid. Both are complete (11/11 importances.json).
    ("MAttr SGD (log)",   "softlog_sgd_lr_1.0",                            "flat"),
    ("MAttr SGD (unif)",  "softuni_sgd_lr_3.0",                            "flat"),
    ("$-c_k$",            "mib_node_detached_tau",                         "flat"),
    ("$-c_k$ (log)",      "mib_node_detached_tau_log",                     "flat"),
    ("+id-STE",           "mib_node_identity_sgd",                         "flat"),
    ("+id-STE (log)",     "mib_node_identity_sgd_log",                     "flat"),
    ("+id-STE gum (log)", "mib_node_identity_gumbel_sgd_log",              "flat"),
    ("+id-STE gum (unif)","mib_node_identity_gumbel_sgd_uniform",          "flat"),
    # NAP-IG at two integration budgets. MIB ships --ig-steps 5 (napig_ref, what every earlier
    # version of this figure showed); napig10 is our re-run differing in that flag ONLY. It is
    # not a cosmetic difference: between the two rungs 27 nodes change SIGN while sitting in the
    # top 10 by |score| (napig_step_convergence.py), and the CPR-AUC row average moves 0.85 ->
    # 1.31. Both rows are here because the pair answers what the eval metrics cannot -- whether
    # under-integration merely adds noise to one ranking, or produces a different ranking that
    # happens to resemble a different family of methods. 30 steps is omitted: it is rho 0.994
    # with zero sign flips vs 10, so its row would be a visual duplicate of the 10-step one.
    ("NAP-IG (5 steps)",  "napig_ref/EAP-IG-inputs_patching_node",         "nested"),
    ("NAP-IG (10 steps)", "napig10/EAP-IG-inputs_patching_node",           "nested"),
    # Same path integral as the two rows above, estimated by ONE draw of alpha ~ U(0,1) per
    # example instead of an m-point grid (run_napig_mc.sh, seed 0 -- the headline dir; _s1/_s2 are
    # replicates and belong in an error bar, not a row here). Its interest is exactly a rank
    # question: on CPR AUC it ties the 30-step grid at a thirtieth of the cost, and only a
    # correlation says whether that is the same ranking recovered cheaply or a different one that
    # scores alike. This is "Expected Gradients" in mib_test_results.tex.
    ("Expected Gradients",       "napig_mc/EAP-IG-inputs-mc_patching_node",       "nested"),
    ("Conductance",       "napig_local/EAP-IG-inputs-local_patching_node", "nested"),
    ("I$\\times$G",       "ig1/EAP-IG-inputs_patching_node",               "nested"),
    ("RelP",              "relp/RelP_patching_node",                       "nested"),
    ("RelP+QK",           "relp_qkgrad/RelP-qkgrad_patching_node",         "nested"),
    ("RelP+Shapley",           "relpshapley/RelPShapley_patching_node",                 "nested"),
    ("AttnLRP",           "attnlrp/AttnLRP_patching_node",                 "nested"),
    ("GIM",               "gim/GIM_patching_node",                         "nested"),
    ("Node Pruning",      EPRUN_BEST[0],                                   "graph"),
    ("DBM",               DBM_BEST,                                        "graph"),
]
TASKS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
         ("arithmetic_subtraction", "llama3"), ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
         ("mcqa", "llama3"), ("arc_easy", "gemma2"), ("arc_easy", "llama3"),
         ("arc_challenge", "llama3")]
labels = [m[0] for m in METHODS]


def load(path):
    if not path.exists():
        return None
    d = json.load(open(path)); nodes = d.get("nodes", d)
    return {n: i["score"] for n, i in nodes.items() if n != "logits" and "score" in i}


def scores_for(spec, task, model):
    _, loc, layout = spec
    if layout == "flat":
        return load(R / loc / f"{task}_{model}_importances.json")
    if layout == "graph":
        # Node Pruning writes its learned per-node mask logits into the graph json under the
        # same {"nodes": {name: {"score": ...}}} schema, so load() needs no special case.
        return load(R / loc / f"graph_{task}_{model}.json")
    return load(R_MIB / loc / f"{task.replace('_', '-')}_{model}" / "importances.json")


LABEL_SPEC = {m[0]: m for m in METHODS}
LOC = {m[0]: m[1] for m in METHODS}          # dir path = stable identity for the cache
KEEP = {
    "all":              lambda n: True,
    "Attention heads":  lambda n: bool(re.fullmatch(r"a\d+\.h\d+", n)),
    "MLPs":             lambda n: bool(re.fullmatch(r"m\d+", n)),
}

# lazy score loading: only touch importances.json for a method if an uncached pair needs it
_scores = {}
def get_scores(label):
    if label not in _scores:
        d = {}
        for task, model in TASKS:
            s = scores_for(LABEL_SPEC[label], task, model)
            if s:
                d[(task, model)] = s
        _scores[label] = d
    return _scores[label]

# cache of pairwise rho keyed by (dir_a, dir_b, task, model, subset) -> stable across renames /
# re-subsetting. delete results/.method_corr_cache.pkl to force a full recompute (e.g. new scores).
import pickle
CACHE = R / ".method_corr_cache.pkl"
_cache = {}
if CACHE.exists():
    try:
        _cache = pickle.load(open(CACHE, "rb"))
    except Exception:
        _cache = {}


def crho(a, b, tm, subname="all"):
    key = (LOC[a], LOC[b], tm[0], tm[1], subname)
    if key in _cache:
        return _cache[key]
    keep = KEEP[subname]
    sa, sb = get_scores(a).get(tm), get_scores(b).get(tm)
    if not sa or not sb:
        v = np.nan
    else:
        common = sorted(n for n in (set(sa) & set(sb)) if keep(n))
        v = np.nan if len(common) < 4 else spearmanr([sa[n] for n in common], [sb[n] for n in common])[0]
    _cache[key] = v
    _cache[(LOC[b], LOC[a], tm[0], tm[1], subname)] = v   # symmetric
    return v


def rho(a, b, tm):
    return crho(a, b, tm, "all")


# ---- (1) averaged heatmap ----
rows = []
for a in labels:
    for b in labels:
        vals = [rho(a, b, tm) for tm in TASKS]
        vals = [v for v in vals if not np.isnan(v)]
        rows.append({"a": a, "b": b, "rho": np.mean(vals) if vals else np.nan})
df = pd.DataFrame(rows)

# ---- hierarchical clustering of methods -> block-diagonal ordering ----
# distance = 1 - avg rank-corr (all nodes); average linkage w/ optimal leaf ordering.
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
M = df.pivot(index="a", columns="b", values="rho").reindex(index=labels, columns=labels).values
M = (M + M.T) / 2.0
M = np.nan_to_num(M, nan=0.0)          # unrelated / missing pair -> 0 corr
D = np.clip(1.0 - M, 0.0, None)
np.fill_diagonal(D, 0.0)
Z = linkage(squareform(D, checks=False), method="average", optimal_ordering=True)
ORDER = [labels[i] for i in leaves_list(Z)]
print("clustered order:", ORDER)

df["a"] = pd.Categorical(df["a"], categories=ORDER, ordered=True)
df["b"] = pd.Categorical(df["b"], categories=ORDER[::-1], ordered=True)
df["lab"] = df["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
p = (ggplot(df, aes("a", "b", fill="rho")) + geom_tile(color="white")
     + geom_text(aes(label="lab"), size=5)
     + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                            midpoint=0, limits=[-1, 1], na_value="#eeeeee")
     + labs(x="", y="", fill="avg ρ", title="Pairwise node-score rank correlation (avg over 11 tasks)")
     + theme(figure_size=(4.8, 4.2), panel_grid=element_blank(),
             axis_text_x=element_text(rotation=45, ha="right", size=6),
             axis_text_y=element_text(size=6)))
p.save(OUT / "method_corr_heatmap.pdf", dpi=300); p.save(OUT / "method_corr_heatmap.png", dpi=150)
print("Saved method_corr_heatmap")

# ---- (1b) MAIN-TEXT figure: curated subset, Attn vs MLP facets only ----
# EIGHT methods, one per mechanism, cut down from eleven on 2026-08-26. The appendix heatmaps
# above keep the full set, so nothing is lost -- this cut is about what a 3.69in-wide main-text
# panel can be read at, and eleven columns put the numbers at 3.6pt against a 6-7pt house style.
#
# What went and why. RelP+QK and GIM are two more gradient baselines that land in the same block
# as I$\times$G and IG-5, so between them and AttnLRP the panel restated one fact three times;
# AttnLRP is kept as the non-IG gradient method (a propagation rule, not a path integral), which
# is the one that could in principle rank differently. NAP-IG (10 steps) went too: Expected Gradients
# now carries the "does the integration budget change the RANKING" question on its own, and it
# is the sharper version of it (one random draw vs a 5-point grid, rather than 5 vs 10).
# "+hard (log)*" is a MAttr ablation whose row said the learned family agrees with itself.
# All three are still above.
#
# What each of the eight is here to answer:
#   I$\times$G, IG-5    the gradient family, at the two budgets the test table leads with
#   Expected Gradients         same integral, one MC draw -- ties IG-30 on CPR, so: same ranking or not?
#   AttnLRP             gradient attribution that is NOT a path integral, so the gradient block
#                       is not just one estimator seen at three budgets
#   DBM, NodePrune      outside mask learners: is "learned" the axis, or is it our parameterization?
#   MAttr, +Adam        the optimizer ablation. Adam and SGD tie on every eval metric (CPR 1.879
#                       vs 1.886, IIA .499 vs .504); only a rank correlation distinguishes "the
#                       same circuit found twice" from "two circuits that score alike".
# LOG-k ONLY throughout: "MAttr SGD (unif)" is in METHODS and in the appendix figures, but a lone
# uniform-k row here would be read against log-k rows and confound the two knobs.
MAIN_LABELS = [
    # ONE MAttr ARM (2026-09-08, requested): the SGD row is dropped and the Adam row is the
    # unmarked "MAttr", matching what plot_mib_test_avg.py now does to the bar chart. Note that
    # makes "MAttr" mean the ADAM dir here while tabs/mib_test_results.tex still uses the name
    # for the SGD one -- same naming split the bar chart carries, and the fix is the same:
    # flip the table to Adam-default rather than renaming further in the figures.
    "MAttr (log)*",                                              # learned, ours (1); * = lr 0.05
    "Node Pruning", "DBM",                                       # learned, external baselines (2)
    "NAP-IG (5 steps)", "Expected Gradients", "AttnLRP", "I$\\times$G", # gradient (4)
]
SUBSETS = ["Attention heads", "MLPs"]
# short display names for the main-text figure (identity labels above stay stable for lookups).
# The IG row keeps its step count in the tick label: the test table carries m=5/10/30 rows and
# a bare "IG" here would not say which of them this is.
# SGD is the default as of 2026-08-24 (see make_mib_table.py's OUR_METHODS comments), so the
# SGD identity label now displays as bare "MAttr" and the Adam identity label as "+Adam" -- the
# swap of make_mib_test_table.py's OUR_NODE_METHODS and make_mib_table.py's emit_ours, applied
# to this figure's DISPLAY mapping instead of a results-dir list.
DISPLAY = {"MAttr (log)*": "MAttr", "+hard (log)*": "+hard",
           "NAP-IG (5 steps)": "IG-5", "NAP-IG (10 steps)": "IG-10",
           "Node Pruning": "NodePrune"}

# ORDER matches paper/tabs/mib_test_results.tex's node-level row order exactly (ascending avg
# CPR AUC within Gradient-based / Mask-based / Ours), NOT a re-cluster -- so a reader holding the
# table finds these seven methods in the same sequence, with the four the table lists between
# them (RelP, RelP+QK, IG m=30, GIM, AttnLRP) simply absent rather than reshuffled.
# Every row now has a table counterpart, which was not true of the eleven-method version.
# Expected Gradients and AttnLRP are adjacent because the table has them at 1.31 and 1.32 -- that near
# tie is the table's own, not a clustering result, and the order between them is the table's.
# Within the Ours pair, SGD-default ("MAttr SGD (log)") leads (2026-08-24 flip, see DISPLAY).
# If the table's row order changes, this list has to be updated by hand to match.
ORDER_MAIN = [
    "I$\\times$G", "NAP-IG (5 steps)", "Expected Gradients", "AttnLRP",
    "DBM", "Node Pruning",
    "MAttr (log)*",
]
assert set(ORDER_MAIN) == set(MAIN_LABELS), "ORDER_MAIN must be a permutation of MAIN_LABELS"
print("main-text order (matches mib_test_results.tex):", ORDER_MAIN)

srows = []
for sublab in SUBSETS:
    for a in MAIN_LABELS:
        for b in MAIN_LABELS:
            vals = [crho(a, b, tm, sublab) for tm in TASKS]
            vals = [v for v in vals if not np.isnan(v)]
            srows.append({"subset": sublab, "a": a, "b": b,
                          "rho": np.mean(vals) if vals else np.nan})
sd = pd.DataFrame(srows)
# relabel to short display names (after all rho lookups, which use identity labels)
sd["a"] = sd["a"].map(lambda x: DISPLAY.get(x, x))
sd["b"] = sd["b"].map(lambda x: DISPLAY.get(x, x))
ORDER_MAIN_D = [DISPLAY.get(x, x) for x in ORDER_MAIN]
sd["subset"] = pd.Categorical(sd["subset"], categories=SUBSETS, ordered=True)
sd["a"] = pd.Categorical(sd["a"], categories=ORDER_MAIN_D, ordered=True)
sd["b"] = pd.Categorical(sd["b"], categories=ORDER_MAIN_D[::-1], ordered=True)
# Leading zero dropped (".69" / "-.22"): every value here is a correlation, so the units digit is
# always 0 and carries nothing.
# TEXT SIZE IS A FUNCTION OF THE COLUMN COUNT, and the figure width is NOT available to trade
# against it -- 3.69in is set by the 0.67*textwidth slot this shares with the scatter, and
# changing it misaligns the two subfigures. Tile width is ~93pt/n_methods, and the widest string
# is "-.22": at the old 11 columns the tile was ~8.5pt and forced size 3.6, right at the limit
# (~8.6pt of glyphs, only just clearing its gutters). At 8 the tile is ~11.6pt, so size 4.5
# (~10.8pt) fits and lands near the 6-7pt-in-the-compiled-PDF band the rest of the figures use.
# The rule if the method count changes again: size ~= 3.6 * 11 / n_methods, then LOOK at it.
sd["lab"] = sd["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}".replace("0.", ".", 1))


def text_color(v):
    """Black on light tiles, white on dark ones -- decided from the TILE's own luminance.

    The fill is scale_fill_gradient2 over [-1, 1], so the tile colour is the linear ramp from
    the midpoint #f7f7f7 to #b2182b (rho -> -1) or #2166ac (rho -> +1). Reproducing that ramp
    here and taking Rec.709 relative luminance is what makes the switch land where the tile
    actually goes dark, rather than at a hand-picked |rho|.

    THE TWO SIDES DO NOT FLIP AT THE SAME |rho|, which is the whole reason this is computed:
    the red end is far darker than the blue at equal distance from the midpoint (its green
    channel falls 223/255 against the blue end's 145), so red crosses the threshold near
    |rho| = 0.57 and blue near 0.83. A single symmetric cutoff would leave the darkest red
    tiles with black text or wash out mid blues with white.

    plotnine interpolates gradient2 in Lab rather than RGB, so this is an approximation of its
    ramp, not a reproduction -- it is accurate near the ends (where the decision matters) and
    the borderline tiles are the ones to LOOK at after changing the palette.
    """
    if pd.isna(v):
        return "#000000"
    mid = (247, 247, 247)
    end = (178, 24, 43) if v < 0 else (33, 102, 172)
    t = min(abs(v), 1.0)
    r, g, b = (m + t * (e - m) for m, e in zip(mid, end))
    lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
    return "#ffffff" if lum < 0.5 else "#000000"


sd["txt"] = sd["rho"].map(text_color)
# SIZED FOR THE THREE-PANEL ROW of the paper's fig:task-transfer, authored at its FINAL page
# size so LaTeX scales nothing and every font here is the font on the page.
#
# EVERY PLOT RECTANGLE IN THAT ROW IS SQUARE (aspect_ratio=1 here, set_box_aspect(1) in
# plot_transfer_vs_corr): the two facets below and the single panel of each neighbour. Sizes
# were solved together from the measured chrome (y labels, rotated x labels, facet strips) at
# one common figure height H:
#   h1 = H-0.537, h2 = H-0.333, h3 = H-0.330  (panel height = panel width, square)
#   W1 = 0.479+2*h1, W2 = 0.373+h2, W3 = 0.391+h3, sum = 0.97*5.5  =>  H = 1.457in
#   heatmap 2.319 x 1.457 | transfer 1.497 x 1.457 | scatter 1.518 x 1.457
# LaTeX subfigure widths: 0.4216 / 0.2722 / 0.2760 of \textwidth.
# These facets come out the SMALLEST square of the four (0.93in against 1.12in) and that is
# forced, not a choice: equal aspect at equal figure height gives the panel with the most
# chrome the least room, and this one pays for both a facet strip and 45-degree x labels.
# CHANGING ANY PANEL IN THE ROW MEANS RESOLVING ALL THREE.
#
# THE COLOURBAR IS GONE, and that is what buys the tile width back. It was ~20% of the old
# figure's width. Every tile is labelled, so the fill is redundant encoding here -- it groups
# the eye, it is not the data -- and the caption carries the scale. Restoring the guide means
# losing about a glyph of tile width; do not simply turn it back on.
# Text size 4.3: at 7 methods the tile is ~9.6pt across and the widest label ("-.20") is ~8.2pt
# at this size. It was 3.9 at 8 methods in the same square facet, where adjacent negative values
# in the MLP row touched. Rule of thumb: size ~= 3.6 * 11 / n_methods, then LOOK at the MLPs
# facet's first row -- it is the densest and fails first.
p1b = (ggplot(sd, aes("a", "b", fill="rho")) + geom_tile(color="white")
       # colour is the PER-TILE text colour computed above, passed through untouched by
       # scale_color_identity -- not a scale to be read, hence guides(color=None).
       + geom_text(aes(label="lab", color="txt"), size=4.3)
       + facet_wrap("subset", ncol=2)
       + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                              midpoint=0, limits=[-1, 1], na_value="#eeeeee", guide=None)
       + scale_color_identity()
       # guide=None ON THE SCALE, not guides(fill=None): the latter is silently ignored by
       # this plotnine version (the colourbar still drew, and dropping labs(fill=) only
       # retitled it "rho"), which is exactly the kind of no-op that ships.
       + guides(color=None)
       + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
       + labs(x="", y="")
       # plot_margin_top, because at figure_size exactly 1.412in the facet strips
       # ("Attention heads" / "MLPs") sat flush on row 0 of the raster and their ascenders
       # were clipped. Margin rather than a taller box: the box IS the layout -- the three
       # subfigure widths in the paper are derived from this PDF's aspect ratio, so growing
       # the height silently unmatches the row.
       + theme(figure_size=(2.319, 1.457), aspect_ratio=1, plot_margin_top=0.025,
               panel_grid=element_blank(),
               axis_text_x=element_text(rotation=45, ha="right", size=5),
               axis_text_y=element_text(size=5)))
p1b.save(OUT / "method_corr_heatmap_bytype.pdf", dpi=300)
p1b.save(OUT / "method_corr_heatmap_bytype.png", dpi=150)
print("Saved method_corr_heatmap_bytype")

# ---- (2) faceted by task ----
frows = []
for task, model in TASKS:
    tm = (task, model); tl = f"{task.replace('arithmetic_subtraction','arith').replace('arc_','arc-')}/{model}"
    for a in labels:
        for b in labels:
            frows.append({"task": tl, "a": a, "b": b, "rho": rho(a, b, tm)})
fd = pd.DataFrame(frows)
tl_order = [f"{t.replace('arithmetic_subtraction','arith').replace('arc_','arc-')}/{m}" for t, m in TASKS]
fd["task"] = pd.Categorical(fd["task"], categories=tl_order, ordered=True)
fd["a"] = pd.Categorical(fd["a"], categories=ORDER, ordered=True)
fd["b"] = pd.Categorical(fd["b"], categories=ORDER[::-1], ordered=True)
p2 = (ggplot(fd, aes("a", "b", fill="rho")) + geom_tile()
      + facet_wrap("task", ncol=4)
      + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                             midpoint=0, limits=[-1, 1], na_value="#eeeeee")
      + labs(x="", y="", fill="ρ", title="Pairwise node-score rank correlation, per task")
      + theme(figure_size=(11, 7.5), panel_grid=element_blank(),
              axis_text_x=element_text(rotation=90, size=4),
              axis_text_y=element_text(size=4)))
p2.save(OUT / "method_corr_heatmap_bytask.pdf", dpi=300)
p2.save(OUT / "method_corr_heatmap_bytask.png", dpi=150)
print("Saved method_corr_heatmap_bytask")

pickle.dump(_cache, open(CACHE, "wb"))
print(f"cached {len(_cache)} pairwise correlations -> {CACHE}")
