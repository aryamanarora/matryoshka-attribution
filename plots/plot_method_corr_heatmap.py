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
                      theme_bw, theme_set, theme,
                      element_text, element_line, element_blank)

R = Path("results")                                             # l2a: flat MAttr importances
R_MIB = Path("/home/guests/aryaman/MIB-circuit-track/results")  # nested gradient baselines
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
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
# ~half the methods, one per mechanism (headline + Pareto learned methods, recognizable
# gradient baselines + the conductance pair). Rest go to the appendix (full-set figures above).
# +Gumbel and +id-STE (log) are dropped: they are MAttr *ablations*, so their rows only restated
# that the learned family agrees with itself, and the row they displace now buys an outside
# mask learner. Both are still in the full-set appendix heatmaps above.
MAIN_LABELS = [
    "MAttr (log)*", "+hard (log)*",                              # learned, ours (2); * = lr 0.05
    # The optimizer ablation. It earns a main-text row on the same grounds as the two IG budgets
    # beside it: the eval metrics say Adam and SGD tie (CPR 1.879 vs 1.886, IIA .499 vs .504),
    # and only a rank correlation can say whether that is the SAME circuit found twice or two
    # different circuits scoring alike -- which is precisely what this panel is for.
    # LOG-k ONLY. "MAttr SGD (unif)" is in METHODS and so in the appendix heatmaps, but its
    # Adam twin is not in this cut, so a lone uniform-k row would be read against log-k rows and
    # confound the two knobs. Adding both would also take the panel to 12 columns; see the
    # tile-width note under sd["lab"] below.
    "MAttr SGD (log)",
    "Node Pruning", "DBM",                                       # learned, external baselines (2)
    "NAP-IG (5 steps)", "NAP-IG (10 steps)",                     # gradient, one method two budgets
    "RelP+QK", "GIM", "AttnLRP", "I$\\times$G",                  # gradient (4)
]
SUBSETS = ["Attention heads", "MLPs"]
# short display names for the main-text figure (identity labels above stay stable for lookups).
# The two IG rows keep their step count in the tick label -- dropping it and relying on the
# clustering to imply the pairing does not work, because they do NOT always land adjacent.
DISPLAY = {"MAttr (log)*": "MAttr", "+hard (log)*": "+hard",
           # matches the label the companion scatter uses for the same dir (softlog_sgd_lr_1.0),
           # so the two subfigures of fig:mib-combined name one method one way
           "MAttr SGD (log)": "+SGD",
           "NAP-IG (5 steps)": "IG-5", "NAP-IG (10 steps)": "IG-10",
           "Node Pruning": "NodePrune"}

# re-cluster the subset on its avg all-node correlation so blocks are tight for these methods
Msub = df.pivot(index="a", columns="b", values="rho").reindex(index=MAIN_LABELS, columns=MAIN_LABELS).values
Msub = np.nan_to_num((Msub + Msub.T) / 2.0, nan=0.0)
Dsub = np.clip(1.0 - Msub, 0.0, None); np.fill_diagonal(Dsub, 0.0)
Zsub = linkage(squareform(Dsub, checks=False), method="average", optimal_ordering=True)
ORDER_MAIN = [MAIN_LABELS[i] for i in leaves_list(Zsub)]
print("main-text clustered order:", ORDER_MAIN)

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
# leading zero dropped (".69" / "-.22") and text one point smaller than the appendix figures.
# At 10 methods each tile is ~9.7pt wide, and a 5-character "-0.22" at size 4.5 is ~10.4pt --
# i.e. the 9-method version was already at the limit and the tenth column made neighbouring
# numbers overlap. Every value here is a correlation, so the units digit is always 0 and
# carries nothing. Do not widen the figure to buy the space back: its 3.69in is set by the
# 0.67*textwidth slot it shares with the scatter, and breaking that misaligns the subfigures.
# At 11 (the +SGD row) the tile is ~8.5pt and the widest string here, "-.22" at size 3.6, is
# ~8.6pt of glyphs but renders inside its tile -- checked at 600dpi, gutters still visible. That
# is the ceiling: a 12th column needs geom_text size ~3.2, so if another method is added here,
# drop one or shrink the text rather than assuming it still fits.
sd["lab"] = sd["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}".replace("0.", ".", 1))
# sized for display at 0.67*textwidth (5.5in) -> ~3.69in wide; fonts/height matched to the
# companion mib_accauc_cpr_scatter (1.65in wide, same base_size) so the subfigures align.
p1b = (ggplot(sd, aes("a", "b", fill="rho")) + geom_tile(color="white")
       + geom_text(aes(label="lab"), size=3.6)
       + facet_wrap("subset", ncol=2)
       + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                              midpoint=0, limits=[-1, 1], na_value="#eeeeee")
       + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
       + labs(x="", y="", fill="avg ρ")
       + theme(figure_size=(3.69, 2.0), panel_grid=element_blank(),
               axis_text_x=element_text(rotation=45, ha="right", size=6),
               axis_text_y=element_text(size=6)))
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
