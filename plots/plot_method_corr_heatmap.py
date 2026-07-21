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

# (label, dir/subfolder, layout). flat = {task}_{model}_importances.json
#                                 nested = <sub>/{stask}_{model}/importances.json
# hard (REINFORCE) and log-k MAttr ablations are dropped to declutter.
METHODS = [
    ("+hard (unif)*",     "htk_lr_0.05",                                   "flat"),   # hard-STE uniform-k (lr=0.05, best from sweep)
    ("+hard (log)*",      "htklog_lr_0.05",                                "flat"),   # hard-STE log-k (lr=0.05, best from sweep)
    ("+Gumbel",           "mib_node_hard_topk_gumbel",                     "flat"),
    ("MAttr (unif)",      "final_node",                                    "flat"),   # soft-fwd uniform-k
    ("MAttr (log)*",      "topklog_lr_0.05",                               "flat"),   # HEADLINE: soft-fwd log-k (lr=0.05, best from sweep)
    ("$-c_k$",            "mib_node_detached_tau",                         "flat"),
    ("$-c_k$ (log)",      "mib_node_detached_tau_log",                     "flat"),
    ("+id-STE",           "mib_node_identity_sgd",                         "flat"),
    ("+id-STE (log)",     "mib_node_identity_sgd_log",                     "flat"),
    ("+id-STE gum (log)", "mib_node_identity_gumbel_sgd_log",              "flat"),
    ("+id-STE gum (unif)","mib_node_identity_gumbel_sgd_uniform",          "flat"),
    ("NAP-IG",            "napig_ref/EAP-IG-inputs_patching_node",         "nested"),
    ("Conductance",       "napig_local/EAP-IG-inputs-local_patching_node", "nested"),
    ("I$\\times$G",       "ig1/EAP-IG-inputs_patching_node",               "nested"),
    ("RelP",              "relp/RelP_patching_node",                       "nested"),
    ("RelP+QK",           "relp_qkgrad/RelP-qkgrad_patching_node",         "nested"),
    ("AttnRLP",           "attnrlp/AttnRLP_patching_node",                 "nested"),
    ("GIM",               "gim/GIM_patching_node",                         "nested"),
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
MAIN_LABELS = [
    "MAttr (log)*", "+hard (log)*", "+Gumbel", "+id-STE (log)",  # learned (4); * = lr 0.05
    "NAP-IG", "RelP+QK", "GIM", "I$\\times$G",                   # gradient (4)
]
SUBSETS = ["Attention heads", "MLPs"]
# short display names for the main-text figure (identity labels above stay stable for lookups)
DISPLAY = {"MAttr (log)*": "MAttr", "+hard (log)*": "+hard", "NAP-IG": "IG"}

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
sd["lab"] = sd["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
# sized for display at 0.67*textwidth (5.5in) -> ~3.69in wide; fonts/height matched to the
# companion mib_accauc_cpr_scatter (1.65in wide, same base_size) so the subfigures align.
p1b = (ggplot(sd, aes("a", "b", fill="rho")) + geom_tile(color="white")
       + geom_text(aes(label="lab"), size=4.5)
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
