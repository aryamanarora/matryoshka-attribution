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

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
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
    ("MAttr",          "mib_node_hard_topk",                            "flat"),
    ("+Gumbel",        "mib_node_hard_topk_gumbel",                     "flat"),
    ("+soft",          "final_node",                                    "flat"),
    ("$-c_k$",         "mib_node_detached_tau",                         "flat"),
    ("+id-STE",        "mib_node_identity_sgd",                         "flat"),
    ("+id-STE log",    "mib_node_identity_sgd_log",                     "flat"),
    ("NAP-IG",         "napig_ref/EAP-IG-inputs_patching_node",         "nested"),
    ("Conductance",    "napig_local/EAP-IG-inputs-local_patching_node", "nested"),
    ("I$\\times$G",    "ig1/EAP-IG-inputs_patching_node",               "nested"),
    ("RelP",           "relp/RelP_patching_node",                       "nested"),
    ("RelP+QK",        "relp_qkgrad/RelP-qkgrad_patching_node",         "nested"),
    ("AttnRLP",        "attnrlp/AttnRLP_patching_node",                 "nested"),
    ("GIM",            "gim/GIM_patching_node",                         "nested"),
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
    return load(R / loc / f"{task.replace('_', '-')}_{model}" / "importances.json")


S = {m[0]: {} for m in METHODS}
for spec in METHODS:
    for task, model in TASKS:
        s = scores_for(spec, task, model)
        if s:
            S[spec[0]][(task, model)] = s


def rho(a, b, tm):
    sa, sb = S[a].get(tm), S[b].get(tm)
    if not sa or not sb:
        return np.nan
    common = sorted(set(sa) & set(sb))
    if len(common) < 4:
        return np.nan
    return spearmanr([sa[n] for n in common], [sb[n] for n in common])[0]


# ---- (1) averaged heatmap ----
rows = []
for a in labels:
    for b in labels:
        vals = [rho(a, b, tm) for tm in TASKS]
        vals = [v for v in vals if not np.isnan(v)]
        rows.append({"a": a, "b": b, "rho": np.mean(vals) if vals else np.nan})
df = pd.DataFrame(rows)
df["a"] = pd.Categorical(df["a"], categories=labels, ordered=True)
df["b"] = pd.Categorical(df["b"], categories=labels[::-1], ordered=True)
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

# ---- (1b) averaged heatmap, faceted by node type (all / attn heads / MLPs) ----
SUBSETS = [
    ("All nodes",       lambda n: True),
    ("Attention heads", lambda n: bool(re.fullmatch(r"a\d+\.h\d+", n))),
    ("MLPs",            lambda n: bool(re.fullmatch(r"m\d+", n))),
]


def rho_sub(a, b, tm, keep):
    sa, sb = S[a].get(tm), S[b].get(tm)
    if not sa or not sb:
        return np.nan
    common = sorted(n for n in (set(sa) & set(sb)) if keep(n))
    if len(common) < 4:
        return np.nan
    return spearmanr([sa[n] for n in common], [sb[n] for n in common])[0]


srows = []
for sublab, keep in SUBSETS:
    for a in labels:
        for b in labels:
            vals = [rho_sub(a, b, tm, keep) for tm in TASKS]
            vals = [v for v in vals if not np.isnan(v)]
            srows.append({"subset": sublab, "a": a, "b": b,
                          "rho": np.mean(vals) if vals else np.nan})
sd = pd.DataFrame(srows)
sd["subset"] = pd.Categorical(sd["subset"], categories=[s[0] for s in SUBSETS], ordered=True)
sd["a"] = pd.Categorical(sd["a"], categories=labels, ordered=True)
sd["b"] = pd.Categorical(sd["b"], categories=labels[::-1], ordered=True)
sd["lab"] = sd["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
p1b = (ggplot(sd, aes("a", "b", fill="rho")) + geom_tile(color="white")
       + geom_text(aes(label="lab"), size=3.2)
       + facet_wrap("subset", ncol=3)
       + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                              midpoint=0, limits=[-1, 1], na_value="#eeeeee")
       + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
       + labs(x="", y="", fill="avg ρ")
       + theme(figure_size=(5.5, 2.0), panel_grid=element_blank(),
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
fd["a"] = pd.Categorical(fd["a"], categories=labels, ordered=True)
fd["b"] = pd.Categorical(fd["b"], categories=labels[::-1], ordered=True)
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
