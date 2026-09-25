"""Combined acc-AUC vs faithfulness-AUC scatter: MIB cells + the 4 SVA tasks (llama3).

Same 4-group coloring as plot_mib_accauc_scatter.py. x = acc-AUC (identical definition
across MIB and SVA — log-weighted mean decision accuracy). y = faithfulness AUC, but the
metric DIFFERS by benchmark: MIB = CPR (raw area, up to ~2.8); SVA = faith_auc (normalized,
~[0,1.3]). So the y-comparison is valid WITHIN a facet, not across the MIB/SVA boundary.

SVA methods (node granularity, logit_diff loss to match MIB training):
  sigmoid log-k = sufficient_hard_topk_adam_bs1;  other MAttr = adam {unif,fixed,ig5} +
  all identity_sgd variants;  other grad = ig/ixg/conductance. (No GIM: MIB-only.)

Run (from MIB-circuit-track, l2a venv):
  ../../.venv/bin/python plots/plot_accauc_scatter_sva.py
"""
import glob, json, os, pickle
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, facet_wrap, labs, theme_set, theme_bw, theme,
    element_text, element_line, element_blank, scale_shape_manual, scale_color_manual,
    guides, guide_legend,
)

theme_set(theme_bw(base_size=9) + theme(
    text=element_text(color="#000", family="Inter"),
    panel_grid_major=element_line(size=0.25, color="#dddddd"), panel_grid_minor=element_blank(),
    strip_background=element_blank(), strip_text=element_text(size=7),
    legend_title=element_text(size=8), legend_text=element_text(size=7), legend_key_size=8))

L2A_RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "sva_sweep")

# ---- MIB baselines + MAttr (same as plot_mib_accauc_scatter) ----
BASELINES = {
    "napig_ref_accauc": "EAP-IG-inputs_patching_node", "napig_local_accauc": "EAP-IG-inputs-local_patching_node",
    "ig1_accauc": "EAP-IG-inputs_patching_node", "relp_accauc": "RelP_patching_node",
    "relp_qkgrad_accauc": "RelP-qkgrad_patching_node", "gim_accauc": "GIM_patching_node",
    "relpshapley_accauc": "RelPShapley_patching_node",
    "attnlrp_accauc": "AttnLRP_patching_node",
}
GIM_DIR = "gim_accauc"
MATTR = {
    "mib_node_hard_topk_log": "siglogk",
    **{v: "otherM" for v in [
        "mib_node_hard_topk", "mib_node_hard_topk_gumbel", "mib_node_topk_log",
        "mib_node_bernoulli_reinforce", "mib_node_bernoulli_reinforce_log",
        "mib_node_detached_tau", "mib_node_detached_tau_log", "mib_node_identity_sgd",
        "mib_node_identity_sgd_log", "mib_node_identity_gumbel_sgd_log",
        "mib_node_identity_gumbel_sgd_uniform"]},
}
TASK_ABBR = {"arc_easy": "arc-e", "arc_challenge": "arc-c", "arithmetic_addition": "arith-add",
             "arithmetic_subtraction": "arith-sub"}

GROUP = {"siglogk": "MAttr sigmoid log-k", "gim": "GIM", "otherM": "other MAttr", "grad": "other grad"}
GROUPS = ["MAttr sigmoid log-k", "GIM", "other MAttr", "other grad"]
COLORS = {"MAttr sigmoid log-k": "#e41a1c", "GIM": "#377eb8", "other MAttr": "#4daf4a", "other grad": "#999999"}


def cell_of(path):
    b = os.path.basename(path).replace("_validation_abs-False.pkl", "")
    model = b.rsplit("_", 1)[1]
    task = b.rsplit("_", 1)[0].replace("-", "_")
    return task, model


rows = []
# baselines (grad) — GIM its own group
for dname, msave in BASELINES.items():
    key = "gim" if dname == GIM_DIR else "grad"
    for p in glob.glob(f"results/{dname}/{msave}/*_validation_abs-False.pkl"):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        if d.get("acc_auc") is None:
            continue
        task, model = cell_of(p)
        rows.append(dict(bench="MIB", gkey=key, task=task, model=model,
                         acc=d["acc_auc"], faith=d["area_under"]))
# MAttr
for v, key in MATTR.items():
    for p in glob.glob(f"results/mattr_accauc/{v}_patching_node/*_validation_abs-False.pkl"):
        try:
            d = pickle.load(open(p, "rb"))
        except Exception:
            continue
        if d.get("acc_auc") is None:
            continue
        task, model = cell_of(p)
        rows.append(dict(bench="MIB", gkey=key, task=task, model=model,
                         acc=d["acc_auc"], faith=d["area_under"]))

# ---- SVA (llama3, 4 tasks, node, logit_diff) ----
SVA = {"nounpp", "rc", "simple", "within_rc"}
def sva_group(stem):
    if stem in ("ig", "ixg", "conductance"):
        return "grad"
    if stem == "sufficient_hard_topk_adam_bs1":
        return "siglogk"
    if stem.startswith("sufficient_hard_topk_"):
        return "otherM"
    return None
for f in glob.glob(f"{L2A_RES}/*_node_*.json"):
    d = json.load(open(f))
    if d.get("task") not in SVA or d.get("loss") != "logit_diff":
        continue
    b = os.path.basename(f).replace(".json", "")
    model = b.split("_node_")[0].rsplit("_", 1)[1]
    stem = b.split("_node_", 1)[1]
    key = sva_group(stem)
    if key is None or d.get("acc_auc") is None:
        continue
    rows.append(dict(bench="SVA", gkey=key, task=d["task"], model=model,
                     acc=d["acc_auc"], faith=d["faith_auc"]))

df = pd.DataFrame(rows)
df["group"] = pd.Categorical(df["gkey"].map(GROUP), categories=GROUPS, ordered=True)
df["gfam"] = df["gkey"].map(lambda k: "grad" if k in ("grad", "gim") else "MAttr")
df["cell"] = df["model"] + "/" + df["task"].map(lambda t: TASK_ABBR.get(t, t))
# order facets: MIB cells first, then the 4 SVA
order = sorted(df.loc[df.bench == "MIB", "cell"].unique()) + \
        [f"llama3/{t}" for t in ["nounpp", "rc", "simple", "within_rc"]]
df["cell"] = pd.Categorical(df["cell"], categories=[c for c in order if c in set(df["cell"])], ordered=True)
print(f"{len(df)} points; MIB={sum(df.bench=='MIB')} SVA={sum(df.bench=='SVA')}; {df['cell'].nunique()} facets")

p = (ggplot(df, aes("acc", "faith", color="group", shape="gfam"))
     + geom_point(size=2.0, alpha=0.8)
     + facet_wrap("~ cell", ncol=4)
     + scale_color_manual(values=COLORS, breaks=GROUPS)
     + scale_shape_manual(values={"MAttr": "^", "grad": "o"})
     + labs(x="acc-AUC ($\\uparrow$)", y="faithfulness AUC  (MIB: CPR  |  SVA: faith-AUC)",
            color="Group", shape="Family")
     + guides(color=guide_legend(nrow=1), shape=guide_legend(nrow=1))
     + theme(figure_size=(8.5, 8.0), legend_position="bottom",
             axis_text=element_text(size=6)))
p.save("results/accauc_scatter_sva.pdf", verbose=False)
p.save("/tmp/accauc_scatter_sva.png", dpi=150, verbose=False)
print("wrote results/accauc_scatter_sva.pdf and /tmp/accauc_scatter_sva.png")
