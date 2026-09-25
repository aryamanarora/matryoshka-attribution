"""Exploratory MIB scatter: acc-AUC (x) vs CPR-AUC (y), one point per (method, model),
faceted by task. All node methods: 7 gradient/reference baselines + MAttr (L2A) variants.

acc-AUC is the honest log-weighted decision accuracy; CPR-AUC is MIB faithfulness area.
Points high-CPR/low-acc = faithful-but-doesn't-recover-the-decision (the gap acc-AUC exposes).

Run (from MIB-circuit-track, with the l2a venv that has plotnine):
  ../../.venv/bin/python plots/plot_mib_accauc_scatter.py
"""
import glob, os, pickle
import pandas as pd
from plotnine import (
    ggplot, aes, geom_point, facet_wrap, labs, theme_set, theme_bw,
    theme, element_text, element_line, element_blank, scale_shape_manual,
    scale_color_manual, guides, guide_legend,
)

theme_set(theme_bw(base_size=9) + theme(
    text=element_text(color="#000", family="Inter"),
    panel_grid_major=element_line(size=0.25, color="#dddddd"), panel_grid_minor=element_blank(),
    strip_background=element_blank(), strip_text=element_text(size=8),
    legend_title=element_text(size=8), legend_text=element_text(size=7), legend_key_size=8))

# baseline dir -> (label, method_saveable subdir)
BASELINES = {
    "napig_ref_accauc":   ("NAP-IG",    "EAP-IG-inputs_patching_node"),
    "napig_local_accauc": ("NAP-local", "EAP-IG-inputs-local_patching_node"),
    "ig1_accauc":         ("IxG",       "EAP-IG-inputs_patching_node"),
    "relp_accauc":        ("RelP",      "RelP_patching_node"),
    "relp_qkgrad_accauc": ("RelP-qk",   "RelP-qkgrad_patching_node"),
    "gim_accauc":         ("GIM",       "GIM_patching_node"),
    "relpshapley_accauc":     ("RelP+Shapley",   "RelPShapley_patching_node"),
    "attnlrp_accauc":     ("AttnLRP",   "AttnLRP_patching_node"),
}
# MAttr variant subdir (under results/mattr_accauc) -> label
MATTR = {
    "mib_node_hard_topk_log": "L2A", "mib_node_hard_topk": "MAttr:htk-unif",
    "mib_node_hard_topk_gumbel": "MAttr:htk-gumbel", "mib_node_topk_log": "MAttr:topk-log",
    "mib_node_bernoulli_reinforce": "MAttr:bern", "mib_node_bernoulli_reinforce_log": "MAttr:bern-log",
    "mib_node_detached_tau": "MAttr:dtau", "mib_node_detached_tau_log": "MAttr:dtau-log",
    "mib_node_identity_sgd": "MAttr:id-sgd", "mib_node_identity_sgd_log": "MAttr:id-sgd-log",
    "mib_node_identity_gumbel_sgd_log": "MAttr:id-gum-log",
    "mib_node_identity_gumbel_sgd_uniform": "MAttr:id-gum-unif",
}


def cell_of(path):
    b = os.path.basename(path).replace("_validation_abs-False.pkl", "")
    model = b.rsplit("_", 1)[1]
    task = b.rsplit("_", 1)[0].replace("-", "_")
    return task, model


def read(path):
    try:
        d = pickle.load(open(path, "rb"))
        return d.get("acc_auc"), d.get("area_under")
    except Exception:
        return None, None


rows = []
for dname, (label, msave) in BASELINES.items():
    for p in glob.glob(f"results/{dname}/{msave}/*_validation_abs-False.pkl"):
        acc, cpr = read(p)
        if acc is None:
            continue
        task, model = cell_of(p)
        rows.append(dict(method=label, family="baseline", task=task, model=model,
                         acc_auc=acc, cpr=cpr))
for v, label in MATTR.items():
    for p in glob.glob(f"results/mattr_accauc/{v}_patching_node/*_validation_abs-False.pkl"):
        acc, cpr = read(p)
        if acc is None:
            continue
        task, model = cell_of(p)
        fam = "L2A" if label == "L2A" else "MAttr"
        rows.append(dict(method=label, family=fam, task=task, model=model,
                         acc_auc=acc, cpr=cpr))

df = pd.DataFrame(rows)
print(f"loaded {len(df)} points across {df['task'].nunique()} tasks, {df['method'].nunique()} methods")
print(df.groupby("family").size().to_string())

TASK_ABBR = {"arc_easy": "arc-e", "arc_challenge": "arc-c", "arithmetic_addition": "arith-add",
             "arithmetic_subtraction": "arith-sub", "ioi": "ioi", "mcqa": "mcqa"}
df["cell"] = df["model"] + "/" + df["task"].map(lambda t: TASK_ABBR.get(t, t))

GROUPS = ["MAttr sigmoid log-k", "GIM", "other MAttr", "other grad"]
def group_of(r):
    if r["method"] == "L2A":
        return "MAttr sigmoid log-k"
    if r["method"] == "GIM":
        return "GIM"
    return "other MAttr" if r["family"] in ("L2A", "MAttr") else "other grad"
df["group"] = df.apply(group_of, axis=1)
df["group"] = pd.Categorical(df["group"], categories=GROUPS, ordered=True)
df["gfam"] = df["family"].map(lambda f: "MAttr" if f in ("L2A", "MAttr") else "grad")

COLORS = {"MAttr sigmoid log-k": "#e41a1c", "GIM": "#377eb8",
          "other MAttr": "#4daf4a", "other grad": "#999999"}
p = (ggplot(df, aes("acc_auc", "cpr", color="group", shape="gfam"))
     + geom_point(size=2.0, alpha=0.8)
     + facet_wrap("~ cell", ncol=6)
     + scale_color_manual(values=COLORS, breaks=GROUPS)
     + scale_shape_manual(values={"MAttr": "^", "grad": "o"})
     + labs(x="acc-AUC ($\\uparrow$)", y="CPR-AUC (faithfulness)",
            color="Group", shape="Family")
     + guides(color=guide_legend(nrow=1), shape=guide_legend(nrow=1))
     + theme(figure_size=(5.5, 2.9), legend_position="bottom",
             legend_box="horizontal", legend_margin=0,
             axis_text=element_text(size=5), axis_title=element_text(size=7),
             strip_text=element_text(size=5.5), legend_text=element_text(size=6),
             legend_title=element_text(size=6.5)))
p.save("results/mib_accauc_scatter.pdf", verbose=False)
p.save("/tmp/mib_accauc_scatter.png", dpi=150, verbose=False)
print("wrote results/mib_accauc_scatter.pdf and /tmp/mib_accauc_scatter.png")
