"""Summary bar chart of mean CPR AUC per method (avg over the 11 MIB tasks) with 95%
CI, grouped by Gradient attribution vs Ours (MAttr). Node + edge facets.
Reads results/. Run on sc."""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from plotnine import (ggplot, aes, geom_col, geom_errorbar, geom_hline, labs, facet_wrap,
                      scale_fill_manual, scale_x_discrete, coord_flip, theme_bw, theme_set,
                      theme, element_text)

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=9) + theme(text=element_text(family="Inter")))

COLUMNS = [("ioi", "gpt2"), ("ioi", "qwen2.5"), ("ioi", "gemma2"), ("ioi", "llama3"),
           ("arithmetic_subtraction", "llama3"), ("mcqa", "qwen2.5"), ("mcqa", "gemma2"),
           ("mcqa", "llama3"), ("arc_easy", "gemma2"), ("arc_easy", "llama3"),
           ("arc_challenge", "llama3")]

# (label, group, level, spec). spec: (dir,) = our flat val.pkl; (evaldir, sub) = grad eval abs-False
METHODS = [
    ("NAP-IG", "Gradient", "node", ("napig_repro_eval", "EAP-IG-inputs_patching_node")),
    ("Conductance", "Gradient", "node", ("napig_local_eval", "EAP-IG-inputs-local_patching_node")),
    ("I×G", "Gradient", "node", ("ig1_eval", "EAP-IG-inputs_patching_node")),
    ("RelP", "Gradient", "node", ("relp_eval", "RelP_patching_node")),
    ("RelP+QK", "Gradient", "node", ("relp_qkgrad_eval", "RelP-qkgrad_patching_node")),
    ("AttnRLP", "Gradient", "node", ("attnrlp_eval", "AttnRLP_patching_node")),
    ("GIM", "Gradient", "node", ("gim_eval", "GIM_patching_node")),
    ("MAttr", "Ours", "node", ("mib_node_hard_topk",)),
    ("+Gumbel", "Ours", "node", ("mib_node_hard_topk_gumbel",)),
    ("+soft", "Ours", "node", ("final_node",)),
    ("+soft −c_k", "Ours", "node", ("mib_node_detached_tau",)),
    ("+hard", "Ours", "node", ("mib_node_bernoulli_reinforce",)),
    ("EAP-IG-inp", "Gradient", "edge", ("eapig_repro_eval", "EAP-IG-inputs_patching_edge")),
    ("MAttr", "Ours", "edge", ("mib_edge_hard_topk_uniform",)),
    ("+soft", "Ours", "edge", ("final_edge",)),
    ("+soft −c_k", "Ours", "edge", ("mib_edge_detached_tau",)),
    ("+hard", "Ours", "edge", ("mib_edge_bernoulli_reinforce",)),
]


def auc(spec, task, model):
    if len(spec) == 1:
        p = R / spec[0] / f"{task}_{model}_validation.pkl"
    else:
        p = R / spec[0] / spec[1] / f"{task.replace('_', '-')}_{model}_validation_abs-False.pkl"
    if not p.exists():
        return None
    try:
        v = pickle.load(open(p, "rb"))["area_under"]
        return float(v) if np.isfinite(v) else None
    except Exception:
        return None


rows = []
for label, group, level, spec in METHODS:
    vals = [auc(spec, t, m) for t, m in COLUMNS]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        continue
    a = np.array(vals); mean = a.mean(); ci = 1.96 * a.std(ddof=1) / np.sqrt(len(a))
    rows.append({"method": label, "group": group, "level": level,
                 "mean": mean, "lo": mean - ci, "hi": mean + ci, "n": len(a)})
df = pd.DataFrame(rows)
df["level"] = df["level"].map({"node": "Node-level", "edge": "Edge-level"})
# order within facet: Gradient first, then Ours, each by descending mean
df["key"] = df["group"].map({"Gradient": 0, "Ours": 1}) * 1e6 - df["mean"]
order = df.sort_values(["level", "key"])["method"].tolist()
# unique ordered labels (method names repeat across facets) -> use a composite for ordering
df["ml"] = df["level"] + " | " + df["method"]
ml_order = df.sort_values(["level", "key"])["ml"].tolist()
df["ml"] = pd.Categorical(df["ml"], categories=ml_order[::-1], ordered=True)

p = (ggplot(df, aes("ml", "mean", fill="group"))
     + geom_col(width=0.7)
     + geom_errorbar(aes(ymin="lo", ymax="hi"), width=0.3, size=0.4)
     + geom_hline(yintercept=1, linetype="dotted", color="#999999", size=0.4)
     + coord_flip()
     + facet_wrap("level", scales="free", ncol=1)
     + scale_fill_manual(values={"Gradient": "#9ecae1", "Ours": "#e41a1c"})
     + scale_x_discrete(labels=lambda xs: [s.split(" | ", 1)[1] for s in xs])
     + labs(x="", y="mean CPR AUC (avg over tasks, 95% CI)", fill="",
            title="MIB CPR: gradient attribution vs. MAttr")
     + theme(figure_size=(5, 5.5), legend_position="top", axis_text_y=element_text(size=7)))
p.save(OUT / "cpr_summary.png", dpi=150); p.save(OUT / "cpr_summary.pdf")
print("Saved cpr_summary")
print(df[["level", "group", "method", "mean", "n"]].to_string(index=False))
