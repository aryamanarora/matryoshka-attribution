"""Faceted top-100 | bottom-100 node-overlap heatmaps for span-mode methods.
Same method order in both panels (clustered on top-100)."""
import torch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from plotnine import (
    ggplot, aes, geom_tile, geom_text, scale_fill_gradient, scale_x_discrete,
    scale_y_discrete, facet_wrap, coord_equal, labs, theme_set, theme_bw, theme,
    element_text, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(text=element_text(color="#000", family="Inter"), figure_size=(9.0, 4.8),
            axis_title=element_blank(), axis_text=element_text(size=5.5),
            axis_text_x=element_text(size=5.5, rotation=45, hjust=1.0, vjust=1.0),
            panel_grid_major=element_blank(), strip_background=element_blank(),
            strip_text=element_text(size=8),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=10, legend_position="right"))

RES = "results/sva/npi_any_subj-relc_llama3_mlp_span_"; TOPK = 100
METHODS = [
    # CE x iso/cause x log/log_both
    ("CE iso", "sufficient_hard_topk_adam_ce_bs1"),
    ("CE iso·lb", "sufficient_hard_topk_adam_ce_logboth_bs1"),
    ("CE cau", "necessary_hard_topk_adam_ce_bs1"),
    ("CE cau·lb", "necessary_hard_topk_adam_ce_logboth_bs1"),
    # logit-diff x iso/cause x log/log_both
    ("LD iso", "sufficient_hard_topk_adam_bs1"),
    ("LD iso·lb", "sufficient_hard_topk_adam_logboth_bs1"),
    ("LD cau", "necessary_hard_topk_adam_bs1"),
    ("LD cau·lb", "necessary_hard_topk_adam_logboth_bs1"),
    # acc T=.5 x iso/cause x log/log_both
    ("acc iso", "sufficient_hard_topk_adam_acc_bs1_t05"),
    ("acc iso·lb", "sufficient_hard_topk_adam_acc_logboth_bs1_t05"),
    ("acc cau", "necessary_hard_topk_adam_acc_bs1_t05"),
    ("acc cau·lb", "necessary_hard_topk_adam_acc_logboth_bs1_t05"),
    # gradient reference
    ("IG", "ig"), ("RelP", "relp"), ("IxG", "ixg"),
]
labels, S = [], []
for lab, tag in METHODS:
    S.append(torch.load(RES + tag + ".scores.pt", map_location="cpu").float()); labels.append(lab)
n = len(labels)


def mat(largest):
    tops = [set(torch.topk(s, TOPK, largest=largest).indices.tolist()) for s in S]
    return np.array([[len(tops[i] & tops[j]) / TOPK for j in range(n)] for i in range(n)])


top, bot = mat(True), mat(False)
# cluster order from the top-100 matrix, applied to both panels
d = 1 - top; np.fill_diagonal(d, 0.0)
order = leaves_list(linkage(squareform(d, checks=False), method="average", optimal_ordering=True))
ordl = [labels[i] for i in order]
top, bot = top[np.ix_(order, order)], bot[np.ix_(order, order)]

rows = []
for panel, m in [("Top 100 (most important)", top), ("Bottom 100 (least important)", bot)]:
    for i in range(n):
        for j in range(n):
            rows.append({"a": ordl[i], "b": ordl[j], "ov": m[i, j] * 100, "panel": panel})
df = pd.DataFrame(rows)
df["a"] = pd.Categorical(df["a"], ordl)
df["b"] = pd.Categorical(df["b"], list(reversed(ordl)))
df["panel"] = pd.Categorical(df["panel"], ["Top 100 (most important)", "Bottom 100 (least important)"])
p = (ggplot(df, aes("a", "b", fill="ov"))
     + geom_tile(color="white", size=0.3)
     + geom_text(aes(label="ov.round().astype(int).astype(str)"), size=3.6)
     + scale_fill_gradient(low="#fff7f3", high="#e41a1c", limits=[0, 100], name="% overlap")
     + facet_wrap("panel")
     + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
     + coord_equal() + labs(x="", y=""))
p.save("plots/npi_overlap_facet_span.pdf", verbose=False)
print("order:", ordl)
print("wrote plots/npi_overlap_facet_span.pdf")
