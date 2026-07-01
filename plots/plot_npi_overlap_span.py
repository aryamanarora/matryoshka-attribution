"""Top-100 node overlap heatmap for span-mode methods, rows/cols ordered by hierarchical
clustering (so similar attribution sets sit adjacent)."""
import torch
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from plotnine import (
    ggplot, aes, geom_tile, geom_text, scale_fill_gradient, scale_x_discrete,
    scale_y_discrete, coord_equal, labs, theme_set, theme_bw, theme,
    element_text, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(text=element_text(color="#000", family="Inter"), figure_size=(4.2, 3.6),
            axis_title=element_blank(), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=1.0, vjust=1.0),
            panel_grid_major=element_blank(),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=10, legend_position="right"))

RES = "results/sva/npi_any_subj-relc_llama3_mlp_span_"; TOPK = 100
METHODS = [
    ("acc T=1", "sufficient_hard_topk_adam_acc_bs1"),
    ("acc T=.5", "sufficient_hard_topk_adam_acc_bs1_t05"),
    ("hinge", "sufficient_hard_topk_adam_hinge_bs1"),
    ("logit-diff", "sufficient_hard_topk_adam_bs1"),
    ("logit", "sufficient_hard_topk_adam_logit_bs1"),
    ("CE", "sufficient_hard_topk_adam_ce_bs1"),
    ("prob", "sufficient_hard_topk_adam_prob_bs1"),
    ("IG", "ig"), ("RelP", "relp"), ("IxG", "ixg"),
]
labels, tops = [], []
for lab, tag in METHODS:
    s = torch.load(RES + tag + ".scores.pt", map_location="cpu").float()
    labels.append(lab); tops.append(set(torch.topk(s, TOPK).indices.tolist()))

n = len(labels)
ov = np.array([[len(tops[i] & tops[j]) / TOPK for j in range(n)] for i in range(n)])
dist = 1 - ov
np.fill_diagonal(dist, 0.0)
order = leaves_list(linkage(squareform(dist, checks=False), method="average",
                            optimal_ordering=True))
labels = [labels[i] for i in order]; ov = ov[np.ix_(order, order)]

rows = [{"a": labels[i], "b": labels[j], "ov": ov[i, j] * 100}
        for i in range(n) for j in range(n)]
df = pd.DataFrame(rows)
df["a"] = pd.Categorical(df["a"], labels)
df["b"] = pd.Categorical(df["b"], list(reversed(labels)))
p = (ggplot(df, aes("a", "b", fill="ov"))
     + geom_tile(color="white", size=0.4)
     + geom_text(aes(label="ov.round().astype(int).astype(str)"), size=6)
     + scale_fill_gradient(low="#fff7f3", high="#e41a1c", limits=[0, 100], name="% overlap")
     + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
     + coord_equal() + labs(x="", y=""))
p.save("plots/npi_top100_overlap_span.pdf", verbose=False)
print("clustered order:", labels)
print("wrote plots/npi_top100_overlap_span.pdf")
