"""Top-100-node overlap heatmap across NPI attribution methods (bs=1 MAttr + gradients).
Overlap(A,B) = |top100(A) ∩ top100(B)| / 100 * 100  (%).  Run: uv run python plots/plot_npi_overlap_heatmap.py
"""
import torch
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, geom_text, scale_fill_gradient, scale_x_discrete,
    scale_y_discrete, coord_equal, labs, theme_set, theme_bw, theme,
    element_text, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 5.2),
        axis_title=element_blank(),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=1.0, vjust=1.0),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=10,
        legend_position="right",
    )
)

RES = "results/sva"
TOPK = 100
B = "npi_any_subj-relc_llama3_mlp_sufficient_hard_topk_adam"
# (label, score-file tag) — order = matrix order
METHODS = [
    ("logit-diff log", f"{B}_bs1"),
    ("logit-diff unif", f"{B}_uniformk_bs1"),
    ("logit-diff adapt", f"{B}_adaptivek_bs1"),
    ("logit log", f"{B}_logit_bs1"),
    ("logit unif", f"{B}_logit_uniformk_bs1"),
    ("CE log", f"{B}_ce_bs1"),
    ("CE unif", f"{B}_ce_uniformk_bs1"),
    ("hinge log", f"{B}_hinge_bs1"),
    ("hinge adapt", f"{B}_hinge_adaptivek_bs1"),
    ("IG", "npi_any_subj-relc_llama3_mlp_ig"),
    ("RelP", "npi_any_subj-relc_llama3_mlp_relp"),
    ("IxG", "npi_any_subj-relc_llama3_mlp_ixg"),
]

labels, topsets = [], []
for lab, tag in METHODS:
    s = torch.load(f"{RES}/{tag}.scores.pt", map_location="cpu").float()
    top = set(torch.topk(s, TOPK).indices.tolist())
    labels.append(lab); topsets.append(top)

rows = []
for i, li in enumerate(labels):
    for j, lj in enumerate(labels):
        ov = 100.0 * len(topsets[i] & topsets[j]) / TOPK
        rows.append({"a": li, "b": lj, "ov": ov})
df = pd.DataFrame(rows)
df["a"] = pd.Categorical(df["a"], categories=labels)
df["b"] = pd.Categorical(df["b"], categories=list(reversed(labels)))  # diag top-left -> bottom-right

p = (
    ggplot(df, aes("a", "b", fill="ov"))
    + geom_tile(color="white", size=0.4)
    + geom_text(aes(label="ov.round().astype(int).astype(str)"), size=5)
    + scale_fill_gradient(low="#fff7f3", high="#e41a1c", limits=[0, 100], name="% overlap")
    + scale_x_discrete(expand=(0, 0))
    + scale_y_discrete(expand=(0, 0))
    + coord_equal()
    + labs(x="", y="")
)
out = "plots/npi_top100_overlap.pdf"
p.save(out, verbose=False)
print("wrote", out)
