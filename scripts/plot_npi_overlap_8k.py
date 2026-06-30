"""Top-100 node overlap heatmap for the 2k/8k step variants of logit-diff/CE/prob (bs=1)."""
import torch
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
        figure_size=(3.6, 3.1),
        axis_title=element_blank(),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=1.0, vjust=1.0),
        panel_grid_major=element_blank(),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=10,
        legend_position="right",
    )
)

RES = "results/sva"; TOPK = 100
B = "npi_any_subj-relc_llama3_mlp_sufficient_hard_topk_adam"
METHODS = [
    ("logit-diff 2k", f"{B}_bs1"),
    ("logit-diff 8k", f"{B}_bs1_s8000"),
    ("CE 2k", f"{B}_ce_bs1"),
    ("CE 8k", f"{B}_ce_bs1_s8000"),
    ("prob 2k", f"{B}_prob_bs1"),
    ("prob 8k", f"{B}_prob_bs1_s8000"),
]
labels, tops = [], []
for lab, tag in METHODS:
    s = torch.load(f"{RES}/{tag}.scores.pt", map_location="cpu").float()
    labels.append(lab); tops.append(set(torch.topk(s, TOPK).indices.tolist()))

print(f"{'':14s}" + "".join(f"{l:>14s}" for l in labels))
rows = []
for i, li in enumerate(labels):
    print(f"{li:14s}" + "".join(f"{len(tops[i]&tops[j]):14d}" for j in range(len(labels))))
    for j, lj in enumerate(labels):
        rows.append({"a": li, "b": lj, "ov": 100.0 * len(tops[i] & tops[j]) / TOPK})

df = pd.DataFrame(rows)
df["a"] = pd.Categorical(df["a"], categories=labels)
df["b"] = pd.Categorical(df["b"], categories=list(reversed(labels)))
p = (
    ggplot(df, aes("a", "b", fill="ov"))
    + geom_tile(color="white", size=0.4)
    + geom_text(aes(label="ov.round().astype(int).astype(str)"), size=6)
    + scale_fill_gradient(low="#fff7f3", high="#e41a1c", limits=[0, 100], name="% overlap")
    + scale_x_discrete(expand=(0, 0)) + scale_y_discrete(expand=(0, 0))
    + coord_equal() + labs(x="", y="")
)
p.save("plots/npi_top100_overlap_8k.pdf", verbose=False)
print("\nwrote plots/npi_top100_overlap_8k.pdf")
