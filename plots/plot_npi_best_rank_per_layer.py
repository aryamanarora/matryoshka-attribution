"""Best (lowest) global rank of any MLP neuron in each layer, per method. Line plot.
For method m and layer L: rank of m's top-scoring node among all nodes in that layer
(rank 1 = the single best node overall lives in L). Run: uv run python plots/plot_npi_best_rank_per_layer.py
"""
import torch
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, labs, scale_y_log10, scale_x_continuous,
    scale_color_brewer, theme_set, theme_bw, theme, element_text, element_line, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.6),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=8,
        legend_position="right",
    )
)

RES = "results/sva"
SEQ, INTER, NLAYER = 9, 14336, 32
B = "npi_any_subj-relc_llama3_mlp_sufficient_hard_topk_adam"
METHODS = [
    ("logit-diff log", f"{B}_bs1"),
    ("logit-diff unif", f"{B}_uniformk_bs1"),
    ("logit-diff adapt", f"{B}_adaptivek_bs1"),
    ("CE log", f"{B}_ce_bs1"),
    ("hinge log", f"{B}_hinge_bs1"),
    ("IG", "npi_any_subj-relc_llama3_mlp_ig"),
    ("RelP", "npi_any_subj-relc_llama3_mlp_relp"),
    ("IxG", "npi_any_subj-relc_llama3_mlp_ixg"),
]

rows = []
for lab, tag in METHODS:
    s = torch.load(f"{RES}/{tag}.scores.pt", map_location="cpu").float()
    layer_max = s.view(NLAYER, SEQ * INTER).max(dim=1).values        # best score per layer
    for L in range(NLAYER):
        best_rank = int((s > layer_max[L]).sum().item()) + 1          # global rank of that node
        rows.append({"method": lab, "layer": L, "rank": best_rank})

df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], [m[0] for m in METHODS])

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
def log_lab(bs):
    return [f"10{str(int(round(np.log10(b)))).translate(_SUP)}" if b > 0 else "0" for b in bs]

p = (
    ggplot(df, aes("layer", "rank", color="method"))
    + geom_line(size=0.5) + geom_point(size=0.7)
    + scale_y_log10(labels=log_lab)
    + scale_x_continuous(breaks=range(0, 32, 4))
    + scale_color_brewer(type="qual", palette="Set1")
    + labs(x="Layer", y="Best rank of any neuron in layer")
)
out = "plots/npi_best_rank_per_layer.pdf"
p.save(out, verbose=False)
print("wrote", out)
