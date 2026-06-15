import torch
import numpy as np
from plotnine import (ggplot, aes, geom_bin2d, geom_abline, labs,
                      theme_minimal, scale_fill_continuous, scale_x_log10,
                      scale_y_log10)
from mizani.transforms import log_trans
import pandas as pd

d300 = torch.load("plots/attribution_months_scores.pt", weights_only=False)
d3k = torch.load("plots/attribution_months_3k_scores.pt", weights_only=False)

from plotnine import facet_wrap, theme, element_text

s300 = d300["scores"]  # [32, seq_len, 14336]
s3k = d3k["scores"]

num_layers = s300.shape[0]
seq_len = s300.shape[1]
d_int = s300.shape[2]

# Compute per-layer ranks
rows = []
for layer in range(num_layers):
    flat300 = s300[layer].flatten()
    flat3k = s3k[layer].flatten()
    r3 = flat300.argsort(descending=True).argsort().numpy() + 1
    rk = flat3k.argsort(descending=True).argsort().numpy() + 1
    layer_label = np.full(len(r3), layer)
    rows.append(np.stack([r3, rk, layer_label], axis=1))

data = np.concatenate(rows, axis=0)
df = pd.DataFrame(data, columns=["rank_300", "rank_3000", "layer"])
df["layer"] = df["layer"].astype(int)

p = (
    ggplot(df, aes(x="rank_300", y="rank_3000"))
    + geom_bin2d(bins=80)
    + geom_abline(slope=1, intercept=0, linetype="dashed", color="red", alpha=0.5, size=0.3)
    + scale_x_log10()
    + scale_y_log10()
    + scale_fill_continuous(trans=log_trans())
    + facet_wrap("layer", ncol=8, labeller="label_both")
    + labs(x="Rank (300 steps)", y="Rank (3000 steps)",
           title="Neuron ranks: 300 vs 3000 steps, by layer")
    + theme_minimal()
    + theme(
        strip_text=element_text(size=7),
        axis_text=element_text(size=5),
        axis_title=element_text(size=8),
    )
)

p.save("plots/rank_scatter_by_layer.png", dpi=150, width=16, height=10)
print("Saved plots/rank_scatter_by_layer.png")
