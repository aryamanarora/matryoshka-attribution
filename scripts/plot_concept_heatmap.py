"""Where do the top SAE features live? Heatmap of the top-1% features' distribution over
(span x layer), facetted by concept x task x mode (necessity/sufficiency).
Cell = fraction of that (mode,task,concept)'s top-1% features at (layer, span)."""
import pickle
import numpy as np
import pandas as pd
import torch
from plotnine import (
    ggplot, aes, geom_tile, facet_grid, labs, scale_fill_gradient,
    scale_x_discrete, scale_y_continuous, theme_set, theme_bw, theme,
    element_text, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(text=element_text(color="#000", family="Inter"),
            axis_title=element_text(size=7), axis_text=element_text(size=5),
            axis_text_x=element_text(size=5, rotation=45, hjust=1.0),
            panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
            panel_spacing_x=0.03, panel_spacing_y=0.03,
            strip_background=element_blank(), strip_text=element_text(size=6),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=8, legend_position="right"))

SRC = {"Necessity": "results/arith_sae_concepts.pkl",
       "Sufficiency": "results/arith_sae_concepts_sufficient.pkl"}
SPANS = ["input", "offset", "last"]
TOP_FRAC = 0.01

rows = []
for mode, fn in SRC.items():
    r = pickle.load(open(fn, "rb"))
    L = r["n_layers"]; S = max(r["num_spans"].values())
    for t in r["tasks"]:
        for c in r["scores"][t]:
            sc = r["scores"][t][c]                      # [L*S*(d_sae+1)]
            W = sc.numel() // (L * S)
            d_sae = W - 1
            feat = sc.view(L, S, W)[:, :, :d_sae]       # drop error node
            k = max(1, int(TOP_FRAC * feat.numel()))
            thr = torch.topk(feat.flatten(), k).values.min()
            cnt = (feat >= thr).sum(dim=2).numpy().astype(float)   # [L, S] count of top feats
            cnt /= cnt.sum()                            # fraction within this facet
            for li in range(L):
                for si in range(S):
                    rows.append({"Mode": mode, "task": t.capitalize(), "Concept": c,
                                 "layer": li, "span": SPANS[si], "frac": cnt[li, si]})

df = pd.DataFrame(rows)
df["task"] = pd.Categorical(df["task"], ["Addition", "Months", "Weekdays", "Hours"])
df["Concept"] = pd.Categorical(df["Concept"], ["input", "offset", "output"], ordered=True)
df["row"] = pd.Categorical(df["Mode"].astype(str) + " / " + df["Concept"].astype(str),
                           categories=[f"{m} / {c}" for m in ["Necessity", "Sufficiency"]
                                       for c in ["input", "offset", "output"]], ordered=True)
df["span"] = pd.Categorical(df["span"], SPANS, ordered=True)

p = (ggplot(df, aes("span", "layer", fill="frac"))
     + geom_tile()
     + facet_grid("row ~ task")
     + scale_fill_gradient(low="#ffffff", high="#E41A1C", name="frac of\ntop-1%")
     + scale_x_discrete(expand=(0, 0))
     + scale_y_continuous(expand=(0, 0), breaks=[0, 8, 16, 24, 31])
     + labs(x="Span", y="Layer")
     + theme(figure_size=(5.5, 8.0)))
p.save("plots/arith_concept_heatmap.pdf", verbose=False)
print("wrote plots/arith_concept_heatmap.pdf")
