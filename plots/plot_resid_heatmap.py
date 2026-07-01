"""Layer x token-position importance heatmap from the per-token resid concept runs.
Scalar resid score per (layer, token); fill = per-facet percentile rank. Facets:
(mode/concept) rows x task cols. x-axis = actual tokens of a reference prompt."""
import pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from plotnine import (
    ggplot, aes, geom_tile, facet_grid, labs, scale_fill_gradient,
    scale_y_continuous, theme_set, theme_bw, theme, element_text, element_blank,
)
import sys
sys.path.insert(0, "src")
from learning_to_attribute.data.arithmetic_wild import ArithmeticWildDataset

ARITH = "/home/guests/aryaman/arithmetic-wild/datasets/Llama-3.1-8B"
SRC = {"Necessity": "results/arith_resid_concepts_necessary.pkl",
       "Sufficiency": "results/arith_resid_concepts_sufficient.pkl"}
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

theme_set(theme_bw(base_size=8) + theme(
    text=element_text(color="#000", family="Inter"),
    axis_title=element_text(size=7), axis_text=element_text(size=4),
    axis_text_x=element_text(size=4, rotation=90, hjust=1, vjust=0.5),
    panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
    panel_spacing_x=0.03, panel_spacing_y=0.03, strip_background=element_blank(),
    strip_text=element_text(size=6), legend_title=element_text(size=7),
    legend_text=element_text(size=6), legend_key_size=8, legend_position="right"))


def token_labels(task):
    ds = ArithmeticWildDataset(task, ARITH); ds.build_schema(tok)
    ids, spans, roles = ds._classify(ds.bases[0], tok)
    labs_ = []
    for sp, role in zip(spans, roles):
        if role in ("input", "offset"):
            labs_.append(f"[{role}]")
        else:
            t = tok.decode(ids[0, sp[0]]).replace("\n", "\\n").strip()
            labs_.append(t if t else "_")
    # disambiguate duplicate labels by position so they stay ordered/distinct
    return [f"{i:02d} {l}" for i, l in enumerate(labs_)]


rows = []
for mode, fn in SRC.items():
    r = pickle.load(open(fn, "rb")); L = r["n_layers"]
    for t in r["tasks"]:
        S = r["num_spans"][t]; labels = token_labels(t)
        for c in r["scores"][t]:
            sc = r["scores"][t][c].view(L, S).numpy()
            rank = sc.flatten().argsort().argsort().reshape(L, S) / (L * S - 1)  # percentile
            pr = np.clip((rank - 0.85) / 0.15, 0, 1)              # show only top ~15% nodes
            for li in range(L):
                for si in range(S):
                    rows.append({"Mode": mode, "task": t.capitalize(), "Concept": c,
                                 "layer": li, "tok": labels[si], "pct": pr[li, si]})

df = pd.DataFrame(rows)
df["task"] = pd.Categorical(df["task"], ["Addition", "Months", "Weekdays", "Hours"])
df["Concept"] = pd.Categorical(df["Concept"], ["input", "offset", "output"], ordered=True)
df["row"] = pd.Categorical(df["Mode"] + " / " + df["Concept"].astype(str),
                           categories=[f"{m} / {c}" for m in ["Necessity", "Sufficiency"]
                                       for c in ["input", "offset", "output"]], ordered=True)

p = (ggplot(df, aes("tok", "layer", fill="pct"))
     + geom_tile()
     + facet_grid("row ~ task", scales="free_x", space="free_x")
     + scale_fill_gradient(low="#ffffff", high="#E41A1C", name="top 15%")
     + scale_y_continuous(expand=(0, 0), breaks=[0, 8, 16, 24, 31])
     + labs(x="Token position", y="Layer")
     + theme(figure_size=(7.5, 9.0)))
p.save("plots/arith_resid_heatmap.pdf", verbose=False)
print("wrote plots/arith_resid_heatmap.pdf")
