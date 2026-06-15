"""Per-layer task analysis for shared-basis multitask DAS.

For each task, find where its circuit lives across the 16 layers: fraction of the
top-5% selected (layer,span,dim) scores falling in each layer. Outputs a task x layer
heatmap (grouped by syntactic family) and per-family layer profiles.
Reads results/pythia1b_multitask_das.pkl. Run on sc.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from plotnine import (
    ggplot, aes, geom_tile, geom_line, geom_point, labs, scale_fill_gradient,
    scale_color_brewer, theme_bw, theme_set, theme, element_text, element_blank,
)

R = Path("results")
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=8) + theme(text=element_text(family="Inter")))

d = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
das_dim = d["das_dim"]
tasks = d["tasks"]
ns = d["num_spans"]
scores = d["scores"]  # task -> tensor (L*S*das_dim,)


def family(t):
    n = t.replace("syntaxgym/", "")
    for fam in ("agr_refl", "agr_sv", "agr", "npi", "garden", "gss", "cleft", "filler_gap"):
        if n.startswith(fam):
            return {"agr_refl": "agr", "agr_sv": "agr"}.get(fam, fam)
    return n.split("_")[0]


# infer n_layers from one task
def nlayers(t):
    return scores[t].numel() // (ns[t] * das_dim)


rows = []
profile = {}  # task -> per-layer fraction
for t in tasks:
    L = nlayers(t)
    s = scores[t].reshape(L, ns[t], das_dim).float()
    flat = s.reshape(-1)
    thr = torch.quantile(flat, 0.95)  # top 5%
    sel = (s >= thr).sum(dim=(1, 2)).numpy().astype(float)  # per layer count
    frac = sel / max(sel.sum(), 1)
    profile[t] = frac
    for l in range(L):
        rows.append({"task": t.replace("syntaxgym/", ""), "layer": l,
                     "frac": frac[l], "family": family(t)})

df = pd.DataFrame(rows)
# order tasks by family then by centroid layer
cent = {t.replace("syntaxgym/", ""): float(np.average(np.arange(len(profile[t])), weights=profile[t]+1e-9))
        for t in tasks}
fam_order = ["agr", "npi", "garden", "gss", "cleft", "filler_gap"]
tlabels = sorted(df["task"].unique(),
                 key=lambda x: (fam_order.index(df[df.task==x]["family"].iloc[0])
                                if df[df.task==x]["family"].iloc[0] in fam_order else 99,
                                cent.get(x, 0)))
df["task"] = pd.Categorical(df["task"], categories=tlabels[::-1], ordered=True)

p = (ggplot(df, aes("factor(layer)", "task", fill="frac"))
     + geom_tile()
     + scale_fill_gradient(low="#ffffff", high="#762a83")
     + labs(x="layer", y="", fill="frac of\ncircuit",
            title="Where each task's circuit lives (top-5% dims per layer)")
     + theme(figure_size=(6, 7), axis_text_y=element_text(size=5),
             axis_text_x=element_text(size=6), panel_grid=element_blank()))
p.save(OUT / "multitask_perlayer.png", dpi=150)
print("Saved", OUT / "multitask_perlayer.png")

# per-family mean profile
frows = []
for t in tasks:
    fam = family(t)
    for l, v in enumerate(profile[t]):
        frows.append({"family": fam, "layer": l, "frac": v})
fd = pd.DataFrame(frows).groupby(["family", "layer"], as_index=False)["frac"].mean()
fd["family"] = pd.Categorical(fd["family"], categories=fam_order, ordered=True)
p2 = (ggplot(fd, aes("layer", "frac", color="family"))
      + geom_line(size=0.8) + geom_point(size=1.2)
      + scale_color_brewer(type="qual", palette="Dark2")
      + labs(x="layer", y="mean frac of circuit", color="",
             title="Per-family layer profile (shared-basis DAS)")
      + theme(figure_size=(5, 2.8), legend_position="right"))
p2.save(OUT / "multitask_perlayer_family.png", dpi=150)
print("Saved", OUT / "multitask_perlayer_family.png")

# text summary: peak layer per family
print("\nPeak layer by family (mean profile argmax):")
for fam in fam_order:
    sub = fd[fd.family == fam]
    if len(sub):
        pk = int(sub.loc[sub.frac.idxmax(), "layer"])
        print(f"  {fam:12s} peak L{pk}")
