"""Span analysis + layer-colored task x feature heatmap for shared-basis multitask DAS.

  eda_span_heatmap.png      - task x span-index, fraction of circuit in each span
  eda_span_position.png     - circuit mass vs normalized span position, by family
  eda_task_feature_layer.png- task x feature heatmap, cells colored by LAYER
Reads results/pythia1b_multitask_das.pkl. Run on sc.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from plotnine import (ggplot, aes, geom_point, geom_smooth, geom_tile, labs,
                      scale_color_brewer, scale_fill_gradient, theme_bw, theme_set,
                      theme, element_text, element_blank)

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=8) + theme(text=element_text(family="Inter")))
d = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
das_dim = d["das_dim"]; tasks = d["tasks"]; ns = d["num_spans"]
spnames = d["span_names"]; feats = d["feature_vectors"]
short = lambda t: t.replace("syntaxgym/", "")
FAM = ("agr", "npi", "garden", "gss", "cleft", "filler_gap")
def family(t):
    n = short(t)
    return next((f for f in FAM if n.startswith(f)), n.split("_")[0])
fam_order = list(FAM)

# ---- per-task per-span circuit fraction (top-5% over layer x dim) ----
span_frac = {}; peak = []
for t in tasks:
    L = scores_len = d["scores"][t].numel() // (ns[t] * das_dim)
    s = d["scores"][t].reshape(L, ns[t], das_dim).float()
    sel = (s >= torch.quantile(s.reshape(-1), 0.95))
    m = sel.sum(dim=(0, 2)).numpy().astype(float)
    f = m / max(m.sum(), 1)
    span_frac[t] = f
    pk = int(np.argmax(f))
    peak.append((short(t), family(t), pk, ns[t], spnames[t][pk] if pk < len(spnames[t]) else "?", f[pk]))

# 1) task x span-index heatmap
maxS = max(ns.values())
rows = []
for t in tasks:
    for i in range(maxS):
        rows.append({"task": short(t), "family": family(t), "span": i,
                     "frac": span_frac[t][i] if i < ns[t] else np.nan})
df = pd.DataFrame(rows)
torder = sorted({short(t) for t in tasks},
                key=lambda x: (fam_order.index(df[df.task==x].family.iloc[0]), x))
df["task"] = pd.Categorical(df["task"], categories=torder[::-1], ordered=True)
p = (ggplot(df.dropna(), aes("factor(span)", "task", fill="frac")) + geom_tile()
     + scale_fill_gradient(low="#ffffff", high="#d94801")
     + labs(x="span index (token position in template)", y="", fill="frac of\ncircuit",
            title="Which span holds each task's circuit")
     + theme(figure_size=(5.5, 7), axis_text_y=element_text(size=5), panel_grid=element_blank()))
p.save(OUT / "eda_span_heatmap.png", dpi=150); print("Saved eda_span_heatmap.png")

# 2) circuit mass vs normalized span position, by family
rows = []
for t in tasks:
    S = ns[t]
    for i in range(S):
        rows.append({"family": family(t), "pos": i/(S-1) if S > 1 else 0.5,
                     "frac": span_frac[t][i]})
pd_ = pd.DataFrame(rows)
pd_["family"] = pd.Categorical(pd_["family"], categories=fam_order, ordered=True)
# binned family means (no loess dependency)
pd_["bin"] = pd.cut(pd_["pos"], bins=np.linspace(0, 1, 7), include_lowest=True)
binned = pd_.groupby(["family", "bin"], observed=True).agg(
    pos=("pos", "mean"), frac=("frac", "mean")).reset_index()
binned["family"] = pd.Categorical(binned["family"], categories=fam_order, ordered=True)
from plotnine import geom_line
p = (ggplot(pd_, aes("pos", "frac", color="family"))
     + geom_point(alpha=0.35, size=0.9)
     + geom_line(binned, aes("pos", "frac", color="family"), size=0.9)
     + scale_color_brewer(type="qual", palette="Dark2")
     + labs(x="normalized span position (0=first token … 1=last)",
            y="frac of circuit", color="",
            title="Circuit mass vs span position, by family (lines=binned mean)")
     + theme(figure_size=(5, 2.9)))
p.save(OUT / "eda_span_position.png", dpi=150); print("Saved eda_span_position.png")

# 3) task x feature heatmap, colored by LAYER
F = np.stack([np.asarray(feats[t]) for t in tasks])  # [29, L*das_dim]
nfeat = F.shape[1]; L = nfeat // das_dim
sel = np.zeros_like(F, bool)
for i in range(len(tasks)):
    sel[i] = F[i] >= np.quantile(F[i], 0.95)
keep = sel.sum(0) > 0
layer_of = (np.arange(nfeat) // das_dim)
S = sel[:, keep]; lay = layer_of[keep]
# cluster tasks (overlap) and features (co-usage)
M = np.array(d["overlap_matrix"]); np.fill_diagonal(M, 1.0)
D = 1 - M; D = (D+D.T)/2; np.fill_diagonal(D, 0)
torder = dendrogram(linkage(squareform(D, checks=False), "average"), no_plot=True)["leaves"]
forder = dendrogram(linkage(S.T.astype(float), "average"), no_plot=True)["leaves"] if S.shape[1] > 2 else list(range(S.shape[1]))
A = np.where(S, lay[None, :].astype(float), np.nan)  # value = layer if selected
A = A[np.ix_(torder, forder)]
fig, ax = plt.subplots(figsize=(7.5, 5))
cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#f0f0f0")
im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=L-1, interpolation="nearest")
ax.set_yticks(range(len(torder))); ax.set_yticklabels([short(tasks[i]) for i in torder], fontsize=5)
ax.set_xlabel(f"shared-basis features (clustered; {keep.sum()} used)")
ax.set_title("Task x feature usage, colored by layer")
cb = fig.colorbar(im, ax=ax, fraction=0.025); cb.set_label("layer", fontsize=7)
plt.tight_layout(); fig.savefig(OUT / "eda_task_feature_layer.png", dpi=150); plt.close(fig)
print("Saved eda_task_feature_layer.png")

# printed: peak span per task
print("\nPeak span per task (name @ index, frac):")
for name, fam, pk, S, sn, fr in sorted(peak, key=lambda x: (fam_order.index(x[1]) if x[1] in fam_order else 99, x[0])):
    print(f"  {name:30s} [{fam:10s}] span {pk}/{S-1} = {sn:18s} {fr:.2f}")
