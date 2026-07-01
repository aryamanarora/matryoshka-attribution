"""Task x feature heatmap, cells colored by the dominant SPAN (normalized token
position in that task's template) of each selected (layer,dim) feature.
Reads results/pythia1b_multitask_das.pkl. Run on sc."""
import pickle
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
d = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
das_dim = d["das_dim"]; tasks = d["tasks"]; ns = d["num_spans"]; feats = d["feature_vectors"]
short = lambda t: t.replace("syntaxgym/", "")

F = np.stack([np.asarray(feats[t]) for t in tasks])      # [29, L*das_dim]
nfeat = F.shape[1]
# selection (top-5% per task, same as before) and dominant-span (normalized) per feature
sel = np.zeros_like(F, bool)
span_pos = np.full_like(F, np.nan, dtype=float)
for i, t in enumerate(tasks):
    sel[i] = F[i] >= np.quantile(F[i], 0.95)
    L = d["scores"][t].numel() // (ns[t] * das_dim)
    full = d["scores"][t].reshape(L, ns[t], das_dim).float()
    arg = full.argmax(dim=1).numpy().reshape(-1)          # [L*das_dim] dominant span idx
    span_pos[i] = arg / max(ns[t] - 1, 1)                 # normalize 0..1

keep = sel.sum(0) > 0
S = sel[:, keep]
P = np.where(S, span_pos[:, keep], np.nan)

# cluster tasks (overlap) + features (co-usage), same orders as layer version
M = np.array(d["overlap_matrix"]); np.fill_diagonal(M, 1.0)
D = 1 - M; D = (D + D.T) / 2; np.fill_diagonal(D, 0)
torder = dendrogram(linkage(squareform(D, checks=False), "average"), no_plot=True)["leaves"]
forder = dendrogram(linkage(S.T.astype(float), "average"), no_plot=True)["leaves"] \
    if S.shape[1] > 2 else list(range(S.shape[1]))
A = P[np.ix_(torder, forder)]

fig, ax = plt.subplots(figsize=(7.5, 5))
cmap = plt.get_cmap("coolwarm").copy(); cmap.set_bad("#f0f0f0")
im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
ax.set_yticks(range(len(torder))); ax.set_yticklabels([short(tasks[i]) for i in torder], fontsize=5)
ax.set_xlabel(f"shared-basis features (clustered; {keep.sum()} used)")
ax.set_title("Task x feature usage, colored by dominant span (0=first token … 1=last)")
cb = fig.colorbar(im, ax=ax, fraction=0.025); cb.set_label("normalized span position", fontsize=7)
plt.tight_layout(); fig.savefig(OUT / "eda_task_feature_span.png", dpi=150)
print("Saved eda_task_feature_span.png")
