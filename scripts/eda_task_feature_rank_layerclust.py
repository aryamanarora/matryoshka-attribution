"""Task x feature heatmap, ALL features, colored by within-task rank (0=top).
Features grouped on x by layer (0..15); WITHIN each layer block, features are
hierarchically clustered by their across-task rank profile. Task rows clustered by
overlap. Reads results/pythia1b_multitask_das.pkl."""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
d = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
das_dim = d["das_dim"]; tasks = d["tasks"]; feats = d["feature_vectors"]
short = lambda t: t.replace("syntaxgym/", "")

F = np.stack([np.asarray(feats[t]) for t in tasks])
ntask, nfeat = F.shape
L = nfeat // das_dim
rank = np.empty_like(F)
for i in range(ntask):
    rank[i, np.argsort(-F[i])] = np.arange(nfeat)

# within-layer clustering on across-task rank profiles
forder = []
for l in range(L):
    idx = np.arange(l * das_dim, (l + 1) * das_dim)
    X = rank[:, idx].T.astype(float)            # [das_dim, ntask]
    sub = dendrogram(linkage(X, "average"), no_plot=True)["leaves"]
    forder.extend(idx[sub].tolist())

# task rows clustered by overlap
M = np.array(d["overlap_matrix"]); np.fill_diagonal(M, 1.0)
Dm = 1 - M; Dm = (Dm + Dm.T) / 2; np.fill_diagonal(Dm, 0)
torder = dendrogram(linkage(squareform(Dm, checks=False), "average"), no_plot=True)["leaves"]

A = rank[np.ix_(torder, forder)]
lay_sorted = (np.array(forder) // das_dim)

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(A, aspect="auto", cmap="viridis_r", vmin=0, vmax=nfeat - 1, interpolation="nearest")
ax.set_yticks(range(ntask)); ax.set_yticklabels([short(tasks[i]) for i in torder], fontsize=5)
bounds = [0] + [k for k in range(1, nfeat) if lay_sorted[k] != lay_sorted[k-1]] + [nfeat]
for b in bounds[1:-1]:
    ax.axvline(b - 0.5, color="white", lw=0.6)
centers = [(bounds[i] + bounds[i+1]) / 2 - 0.5 for i in range(len(bounds)-1)]
ax.set_xticks(centers); ax.set_xticklabels([f"L{lay_sorted[bounds[i]]}" for i in range(len(bounds)-1)], fontsize=6)
ax.set_xlabel("features grouped by layer; clustered within each layer")
ax.set_title("Task x feature, colored by within-task rank (0=top); all features")
cb = fig.colorbar(im, ax=ax, fraction=0.025); cb.set_label("within-task rank (0=most important)", fontsize=7)
plt.tight_layout(); fig.savefig(OUT / "eda_task_feature_rank_layerclust.png", dpi=150)
print("Saved eda_task_feature_rank_layerclust.png")
