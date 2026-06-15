"""Task x feature heatmap, cells colored by the dominant SPAN NAME (linguistic role,
e.g. {det1}, {verb}, {the}) of each selected feature, with a legend.
Reads results/pythia1b_multitask_das.pkl. Run on sc."""
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
d = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
das_dim = d["das_dim"]; tasks = d["tasks"]; ns = d["num_spans"]
feats = d["feature_vectors"]; spnames = d["span_names"]
short = lambda t: t.replace("syntaxgym/", "")
norm = lambda s: s.strip("{}_ ").lower()  # canonicalize span names a bit

F = np.stack([np.asarray(feats[t]) for t in tasks])
nfeat = F.shape[1]
sel = np.zeros_like(F, bool)
name_grid = np.empty(F.shape, dtype=object)
for i, t in enumerate(tasks):
    sel[i] = F[i] >= np.quantile(F[i], 0.95)
    L = d["scores"][t].numel() // (ns[t] * das_dim)
    full = d["scores"][t].reshape(L, ns[t], das_dim).float()
    arg = full.argmax(dim=1).numpy().reshape(-1)  # dominant span idx per feature
    nm = spnames[t]
    for f in range(nfeat):
        if sel[i, f]:
            idx = arg[f]
            name_grid[i, f] = norm(nm[idx]) if idx < len(nm) else "?"

keep = sel.sum(0) > 0
S = sel[:, keep]; NG = name_grid[:, keep]

# frequency of names among selected cells -> top K + other
freq = Counter(NG[S])
K = 18
top = [n for n, _ in freq.most_common(K)]
code = {n: i for i, n in enumerate(top)}
OTHER = len(top)
print(f"{len(freq)} unique span names; top {len(top)} kept. Most common:")
for n, c in freq.most_common(K):
    print(f"   {n:16s} {c}")

A = np.full(S.shape, np.nan)
for i in range(S.shape[0]):
    for f in range(S.shape[1]):
        if S[i, f]:
            A[i, f] = code.get(NG[i, f], OTHER)

# cluster order (tasks by overlap, features by co-usage)
M = np.array(d["overlap_matrix"]); np.fill_diagonal(M, 1.0)
D = 1 - M; D = (D + D.T) / 2; np.fill_diagonal(D, 0)
torder = dendrogram(linkage(squareform(D, checks=False), "average"), no_plot=True)["leaves"]
forder = dendrogram(linkage(S.T.astype(float), "average"), no_plot=True)["leaves"] \
    if S.shape[1] > 2 else list(range(S.shape[1]))
A = A[np.ix_(torder, forder)]

base = list(plt.get_cmap("tab20").colors)[:len(top)] + [(0.7, 0.7, 0.7)]  # +other grey
cmap = ListedColormap(base); cmap.set_bad("#ffffff")
fig, ax = plt.subplots(figsize=(8.5, 5))
im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(top)+0.5, interpolation="nearest")
ax.set_yticks(range(len(torder))); ax.set_yticklabels([short(tasks[i]) for i in torder], fontsize=5)
ax.set_xlabel(f"shared-basis features (clustered; {keep.sum()} used)")
ax.set_title("Task x feature usage, colored by dominant span name")
handles = [Patch(color=base[i], label=top[i]) for i in range(len(top))] + \
          [Patch(color=base[-1], label="other")]
ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
          fontsize=6, title="span name", title_fontsize=7, ncol=1)
plt.tight_layout(); fig.savefig(OUT / "eda_task_feature_spanname.png", dpi=150, bbox_inches="tight")
print("Saved eda_task_feature_spanname.png")
