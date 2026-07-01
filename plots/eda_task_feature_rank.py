"""Task x feature heatmap, cells colored by the RANK of that feature within the task
(0 = the task's highest-scored feature). Only top-5% selected cells shown.
Reads results/pythia1b_multitask_das.pkl. Run on sc."""
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
tasks = d["tasks"]; feats = d["feature_vectors"]
short = lambda t: t.replace("syntaxgym/", "")

F = np.stack([np.asarray(feats[t]) for t in tasks])  # [29, nfeat]
n = F.shape[1]
rank = np.empty_like(F)            # rank of each feature within its task (0=best)
sel = np.zeros_like(F, bool)
for i in range(len(tasks)):
    order = np.argsort(-F[i])
    rank[i, order] = np.arange(n)
    sel[i] = F[i] >= np.quantile(F[i], 0.95)

keep = sel.sum(0) > 0
S = sel[:, keep]; Rk = rank[:, keep]
A = np.where(S, Rk, np.nan)
vmax = float(np.nanmax(np.where(S, Rk, np.nan)))  # ~ top-5% size

# cluster tasks (overlap) + features (co-usage), same orders as other versions
M = np.array(d["overlap_matrix"]); np.fill_diagonal(M, 1.0)
D = 1 - M; D = (D + D.T) / 2; np.fill_diagonal(D, 0)
torder = dendrogram(linkage(squareform(D, checks=False), "average"), no_plot=True)["leaves"]
forder = dendrogram(linkage(S.T.astype(float), "average"), no_plot=True)["leaves"] \
    if S.shape[1] > 2 else list(range(S.shape[1]))
A = A[np.ix_(torder, forder)]

fig, ax = plt.subplots(figsize=(7.5, 5))
cmap = plt.get_cmap("viridis_r").copy(); cmap.set_bad("#f0f0f0")
im = ax.imshow(A, aspect="auto", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
ax.set_yticks(range(len(torder))); ax.set_yticklabels([short(tasks[i]) for i in torder], fontsize=5)
ax.set_xlabel(f"shared-basis features (clustered; {keep.sum()} used)")
ax.set_title("Task x feature usage, colored by within-task rank (0 = top feature)")
cb = fig.colorbar(im, ax=ax, fraction=0.025); cb.set_label("rank within task (0=most important)", fontsize=7)
plt.tight_layout(); fig.savefig(OUT / "eda_task_feature_rank.png", dpi=150)
print(f"Saved eda_task_feature_rank.png (max selected rank ~{vmax:.0f})")
