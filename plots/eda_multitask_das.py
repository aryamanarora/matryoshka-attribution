"""Exploratory data analysis of the shared-basis multitask DAS run.

Produces:
  eda_dendrogram.png       - hierarchical clustering of tasks (1 - Jaccard overlap)
  eda_universality.png     - how many tasks share each (layer,dim) feature
  eda_task_feature.png     - clustered task x selected-feature heatmap
  eda_circuit_size.png     - effective circuit size (% dims to near-floor CE) by family
  eda_localizability.png   - learned-vs-random CE gap (how findable each task's circuit is)
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
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from plotnine import (ggplot, aes, geom_col, geom_point, labs, coord_flip,
                      scale_fill_brewer, scale_color_brewer, theme_bw, theme_set,
                      theme, element_text)

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=8) + theme(text=element_text(family="Inter")))
d = pickle.load(open(R / "pythia1b_multitask_das.pkl", "rb"))
das_dim = d["das_dim"]; tasks = d["tasks"]; ns = d["num_spans"]
sp = np.array(d["sparsities"]) * 100
short = lambda t: t.replace("syntaxgym/", "")
def family(t):
    n = short(t)
    for f in ("agr", "npi", "garden", "gss", "cleft", "filler_gap"):
        if n.startswith(f):
            return f
    return n.split("_")[0]
labels = [short(t) for t in tasks]

# ---- 1. dendrogram from overlap matrix ----
M = np.array(d["overlap_matrix"]); np.fill_diagonal(M, 1.0)
D = 1 - M; D = (D + D.T) / 2; np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
fig, ax = plt.subplots(figsize=(7, 5))
dendrogram(Z, labels=labels, leaf_font_size=5, ax=ax, color_threshold=0.6)
ax.set_title("Task clustering on shared-basis feature overlap (1 - Jaccard)")
ax.set_ylabel("distance"); plt.tight_layout()
fig.savefig(OUT / "eda_dendrogram.png", dpi=150); plt.close(fig)
print("Saved eda_dendrogram.png")

# ---- feature selection matrix: task x (L*das_dim), binary top-5% ----
feats = d["feature_vectors"]  # task -> np array len L*das_dim
F = np.stack([np.asarray(feats[t]) for t in tasks])  # [29, 1024]
sel = np.zeros_like(F, dtype=bool)
for i, t in enumerate(tasks):
    thr = np.quantile(F[i], 0.95)
    sel[i] = F[i] >= thr

# ---- 2. universality histogram ----
share_count = sel.sum(axis=0)              # per feature: #tasks selecting it
used = share_count[share_count > 0]
fig, ax = plt.subplots(figsize=(4.5, 2.8))
ax.hist(used, bins=range(1, len(tasks)+2), color="#762a83", edgecolor="white")
ax.set_xlabel("# tasks selecting a feature (of 29)"); ax.set_ylabel("# features")
ax.set_title(f"Feature universality: {(share_count>0).sum()}/{F.shape[1]} dims used; "
             f"{(share_count>=10).sum()} shared by >=10 tasks")
plt.tight_layout(); fig.savefig(OUT / "eda_universality.png", dpi=150); plt.close(fig)
print("Saved eda_universality.png")

# ---- 3. clustered task x selected-feature heatmap ----
keep = share_count > 0
S = sel[:, keep].astype(float)
order = dendrogram(Z, no_plot=True)["leaves"]
# order features by linkage too
if S.shape[1] > 2:
    Zf = linkage(S.T, method="average")
    forder = dendrogram(Zf, no_plot=True)["leaves"]
else:
    forder = list(range(S.shape[1]))
fig, ax = plt.subplots(figsize=(7, 5))
ax.imshow(S[np.ix_(order, forder)], aspect="auto", cmap="Purples", interpolation="nearest")
ax.set_yticks(range(len(order))); ax.set_yticklabels([labels[i] for i in order], fontsize=5)
ax.set_xlabel(f"shared-basis features (clustered; {keep.sum()} used dims)")
ax.set_title("Which tasks select which features (top-5%)")
plt.tight_layout(); fig.savefig(OUT / "eda_task_feature.png", dpi=150); plt.close(fig)
print("Saved eda_task_feature.png")

# ---- 4. effective circuit size: smallest % to reach floor+0.1 CE ----
rows = []
for t in tasks:
    ce = np.array(d["per_task_eval"][t]["eval_learned_ce"])
    floor = ce.min(); thr = floor + 0.1
    idx = np.argmax(ce <= thr) if (ce <= thr).any() else len(ce)-1
    rows.append({"task": short(t), "family": family(t), "pct": sp[idx], "floor": floor})
cs = pd.DataFrame(rows).sort_values("pct")
cs["task"] = pd.Categorical(cs["task"], categories=cs["task"], ordered=True)
p = (ggplot(cs, aes("task", "pct", fill="family")) + geom_col()
     + scale_fill_brewer(type="qual", palette="Dark2") + coord_flip()
     + labs(x="", y="% dims to reach (CE floor + 0.1)", fill="",
            title="Effective circuit size per task")
     + theme(figure_size=(5.5, 6), axis_text_y=element_text(size=5)))
p.save(OUT / "eda_circuit_size.png", dpi=150); print("Saved eda_circuit_size.png")

# ---- 5. localizability: mean(random - learned) CE over sweep ----
rows = []
for t in tasks:
    ev = d["per_task_eval"][t]
    gap = np.mean(np.array(ev["eval_random_ce"]) - np.array(ev["eval_learned_ce"]))
    rows.append({"task": short(t), "family": family(t), "gap": gap})
lz = pd.DataFrame(rows).sort_values("gap")
lz["task"] = pd.Categorical(lz["task"], categories=lz["task"], ordered=True)
p = (ggplot(lz, aes("task", "gap", fill="family")) + geom_col()
     + scale_fill_brewer(type="qual", palette="Dark2") + coord_flip()
     + labs(x="", y="mean CE(random) - CE(learned)  [higher = more localizable]",
            fill="", title="How localizable is each task's circuit?")
     + theme(figure_size=(5.5, 6), axis_text_y=element_text(size=5)))
p.save(OUT / "eda_localizability.png", dpi=150); print("Saved eda_localizability.png")

# ---- printed summary ----
print(f"\nDims used (>=1 task): {(share_count>0).sum()}/{F.shape[1]}")
print(f"Universal dims (>=20 tasks): {(share_count>=20).sum()}")
cl = fcluster(Z, t=0.6, criterion="distance")
print(f"Clusters at d=0.6: {len(set(cl))}")
for c in sorted(set(cl)):
    mem = [labels[i] for i in range(len(labels)) if cl[i] == c]
    if len(mem) > 1:
        print(f"  cluster {c}: {', '.join(mem)}")
