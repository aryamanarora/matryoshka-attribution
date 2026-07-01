"""Pairwise per-node score correlation among the ioi/gpt2 runs:
sufficient, necessary (w/ input), necessary (no input). Spearman heatmap + scatter grid.
Run on sc."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from plotnine import (ggplot, aes, geom_tile, geom_text, geom_point, geom_smooth, labs,
                      facet_wrap, scale_fill_gradient2, theme_bw, theme_set, theme,
                      element_text, element_blank)

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
theme_set(theme_bw(base_size=9) + theme(text=element_text(family="Inter")))
RUNS = [("sufficient", "mib_node_hard_topk"),
        ("necessary", "mib_node_necessary_ioi_gpt2"),
        ("necessary\n(no input)", "mib_node_necessary_ioi_gpt2_noinput")]

def load(d):
    j = json.load(open(R / d / "ioi_gpt2_importances.json")); n = j.get("nodes", j)
    return {k: v["score"] for k, v in n.items() if k != "logits" and "score" in v}

S = {name: load(d) for name, d in RUNS}
common = sorted(set.intersection(*[set(s) for s in S.values()]))  # input dropped (absent in no-input)
M = pd.DataFrame({name: [S[name][n] for n in common] for name in S})
labels = [r[0] for r in RUNS]

# 1) Spearman corr heatmap
rows = []
for a in labels:
    for b in labels:
        rows.append({"a": a, "b": b, "rho": spearmanr(M[a], M[b])[0]})
cm = pd.DataFrame(rows)
cm["a"] = pd.Categorical(cm["a"], categories=labels, ordered=True)
cm["b"] = pd.Categorical(cm["b"], categories=labels[::-1], ordered=True)
p = (ggplot(cm, aes("a", "b", fill="rho")) + geom_tile()
     + geom_text(aes(label="rho"), format_string="{:.2f}", size=9)
     + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac", midpoint=0, limits=[-1, 1])
     + labs(x="", y="", fill="Spearman ρ",
            title=f"ioi/gpt2 node-score correlations (n={len(common)})")
     + theme(figure_size=(3.8, 3.2), panel_grid=element_blank()))
p.save(OUT / "nec_suf_corr_matrix.png", dpi=150)
print("Saved nec_suf_corr_matrix.png")

# 2) pairwise scatter grid
pr = []
for a, b in combinations(labels, 2):
    r = spearmanr(M[a], M[b])[0]
    for n in common:
        pr.append({"pair": f"{a.replace(chr(10),' ')} vs {b.replace(chr(10),' ')} (ρ={r:.2f})",
                   "x": S[a][n], "y": S[b][n]})
pf = pd.DataFrame(pr)
p2 = (ggplot(pf, aes("x", "y")) + geom_point(alpha=0.5, size=1.0, color="#377eb8")
      + geom_smooth(method="lm", se=False, color="#999999", linetype="dashed", size=0.5)
      + facet_wrap("pair", scales="free")
      + labs(x="score", y="score", title="Pairwise node-score scatters")
      + theme(figure_size=(8, 3)))
p2.save(OUT / "nec_suf_corr_scatters.png", dpi=150)
print("Saved nec_suf_corr_scatters.png")
print("Spearman:", {f"{a[:4]}-{b[:4]}": round(spearmanr(M[a],M[b])[0],3) for a,b in combinations(labels,2)})
