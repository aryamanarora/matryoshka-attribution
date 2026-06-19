"""Scatter of per-node learned scores: sufficient vs necessary mode (ioi/gpt2).
Reads importances.json from both runs. Run on sc."""
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from plotnine import (ggplot, aes, geom_point, geom_smooth, labs, theme_bw, theme_set,
                      theme, element_text, scale_color_manual)

R = Path("results")
SUF = R / "mib_node_hard_topk" / "ioi_gpt2_importances.json"
NEC = R / "mib_node_necessary_ioi_gpt2" / "ioi_gpt2_importances.json"
theme_set(theme_bw(base_size=9) + theme(text=element_text(family="Inter")))


def load(p):
    d = json.load(open(p)); nodes = d.get("nodes", d)
    return {n: i["score"] for n, i in nodes.items() if n != "logits" and "score" in i}


suf, nec = load(SUF), load(NEC)
common = sorted(set(suf) & set(nec))
def typ(n):
    return "Input" if n == "input" else ("MLP" if n.startswith("m") else "Attn head")
df = pd.DataFrame([{"node": n, "suf": suf[n], "nec": nec[n], "type": typ(n)} for n in common])

pr = pearsonr(df.suf, df.nec)[0]; sr = spearmanr(df.suf, df.nec)[0]
pal = {"Attn head": "#377eb8", "MLP": "#e41a1c", "Input": "#ff7f00"}
p = (ggplot(df, aes("suf", "nec", color="type"))
     + geom_smooth(method="lm", se=False, color="#999999", linetype="dashed", size=0.5)
     + geom_point(alpha=0.6, size=1.4)
     + scale_color_manual(values=pal)
     + labs(x="score — sufficient (denoising)", y="score — necessary (noising)", color="",
            title=f"ioi/gpt2 node scores: sufficient vs necessary\nPearson r={pr:.2f}, Spearman ρ={sr:.2f}")
     + theme(figure_size=(4.2, 3.6), legend_position="top"))
out = Path("paper/figs/suf_vs_nec_scatter.png")
p.save(out, dpi=150)
print(f"Saved {out}  (n={len(common)} nodes, Pearson {pr:.3f}, Spearman {sr:.3f})")
