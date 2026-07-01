"""Histogram of learned node-score distributions (361 per-node scores) for ioi/qwen2.5,
across four id-STE SGD methods: {uniform,log} k-schedule x {deterministic, gumbel} forward.
Shows the uniform-deterministic collapse (degenerate spike at 0, CPR floor) vs the spread,
well-separated distributions of the working methods.
Reproduce: uv run python plots/qwen_ioi_score_hist.py
"""
import torch
import pandas as pd
from plotnine import (
    ggplot, aes, geom_histogram, geom_vline, facet_wrap, labs,
    scale_fill_brewer, scale_y_sqrt, theme_set, theme_bw, theme,
    element_text, element_line, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.0),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.04,
        panel_spacing_y=0.06,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_position="none",
    )
)

# (label, results dir, CPR AUC) — AUC from each dir's ioi_qwen2.5_validation.pkl
METHODS = [
    ("Uniform-k, deterministic (AUC 0.25)", "results/mib_node_identity_sgd"),
    ("Uniform-k, Gumbel (AUC 1.45)",        "results/mib_node_identity_gumbel_sgd_uniform"),
    ("Log-k, deterministic (AUC 1.52)",     "results/mib_node_identity_sgd_log"),
    ("Log-k, Gumbel (AUC 1.33)",            "results/mib_node_identity_gumbel_sgd_log"),
]

rows = []
for label, d in METHODS:
    s = torch.load(f"{d}/ioi_qwen2.5_scores.pt")["scores"].float().flatten().tolist()
    rows += [{"method": label, "score": v} for v in s]
df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], categories=[m[0] for m in METHODS], ordered=True)

p = (
    ggplot(df, aes("score", fill="method"))
    + geom_vline(xintercept=0, color="#999999", size=0.3, linetype="dashed")
    + geom_histogram(bins=40, color="none")
    + facet_wrap("method", ncol=2, scales="free")
    + scale_fill_brewer(type="qual", palette="Set1")
    + scale_y_sqrt()
    + labs(x="Learned node score", y="Count (√ scale)")
)
p.save("plots/qwen_ioi_score_hist.pdf", verbose=False)
print("saved plots/qwen_ioi_score_hist.pdf")
