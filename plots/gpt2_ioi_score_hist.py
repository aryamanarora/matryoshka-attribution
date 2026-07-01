"""Histogram of learned node-score distributions (157 per-node scores) for ioi/gpt2,
2x2 over masking (id-STE vs hard_topk) x optimizer (SGD vs Adam), log-k schedule.
Shared x-axis (ranges are comparable) so the distributions compare directly: id-STE keeps
most scores positive, hard_topk pushes most negative (sparser); optimizer mainly shifts spread.
Reproduce: uv run python plots/gpt2_ioi_score_hist.py
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
        axis_text_x=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.04,
        panel_spacing_y=0.06,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_position="none",
    )
)

# (label, results dir) — all log-k; naming: no "sgd" in dir => Adam.
METHODS = [
    ("id-STE, SGD (AUC 1.77)",     "mib_node_identity_sgd_log"),
    ("id-STE, Adam (AUC 1.27)",    "mib_node_identity_log"),
    ("hard_topk, SGD (AUC 1.79)",  "mib_node_hard_topk_log_sgd"),
    ("hard_topk, Adam (AUC 1.81)", "mib_node_hard_topk_log_local"),
]

rows = []
for label, d in METHODS:
    s = torch.load(f"results/{d}/ioi_gpt2_scores.pt")["scores"].float().flatten().tolist()
    rows += [{"method": label, "score": v} for v in s]
df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], categories=[m[0] for m in METHODS], ordered=True)

p = (
    ggplot(df, aes("score", fill="method"))
    + geom_vline(xintercept=0, color="#999999", size=0.3, linetype="dashed")
    + geom_histogram(bins=30, color="none")
    + facet_wrap("method", ncol=2)          # shared x and y
    + scale_fill_brewer(type="qual", palette="Set1")
    + scale_y_sqrt()
    + labs(x="Learned node score", y="Count (√ scale)")
)
p.save("plots/gpt2_ioi_score_hist.pdf", verbose=False)
print("saved plots/gpt2_ioi_score_hist.pdf")
