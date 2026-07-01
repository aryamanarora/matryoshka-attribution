"""Plot per-layer score distributions for DAS, facetted by layer, one line per span.

Usage:
    uv run python plots/plot_das_score_dist.py results/foo_scores.pt results/bar_scores.pt
"""

import sys
from pathlib import Path

import torch
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, facet_wrap, labs,
    theme_bw, theme_set, theme, element_text, element_blank,
)

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(16, 10),
        axis_title=element_text(size=10),
        axis_text=element_text(size=7),
        legend_text=element_text(size=8),
        legend_title=element_text(size=9),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=9, face="plain"),
    )
)


def load_and_build_df(path, label):
    data = torch.load(path, weights_only=False, map_location="cpu")
    scores = data["scores"]
    span_names = data.get("span_names", None)
    args = data["args"]

    model_name = args.get("model", "")
    das_dim = args.get("das_dim", None)
    if "pythia-1b" in model_name:
        num_layers = 16
        hidden_size = das_dim or 2048
    elif "pythia-410m" in model_name:
        num_layers = 24
        hidden_size = das_dim or 1024
    else:
        num_layers = 16
        hidden_size = das_dim or 2048

    total = scores.shape[0]
    num_spans = total // (num_layers * hidden_size)
    scores_3d = scores.view(num_layers, num_spans, hidden_size)

    rows = []
    for li in range(num_layers):
        for si in range(num_spans):
            span_scores = scores_3d[li, si].sort(descending=True).values
            span_label = span_names[si] if span_names else f"span_{si}"
            for rank, val in enumerate(span_scores.tolist()):
                rows.append({
                    "layer": f"L{li}",
                    "span": span_label,
                    "rank": rank + 1,
                    "score": val,
                    "run": label,
                })

    return pd.DataFrame(rows), num_layers


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: plot_das_score_dist.py <scores1.pt> [<scores2.pt> ...]")
        return

    dfs = []
    for p in paths:
        path = Path(p)
        label = path.stem.replace("_scores", "").replace("pythia1b_", "")
        df, num_layers = load_and_build_df(path, label)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    layer_order = [f"L{i}" for i in range(num_layers)]
    df["layer"] = pd.Categorical(df["layer"], categories=layer_order, ordered=True)

    if len(paths) == 1:
        p = (
            ggplot(df, aes(x="rank", y="score", color="span"))
            + geom_line(alpha=0.7, size=0.5)
            + facet_wrap("layer", ncol=4, scales="free_y")
            + labs(x="Rank (most important →)", y="Score", color="Span")
        )
    else:
        p = (
            ggplot(df, aes(x="rank", y="score", color="span", linetype="run"))
            + geom_line(alpha=0.7, size=0.5)
            + facet_wrap("layer", ncol=4, scales="free_y")
            + labs(x="Rank (most important →)", y="Score", color="Span", linetype="Run")
        )

    out = Path(paths[0]).with_name(Path(paths[0]).stem.replace("_scores", "") + "_score_dist.png")
    p.save(out, dpi=150)
    print(f"Saved {out}")
    p.save(out.with_suffix(".pdf"), dpi=300)


if __name__ == "__main__":
    main()
