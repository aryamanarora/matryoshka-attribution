"""Plot # of active rotated dimensions per (layer, span) at varying sparsities.

Usage:
    uv run python plots/plot_das_dims.py results/pythia1b_npi_any_das_scores.pt
"""

import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_tile, geom_text, facet_wrap, labs, scale_fill_gradient,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
)

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(12, 8),
        axis_title=element_text(size=10),
        axis_text=element_text(size=7),
        axis_text_x=element_text(rotation=90, hjust=1, vjust=0.5),
        legend_text=element_text(size=8),
        legend_title=element_text(size=9),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=9, face="plain"),
    )
)


def main():
    path = Path(sys.argv[1])
    data = torch.load(path, weights_only=False, map_location="cpu")
    scores = data["scores"]  # [num_layers * num_spans * hidden_size]
    args = data["args"]
    span_names = data.get("span_names", None)

    # Infer dimensions from config
    # Need to figure out num_layers, num_spans, hidden_size
    # Total = num_layers * num_spans * hidden_size
    total = scores.shape[0]

    # Try to get from model config
    model_name = args.get("model", "")
    if "pythia-1b" in model_name:
        num_layers, hidden_size = 16, 2048
    elif "pythia-410m" in model_name:
        num_layers, hidden_size = 24, 1024
    elif "llama" in model_name.lower():
        num_layers, hidden_size = 32, 4096
    else:
        # Guess from span_names
        num_spans = len(span_names) if span_names else 9
        # total = L * S * d, try common configs
        for L, d in [(16, 2048), (24, 1024), (32, 4096), (12, 768)]:
            if total == L * num_spans * d:
                num_layers, hidden_size = L, d
                break
        else:
            print(f"Cannot infer dimensions from total={total}")
            return

    num_spans = total // (num_layers * hidden_size)
    print(f"Dimensions: {num_layers} layers × {num_spans} spans × {hidden_size} dims = {total}")

    # Reshape scores to [num_layers, num_spans, hidden_size]
    scores_3d = scores.view(num_layers, num_spans, hidden_size)

    # At various sparsities, count active dims per (layer, span)
    sparsities = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    sorted_scores, _ = scores.sort(descending=True)

    rows = []
    for pct in sparsities:
        k = max(1, int(total * pct))
        threshold = sorted_scores[k - 1].item()

        for li in range(num_layers):
            for si in range(num_spans):
                n_active = (scores_3d[li, si] >= threshold).sum().item()
                span_label = span_names[si] if span_names else f"span_{si}"
                rows.append({
                    "layer": li,
                    "span": span_label,
                    "n_dims": n_active,
                    "sparsity": f"top {pct*100:.1f}%",
                })

    df = pd.DataFrame(rows)

    # Order spans
    if span_names:
        df["span"] = pd.Categorical(df["span"], categories=span_names, ordered=True)

    # Order sparsities
    sparsity_labels = [f"top {p*100:.1f}%" for p in sparsities]
    df["sparsity"] = pd.Categorical(df["sparsity"], categories=sparsity_labels, ordered=True)

    p = (
        ggplot(df, aes(x="span", y="factor(layer)", fill="n_dims"))
        + geom_tile()
        + geom_text(aes(label="n_dims"), size=5, color="white")
        + facet_wrap("sparsity", ncol=3)
        + scale_fill_gradient(low="#1a1a2e", high="#e94560",
                              trans="log1p")
        + labs(x="", y="Layer", fill="# dims")
        + theme(legend_position="right")
    )

    out = path.with_name(path.stem.replace("_scores", "_dims") + ".png")
    p.save(out, dpi=150)
    print(f"Saved {out}")

    # Also save PDF
    p.save(out.with_suffix(".pdf"), dpi=300)


if __name__ == "__main__":
    main()
