"""Plot how active DAS dims are shared across spans at each layer.

For each sparsity level and layer, shows stacked bars of:
- dims unique to each span
- dims shared by 2+ spans

Usage:
    uv run python plots/plot_das_sharing.py results/pythia1b_npi_any_das_scores.pt
"""

import sys
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, facet_wrap, labs, position_stack,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_fill_brewer,
)

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(14, 8),
        axis_title=element_text(size=10),
        axis_text=element_text(size=7),
        legend_text=element_text(size=8),
        legend_title=element_text(size=9),
        panel_grid_major=element_line(size=0.5, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=8, face="plain"),
    )
)


def main():
    path = Path(sys.argv[1])
    data = torch.load(path, weights_only=False, map_location="cpu")
    scores = data["scores"]
    span_names = data.get("span_names", None)

    # Infer dims
    model_name = data["args"].get("model", "")
    if "pythia-1b" in model_name:
        L, D = 16, 2048
    elif "pythia-410m" in model_name:
        L, D = 24, 1024
    elif "llama" in model_name.lower():
        L, D = 32, 4096
    else:
        L, D = 16, 2048  # fallback

    S = scores.shape[0] // (L * D)
    print(f"{L} layers × {S} spans × {D} dims = {scores.shape[0]}")

    scores_3d = scores.view(L, S, D)
    sorted_scores, _ = scores.sort(descending=True)

    sparsities = [0.001, 0.005, 0.01, 0.05, 0.1]

    rows = []
    for pct in sparsities:
        k = max(1, int(scores.shape[0] * pct))
        threshold = sorted_scores[k - 1].item()

        for li in range(L):
            # For this layer, find which dims are active at each span
            active_per_span = {}  # dim -> set of spans
            for si in range(S):
                active_dims = (scores_3d[li, si] >= threshold).nonzero(as_tuple=True)[0]
                for d in active_dims.tolist():
                    if d not in active_per_span:
                        active_per_span[d] = set()
                    active_per_span[d].add(si)

            # Classify each active dim
            unique_counts = Counter()  # span_name -> count of dims unique to it
            shared_count = 0
            for d, spans in active_per_span.items():
                if len(spans) == 1:
                    si = next(iter(spans))
                    sname = span_names[si] if span_names else f"span_{si}"
                    unique_counts[sname] += 1
                else:
                    shared_count += 1

            # Add rows
            for sname, count in unique_counts.items():
                rows.append({
                    "layer": li,
                    "category": f"unique: {sname}",
                    "count": count,
                    "sparsity": f"top {pct*100:.1f}%",
                })
            if shared_count > 0:
                rows.append({
                    "layer": li,
                    "category": "shared (2+ spans)",
                    "count": shared_count,
                    "sparsity": f"top {pct*100:.1f}%",
                })

    df = pd.DataFrame(rows)

    # Order
    sparsity_labels = [f"top {p*100:.1f}%" for p in sparsities]
    df["sparsity"] = pd.Categorical(df["sparsity"], categories=sparsity_labels, ordered=True)

    # Build category order: shared first, then unique per span
    all_cats = sorted(df["category"].unique())
    shared_cats = [c for c in all_cats if "shared" in c]
    unique_cats = [c for c in all_cats if "unique" in c]
    cat_order = shared_cats + unique_cats
    df["category"] = pd.Categorical(df["category"], categories=cat_order, ordered=True)

    p = (
        ggplot(df, aes(x="factor(layer)", y="count", fill="category"))
        + geom_col(position=position_stack())
        + facet_wrap("sparsity", ncol=3, scales="free_y")
        + scale_fill_brewer(type="qual", palette="Set3")
        + labs(x="Layer", y="# active dims", fill="")
        + theme(legend_position="bottom")
    )

    out = path.with_name(path.stem.replace("_scores", "_sharing") + ".png")
    p.save(out, dpi=150)
    print(f"Saved {out}")
    p.save(out.with_suffix(".pdf"), dpi=300)


if __name__ == "__main__":
    main()
