"""Scatter plot of node ranks: Ours vs NAP-IG on IOI/GPT-2.

Points colored by IOI circuit role. Diagonal = agreement.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_point, geom_abline, geom_text, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_size_manual, scale_shape_manual, guides, guide_legend,
)

RESULTS_BASE = Path("results")
LOCAL_BASE = Path("/tmp/mib_pkls/results")

CATEGORIES = [
    ("Known heads", [
        "a9.h9", "a9.h6", "a10.h0",  # name mover
        "a10.h7", "a11.h10",          # neg name mover
        "a2.h2", "a4.h11",            # previous token
        "a0.h1", "a3.h0", "a0.h10",   # duplicate token
        "a5.h5", "a6.h9", "a5.h8", "a5.h9",  # induction
        "a7.h3", "a7.h9", "a8.h6", "a8.h10",  # s-inhibition
        "a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10", "a11.h2", "a11.h9",  # backup name mover
    ]),
]

NODE_TO_CAT = {}
for cat, nodes in CATEGORIES:
    for n in nodes:
        NODE_TO_CAT[n] = cat

PALETTE = {
    "Known heads": "#377eb8",
    "Other hd.": "#bbbbbb",
    "MLP": "#e41a1c",
    "Input": "#ff7f00",
}

CAT_ORDER = [cat for cat, _ in CATEGORIES] + ["Other hd.", "MLP", "Input"]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(1.83, 1.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        legend_text=element_text(size=5),
        legend_title=element_text(size=5),
        legend_key_size=5,
        legend_position="bottom",
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
    )
)


def load_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {name: info["score"] for name, info in nodes.items()
            if name != "logits" and "score" in info}


def load_our_scores():
    rel = Path("mib_node_hard_topk") / "ioi_gpt2_importances.json"
    for base in (LOCAL_BASE, RESULTS_BASE):
        if (base / rel).exists():
            return load_scores(base / rel)
    return load_scores(RESULTS_BASE / rel)


def load_napig_scores():
    rel = Path("napig_repro") / "EAP-IG-inputs_patching_node" / "ioi_gpt2" / "importances.json"
    for base in (LOCAL_BASE, RESULTS_BASE):
        if (base / rel).exists():
            return load_scores(base / rel)
    return load_scores(RESULTS_BASE / rel)


def main():
    ours = load_our_scores()
    napig = load_napig_scores()

    common = sorted(set(ours.keys()) & set(napig.keys()))
    total = len(common)

    # Compute ranks (1 = highest score)
    ours_sorted = sorted(common, key=lambda n: ours[n], reverse=True)
    napig_sorted = sorted(common, key=lambda n: napig[n], reverse=True)
    ours_rank = {n: i + 1 for i, n in enumerate(ours_sorted)}
    napig_rank = {n: i + 1 for i, n in enumerate(napig_sorted)}

    rows = []
    for node in common:
        if node == "input":
            cat = "Input"
        else:
            cat = NODE_TO_CAT.get(node, None)
            if cat is None:
                cat = "MLP" if node.startswith("m") else "Other hd."
        rows.append({
            "node": node,
            "ours_rank": ours_rank[node],
            "napig_rank": napig_rank[node],
            "category": cat,
            "rank_diff": abs(ours_rank[node] - napig_rank[node]),
            "type": "Input" if node == "input" else ("MLP" if node.startswith("m") else "Attn head"),
        })
    df = pd.DataFrame(rows)
    df["category"] = pd.Categorical(df["category"], categories=CAT_ORDER, ordered=True)
    df["is_labeled"] = df["category"] != "Other"

    # Label + enlarge only nodes far from diagonal
    DISAGREE_THRESHOLD = 40
    df["is_disagree"] = df["rank_diff"] > DISAGREE_THRESHOLD
    df["label"] = ""
    df.loc[df["is_disagree"], "label"] = df.loc[df["is_disagree"], "node"]
    df["is_highlight"] = df["is_disagree"]

    # Sort so highlighted points are drawn on top
    df = df.sort_values("is_highlight", ascending=True).reset_index(drop=True)

    from scipy.stats import spearmanr
    rho, pval = spearmanr(df["ours_rank"], df["napig_rank"])

    p = (
        ggplot(df, aes(x="ours_rank", y="napig_rank", color="category"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="#999999", size=0.4)
        + geom_point(aes(shape="type"), alpha=0.7, size=1.2)
        + scale_color_manual(values=PALETTE)
        + scale_shape_manual(values={"Attn head": "o", "MLP": "s", "Input": "D"}, guide=None)
        + labs(
            x="Rank (MAttr)",
            y="Rank (NAP-IG)",
            color="",
        )
        + guides(color=guide_legend(ncol=3, override_aes={"size": 2}))
        + theme(
            legend_position="top",
            legend_margin=2,
            legend_box_spacing=0.05,
        )
    )

    out = Path("paper/figs/ioi_gpt2_rank_scatter.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")
    print(f"Spearman rho={rho:.3f}, p={pval:.2e}")


if __name__ == "__main__":
    main()
