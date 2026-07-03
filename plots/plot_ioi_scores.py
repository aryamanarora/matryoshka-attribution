"""Bar chart comparing node scores for Ours vs NAP-IG on IOI/GPT-2.

Only shows nodes with known IOI circuit roles (Wang et al., 2022).
Same x-axis for both methods, grouped by category with dashed separators.
"""

import json
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, geom_vline, facet_wrap, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_fill_manual, scale_x_discrete, guides, guide_legend,
)

RESULTS_BASE = Path("results")
LOCAL_DIR = Path("/tmp/mib_scores")

# IOI circuit categories in causal order (Wang et al., 2022)
CATEGORIES = [
    ("Previous token", ["a2.h2", "a4.h11"]),
    ("Duplicate token", ["a0.h1", "a3.h0", "a0.h10"]),
    ("Induction", ["a5.h5", "a6.h9", "a5.h8", "a5.h9"]),
    ("S-inhibition", ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("Neg. name mover", ["a10.h7", "a11.h10"]),
    ("Name mover", ["a9.h9", "a9.h6", "a10.h0"]),
    ("Backup name mover", ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10", "a11.h2", "a11.h9"]),
]

NODE_TO_CAT = {}
for cat, nodes in CATEGORIES:
    for n in nodes:
        NODE_TO_CAT[n] = cat

# Build ordered node list: grouped by category
ALL_NODES = []
for _, nodes in CATEGORIES:
    ALL_NODES.extend(nodes)

PALETTE = {
    "Previous token": "#e41a1c",
    "Duplicate token": "#377eb8",
    "Induction": "#4daf4a",
    "S-inhibition": "#984ea3",
    "Neg. name mover": "#ff7f00",
    "Name mover": "#a65628",
    "Backup name mover": "#f781bf",
    "Other": "#999999",
}

CAT_ORDER = [cat for cat, _ in CATEGORIES] + ["Other"]

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(16, 5),
        axis_title=element_text(size=10),
        axis_text=element_text(size=8),
        axis_text_x=element_text(rotation=90, hjust=1, vjust=0.5, size=5),
        legend_text=element_text(size=8),
        legend_title=element_text(size=9),
        panel_grid_major=element_line(size=0.5, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=9, face="plain"),
    )
)


def load_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {name: info["score"] for name, info in nodes.items()
            if name not in ("input", "logits")}


def load_our_scores():
    if LOCAL_DIR.exists():
        return load_scores(LOCAL_DIR / "ours.json")
    return load_scores(RESULTS_BASE / "mib_node_hard_topk" / "ioi_gpt2_importances.json")


def load_napig_scores():
    if LOCAL_DIR.exists():
        return load_scores(LOCAL_DIR / "napig.json")
    return load_scores(RESULTS_BASE / "napig_repro" / "EAP-IG-inputs_patching_node" / "ioi_gpt2" / "importances.json")


def main():
    ours = load_our_scores()
    napig = load_napig_scores()

    # All nodes present in both methods
    all_node_names = sorted(set(ours.keys()) | set(napig.keys()))
    # "Other" nodes: everything not in a known category, sorted by name
    other_nodes = sorted([n for n in all_node_names if n not in NODE_TO_CAT])
    node_order = ALL_NODES + other_nodes
    total = len(node_order)

    rows = []
    for node in node_order:
        cat = NODE_TO_CAT.get(node, "Other")
        if node in ours:
            # Rank descending by score, convert to (total - rank)
            rows.append({"node": node, "score": ours[node], "category": cat,
                          "method": "Ours"})
        if node in napig:
            rows.append({"node": node, "score": napig[node], "category": cat,
                          "method": "NAP-IG (repro)"})
    df = pd.DataFrame(rows)

    # Compute rank per method: rank 1 = highest score, then y = total - rank
    for method in df["method"].unique():
        mask = df["method"] == method
        df.loc[mask, "rank"] = df.loc[mask, "score"].rank(ascending=False).astype(int)
    df["y"] = total - df["rank"]

    # Shared x-axis: categorized nodes first, then all others
    df["node"] = pd.Categorical(df["node"], categories=node_order, ordered=True)
    df["category"] = pd.Categorical(df["category"], categories=CAT_ORDER, ordered=True)
    df["method"] = pd.Categorical(
        df["method"],
        categories=["Ours", "NAP-IG (repro)"],
        ordered=True,
    )

    # Compute category boundary positions for dashed separator lines
    boundaries = []
    pos = 0
    for cat, nodes in CATEGORIES:
        pos += len(nodes)
        boundaries.append(pos + 0.5)
    # Keep the last boundary (separates backup name movers from Other)

    p = (
        ggplot(df, aes(x="node", y="y", fill="category"))
        + geom_col(width=0.8)
        + geom_vline(xintercept=boundaries, linetype="dashed", color="#999999", size=0.4)
        + facet_wrap("method", ncol=1)
        + scale_fill_manual(values=PALETTE)
        + labs(x="", y="Total nodes $-$ rank", fill="Circuit role")
        + guides(fill=guide_legend(nrow=1))
        + theme(
            legend_position="bottom",
            panel_grid_major_x=element_blank(),
        )
    )

    out = Path("paper/figs/ioi_gpt2_scores.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")

    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
