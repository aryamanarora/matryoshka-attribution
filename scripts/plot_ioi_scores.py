"""Bar chart comparing top-20 + bottom-20 node scores for Ours vs NAP-IG on IOI/GPT-2.

Bars are colored by known IOI circuit role from prior work (Wang et al., 2022).
"""

import json
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, geom_text, facet_grid, labs, coord_flip,
    theme_minimal, theme, element_text, element_blank, element_line,
    scale_fill_manual, scale_x_discrete, guides, guide_legend,
)

RESULTS_BASE = Path("results")

# IOI circuit categories (Wang et al., 2022)
CATEGORIES = {
    "Previous token": ["a2.h2", "a4.h11"],
    "Duplicate token": ["a0.h1", "a3.h0", "a0.h10"],
    "Induction": ["a5.h5", "a6.h9", "a5.h8", "a5.h9"],
    "S-inhibition": ["a7.h3", "a7.h9", "a8.h6", "a8.h10"],
    "Neg. name mover": ["a10.h7", "a11.h10"],
    "Name mover": ["a9.h9", "a9.h6", "a10.h0"],
    "Backup name mover": ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10", "a11.h2", "a11.h9"],
}

# Build reverse lookup
NODE_TO_CAT = {}
for cat, nodes in CATEGORIES.items():
    for n in nodes:
        NODE_TO_CAT[n] = cat

# Set1-ish palette (7 categories + gray for uncategorized)
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


def load_our_scores():
    path = RESULTS_BASE / "mib_node_hard_topk" / "ioi_gpt2_importances.json"
    d = json.load(open(path))
    nodes = d["nodes"]
    scores = {}
    for name, info in nodes.items():
        if name in ("input", "logits"):
            continue
        scores[name] = info["score"]
    return scores


def load_napig_scores():
    path = RESULTS_BASE / "napig_repro" / "EAP-IG-inputs_patching_node" / "ioi_gpt2" / "importances.json"
    d = json.load(open(path))
    nodes = d["nodes"]
    scores = {}
    for name, info in nodes.items():
        if name in ("input", "logits"):
            continue
        scores[name] = info["score"]
    return scores


def make_df(scores, method_name, n=20):
    """Get top-n and bottom-n nodes, return DataFrame."""
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_nodes[:n]
    bottom = sorted_nodes[-n:]

    rows = []
    for rank, (name, score) in enumerate(top):
        cat = NODE_TO_CAT.get(name, "Other")
        rows.append({"node": name, "score": score, "category": cat,
                      "method": method_name, "rank": rank})
    for rank, (name, score) in enumerate(reversed(bottom)):
        cat = NODE_TO_CAT.get(name, "Other")
        rows.append({"node": name, "score": score, "category": cat,
                      "method": method_name, "rank": n + rank})
    return pd.DataFrame(rows)


def main():
    ours = load_our_scores()
    napig = load_napig_scores()

    df_ours = make_df(ours, "Ours (hard top-k + ST)", n=20)
    df_napig = make_df(napig, "NAP-IG (CF, repro)", n=20)
    df = pd.concat([df_ours, df_napig], ignore_index=True)

    # Order nodes by score within each method
    # For each method, create ordered factor levels
    for method in df["method"].unique():
        mask = df["method"] == method
        sub = df[mask].sort_values("score", ascending=True)
        df.loc[mask, "order"] = range(len(sub))

    # Use node name + method as unique id for ordering
    df["node_label"] = df["node"]

    # Create separate ordered categories per facet
    # plotnine facet_grid with free scales handles this
    df["method"] = pd.Categorical(
        df["method"],
        categories=["Ours (hard top-k + ST)", "NAP-IG (CF, repro)"],
        ordered=True,
    )

    # For the label positioning
    df["hjust"] = df["score"].apply(lambda x: 1.1 if x >= 0 else -0.1)
    df["label_y"] = df["score"].apply(lambda x: x + 0.02 * abs(x) if x >= 0 else x - 0.02 * abs(x))

    # Build plot per method to get correct node ordering
    dfs = []
    for method in ["Ours (hard top-k + ST)", "NAP-IG (CF, repro)"]:
        sub = df[df["method"] == method].copy()
        sub = sub.sort_values("score", ascending=True)
        sub["node_label"] = pd.Categorical(sub["node_label"], categories=sub["node_label"].tolist(), ordered=True)
        dfs.append(sub)
    df = pd.concat(dfs, ignore_index=True)

    p = (
        ggplot(df, aes(x="node_label", y="score", fill="category"))
        + geom_col(width=0.75)
        + geom_text(
            aes(label="node_label"),
            size=5.5, ha="left",
            nudge_y=0.05,
        )
        + facet_grid("method ~ .", scales="free")
        + coord_flip()
        + scale_fill_manual(values=PALETTE)
        + labs(x="", y="Node score", fill="Circuit role",
               title="IOI / GPT-2: Top-20 and Bottom-20 node scores")
        + theme_minimal()
        + theme(
            figure_size=(7, 10),
            axis_text_y=element_blank(),
            axis_ticks_major_y=element_blank(),
            panel_grid_major_y=element_blank(),
            strip_text=element_text(size=10, weight="bold"),
            legend_position="bottom",
            legend_title=element_text(size=8),
            legend_text=element_text(size=7),
            plot_title=element_text(size=11, weight="bold"),
            axis_title_x=element_text(size=9),
        )
        + guides(fill=guide_legend(nrow=2))
    )

    out = Path("paper/figs/ioi_gpt2_scores.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")

    # Also save png for preview
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
