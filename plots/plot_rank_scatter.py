"""Scatter plot of node ranks: Ours vs NAP-IG, for any task/model.

Usage:
    uv run python plots/plot_rank_scatter.py --task ioi --model gpt2
    uv run python plots/plot_rank_scatter.py --task ioi --model qwen2.5
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_point, geom_abline, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_size_manual, scale_shape_manual, guides, guide_legend,
)
from scipy.stats import spearmanr
from adjustText import adjust_text

LOCAL_DIR = Path("/tmp/mib_pkls/results")
CLUSTER = Path("results")

# IOI circuit categories (only apply to GPT-2 IOI)
IOI_CATEGORIES = [
    ("Previous token", ["a2.h2", "a4.h11"]),
    ("Duplicate token", ["a0.h1", "a3.h0", "a0.h10"]),
    ("Induction", ["a5.h5", "a6.h9", "a5.h8", "a5.h9"]),
    ("S-inhibition", ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("Neg. name mover", ["a10.h7", "a11.h10"]),
    ("Name mover", ["a9.h9", "a9.h6", "a10.h0"]),
    ("Backup name mover", ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10", "a11.h2", "a11.h9"]),
]

IOI_NODE_TO_CAT = {}
for cat, nodes in IOI_CATEGORIES:
    for n in nodes:
        IOI_NODE_TO_CAT[n] = cat

PALETTE = {
    "Previous token": "#e41a1c",
    "Duplicate token": "#377eb8",
    "Induction": "#4daf4a",
    "S-inhibition": "#984ea3",
    "Neg. name mover": "#ff7f00",
    "Name mover": "#a65628",
    "Backup name mover": "#f781bf",
    "Other heads": "#bbbbbb",
    "MLP": "#e6ab02",
}

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 5),
        axis_title=element_text(size=10),
        axis_text=element_text(size=8),
        legend_text=element_text(size=8),
        legend_title=element_text(size=9),
        panel_grid_major=element_line(size=0.5, color="#dddddd"),
        panel_grid_minor=element_blank(),
    )
)


def load_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {n: info["score"] for n, info in nodes.items() if n not in ("input", "logits")}


def resolve(base, *parts):
    p = base / Path(*parts)
    if p.exists():
        return p
    p2 = CLUSTER / Path(*parts)
    return p2 if p2.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    task, model = args.task, args.model
    stask = task.replace("_", "-")
    use_ioi_colors = (task == "ioi" and model == "gpt2")

    ours_path = resolve(LOCAL_DIR, "mib_node_hard_topk", f"{task}_{model}_importances.json")
    napig_path = resolve(LOCAL_DIR, "napig_repro", "EAP-IG-inputs_patching_node", f"{stask}_{model}", "importances.json")

    if not ours_path or not ours_path.exists():
        print(f"No ours importances for {task}/{model}")
        return
    if not napig_path or not napig_path.exists():
        print(f"No NAP-IG importances for {task}/{model}")
        return

    ours = load_scores(ours_path)
    napig = load_scores(napig_path)
    common = sorted(set(ours) & set(napig))
    total = len(common)

    ours_sorted = sorted(common, key=lambda n: ours[n], reverse=True)
    napig_sorted = sorted(common, key=lambda n: napig[n], reverse=True)
    ours_rank = {n: i + 1 for i, n in enumerate(ours_sorted)}
    napig_rank = {n: i + 1 for i, n in enumerate(napig_sorted)}

    rows = []
    for node in common:
        if use_ioi_colors:
            cat = IOI_NODE_TO_CAT.get(node, None)
            if cat is None:
                cat = "MLP" if node.startswith("m") else "Other heads"
        else:
            cat = "MLP" if node.startswith("m") else "Other heads"
        rows.append({
            "node": node,
            "ours_rank": ours_rank[node],
            "napig_rank": napig_rank[node],
            "category": cat,
            "rank_diff": abs(ours_rank[node] - napig_rank[node]),
            "type": "MLP" if node.startswith("m") else "Attn head",
        })
    df = pd.DataFrame(rows)

    cat_order = [c for c, _ in IOI_CATEGORIES] + ["Other heads", "MLP"] if use_ioi_colors else ["Other heads", "MLP"]
    df["category"] = pd.Categorical(df["category"], categories=cat_order, ordered=True)

    DISAGREE_THRESHOLD = max(40, int(total * 0.05))
    df["is_disagree"] = df["rank_diff"] > DISAGREE_THRESHOLD
    df["label"] = ""
    df.loc[df["is_disagree"], "label"] = df.loc[df["is_disagree"], "node"]
    df["is_highlight"] = df["is_disagree"]

    df = df.sort_values("is_highlight", ascending=True).reset_index(drop=True)

    rho, pval = spearmanr(df["ours_rank"], df["napig_rank"])

    MODEL_NAMES = {"gpt2": "GPT-2", "qwen2.5": "Qwen-2.5", "gemma2": "Gemma-2", "llama3": "Llama-3.1"}
    TASK_NAMES = {"ioi": "IOI", "mcqa": "MCQA", "arithmetic_subtraction": "Arithmetic",
                  "arc_easy": "ARC (E)", "arc_challenge": "ARC (C)"}
    title = f"{TASK_NAMES.get(task, task)} / {MODEL_NAMES.get(model, model)} node ranks (Spearman ρ = {rho:.2f})"

    p = (
        ggplot(df, aes(x="ours_rank", y="napig_rank", color="category"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="#999999", size=0.4)
        + geom_point(aes(size="is_highlight", shape="type"), alpha=0.7)
        + scale_color_manual(values=PALETTE)
        + scale_shape_manual(values={"Attn head": "o", "MLP": "s"})
        + scale_size_manual(values={False: 1.5, True: 3.5}, guide=None)
        + labs(x="Rank (Ours)", y="Rank (NAP-IG repro)", color="Circuit role", title=title)
        + guides(color=guide_legend(ncol=1), shape=guide_legend(ncol=1))
        + theme(legend_position="right", plot_title=element_text(size=10))
    )

    fig = p.draw()
    ax = fig.axes[0]

    labeled = df[df["label"] != ""]
    texts = []
    for _, row in labeled.iterrows():
        texts.append(ax.text(row["ours_rank"], row["napig_rank"], row["label"],
                             fontsize=6, fontfamily="Inter", color="#333333"))
    if texts:
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.5),
                    expand=(1.5, 1.5))

    out = Path(f"paper/figs/{task}_{model}_rank_scatter.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"Saved {out.with_suffix('.png')}")
    print(f"Spearman rho={rho:.3f}, p={pval:.2e}, n={total}")


if __name__ == "__main__":
    main()
