"""Full-width facetted scatter plot of node ranks: MAttr vs NAP-IG across all tasks."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from plotnine import (
    ggplot, aes, geom_point, geom_abline, geom_text, labs, facet_wrap,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_shape_manual, guides, guide_legend,
)

LOCAL_OURS = Path("/tmp/mib_pkls/results/mib_node_hard_topk")
LOCAL_NAPIG = Path("/tmp/mib_pkls/results/napig_repro/EAP-IG-inputs_patching_node")
CLUSTER_OURS = Path("results/mib_node_hard_topk")
CLUSTER_NAPIG = Path("results/napig_repro/EAP-IG-inputs_patching_node")

TASKS = [
    ("ioi", "gpt2", "IOI / GPT-2"),
    ("ioi", "qwen2.5", "IOI / Qwen"),
    ("ioi", "gemma2", "IOI / Gemma"),
    ("arithmetic_subtraction", "llama3", "Arith / Llama"),
    ("mcqa", "qwen2.5", "MCQA / Qwen"),
    ("mcqa", "gemma2", "MCQA / Gemma"),
    ("arc_easy", "gemma2", "ARC-E / Gemma"),
    ("arc_easy", "llama3", "ARC-E / Llama"),
    ("arc_challenge", "llama3", "ARC-C / Llama"),
]

PALETTE = {
    "Attn": "#bbbbbb",
    "MLP": "#e41a1c",
    "Input": "#ff7f00",
}

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.5),
        axis_title=element_text(size=7),
        axis_text=element_text(size=5),
        legend_text=element_text(size=6),
        legend_key_size=6,
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=6, face="plain"),
    )
)


def load_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {n: info["score"] for n, info in nodes.items()
            if n != "logits" and "score" in info}


def load_ours(task, model):
    p = LOCAL_OURS / f"{task}_{model}_importances.json"
    if p.exists():
        return load_scores(p)
    p = CLUSTER_OURS / f"{task}_{model}_importances.json"
    if p.exists():
        return load_scores(p)
    return None


def load_napig(task, model):
    stask = task.replace("_", "-")
    for base in [LOCAL_NAPIG, CLUSTER_NAPIG]:
        p = base / f"{stask}_{model}" / "importances.json"
        if p.exists():
            return load_scores(p)
        p = base / f"{task}_{model}" / "importances.json"
        if p.exists():
            return load_scores(p)
    return None


def main():
    all_rows = []
    task_order = []

    for task, model, label in TASKS:
        ours = load_ours(task, model)
        napig = load_napig(task, model)
        if ours is None or napig is None:
            continue

        common = sorted(set(ours) & set(napig))
        if len(common) < 4:
            continue

        ours_sorted = sorted(common, key=lambda n: ours[n], reverse=True)
        napig_sorted = sorted(common, key=lambda n: napig[n], reverse=True)
        ours_rank = {n: i + 1 for i, n in enumerate(ours_sorted)}
        napig_rank = {n: i + 1 for i, n in enumerate(napig_sorted)}

        rho, _ = spearmanr([ours[n] for n in common], [napig[n] for n in common])
        facet_label = f"{label} (ρ={rho:.2f})"
        task_order.append(facet_label)

        for node in common:
            cat = "Input" if node == "input" else ("MLP" if node.startswith("m") else "Attn")
            all_rows.append({
                "ours_rank": ours_rank[node],
                "napig_rank": napig_rank[node],
                "category": cat,
                "type": cat,
                "facet": facet_label,
            })

    df = pd.DataFrame(all_rows)
    df["category"] = pd.Categorical(df["category"], categories=["Attn", "MLP", "Input"], ordered=True)
    df["facet"] = pd.Categorical(df["facet"], categories=task_order, ordered=True)
    df = df.sort_values("category", ascending=True).reset_index(drop=True)

    p = (
        ggplot(df, aes(x="ours_rank", y="napig_rank", color="category"))
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="#999999", size=0.3)
        + geom_point(aes(shape="type"), alpha=0.6, size=0.8)
        + facet_wrap("facet", ncol=5, scales="free")
        + scale_color_manual(values=PALETTE)
        + scale_shape_manual(values={"Attn": "o", "MLP": "s", "Input": "D"}, guide=None)
        + labs(x="Rank (MAttr)", y="Rank (NAP-IG)", color="")
        + guides(color=guide_legend(override_aes={"size": 2}))
        + theme(
            legend_position="top",
            legend_margin=0,
            legend_box_spacing=0.05,
        )
    )

    out = Path("paper/figs/rank_scatter_all.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
