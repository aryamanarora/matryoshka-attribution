"""Full-width facetted scatter plot of raw node scores: MAttr vs natural-k MAttr across all tasks."""

import json
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from plotnine import (
    ggplot, aes, geom_point, geom_abline, labs, facet_wrap,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_shape_manual, guides, guide_legend,
)

LOCAL_MAttr = Path("/tmp/mib_pkls/results/mib_node_hard_topk_log")
LOCAL_NK = Path("/tmp/mib_pkls/results/mib_node_natural_k")
CLUSTER_MAttr = Path("results/mib_node_hard_topk_log")
CLUSTER_NK = Path("results/mib_node_natural_k")

TASKS = [
    ("ioi", "gpt2", "IOI / GPT-2"),
    ("ioi", "qwen2.5", "IOI / Qwen"),
    ("ioi", "gemma2", "IOI / Gemma"),
    ("arithmetic_subtraction", "llama3", "Arith / Llama"),
    ("mcqa", "qwen2.5", "MCQA / Qwen"),
    ("mcqa", "gemma2", "MCQA / Gemma"),
    ("mcqa", "llama3", "MCQA / Llama"),
    ("arc_easy", "gemma2", "ARC-E / Gemma"),
    ("arc_easy", "llama3", "ARC-E / Llama"),
    ("arc_challenge", "llama3", "ARC-C / Llama"),
]

PALETTE = {
    "Attn": "#bbbbbb",
    "MLP": "#e41a1c",
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
            if n not in ("input", "logits")}


def load_one(task, model, local_base, cluster_base):
    for base in [local_base, cluster_base]:
        p = base / f"{task}_{model}_importances.json"
        if p.exists():
            return load_scores(p)
    return None


def main():
    all_rows = []
    task_order = []

    for task, model, label in TASKS:
        l2a = load_one(task, model, LOCAL_MAttr, CLUSTER_MAttr)
        nk = load_one(task, model, LOCAL_NK, CLUSTER_NK)
        if l2a is None or nk is None:
            continue

        common = sorted(set(l2a) & set(nk))
        if len(common) < 4:
            continue

        rho, _ = spearmanr([l2a[n] for n in common], [nk[n] for n in common])
        facet_label = f"{label} (ρ={rho:.2f})"
        task_order.append(facet_label)

        for node in common:
            cat = "MLP" if node.startswith("m") else "Attn"
            all_rows.append({
                "l2a_score": l2a[node],
                "nk_score": nk[node],
                "category": cat,
                "type": cat,
                "facet": facet_label,
            })

    df = pd.DataFrame(all_rows)
    df["category"] = pd.Categorical(df["category"], categories=["Attn", "MLP"], ordered=True)
    df["facet"] = pd.Categorical(df["facet"], categories=task_order, ordered=True)
    df = df.sort_values("category", ascending=True).reset_index(drop=True)

    p = (
        ggplot(df, aes(x="l2a_score", y="nk_score", color="category"))
        + geom_point(aes(shape="type"), alpha=0.6, size=0.8)
        + facet_wrap("facet", ncol=5, scales="free")
        + scale_color_manual(values=PALETTE)
        + scale_shape_manual(values={"Attn": "o", "MLP": "s"}, guide=None)
        + labs(x="Score (MAttr)", y="Score (MAttr, natural $k$)", color="")
        + guides(color=guide_legend(override_aes={"size": 2}))
        + theme(
            legend_position="top",
            legend_margin=0,
            legend_box_spacing=0.05,
        )
    )

    out = Path("paper/figs/score_scatter_naturalk.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
