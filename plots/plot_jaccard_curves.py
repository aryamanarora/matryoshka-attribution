"""Jaccard overlap at varying top/bottom-k% between Ours and baselines.

Faceted: rows = node/edge level, columns = task/model.

Usage:
    uv run python plots/plot_jaccard_curves.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_linetype_manual,
)

LOCAL = Path("/tmp/mib_pkls/results")
CLUSTER = Path("results")
R = LOCAL if LOCAL.exists() else CLUSTER

COLUMNS = [
    ("ioi", "gpt2", "IOI/GPT-2"),
    ("ioi", "qwen2.5", "IOI/Qwen"),
    ("ioi", "gemma2", "IOI/Gemma"),
    ("mcqa", "qwen2.5", "MCQA/Qwen"),
    ("mcqa", "gemma2", "MCQA/Gemma"),
]

KS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

PALETTE = {"top-k": "#e41a1c", "bottom-k": "#377eb8", "random": "#999999"}
LINETYPES = {"top-k": "solid", "bottom-k": "dashed", "random": "dotted"}

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(12, 5),
        axis_title=element_text(size=10),
        axis_text=element_text(size=7),
        legend_text=element_text(size=9),
        legend_title=element_text(size=10),
        panel_grid_major=element_line(size=0.5, color="#dddddd"),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=8, face="plain"),
    )
)


def load_node(path):
    d = json.load(open(path))
    nodes = d.get("nodes", d)
    return np.array([info["score"] for n, info in nodes.items() if n not in ("input", "logits")])


def jaccard_at_k(a, b, k_frac, bottom=False):
    n = len(a)
    k = max(1, int(n * k_frac))
    order = np.argsort(a if bottom else -a)
    sel_a = set(order[:k].tolist())
    order = np.argsort(b if bottom else -b)
    sel_b = set(order[:k].tolist())
    return len(sel_a & sel_b) / len(sel_a | sel_b)


def resolve(base, *parts):
    p = base / Path(*parts)
    if p.exists():
        return p
    p2 = CLUSTER / Path(*parts)
    return p2 if p2.exists() else None


def main():
    rows = []
    facet_cols = []

    for task, model, label in COLUMNS:
        stask = task.replace("_", "-")

        # Node: Ours vs NAP-IG
        ours_n = resolve(R, "mib_node_hard_topk", f"{task}_{model}_importances.json")
        napig_n = resolve(R, "napig_repro", "EAP-IG-inputs_patching_node", f"{stask}_{model}", "importances.json")

        if ours_n and napig_n and ours_n.exists() and napig_n.exists():
            a = load_node(ours_n)
            b = load_node(napig_n)
            if label not in facet_cols:
                facet_cols.append(label)
            for k in KS:
                for direction in ["top-k", "bottom-k"]:
                    j = jaccard_at_k(a, b, k, bottom=(direction == "bottom-k"))
                    rows.append({"task": label, "k_pct": k * 100, "jaccard": j,
                                  "direction": direction, "level": "Node"})
                n = len(a)
                ki = max(1, int(n * k))
                j_rand = ki / (2 * n - ki)
                rows.append({"task": label, "k_pct": k * 100, "jaccard": j_rand,
                              "direction": "random", "level": "Node"})

        # Edge: Ours vs EAP-IG
        ours_e = resolve(R, "mib_edge_hard_topk", f"{task}_{model}_scores.pt")
        eapig_e = resolve(R, "eapig_repro", "EAP-IG-inputs_patching_edge", f"{stask}_{model}", "importances.json")

        if ours_e and eapig_e and ours_e.exists() and eapig_e.exists():
            a = torch.load(ours_e, weights_only=False, map_location="cpu")["scores"].numpy()
            b_data = json.load(open(eapig_e))
            b = np.array([e["score"] for e in b_data["edges"].values()])
            if label not in facet_cols:
                facet_cols.append(label)
            for k in KS:
                for direction in ["top-k", "bottom-k"]:
                    j = jaccard_at_k(a, b, k, bottom=(direction == "bottom-k"))
                    rows.append({"task": label, "k_pct": k * 100, "jaccard": j,
                                  "direction": direction, "level": "Edge"})
                n = len(a)
                ki = max(1, int(n * k))
                j_rand = ki / (2 * n - ki)
                rows.append({"task": label, "k_pct": k * 100, "jaccard": j_rand,
                              "direction": "random", "level": "Edge"})

    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=facet_cols, ordered=True)
    df["level"] = pd.Categorical(df["level"], categories=["Node", "Edge"], ordered=True)

    p = (
        ggplot(df, aes(x="k_pct", y="jaccard", color="direction", linetype="direction"))
        + geom_line(size=0.8)
        + geom_point(size=1.5)
        + facet_grid("level ~ task")
        + scale_color_manual(values=PALETTE)
        + scale_linetype_manual(values=LINETYPES)
        + labs(x="k%", y="Jaccard overlap", color="", linetype="")
        + theme(legend_position="bottom")
    )

    out = Path("paper/figs/jaccard_curves.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
