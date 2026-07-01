"""Bar chart showing effect of swapping MLP rankings between methods.

Usage:
    uv run python plots/plot_hybrid_swap.py
"""

import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, geom_hline, geom_text, labs, position_dodge,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_fill_manual,
)

LOCAL = Path("/tmp/mib_pkls/results/hybrid_eval")
CLUSTER = Path("results/hybrid_eval")
RESULTS_BASE = LOCAL if LOCAL.exists() else CLUSTER

COLUMNS = [
    ("ioi", "gpt2", "IOI\nGPT-2"),
    ("ioi", "qwen2.5", "IOI\nQwen"),
    ("ioi", "gemma2", "IOI\nGemma"),
    ("ioi", "llama3", "IOI\nLlama"),
    ("arithmetic_subtraction", "llama3", "Arith\nLlama"),
    ("mcqa", "qwen2.5", "MCQA\nQwen"),
    ("mcqa", "gemma2", "MCQA\nGemma"),
    ("mcqa", "llama3", "MCQA\nLlama"),
    ("arc_easy", "gemma2", "ARC-E\nGemma"),
    ("arc_easy", "llama3", "ARC-E\nLlama"),
    ("arc_challenge", "llama3", "ARC-C\nLlama"),
]

METHODS = [
    ("ours", "MAttr"),
    ("napig_ours_mlp", "NAP-IG + MAttr MLPs"),
    ("ours_napig_mlp", "MAttr + NAP-IG MLPs"),
    ("napig", "NAP-IG"),
]

PALETTE = {
    "MAttr": "#bbbbbb",
    "NAP-IG + MAttr MLPs": "#e41a1c",
    "MAttr + NAP-IG MLPs": "#377eb8",
    "NAP-IG": "#888888",
}

OURS_DIR = Path("/tmp/mib_pkls/results/mib_node_hard_topk")
NAPIG_DIR = Path("/tmp/mib_pkls/results/napig_repro_eval/EAP-IG-inputs_patching_node")
OURS_CLUSTER = Path("results/mib_node_hard_topk")
NAPIG_CLUSTER = Path("results/napig_repro_eval/EAP-IG-inputs_patching_node")

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(3.67, 1.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, ha="center"),
        legend_text=element_text(size=5.5),
        legend_title=element_blank(),
        legend_key_size=8,
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_grid_major_x=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=7, face="plain"),
    )
)


def load(task, model, method):
    stask = task.replace("_", "-")
    if method == "ours":
        for d in [OURS_DIR, OURS_CLUSTER]:
            p = d / f"{task}_{model}_validation.pkl"
            if p.exists():
                return pickle.load(open(p, "rb"))["area_under"]
        return None
    elif method == "napig":
        for d in [NAPIG_DIR, NAPIG_CLUSTER]:
            p = d / f"{stask}_{model}_validation_abs-False.pkl"
            if p.exists():
                return pickle.load(open(p, "rb"))["area_under"]
        return None
    else:
        for d in [RESULTS_BASE, Path("results/hybrid_eval")]:
            p = d / f"{task}_{model}_{method}_validation.pkl"
            if p.exists():
                return pickle.load(open(p, "rb"))["area_under"]
        return None


def main():
    rows = []
    task_order = []
    method_order = [m[1] for m in METHODS]

    for task, model, label in COLUMNS:
        task_rows = []
        for hybrid, name in METHODS:
            v = load(task, model, hybrid)
            if v is not None:
                task_rows.append({"task": label, "method": name, "cpr_auc": round(v, 2)})
        if len(task_rows) == len(METHODS):
            rows.extend(task_rows)
            task_order.append(label)

    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=task_order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    p = (
        ggplot(df, aes(x="task", y="cpr_auc", fill="method"))
        + geom_col(position=position_dodge(width=0.8), width=0.7)
        + geom_hline(yintercept=1, linetype="dotted", color="#999999", size=0.4)
        + geom_text(
            aes(label="cpr_auc"),
            position=position_dodge(width=0.8),
            size=3, va="bottom",
            format_string="{:.2f}",
        )
        + scale_fill_manual(values=PALETTE)
        + labs(x="", y="CPR AUC")
        + theme(legend_position="top")
    )

    out = Path("paper/figs/hybrid_swap.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
