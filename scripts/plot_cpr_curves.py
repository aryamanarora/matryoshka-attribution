"""CPR curves: Ours vs NAP-IG (repro) across all node-level tasks.

Plots faithfulness (CPR) at each sparsity percentage, faceted by task/model.

Usage:
    uv run python scripts/plot_cpr_curves.py
"""

import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_wrap, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_x_log10,
)

# Try local cache first, then cluster paths
LOCAL_BASE = Path("/tmp/mib_pkls/results")
CLUSTER_BASE = Path("results")
RESULTS_BASE = LOCAL_BASE if LOCAL_BASE.exists() else CLUSTER_BASE

COLUMNS = [
    ("ioi", "gpt2", "IOI / GPT-2"),
    ("ioi", "qwen2.5", "IOI / Qwen-2.5"),
    ("ioi", "gemma2", "IOI / Gemma-2"),
    ("arithmetic_subtraction", "llama3", "Arith. / Llama-3.1"),
    ("mcqa", "qwen2.5", "MCQA / Qwen-2.5"),
    ("mcqa", "gemma2", "MCQA / Gemma-2"),
    ("mcqa", "llama3", "MCQA / Llama-3.1"),
    ("arc_easy", "gemma2", "ARC (E) / Gemma-2"),
    ("arc_easy", "llama3", "ARC (E) / Llama-3.1"),
    ("arc_challenge", "llama3", "ARC (C) / Llama-3.1"),
]

SPARSITIES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

PALETTE = {
    "Ours": "#e41a1c",
    "NAP-IG (repro)": "#377eb8",
}

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(12, 8),
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


def load_pkl(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    rows = []
    facet_order = []

    for task, model, label in COLUMNS:
        # Ours
        ours_pkl = RESULTS_BASE / "mib_node_hard_topk" / f"{task}_{model}_validation.pkl"
        ours = load_pkl(ours_pkl)

        # NAP-IG repro
        stask = task.replace("_", "-")
        napig_pkl = RESULTS_BASE / "napig_repro_eval" / "EAP-IG-inputs_patching_node" / f"{stask}_{model}_validation_abs-False.pkl"
        napig = load_pkl(napig_pkl)

        if ours is None and napig is None:
            continue

        facet_order.append(label)

        for method, data, name in [("Ours", ours, "Ours"), ("NAP-IG (repro)", napig, "NAP-IG (repro)")]:
            if data is None:
                continue
            faiths = data["faithfulnesses"]
            for pct, faith in zip(SPARSITIES, faiths):
                rows.append({
                    "task": label,
                    "method": name,
                    "sparsity": pct * 100,
                    "cpr": faith,
                })

    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=facet_order, ordered=True)

    p = (
        ggplot(df, aes(x="sparsity", y="cpr", color="method"))
        + geom_line(size=0.8)
        + geom_point(size=1.5)
        + geom_hline(yintercept=1, linetype="dotted", color="#999999", size=0.4)
        + facet_wrap("task", ncol=4, scales="free_y")
        + scale_x_log10()
        + scale_color_manual(values=PALETTE)
        + labs(x="Circuit size (% of total)", y="CPR", color="Method")
        + theme(legend_position="bottom")
    )

    out = Path("paper/figs/cpr_curves_node.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
