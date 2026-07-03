"""CPR curves: all our node-level methods + NAP-IG, faceted by task/model.

Usage:
    uv run python plots/plot_cpr_curves.py
"""

import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_wrap, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_x_log10, scale_linetype_manual,
)

LOCAL_BASE = Path("/tmp/mib_pkls/results")
CLUSTER_BASE = Path("results")
R = LOCAL_BASE if LOCAL_BASE.exists() else CLUSTER_BASE

COLUMNS = [
    ("ioi", "gpt2", "IOI / GPT"),
    ("ioi", "qwen2.5", "IOI / Qwen"),
    ("ioi", "gemma2", "IOI / Gemma"),
    ("arithmetic_subtraction", "llama3", "Arith. / Llama"),
    ("mcqa", "qwen2.5", "MCQA / Qwen"),
    ("mcqa", "gemma2", "MCQA / Gemma"),
    ("mcqa", "llama3", "MCQA / Llama"),
    ("arc_easy", "gemma2", "ARC (E) / Gemma"),
    ("arc_easy", "llama3", "ARC (E) / Llama"),
    ("arc_challenge", "llama3", "ARC (C) / Llama"),
]

SPARSITIES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

# Methods: (display_name, results_subdir, pkl_pattern)
METHODS = [
    ("Ours", "mib_node_hard_topk", "{task}_{model}_validation.pkl"),
    ("Ours (log k)", "mib_node_hard_topk_log", "{task}_{model}_validation.pkl"),
    ("+ soft fwd", "final_node", "{task}_{model}_validation.pkl"),
    ("+ soft fwd, - c_k grad", "mib_node_detached_tau", "{task}_{model}_validation.pkl"),
    ("+ hard bwd", "mib_node_bernoulli_reinforce", "{task}_{model}_validation.pkl"),
    ("NAP-IG (repro)", "napig_repro_eval/EAP-IG-inputs_patching_node",
     "{stask}_{model}_validation_abs-False.pkl"),
]

PALETTE = {
    "Ours": "#e41a1c",
    "Ours (log k)": "#ff7f00",
    "+ soft fwd": "#984ea3",
    "+ soft fwd, - c_k grad": "#4daf4a",
    "+ hard bwd": "#a65628",
    "NAP-IG (repro)": "#377eb8",
}

LINETYPES = {
    "Ours": "solid",
    "Ours (log k)": "solid",
    "+ soft fwd": "dashed",
    "+ soft fwd, - c_k grad": "dashed",
    "+ hard bwd": "dotted",
    "NAP-IG (repro)": "solid",
}

theme_set(
    theme_bw(base_size=10)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(12, 8),
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


def load_pkl(path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def resolve(base, *parts):
    p = base / Path(*parts)
    if p.exists():
        return p
    p2 = CLUSTER_BASE / Path(*parts)
    return p2 if p2.exists() else None


def main():
    rows = []
    facet_order = []
    method_order = [m[0] for m in METHODS]

    for task, model, label in COLUMNS:
        stask = task.replace("_", "-")
        has_any = False

        for method_name, subdir, pattern in METHODS:
            fname = pattern.format(task=task, model=model, stask=stask)
            pkl_path = resolve(R, subdir, fname)

            if pkl_path is None or not pkl_path.exists():
                continue

            data = load_pkl(pkl_path)
            if data is None:
                continue

            has_any = True
            faiths = data["faithfulnesses"]
            for pct, faith in zip(SPARSITIES, faiths):
                rows.append({
                    "task": label,
                    "method": method_name,
                    "sparsity": pct * 100,
                    "cpr": faith,
                })

        if has_any and label not in facet_order:
            facet_order.append(label)

    df = pd.DataFrame(rows)
    df["task"] = pd.Categorical(df["task"], categories=facet_order, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    p = (
        ggplot(df, aes(x="sparsity", y="cpr", color="method", linetype="method"))
        + geom_line(size=0.7)
        + geom_point(size=1.2)
        + geom_hline(yintercept=1, linetype="dotted", color="#999999", size=0.3)
        + facet_wrap("task", ncol=4, scales="free_y")
        + scale_x_log10()
        + scale_color_manual(values=PALETTE)
        + scale_linetype_manual(values=LINETYPES)
        + labs(x="Circuit size (% of total)", y="CPR", color="Method", linetype="Method")
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
