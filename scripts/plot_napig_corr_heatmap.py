"""Heatmap of per-task Spearman correlation between each node ablation's scores and NAP-IG.

rows = ablation methods, cols = task/model, fill = Spearman rho (node importances vs NAP-IG).
Run on sc (reads results/).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from plotnine import (
    ggplot, aes, geom_tile, geom_text, labs, scale_fill_gradient2,
    theme_bw, theme_set, theme, element_text, element_blank, scale_x_discrete,
    scale_y_discrete,
)

R = Path("/tmp/mib_pkls/results")
RB = R if R.exists() else Path("results")

# (label, results-dir)
METHODS = [
    ("MAttr (log)", "mib_node_hard_topk_log"),
    ("+soft (log)", "mib_node_topk_log"),
    ("+soft -ck (log)", "mib_node_detached_tau_log"),
    ("+hard bwd (log)", "mib_node_bernoulli_reinforce_log"),
    ("MAttr (unif)", "mib_node_hard_topk"),
    ("+soft (unif)", "final_node"),
    ("+soft -ck (unif)", "mib_node_detached_tau"),
    ("+hard bwd (unif)", "mib_node_bernoulli_reinforce"),
]

TASKS = [
    ("ioi", "gpt2", "IOI/GPT"),
    ("ioi", "qwen2.5", "IOI/Qwen"),
    ("ioi", "gemma2", "IOI/Gemma"),
    ("arithmetic_subtraction", "llama3", "Arith/Llama"),
    ("mcqa", "qwen2.5", "MCQA/Qwen"),
    ("mcqa", "gemma2", "MCQA/Gemma"),
    ("mcqa", "llama3", "MCQA/Llama"),
    ("arc_easy", "gemma2", "ARC-E/Gemma"),
    ("arc_easy", "llama3", "ARC-E/Llama"),
    ("arc_challenge", "llama3", "ARC-C/Llama"),
]

theme_set(
    theme_bw(base_size=9)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(7, 4),
        axis_text_x=element_text(rotation=40, ha="right", size=7),
        axis_text_y=element_text(size=8),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
    )
)


def load_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {n: info["score"] for n, info in nodes.items() if n not in ("input", "logits")}


def load_method(d, task, model):
    p = RB / d / f"{task}_{model}_importances.json"
    return load_scores(p) if p.exists() else None


def load_napig(task, model):
    stask = task.replace("_", "-")
    for name in (f"{stask}_{model}", f"{task}_{model}"):
        p = RB / "napig_repro" / "EAP-IG-inputs_patching_node" / name / "importances.json"
        if p.exists():
            return load_scores(p)
    return None


def main():
    rows = []
    for task, model, tlabel in TASKS:
        nap = load_napig(task, model)
        if nap is None:
            continue
        for mlabel, d in METHODS:
            s = load_method(d, task, model)
            if s is None:
                rows.append({"method": mlabel, "task": tlabel, "rho": np.nan})
                continue
            common = sorted(set(s) & set(nap))
            rho = spearmanr([s[n] for n in common], [nap[n] for n in common])[0] if len(common) >= 4 else np.nan
            rows.append({"method": mlabel, "task": tlabel, "rho": rho})

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=[m[0] for m in METHODS][::-1], ordered=True)
    df["task"] = pd.Categorical(df["task"], categories=[t[2] for t in TASKS], ordered=True)
    df["label"] = df["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    p = (
        ggplot(df, aes(x="task", y="method", fill="rho"))
        + geom_tile(color="white")
        + geom_text(aes(label="label"), size=7)
        + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                               midpoint=0, limits=[-1, 1], na_value="#eeeeee")
        + labs(x="", y="", fill="Spearman ρ\nvs NAP-IG")
    )
    out = Path("paper/figs/napig_corr_heatmap.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
