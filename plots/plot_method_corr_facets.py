"""Per-task subplots: full pairwise Spearman correlation matrix among all methods
(node ablations + NAP-IG). One facet per task/model. Run on sc (reads results/).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from plotnine import (
    ggplot, aes, geom_tile, geom_text, labs, facet_wrap, scale_fill_gradient2,
    theme_bw, theme_set, theme, element_text, element_blank, coord_equal,
)

R = Path("/tmp/mib_pkls/results")
RB = R if R.exists() else Path("results")

# (short label, kind, dir). kind: 'napig' or 'node'
METHODS = [
    ("NAP-IG", "napig", "napig_repro/EAP-IG-inputs_patching_node"),
    ("MAttr", "node", "mib_node_hard_topk_log"),
    ("soft", "node", "mib_node_topk_log"),
    ("soft-ck", "node", "mib_node_detached_tau_log"),
    ("hard", "node", "mib_node_bernoulli_reinforce_log"),
    ("MAttr-u", "node", "mib_node_hard_topk"),
    ("soft-u", "node", "final_node"),
    ("sck-u", "node", "mib_node_detached_tau"),
    ("hard-u", "node", "mib_node_bernoulli_reinforce"),
]
ORDER = [m[0] for m in METHODS]

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
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(13, 6),
        axis_text_x=element_text(rotation=90, size=5),
        axis_text_y=element_text(size=5),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        strip_background=element_blank(),
        strip_text=element_text(size=8),
    )
)


def load_scores(path):
    d = json.load(open(path))
    nodes = d["nodes"] if "nodes" in d else d
    return {n: info["score"] for n, info in nodes.items() if n not in ("input", "logits")}


def load(kind, d, task, model):
    if kind == "napig":
        stask = task.replace("_", "-")
        for name in (f"{stask}_{model}", f"{task}_{model}"):
            p = RB / d / name / "importances.json"
            if p.exists():
                return load_scores(p)
        return None
    p = RB / d / f"{task}_{model}_importances.json"
    return load_scores(p) if p.exists() else None


def main():
    rows = []
    facet_order = []
    for task, model, tlabel in TASKS:
        scores = {lbl: load(kind, d, task, model) for lbl, kind, d in METHODS}
        scores = {k: v for k, v in scores.items() if v is not None}
        if len(scores) < 2:
            continue
        facet_order.append(tlabel)
        for a in ORDER:
            for b in ORDER:
                if a in scores and b in scores:
                    common = sorted(set(scores[a]) & set(scores[b]))
                    rho = spearmanr([scores[a][n] for n in common],
                                    [scores[b][n] for n in common])[0] if len(common) >= 4 else np.nan
                else:
                    rho = np.nan
                rows.append({"task": tlabel, "m1": a, "m2": b, "rho": rho})

    df = pd.DataFrame(rows)
    df["m1"] = pd.Categorical(df["m1"], categories=ORDER, ordered=True)
    df["m2"] = pd.Categorical(df["m2"], categories=ORDER[::-1], ordered=True)
    df["task"] = pd.Categorical(df["task"], categories=facet_order, ordered=True)
    df["label"] = df["rho"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")

    p = (
        ggplot(df, aes(x="m1", y="m2", fill="rho"))
        + geom_tile(color="white")
        + geom_text(aes(label="label"), size=3.2)
        + facet_wrap("task", ncol=5)
        + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                               midpoint=0, limits=[-1, 1], na_value="#eeeeee")
        + coord_equal()
        + labs(x="", y="", fill="Spearman ρ")
    )
    out = Path("paper/figs/method_corr_facets.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
