"""CPR curves for all node ablations, faceted by dataset (task/model).
Color = base method, linetype = k-schedule (log solid / uniform dashed). Run on sc.
"""

import pickle
from pathlib import Path

import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_wrap, labs,
    theme_bw, theme_set, theme, element_text, element_blank, element_line,
    scale_color_manual, scale_x_log10, scale_linetype_manual, guides, guide_legend,
)

R = Path("/tmp/mib_pkls/results")
RB = R if R.exists() else Path("results")

COLUMNS = [
    ("ioi", "gpt2", "IOI / GPT"),
    ("ioi", "qwen2.5", "IOI / Qwen"),
    ("ioi", "gemma2", "IOI / Gemma"),
    ("ioi", "llama3", "IOI / Llama"),
    ("arithmetic_subtraction", "llama3", "Arith / Llama"),
    ("mcqa", "qwen2.5", "MCQA / Qwen"),
    ("mcqa", "gemma2", "MCQA / Gemma"),
    ("mcqa", "llama3", "MCQA / Llama"),
    ("arc_easy", "gemma2", "ARC-E / Gemma"),
    ("arc_easy", "llama3", "ARC-E / Llama"),
    ("arc_challenge", "llama3", "ARC-C / Llama"),
]

SPARSITIES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

# (base, schedule, dir, is_napig)
METHODS = [
    ("MAttr", "log", "mib_node_hard_topk_log", False),
    ("MAttr", "uniform", "mib_node_hard_topk", False),
    ("+soft fwd", "log", "mib_node_topk_log", False),
    ("+soft fwd", "uniform", "final_node", False),
    ("+soft -ck", "log", "mib_node_detached_tau_log", False),
    ("+soft -ck", "uniform", "mib_node_detached_tau", False),
    ("+hard bwd", "log", "mib_node_bernoulli_reinforce_log", False),
    ("+hard bwd", "uniform", "mib_node_bernoulli_reinforce", False),
    ("+Gumbel", "uniform", "mib_node_hard_topk_gumbel", False),
    ("+id-STE SGD", "log", "mib_node_identity_sgd_log", False),
    ("+id-STE SGD", "uniform", "mib_node_identity_sgd", False),
    ("NAP-IG", "log", "napig_repro_eval/EAP-IG-inputs_patching_node", True),
]
BASE_ORDER = ["MAttr", "+Gumbel", "+soft fwd", "+soft -ck", "+hard bwd", "+id-STE SGD", "NAP-IG"]

PALETTE = {
    "MAttr": "#e41a1c", "+Gumbel": "#ff7f00", "+soft fwd": "#984ea3",
    "+soft -ck": "#4daf4a", "+hard bwd": "#a65628", "+id-STE SGD": "#f781bf", "NAP-IG": "#377eb8",
}

theme_set(
    theme_bw(base_size=9)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(12, 5.5),
        panel_grid_minor=element_blank(),
        panel_grid_major=element_line(size=0.3, color="#dddddd"),
        strip_background=element_blank(),
        strip_text=element_text(size=8),
        legend_position="bottom",
    )
)


def load_faith(d, task, model, is_napig):
    if is_napig:
        stask = task.replace("_", "-")
        p = RB / d / f"{stask}_{model}_validation_abs-False.pkl"
    else:
        p = RB / d / f"{task}_{model}_validation.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)["faithfulnesses"]


def main():
    rows = []
    facet_order = []
    for task, model, label in COLUMNS:
        has = False
        for base, sched, d, is_napig in METHODS:
            faith = load_faith(d, task, model, is_napig)
            if faith is None:
                continue
            has = True
            for pct, fv in zip(SPARSITIES, faith):
                rows.append({"task": label, "base": base, "sched": sched,
                             "sparsity": pct * 100, "cpr": fv})
        if has:
            facet_order.append(label)

    df = pd.DataFrame(rows)
    df["base"] = pd.Categorical(df["base"], categories=BASE_ORDER, ordered=True)
    df["task"] = pd.Categorical(df["task"], categories=facet_order, ordered=True)

    p = (
        ggplot(df, aes(x="sparsity", y="cpr", color="base", linetype="sched"))
        + geom_line(size=0.6)
        + geom_point(size=0.8)
        + geom_hline(yintercept=1, linetype="dotted", color="#999999", size=0.3)
        + facet_wrap("task", ncol=4, scales="free_y")
        + scale_x_log10()
        + scale_color_manual(values=PALETTE)
        + scale_linetype_manual(values={"log": "solid", "uniform": "dashed"})
        + labs(x="Circuit size (% of nodes)", y="CPR", color="Method", linetype="k-schedule")
        + guides(color=guide_legend(nrow=1), linetype=guide_legend(nrow=1))
    )
    out = Path("paper/figs/cpr_curves_ablations.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, dpi=300)
    print(f"Saved {out}")
    p.save(out.with_suffix(".png"), dpi=150)
    print(f"Saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
