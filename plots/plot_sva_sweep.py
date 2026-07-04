"""Paper-ready SVA sweep figure: facet grid of metrics (rows) x task (cols), one PDF
per substrate. Bars = attribution method (IG / IxG / MAttr-log / MAttr-uniform), filled
by training/target loss (logit_diff / ce / acc). Data: results/sva_sweep/*.json.

Run:  uv run python plots/plot_sva_sweep.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_col, geom_hline, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, position_dodge, scale_fill_brewer,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),  # ~70% — LaTeX upscales so text/lines feel larger
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.02,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

RES = Path("results/sva_sweep")
# SVA tasks are llama3; MIB tasks carry a /model suffix (node substrate only).
TASKS = ["nounpp", "rc", "simple", "within_rc", "arc_easy", "ioi/qwen2.5"]
# metrics that have a measured random-ranking baseline (drawn as a dashed red line)
RANDOM_METRICS = {"acc_auc", "faith_auc"}
# MAttr split into gate/optimizer family: soft = hard_topk + Adam (sigmoid-STE);
# idSTE = hard_topk_identity + SGD (identity-STE). Each x {log, uniform} k; -IG = mask-space
# integrated-gradient score update (--mattr-ig-steps>1), swept on log-k only.
METHOD_ORDER = ["IG", "IxG", "Cond",
                "soft-log", "soft-unif", "soft-fixed", "idSTE-log", "idSTE-unif", "idSTE-fixed",
                "soft-log-IG", "idSTE-log-IG"]   # -fixed = trained at fixed k=10% (no schedule)
LOSS_ORDER = ["logit_diff", "ce", "acc"]
# (json key, facet-strip label, log10-transform?)
METRICS = [
    ("acc_auc", "Accuracy AUC (↑)", False),
    ("kstar_50", "log₁₀ k* iso (↓)", True),      # iso: first k s.t. acc_base>=0.5 (in JSON)
    ("faith_auc", "Faithfulness AUC (↑)", False),
    ("cause_kstar", "log₁₀ k* cause (↓)", True), # cause: first k s.t. acc_source>=0.5 (computed)
]
CAUSE_THR = 0.5


def parse_method(fname: str, d: dict) -> str:
    """Method label from the filename tag (JSON doesn't store ig/ixg/mattr)."""
    nodes_safe = d["nodes"].replace("+", "-")   # filenames use '-' not '+'
    tag = fname.split(f"_{nodes_safe}_", 1)[1].rsplit(".json", 1)[0]
    if tag.startswith("random"):
        return "RANDOM"
    if tag.startswith("conductance"):
        return "Cond"
    if "hard_topk" in tag:
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "fixed" if "fixedk" in tag else ("unif" if "uniformk" in tag else "log")
        ig = "-IG" if re.search(r"_ig\d+", tag) else ""
        return f"{fam}-{ks}{ig}"
    return "IxG" if tag.startswith("ixg") else "IG"


def _task_label(d):
    """SVA (llama3) tasks keep their name; MIB tasks on other models get a /model suffix."""
    return d["task"] if d["model"] == "llama3" else f"{d['task']}/{d['model']}"


def load() -> pd.DataFrame:
    """Method runs only (random-baseline files are excluded here; see random_lines())."""
    rows = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        d = json.load(open(f))
        method = parse_method(Path(f).name, d)
        if method == "RANDOM":
            continue
        rec = {"task": _task_label(d), "nodes": d["nodes"], "loss": d["loss"], "method": method}
        # cause k*: first k (n_nodes) at which corrupting the top-k gives acc_source >= thr
        cs = d["cause_metrics"]["acc_source"]
        cause_kstar = next((float(k) for k, a in zip(d["n_nodes"], cs) if a >= CAUSE_THR), None)
        for key, _, is_log in METRICS:
            v = cause_kstar if key == "cause_kstar" else d.get(key)
            rec[key] = (np.nan if v is None
                        else float(np.log10(v)) if is_log else float(v))
        rows.append(rec)
    return pd.DataFrame(rows)


def random_lines() -> pd.DataFrame:
    """Measured random-ranking baselines -> dashed lines, one per (nodes, task, metric)."""
    lbl = {k: l for k, l, _ in METRICS}
    acc = {}   # (nodes, task) -> list of (acc_auc, faith_auc)
    for f in glob.glob(str(RES / "*.json")):
        d = json.load(open(f))
        if parse_method(Path(f).name, d) != "RANDOM":
            continue
        acc.setdefault((d["nodes"], _task_label(d)), []).append((d["acc_auc"], d["faith_auc"]))
    rows = []
    for (nodes, task), vals in acc.items():
        arr = np.array(vals)
        for key, col in (("acc_auc", 0), ("faith_auc", 1)):
            rows.append({"nodes": nodes, "task": task, "metric": lbl[key],
                         "yintercept": float(arr[:, col].mean())})
    return pd.DataFrame(rows)


def long_form(df: pd.DataFrame) -> pd.DataFrame:
    m = df.melt(id_vars=["task", "nodes", "loss", "method"],
                value_vars=[k for k, _, _ in METRICS],
                var_name="metric", value_name="value")
    m["metric"] = pd.Categorical(m["metric"].map({k: lbl for k, lbl, _ in METRICS}),
                                 categories=[lbl for _, lbl, _ in METRICS], ordered=True)
    m["task"] = pd.Categorical(m["task"], categories=TASKS, ordered=True)
    m["method"] = pd.Categorical(m["method"], categories=METHOD_ORDER, ordered=True)
    m["loss"] = pd.Categorical(m["loss"], categories=LOSS_ORDER, ordered=True)
    return m


def plot_substrate(m: pd.DataFrame, nodes: str, out: Path, rand: pd.DataFrame):
    sub = m[m["nodes"] == nodes].copy()
    sub["task"] = sub["task"].cat.remove_unused_categories()   # drop tasks absent for substrate
    keep = list(sub["task"].cat.categories)
    n_tasks = len(keep)
    p = (
        ggplot(sub, aes("method", "value", fill="loss"))
        + geom_col(position=position_dodge(width=0.8), width=0.72)
        + facet_grid("metric ~ task", scales="free_y")
        + scale_fill_brewer(type="qual", palette="Set1")
        + labs(x="", y="", fill="Loss")
        + theme(figure_size=(1.6 * n_tasks + 1.1, 5.2))   # 8 methods; width scales with #tasks
    )
    # dashed red random-ranking baseline where measured (acc/faith rows only)
    rl = rand[(rand["nodes"] == nodes) & (rand["task"].isin(keep))].copy()
    if len(rl):
        rl["metric"] = pd.Categorical(rl["metric"], categories=[l for _, l, _ in METRICS], ordered=True)
        rl["task"] = pd.Categorical(rl["task"], categories=keep, ordered=True)
        p = p + geom_hline(rl, aes(yintercept="yintercept"), linetype="dashed",
                           color="red", size=0.4)
    p.save(out, verbose=False)
    print(f"wrote {out}")


def main():
    df = load()
    rand = random_lines()
    print(f"loaded {len(df)} runs; nodes={sorted(df.nodes.unique())}; random lines={len(rand)}")
    m = long_form(df)
    for nodes in sorted(df["nodes"].unique()):
        plot_substrate(m, nodes, RES / f"sva_sweep_facet_{nodes.replace('+', '-')}.pdf", rand)


if __name__ == "__main__":
    main()
