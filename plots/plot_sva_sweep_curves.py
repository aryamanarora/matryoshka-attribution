"""SVA sweep curves: metric vs k (the curves whose log-k integral is the AUC), one PDF per
substrate. facet_grid(metric ~ task); a line per (method, loss): colour = method, linetype =
loss. Data: results/sva_sweep/*.json (n_nodes + iso_metrics/cause_metrics per-k curves).

Run:  uv run python plots/plot_sva_sweep_curves.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, facet_grid, labs, theme, theme_set, theme_bw,
    element_text, element_line, element_blank, scale_x_log10, scale_color_brewer,
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
TASKS = ["nounpp", "rc", "simple", "within_rc", "arc_easy"]   # arc_easy: MIB, node substrate only
METHOD_ORDER = ["IG", "IxG", "Cond",
                "soft-log", "soft-unif", "soft-fixed", "idSTE-log", "idSTE-unif", "idSTE-fixed",
                "soft-log-IG", "idSTE-log-IG"]   # Cond = conductance; -fixed = fixed k=10%
LOSS_ORDER = ["logit_diff", "ce", "acc"]
# (metrics dict in JSON, key, facet-strip label) — the curves behind the AUC rows.
CURVES = [
    ("iso_metrics", "acc_base", "Accuracy (↑)"),
    ("iso_metrics", "faithfulness", "Faithfulness (↑)"),
    ("cause_metrics", "acc_source", "Cause acc-source (↑)"),
]

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci_labels(breaks):
    out = []
    for b in breaks:
        if b is None or b <= 0 or not np.isfinite(b):
            out.append("")
        else:
            out.append("10" + str(int(round(np.log10(b)))).translate(_SUP))
    return out


def parse_method(fname: str, d: dict) -> str:
    nodes_safe = d["nodes"].replace("+", "-")
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


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        d = json.load(open(f))
        method = parse_method(Path(f).name, d)
        if method == "RANDOM":
            continue
        ks = d["n_nodes"]
        task = d["task"] if d["model"] == "llama3" else f"{d['task']}/{d['model']}"
        for dictname, key, label in CURVES:
            ys = d[dictname][key]
            for k, y in zip(ks, ys):
                rows.append({"task": task, "nodes": d["nodes"], "loss": d["loss"],
                             "method": method, "metric": label, "k": float(k),
                             "value": float(y)})
    return pd.DataFrame(rows)


def cat(df: pd.DataFrame) -> pd.DataFrame:
    df["metric"] = pd.Categorical(df["metric"], categories=[l for _, _, l in CURVES], ordered=True)
    df["task"] = pd.Categorical(df["task"], categories=TASKS, ordered=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    df["loss"] = pd.Categorical(df["loss"], categories=LOSS_ORDER, ordered=True)
    return df


def plot_substrate(df: pd.DataFrame, nodes: str, out: Path):
    sub = df[df["nodes"] == nodes].copy()
    sub["task"] = sub["task"].cat.remove_unused_categories()   # drop tasks absent for substrate
    n_tasks = sub["task"].nunique()
    brk = [10.0 ** e for e in range(0, 8) if 10 ** e <= sub["k"].max() * 1.5]
    p = (
        ggplot(sub, aes("k", "value", color="method", linetype="loss"))
        + geom_line(size=0.4)
        + facet_grid("metric ~ task", scales="free_y")
        + scale_x_log10(breaks=brk, labels=sci_labels)
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="k (nodes kept clean)", y="", color="Method", linetype="Loss")
        + theme(figure_size=(1.35 * n_tasks + 1.0, 4.6))
    )
    p.save(out, verbose=False)
    print(f"wrote {out}")


def main():
    df = cat(load())
    print(f"loaded curves for {df.groupby('nodes').ngroups} substrate(s); "
          f"{df[['task','nodes','method','loss']].drop_duplicates().shape[0]} runs")
    for nodes in sorted(df["nodes"].unique()):
        plot_substrate(df, nodes, RES / f"sva_sweep_curves_{nodes.replace('+', '-')}.pdf")


if __name__ == "__main__":
    main()
