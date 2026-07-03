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
    ggplot, aes, geom_col, facet_grid, labs, theme, theme_set, theme_bw,
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
TASKS = ["nounpp", "rc", "simple", "within_rc"]
# MAttr split into gate/optimizer family: soft = hard_topk + Adam (sigmoid-STE);
# idSTE = hard_topk_identity + SGD (identity-STE). Each x {log, uniform} k; -IG = mask-space
# integrated-gradient score update (--mattr-ig-steps>1), swept on log-k only.
METHOD_ORDER = ["IG", "IxG", "soft-log", "soft-unif", "idSTE-log", "idSTE-unif",
                "soft-log-IG", "idSTE-log-IG"]
LOSS_ORDER = ["logit_diff", "ce", "acc"]
# (json key, facet-strip label, log10-transform?)
METRICS = [
    ("acc_auc", "Accuracy AUC (↑)", False),
    ("kstar_50", "log₁₀ k* (↓)", True),          # huge dynamic range -> log10 in load()
    ("faith_auc", "Faithfulness AUC (↑)", False),
    ("cause_accsrc_auc", "Cause acc-source AUC (↑)", False),
]


def parse_method(fname: str, d: dict) -> str:
    """Method label from the filename tag (JSON doesn't store ig/ixg/mattr)."""
    nodes_safe = d["nodes"].replace("+", "-")   # filenames use '-' not '+'
    tag = fname.split(f"_{nodes_safe}_", 1)[1].rsplit(".json", 1)[0]
    if "hard_topk" in tag:
        fam = "idSTE" if "identity" in tag else "soft"
        ks = "unif" if "uniformk" in tag else "log"
        ig = "-IG" if re.search(r"_ig\d+", tag) else ""
        return f"{fam}-{ks}{ig}"
    return "IxG" if tag.startswith("ixg") else "IG"


def load() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RES / "*.json"))):
        d = json.load(open(f))
        rec = {"task": d["task"], "nodes": d["nodes"], "loss": d["loss"],
               "method": parse_method(Path(f).name, d)}
        for key, _, is_log in METRICS:
            v = d.get(key)
            rec[key] = (np.nan if v is None
                        else float(np.log10(v)) if is_log else float(v))
        rows.append(rec)
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


def plot_substrate(m: pd.DataFrame, nodes: str, out: Path):
    p = (
        ggplot(m[m["nodes"] == nodes], aes("method", "value", fill="loss"))
        + geom_col(position=position_dodge(width=0.8), width=0.72)
        + facet_grid("metric ~ task", scales="free_y")
        + scale_fill_brewer(type="qual", palette="Set1")
        + labs(x="", y="", fill="Loss")
        + theme(figure_size=(7.5, 5.2))   # 8 methods x 4 tasks
    )
    p.save(out, verbose=False)
    print(f"wrote {out}")


def main():
    df = load()
    print(f"loaded {len(df)} runs; nodes={sorted(df.nodes.unique())}")
    m = long_form(df)
    for nodes in sorted(df["nodes"].unique()):
        plot_substrate(m, nodes, RES / f"sva_sweep_facet_{nodes.replace('+', '-')}.pdf")


if __name__ == "__main__":
    main()
