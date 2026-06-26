"""Paper-ready: all SVA eval metrics vs circuit size, faceted by metric, coloured by method,
one figure per eval direction (iso = keep top-k clean; cause = corrupt top-k).
Reproduce: uv run python plots/sva_eval_metrics.py
"""
import json
import numpy as np
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, geom_hline, facet_grid, labs,
    scale_x_log10, scale_color_brewer, theme_set, theme_bw, theme,
    element_text, element_line, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(2.4, 1.7),
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

RES = "results/sva"
TASK, MODEL = "nounpp", "llama3"
METHODS = [  # (label, file tag) — order = legend/colour order
    ("MAttr suff", "sufficient_hard_topk_adam"),
    ("MAttr nec", "necessary_hard_topk_adam"),
    ("RelP", "relp"),
    ("IxG", "ixg"),
    ("IG", "ig"),
]
NODESETS = [("mlp", "MLP"), ("mlp-attn_dim", "MLP + attn")]  # (file key, facet label)
# metric key -> facet title (proper capitalisation)
METRICS = {
    "faithfulness": "Faithfulness (norm. logit diff)",
    "logit_diff": "Logit difference",
    "p_base": "P(base)",
    "p_source": "P(source)",
    "acc_base": "Acc: P(base) > P(source)",
    "acc_source": "Acc: P(source) > P(base)",
}

_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
def log_labels(breaks):
    return [f"10{str(int(round(np.log10(b)))).translate(_SUP)}" if b > 0 else "0" for b in breaks]


def make(direction, fname):
    rows = []
    for label, tag in METHODS:
        for nkey, nlabel in NODESETS:
            d = json.load(open(f"{RES}/{TASK}_{MODEL}_{nkey}_{tag}.json"))
            cur = d[f"{direction}_metrics"]
            for mkey, mtitle in METRICS.items():
                for n, v in zip(d["n_nodes"], cur[mkey]):
                    rows.append({"method": label, "nodes": nlabel, "n_nodes": n,
                                 "metric": mtitle, "value": v})
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], [m[0] for m in METHODS])
    df["nodes"] = pd.Categorical(df["nodes"], [n[1] for n in NODESETS])
    df["metric"] = pd.Categorical(df["metric"], list(METRICS.values()))
    p = (
        ggplot(df, aes("n_nodes", "value", color="method"))
        + geom_hline(yintercept=[0, 1], linetype="dashed", color="#cccccc", size=0.25)
        + geom_line(size=0.5)
        + geom_point(size=0.4)
        + facet_grid("metric ~ nodes", scales="free_y")
        + scale_x_log10(labels=log_labels)
        + scale_color_brewer(type="qual", palette="Set1")
        + labs(x="Circuit size (nodes)", y="Value", color="")
        + theme(figure_size=(5.5, 7.2))
    )
    p.save(fname, verbose=False)
    print("wrote", fname)


make("iso", "plots/sva_eval_metrics_iso.pdf")
make("cause", "plots/sva_eval_metrics_cause.pdf")
