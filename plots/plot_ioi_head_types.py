"""IOI / GPT-2 small: where each attribution method ranks the known circuit heads.

One heatmap, test-split rankings: rows are the key MIB node-level methods (ordered by their
test-table CPR average), columns are the 26 attention heads of the Wang et al. (2023) IOI
circuit grouped by head class, cell = the head's rank among all 157 nodes under that method
(1 = most important; colour on a log scale, text = rank). A method that "finds the circuit"
reads as a dark row; a class it misses reads as a light block.

Head classes are Wang et al.'s (Figure 2 / Table 2). The negative name movers are the heads
McDougall et al. (2023) later characterised as copy-suppression heads, so that column is
labelled with both names.

Rank = 1 + #nodes with a strictly larger score, over all 157 nodes (input, 144 heads, 12 MLPs),
i.e. exactly the order MIB's evaluate_area_under_curve consumes (absolute=False). Gradient
baselines are attributed on the train split and are split-independent (the fork's
importances.json); MAttr / IntInv / no-learning read their test-run importances; Node Pruning
and DBM read the learned mask logits in graph_ioi_gpt2.json.

    uv run python plots/plot_ioi_head_types.py   ->  paper/figs/ioi_head_types.pdf
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotnine import (aes, element_blank, element_rect, element_text, facet_grid, geom_text,
                      geom_tile, ggplot, labs, scale_color_identity, scale_fill_cmap,
                      scale_x_discrete, scale_y_discrete, theme, theme_bw, theme_set)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from learning_to_attribute.deps import mib_results_dir  # noqa: E402

R = Path("results")
R_MIB = mib_results_dir()
OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.35),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=5.5, rotation=90, hjust=0.5, vjust=0.5),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.015,
        panel_border=element_rect(color="#000", size=0.4),
        strip_background=element_blank(),
        strip_text=element_text(size=6.2),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="right",
        legend_direction="vertical",
        legend_box_margin=0,
    )
)

# --- the circuit (Wang et al. 2023, GPT-2 small) ---------------------------------------
# Column order follows the flow of the circuit: early heads that detect the duplicated name,
# the induction/S-inhibition path, then the heads that write the answer. Class labels are
# the strip text; "\n" keeps each strip to two short lines over its narrow block of columns.
CLASSES = [
    ("Duplicate\ntoken",          ["a0.h1", "a0.h10", "a3.h0"]),
    ("Previous\ntoken",           ["a2.h2", "a4.h11"]),
    ("Induction",                 ["a5.h5", "a5.h8", "a5.h9", "a6.h9"]),
    ("S-inhibition",              ["a7.h3", "a7.h9", "a8.h6", "a8.h10"]),
    ("Name\nmover",               ["a9.h6", "a9.h9", "a10.h0"]),
    ("Backup\nname mover",        ["a9.h0", "a9.h7", "a10.h1", "a10.h2", "a10.h6", "a10.h10",
                                   "a11.h2", "a11.h9"]),
    ("Negative NM /\ncopy suppr.", ["a10.h7", "a11.h10"]),
]
CLASS_OF = {h: c for c, hs in CLASSES for h in hs}
HEADS = [h for _, hs in CLASSES for h in hs]
NEG = CLASSES[-1][1]                       # the two heads every method should rank LAST
# Summary facet: one column, the median rank over the 24 positive circuit heads. The negative
# name movers are excluded because the sufficiency ranking puts them at the bottom by design
# (ranks 151-157 for every method), which would only pull the median away from the reading.
SUMMARY = "Median\n(24 heads)"

# --- methods: (row label, path, layout), top-to-bottom by test-table CPR Avg -----------------
# flat   = results/<dir>/ioi_gpt2_importances.json   (eval_mib.py layout; MAttr, IntInv, no-learning)
# graph  = results/<dir>/graph_ioi_gpt2.json          (Node Pruning / DBM mask logits)
# nested = <fork results>/<dir>/ioi_gpt2/importances.json (run_attribution.py baselines)
METHODS = [
    ("MAttr",              "test_node_topk_uniform_lr05",                    "flat"),
    ("IntInv",             "test_node_actpatch_noise",                       "flat"),
    ("Node Pruning",       "eprun_node_s0.5_ld",                             "graph"),
    ("DBM",                "eprun_node_ld_sig_lr0.3_l16.0",                  "graph"),
    ("AttnLRP",            "attnlrp/AttnLRP_patching_node",                  "nested"),
    ("Expected Gradients", "napig_mc/EAP-IG-inputs-mc_patching_node",        "nested"),
    ("IG (m=10)",          "napig10/EAP-IG-inputs_patching_node",            "nested"),
    ("I×G",                "ig1/EAP-IG-inputs_patching_node",                "nested"),
    ("− learning",         "test_node_topkid_uniform_frozen",                "flat"),
    ("Random",             "random_s42/Random_patching_node",                "nested"),
]
N_NODES = 157


def load(path):
    d = json.load(open(path)); nodes = d.get("nodes", d)
    s = {n: v["score"] for n, v in nodes.items() if n != "logits" and "score" in v}
    assert len(s) == N_NODES, f"{path}: {len(s)} scored nodes, expected {N_NODES}"
    return s


def scores(spec):
    _, loc, layout = spec
    if layout == "flat":
        return load(R / loc / "ioi_gpt2_importances.json")
    if layout == "graph":
        return load(R / loc / "graph_ioi_gpt2.json")
    return load(R_MIB / loc / "ioi_gpt2" / "importances.json")


def ranks(s):
    """1 + number of nodes with a strictly larger score (ties share the better rank)."""
    v = np.array(list(s.values()))
    return {n: int((v > s[n]).sum()) + 1 for n in s}


rows = []
for spec in METHODS:
    rk = ranks(scores(spec))
    for h in HEADS:
        rows.append(dict(method=spec[0], head=h, cls=CLASS_OF[h], rank=rk[h]))
    rows.append(dict(method=spec[0], head="all", cls=SUMMARY,
                     rank=float(np.median([rk[h] for h in HEADS if h not in NEG]))))
df = pd.DataFrame(rows)
df["method"] = pd.Categorical(df["method"], [m[0] for m in METHODS][::-1])   # first method on top
df["head"] = pd.Categorical(df["head"], HEADS + ["all"])
df["cls"] = pd.Categorical(df["cls"], [c for c, _ in CLASSES] + [SUMMARY])
df["log_rank"] = np.log10(df["rank"])
df["txt"] = df["rank"].map(lambda r: f"{r:g}")
df["txt_col"] = np.where(df["rank"] <= 12, "#ffffff", "#000000")   # dark cells get white text

# Per-class median rank, printed so the figure's reading can be quoted.
med = df[df["cls"] != SUMMARY].groupby(["method", "cls"], observed=True)["rank"].median().unstack()
print(med.reindex([m[0] for m in METHODS]).to_string())

p = (
    ggplot(df, aes("head", "method", fill="log_rank"))
    + geom_tile(color="#ffffff", size=0.3)
    + geom_text(aes(label="txt", color="txt_col"), size=4.6)
    + facet_grid(cols="cls", scales="free_x", space="free_x")
    + scale_fill_cmap("Blues_r", name="Rank", limits=(0, np.log10(N_NODES)),
                      breaks=[0, 1, 2], labels=["1", "10", "100"])
    + scale_x_discrete(expand=(0, 0))
    + scale_y_discrete(expand=(0, 0))
    + labs(x="", y="")
    + theme(legend_key_height=14, legend_key_width=5)
)
# geom_text's colour is data-driven (white on dark cells); take it verbatim, no legend.
p = p + scale_color_identity()

out = OUT / "ioi_head_types.pdf"
p.save(out, verbose=False)
p.save(out.with_suffix(".png"), dpi=200, verbose=False)
print("wrote", out)
