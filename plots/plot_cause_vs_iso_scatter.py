"""Per-cell iso acc-AUC (x) vs cause acc-AUC (y) for MAttr trained iso-only / cause-only / joint,
with IG as the reference. One point per (task, substrate) cell; SGD lr1, acc loss, seed 42.
Shows the trade-off: cause-only training moves points UP and LEFT (better cause, worse iso);
joint moves them UP without moving left. Run: uv run python plots/plot_cause_vs_iso_scatter.py
"""
import glob, json, os, sys
import pandas as pd
from plotnine import (ggplot, aes, geom_point, geom_segment, facet_wrap, labs, theme, theme_set,
                      theme_bw, element_text, element_line, element_blank, scale_color_manual,
                      scale_shape_manual, guides, guide_legend)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P
theme_set(theme_bw(base_size=8) + theme(
    text=element_text(color="#000", family="Inter"), figure_size=(5.4, 2.4),
    axis_title=element_text(size=7), axis_text=element_text(size=6),
    panel_grid_major=element_line(size=0.25, color="#dddddd"), panel_grid_minor=element_blank(),
    strip_background=element_blank(), strip_text=element_text(size=7),
    legend_title=element_text(size=7), legend_text=element_text(size=6), legend_key_size=6,
    legend_position="top", legend_direction="horizontal", legend_box="horizontal", legend_box_margin=0))
SUBS = {"mlp": "MLP neurons", "mlp+attn_head": "MLP neurons + attn heads"}
TAGS = {"sufficient_topk_sgd_acc_bs1": ("MAttr (SGD)", "iso-only"),
        "necessary_topk_sgd_acc_bs1": ("MAttr (SGD)", "cause-only"),
        "joint_topk_sgd_acc_bs1": ("MAttr (SGD)", "joint"), "ig_acc": ("IG", "n/a")}
rows = []
for f in glob.glob("results/sva_sweep/*.json") + glob.glob("results/sva_sweep_cause/*.json"):
    d = json.load(open(f))
    if d["nodes"] not in SUBS or d["model"] != "llama3": continue
    tag = os.path.basename(f).split("_" + d["nodes"].replace("+", "-") + "_", 1)[1][:-5]
    if tag not in TAGS: continue
    m, tr = TAGS[tag]
    rows.append(dict(sub=SUBS[d["nodes"]], task=d["task"], method=m, trained=tr, x=d["acc_auc"], y=d["cause_accsrc_auc"]))
df = pd.DataFrame(rows)
base = df[df.trained == "iso-only"][["sub", "task", "x", "y"]].rename(columns={"x": "x0", "y": "y0"})
seg = df[df.trained.isin(["cause-only", "joint"])].merge(base, on=["sub", "task"])
df["trained"] = pd.Categorical(df["trained"], ["iso-only", "cause-only", "joint", "n/a"])
p = (ggplot(df, aes("x", "y"))
     + geom_segment(seg, aes(x="x0", y="y0", xend="x", yend="y", linetype="trained"), color="#888888", size=0.25)
     + geom_point(aes(color="method", shape="trained"), size=1.8, alpha=0.9, stroke=0.3)
     + facet_wrap("~sub", scales="free")
     + scale_color_manual(values={"MAttr (SGD)": P.color("MAttr (SGD)"), "IG": P.color("IG")}, name="Method")
     + scale_shape_manual(values={"iso-only": "o", "cause-only": "^", "joint": "s", "n/a": "*"}, name="MAttr trained")
     + guides(linetype=False)
     + labs(x="Iso acc-AUC (keep top-k clean)", y="Cause acc-AUC (patch top-k to source)"))
p.save("plots/cause_vs_iso_scatter.pdf", dpi=300, verbose=False); p.save("plots/cause_vs_iso_scatter.png", dpi=170, verbose=False)
print(len(df), "points")
