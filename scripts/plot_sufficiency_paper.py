"""Paper-themed sufficiency plot: accuracy vs # features/dims kept (log), per CausalGym
subtask, SAE-feature MAttr vs DAS-64 (+ random-mask baselines), layer 12. Saves PDF.
"""
import glob, json, os
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_hline, facet_wrap, labs,
    scale_x_log10, scale_y_continuous, scale_color_brewer, scale_linetype_manual,
    theme_set, theme_bw, theme, element_text, element_line, element_blank, guides,
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
        strip_text=element_text(size=6),
        legend_title=element_text(size=7),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

SRC = {"SAE": "results/sae_node_sweep", "DAS-64": "results/das64_sweep"}


def wrap_label(t):  # keep facet strips narrow: break long names onto 2 lines
    return t.replace("_", " ").replace(" ", "\n", 1) if len(t) > 14 else t.replace("_", " ")


rows = []
tasks = sorted(os.path.basename(os.path.dirname(p))
               for p in glob.glob(f"{SRC['SAE']}/*/results.json"))
for t in tasks:
    for method, d in SRC.items():
        p = f"{d}/{t}/results.json"
        if not os.path.exists(p):
            continue
        c = json.load(open(p)).get("curve") or {}
        for mask, key in [("Learned", "learned_acc"), ("Random", "random_acc")]:
            if key in c:
                for k, a in zip(c["k"], c[key]):
                    rows.append({"task": wrap_label(t), "Method": method,
                                 "Mask": mask, "k": k, "acc": a})

df = pd.DataFrame(rows)
df["Method"] = pd.Categorical(df["Method"], categories=["SAE", "DAS-64"], ordered=True)
df["Mask"] = pd.Categorical(df["Mask"], categories=["Learned", "Random"], ordered=True)

p = (
    ggplot(df, aes("k", "acc", color="Method", linetype="Mask"))
    + geom_hline(yintercept=0.9, size=0.2, color="#bbbbbb")
    + geom_line(size=0.45)
    + facet_wrap("task", ncol=5)
    + scale_x_log10(breaks=[1, 10, 100, 1000],
                    labels=["10⁰", "10¹", "10²", "10³"])
    + scale_y_continuous(breaks=[0, 0.5, 1.0], limits=[-0.03, 1.03])
    + scale_color_brewer(type="qual", palette="Set1")
    + scale_linetype_manual(values={"Learned": "solid", "Random": "dotted"})
    + labs(x="Number of features kept", y="Accuracy")
    + theme(figure_size=(5.5, 6.8))
)
p.save("plots/sae_sufficiency.pdf", verbose=False)
print(f"wrote plots/sae_sufficiency.pdf ({df['task'].nunique()} subtasks)")
