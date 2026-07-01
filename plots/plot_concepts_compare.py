"""Necessity vs sufficiency per-concept accuracy vs % nodes, facet_grid(mode ~ task).
Usage: plot_concepts_compare.py [nec.pkl suf.pkl out.pdf]"""
import pickle, sys
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, facet_grid, labs, scale_x_log10, scale_y_continuous,
    scale_color_brewer, theme_set, theme_bw, theme, element_text, element_line, element_blank,
)

theme_set(
    theme_bw(base_size=8)
    + theme(text=element_text(color="#000", family="Inter"), figure_size=(2.4, 1.7),
            axis_title=element_text(size=7), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(), panel_spacing_x=0.02, panel_spacing_y=0.02,
            strip_background=element_blank(), strip_text=element_text(size=7),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=6, legend_position="top", legend_direction="horizontal",
            legend_box_margin=0))

_a = sys.argv[1:] if len(sys.argv) > 3 else ["results/arith_sae_concepts.pkl",
                                             "results/arith_sae_concepts_sufficient.pkl",
                                             "plots/arith_concepts_compare.pdf"]
SRC = {"Necessity": _a[0], "Sufficiency": _a[1]}
OUT = _a[2]
rows = []
for mode, fn in SRC.items():
    r = pickle.load(open(fn, "rb"))
    for t in r["tasks"]:
        for c, e in r["per_eval"][t].items():
            for s, a in zip(e["sparsities"], e["acc"]):
                rows.append({"Mode": mode, "task": t.capitalize(), "Concept": c,
                             "pct": s * 100, "acc": a})
df = pd.DataFrame(rows)
df["task"] = pd.Categorical(df["task"], ["Addition", "Months", "Weekdays", "Hours"])
df["Mode"] = pd.Categorical(df["Mode"], ["Necessity", "Sufficiency"])
df["Concept"] = pd.Categorical(df["Concept"], ["input", "offset", "output"], ordered=True)

p = (ggplot(df, aes("pct", "acc", color="Concept"))
     + geom_line(size=0.5)
     + facet_grid("Mode ~ task")
     + scale_x_log10(breaks=[0.01, 0.1, 1, 10, 100], labels=["0.01", "0.1", "1", "10", "100"])
     + scale_y_continuous(breaks=[0, 0.5, 1.0], limits=[-0.03, 1.03])
     + scale_color_brewer(type="qual", palette="Set1")
     + labs(x="Units in mask (%)", y="Accuracy")
     + theme(figure_size=(5.5, 2.7)))
p.save(OUT, verbose=False)
print("wrote", OUT)
