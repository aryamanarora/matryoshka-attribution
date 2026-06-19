"""Accuracy vs #MLP neurons kept (sufficient) / corrupted (necessary), faceted by
intervention type x dataset. Reads results/arith_accuracy.csv (see
scripts/eval_arith_accuracy.py). Saves a textwidth-suitable PDF.
"""
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, geom_point, facet_grid, labs,
    scale_x_log10, scale_y_continuous, scale_color_brewer,
    theme_set, theme_bw, theme, element_text, element_line, element_blank,
    guides,
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

df = pd.read_csv("results/arith_accuracy.csv")
df["Type"] = pd.Categorical(
    df["mode"].map({"sufficient": "Sufficient (kept)", "necessary": "Necessary (corrupted)"}),
    categories=["Sufficient (kept)", "Necessary (corrupted)"], ordered=True,
)
df["Dataset"] = pd.Categorical(
    df["task"].str.capitalize(),
    categories=["Addition", "Months", "Weekdays", "Hours"], ordered=True,
)

p = (
    ggplot(df, aes("n_neurons", "accuracy", color="Type"))
    + geom_line(size=0.5)
    + geom_point(size=0.7)
    + facet_grid("Type ~ Dataset")
    + scale_x_log10(
        breaks=[1e2, 1e3, 1e4, 1e5, 1e6],
        labels=["10²", "10³", "10⁴", "10⁵", "10⁶"],
    )
    + scale_y_continuous(breaks=[0, 0.5, 1.0], limits=[0, 1])
    + scale_color_brewer(type="qual", palette="Set1")
    + guides(color=None)
    + labs(x="Number of MLP neurons", y="Accuracy")
    + theme(figure_size=(5.5, 2.6))
)
p.save("plots/mlp_suf_nec_accuracy.pdf", verbose=False)
print("wrote plots/mlp_suf_nec_accuracy.pdf")
