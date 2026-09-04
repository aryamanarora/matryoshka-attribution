"""Containment curves: transfer = the target circuit sitting inside the source's top-k.

Mechanism panel for the transfer figures. One target task (Simple), accuracy vs kept-nodes k
under rankings from four sources: its own, a same-family one (Within RC), a cross-family one
that transfers (Weekdays), and one that does not (MCQA). The own-circuit curve snaps to 1.0
at k~38; good foreign sources reach it one-to-two grid steps later (their top-k CONTAINS
Simple's circuit at a small k premium); MCQA's ranking never recovers it -- MAttr pushed
Simple's nodes into MCQA's tail. The heatmap's cell values are the log-k average of exactly
these curves, so this panel is one row of the matrix opened up.

Data: results/transfer_sva_adam/simple_llama3_node_xfer_{source}.json (the MAttr+Adam
round, matching the node-scatter panel's scores; the SGD round's curves are
indistinguishable), 200 eval examples, iso (sufficiency) accuracy curves.

Out: paper/figs/containment_curves.pdf (1.45x1.5in, the third-width row's box)
"""
import json
from pathlib import Path

import pandas as pd
from plotnine import (ggplot, aes, geom_line, geom_point, labs, scale_x_log10,
                      scale_color_manual, theme_bw, theme_set, theme,
                      element_text, element_line, element_blank)

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
R = Path("results/transfer_sva_adam")

TARGET = "simple"
# (source, display, colour): lightness-separated; own-circuit black, the two transferring
# sources in the project's cool/warm pair, the failing one light grey-blue.
SOURCES = [("simple",    "Simple (own)", "#000000"),
           ("within_rc", "Within RC",    "#0072b2"),
           ("weekdays",  "Weekdays",     "#e69f00"),
           ("mcqa",      "MCQA",         "#999999")]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(1.45, 1.5),
        axis_title=element_text(size=6),
        axis_text=element_text(size=5),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        legend_title=element_blank(),
        legend_text=element_text(size=5),
        legend_key_size=6,
        legend_position=(0.32, 0.72),
        legend_background=element_blank(),
    )
)


def main():
    rows = []
    for src, lab, _c in SOURCES:
        j = json.load(open(R / f"{TARGET}_llama3_node_xfer_{src}.json"))
        for k, acc in zip(j["n_nodes"], j["iso_metrics"]["acc_base"]):
            rows.append(dict(k=k, acc=acc, src=lab))
    df = pd.DataFrame(rows)
    df["src"] = pd.Categorical(df["src"], categories=[l for _s, l, _c in SOURCES],
                               ordered=True)
    p = (ggplot(df, aes("k", "acc", color="src"))
         + geom_line(size=0.5) + geom_point(size=0.7, stroke=0)
         + scale_color_manual(values={l: c for _s, l, c in SOURCES})
         + scale_x_log10()
         + labs(x="Nodes kept k", y="Accuracy (Simple)"))
    p.save(OUT / "containment_curves.pdf", dpi=300, verbose=False)
    print("wrote", OUT / "containment_curves.pdf")


if __name__ == "__main__":
    main()
