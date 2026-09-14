"""Figures for the cross-format arithmetic replication (arXiv:2609.04463 with MAttr).

Reads results/arith_formats/summary.json (written by scripts/arith_formats/analyse.py) and
writes to paper/figs/:
  arith_formats_acc.pdf       accuracy per format, one line per model            (paper Fig. 2)
  arith_formats_overlap.pdf   top-1% overlap with the numeric circuit vs accuracy (Sec. 3.2)
  arith_formats_pbr.pdf       item-level point-biserial r, MAttr vs AP circuits    (Fig. 3A)
  arith_formats_loading.pdf   accuracy vs decile of numeric-circuit loading       (Fig. 3B)
  arith_formats_patch.pdf     causal validation: circuit vs random-matched units  (App. C)

plotnine; colours from plots/palette.py (MAttr(SGD) = black, attribution patching takes the
I×G wine because it is the same gradient x activation-difference estimator at the neuron level).
"""

import json
import sys
from pathlib import Path

import pandas as pd
from plotnine import (aes, element_blank, element_line, element_text, facet_wrap, geom_abline,
                      geom_hline, geom_line, geom_point, geom_smooth, ggplot, labs,
                      position_dodge, scale_color_manual, scale_shape_manual, scale_x_discrete,
                      scale_y_continuous, theme, theme_bw, theme_set)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import METHOD  # noqa: E402

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.4, 2.0),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6, rotation=45, hjust=1.0, vjust=1.0),
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

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "results/arith_formats")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "paper/figs")
OUT.mkdir(parents=True, exist_ok=True)
S = json.load(open(ROOT / "summary.json"))
FORMATS = ["numeric", "english", "spanish", "italian"]
FMT_LABEL = {"numeric": "Numeric", "english": "English", "spanish": "Spanish", "italian": "Italian"}
VERBAL = FORMATS[1:]
METH_LABEL = {"mattr": "MAttr (SGD)", "ap": "Attribution patching", "item": "MAttr (per item)"}
COL = {"MAttr (SGD)": METHOD["MAttr (SGD)"], "Attribution patching": METHOD["I×G"],
       "MAttr (per item)": METHOD["MAttr"]}


def save(p, name, size):
    p = p + theme(figure_size=size)
    p.save(OUT / name, verbose=False)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- accuracy (Fig. 2)
rows = [{"model": m, "format": FMT_LABEL[f], "acc": 100 * a}
        for m, d in S["acc"].items() for f, a in d.items()]
df = pd.DataFrame(rows)
df["format"] = pd.Categorical(df["format"], [FMT_LABEL[f] for f in FORMATS])
p = (ggplot(df, aes("format", "acc", group="model"))
     + geom_line(color="#999999", size=0.3)
     + geom_point(size=1.2, color="#000000")
     + labs(x="Format", y="Accuracy (%)")
     + theme(axis_text_x=element_text(rotation=0)))
save(p, "arith_formats_acc.pdf", (2.4, 1.7))

# ---------------------------------------------------------------- overlap vs accuracy (Sec. 3.2)
rows = []
for m, d in S["jaccard"].items():
    for meth in ("mattr", "ap"):
        for f in VERBAL:
            rows.append({"model": m, "method": METH_LABEL[meth], "format": FMT_LABEL[f],
                         "jaccard": d[f"{meth}:{f}"], "acc": 100 * S["acc"][m][f]})
df = pd.DataFrame(rows).dropna()
df["format"] = pd.Categorical(df["format"], [FMT_LABEL[f] for f in VERBAL])
p = (ggplot(df, aes("jaccard", "acc", color="method"))
     + geom_smooth(method="lm", se=False, size=0.5)
     + geom_point(size=1.2)
     + facet_wrap("~format", scales="free_x")
     + scale_color_manual(values=COL, name="")
     + labs(color="", shape="")
     + labs(x="Top-1% Jaccard overlap with the numeric circuit", y="Accuracy (%)")
     + theme(axis_text_x=element_text(rotation=0)))
save(p, "arith_formats_overlap.pdf", (5.4, 2.0))

# ---------------------------------------------------------------- point-biserial (Fig. 3A)
rows = []
for m, d in S["pbr"].items():
    for key, v in d.items():
        meth, f = key.split(":")
        rows.append({"model": m, "method": METH_LABEL[meth], "format": FMT_LABEL[f],
                     "r": v["r"], "sig": "p < 0.05" if v["p"] < 0.05 else "n.s."})
df = pd.DataFrame(rows).dropna()
df["format"] = pd.Categorical(df["format"], [FMT_LABEL[f] for f in VERBAL])
p = (ggplot(df, aes("model", "r", color="method", shape="sig"))
     + geom_hline(yintercept=0, size=0.25, color="#000000")
     + geom_point(size=1.3, position=position_dodge(width=0.6))
     + facet_wrap("~format")
     + scale_color_manual(values=COL, name="")
     + labs(color="", shape="")
     + scale_shape_manual(values={"p < 0.05": "o", "n.s.": "x"}, name="")
     + labs(x="Model", y="Point-biserial r (loading, correct)"))
save(p, "arith_formats_pbr.pdf", (5.4, 2.3))

# ---------------------------------------------------------------- loading deciles (Fig. 3B)
rows = []
for key, lst in S["loading_bins"].items():
    meth, f = key.split(":")
    for r in lst:
        rows.append({**r, "method": METH_LABEL[meth], "format": FMT_LABEL[f]})
df = pd.DataFrame(rows)
if len(df):
    df["format"] = pd.Categorical(df["format"], [FMT_LABEL[f] for f in VERBAL])
    df["acc"] = 100 * df["acc"]
    p = (ggplot(df, aes("bin", "acc", color="method", group="model + method"))
         + geom_line(size=0.3, alpha=0.6)
         + facet_wrap("~format")
         + scale_color_manual(values=COL, name="")
     + labs(color="", shape="")
         + labs(x="Decile of numeric-circuit loading (per model)", y="Accuracy (%)")
         + theme(axis_text_x=element_text(rotation=0)))
    save(p, "arith_formats_loading.pdf", (5.4, 2.0))

# ---------------------------------------------------------------- causal validation (App. C)
rows = []
for m, d in S["patch"].items():
    for key, v in d.items():
        meth, f, k = key.split(":")
        rows.append({"model": m, "method": METH_LABEL[meth], "format": FMT_LABEL[f], "kind": k, "r": v["r"]})
df = pd.DataFrame(rows).dropna()
if len(df):
    w = df.pivot_table(index=["model", "method", "format"], columns="kind", values="r").reset_index()
    w["format"] = pd.Categorical(w["format"], [FMT_LABEL[f] for f in VERBAL])
    p = (ggplot(w, aes("rand", "circ", color="method"))
         + geom_abline(slope=1, intercept=0, size=0.25, color="#000000")
         + geom_point(size=1.2)
         + facet_wrap("~format")
         + scale_color_manual(values=COL, name="")
     + labs(color="", shape="")
         + labs(x="r with correctness: random matched units", y="r with correctness: numeric circuit")
         + theme(axis_text_x=element_text(rotation=0)))
    save(p, "arith_formats_patch.pdf", (5.4, 2.0))
