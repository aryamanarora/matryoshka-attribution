"""Does MAttr's edge come from the corruption being a *distribution*, or from its keeping
local image structure? -- the 2x2 of baseline choices.

Each cell trains/scores every method against one corruption and evaluates the sufficiency
sweep under that same corruption:

                        structured (masked unit keeps      flat (masked unit becomes
                        its own local colour)              the same hole everywhere)
    resampled           pixelate                           solid  (uniform random colour)
    fixed               pixelate --fixed-corrupt           white

AUCs are NOT comparable across cells -- each is a different intervention with its own
margin scale -- so read each panel by the *gap* between MAttr and the baselines, not by
the absolute height. That is why the baselines are drawn too.

In the pixel row "MAttr" is `mattr_pixel` and the patch-only baselines drop out; the two
rows therefore share a colour but not a unit of analysis.

    uv run python plots/plot_vit_baseline_grid.py
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (aes, element_blank, element_line, element_text, facet_grid, geom_line,
                      geom_point, ggplot, labs, scale_color_manual, theme, theme_bw,
                      theme_set)

from palette import METHOD

plt.rcParams["pdf.fonttype"] = 42

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.2, 2.9),
        axis_title=element_text(size=7),
        axis_text=element_text(size=6),
        axis_text_x=element_text(size=6),
        panel_grid_major=element_line(size=0.25, color="#dddddd"),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.03,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
        legend_title=element_blank(),
        legend_text=element_text(size=6),
        legend_key_size=6,
        legend_position="top",
        legend_direction="horizontal",
        legend_box_margin=0,
    )
)

# (corruption label, results dir per target). `vit_teaser_pixelate` / `vit_teaser_cat` are
# the original resampled-pixelate pair, named before the other cells existed.
CELLS = [
    ("pixelate\n(resampled)", {"dog": "results/vit_teaser_pixelate",
                               "cat": "results/vit_teaser_cat"}),
    ("pixelate\n(fixed)", {"dog": "results/vit_teaser_pixfixed_dog",
                           "cat": "results/vit_teaser_pixfixed_cat"}),
    ("solid colour\n(resampled)", {"dog": "results/vit_teaser_solid_dog",
                                   "cat": "results/vit_teaser_solid_cat"}),
    ("white\n(fixed)", {"dog": "results/vit_teaser_white_dog",
                        "cat": "results/vit_teaser_white_cat"}),
]
SERIES = {"mattr": "MAttr", "mattr_pixel": "MAttr", "attnlrp": "AttnLRP",
          "kernelshap": "KernelSHAP"}
GRAN = {"patch": ("faithfulness.json", ["mattr", "kernelshap", "attnlrp"]),
        "pixel": ("faithfulness_pixel.json", ["mattr_pixel", "attnlrp"])}


def auc(curve, fracs):
    logf = np.log(fracs)
    return float(np.trapezoid(curve, logf) / (logf[-1] - logf[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="paper/figs/vit_baseline_grid.pdf")
    args = ap.parse_args()

    rows, missing = [], []
    for cell, dirs in CELLS:
        for target, d in dirs.items():
            for gran, (fname, methods) in GRAN.items():
                f = Path(d) / fname
                if not f.exists():
                    missing.append(str(f))
                    continue
                blob = json.loads(f.read_text())
                label = json.loads((Path(d) / "meta.json").read_text())["pos"]["label"]
                for m in methods:
                    if m not in blob["curves"]:
                        continue
                    rows.append({"cell": cell, "target": f"explaining “{label}”",
                                 "granularity": f"{gran} units",
                                 "method": SERIES[m],
                                 "auc": auc(blob["curves"][m], blob["fracs"])})
    if missing:
        print("missing (skipped):\n  " + "\n  ".join(missing))
    df = pd.DataFrame(rows)
    df["cell"] = pd.Categorical(df["cell"], [c for c, _ in CELLS], ordered=True)

    p = (ggplot(df, aes("cell", "auc", color="method", group="method"))
         + geom_line(size=0.4, alpha=0.55)
         + geom_point(size=1.5)
         + facet_grid("granularity ~ target", scales="free_y")
         + scale_color_manual(values={k: METHOD[k] for k in ("MAttr", "AttnLRP")}
                              | {"KernelSHAP": METHOD["Node Pruning"]})
         + labs(x="Corruption the circuit is trained and scored against",
                y="Sufficiency AUC"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.save(out, verbose=False)
    p.save(out.with_suffix(".png"), dpi=300, verbose=False)
    print(f"wrote {out}")
    piv = df.pivot_table(index=["granularity", "cell"], columns=["target", "method"],
                         values="auc", observed=True)
    print(piv.round(2).to_string())


if __name__ == "__main__":
    main()
