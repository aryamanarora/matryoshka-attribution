"""Teaser figure in the style of Figure 1 of AttnLRP (Achtibat et al., ICML 2024),
with MAttr (soft top-k forward, log-k schedule, lr 0.05 -- the headline variant) as "ours".

A row of attribution heatmaps for one image containing a dog and a cat, explained for
"dog", followed by a method x property table of qualitative suitability marks (+ / o / -).

Data comes from `scripts/vit_teaser_attr.py` (see that file for the substrate and the
explained scalar). Regenerate end to end with:

    sbatch -J teaser vit_teaser.sbatch --baseline pixelate --out results/vit_teaser_pixelate
    uv run python plots/plot_vit_teaser.py

(Only the attribution step needs `.venv-vit`; plotting reads the npz and runs in the
project `.venv`.)

Raw matplotlib rather than plotnine: an image strip with hand-placed marker glyphs beneath
it is not a grammar-of-graphics figure, and the panel/table alignment has to be set in
inches so the columns line up across two visually different bands.

The RATINGS table below reproduces AttnLRP's own marks for the four methods it compared,
with two deliberate changes. (1) A MAttr column is added. (2) AttnLRP's faithfulness drops
from + to o: on MIB, MAttr's CPR AUC beats AttnLRP's (paper/tabs/mib_results.tex), so the
top mark in that row goes to the method that wins the benchmark this paper is about.
Everything else is left exactly as AttnLRP had it.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import RATING  # noqa: E402

# display name -> key in the npz / meta.json ("input" is the photo itself)
COLUMNS = [
    ("input", "input"),
    ("MAttr (ours)", "mattr"),
    ("AttnLRP", "attnlrp"),
    ("SmoothGrad", "smoothgrad"),
    ("Grad×AttnRoll", "gradattnroll"),
    ("KernelSHAP", "kernelshap"),
]
# property -> mark per method (order matches COLUMNS[1:])
RATINGS = {
    "faithfulness":             ["+", "o", "-", "o", "o"],
    "computational\nefficiency": ["-", "+", "o", "+", "-"],
    "latent\nattributions":     ["+", "+", "+", "-", "o"],
}
MARKER = {"+": dict(marker="P", ms=5.0, mew=0),
          "o": dict(marker="o", ms=4.4, mew=1.5, mfc="none"),
          "-": dict(marker="_", ms=5.0, mew=2.2)}

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})


def load_map(data, key, granularity, smoothgrad_variant):
    """Relevance map for one method, at its native resolution or pooled to 14x14 patches."""
    if granularity == "patch":
        return data[f"{key}/patch"]
    if key == "smoothgrad" and smoothgrad_variant == "xinput":
        return data["smoothgrad/pixel_xinput"]
    for suffix in ("pixel", "patch"):
        if f"{key}/{suffix}" in data.files:
            return data[f"{key}/{suffix}"]
    raise KeyError(key)


def draw_heatmap(ax, r, clip, overlay_rgb=None):
    """Symmetric blue-white-red heatmap, scaled by a high quantile of |r| so that a couple
    of extreme patches cannot wash the rest of the map out."""
    v = np.quantile(np.abs(r), clip)
    if overlay_rgb is not None:
        ax.imshow(overlay_rgb.mean(-1), cmap="gray", vmin=-0.2, vmax=1.6,
                  extent=(0, 1, 1, 0))
    ax.imshow(r, cmap="bwr", vmin=-v, vmax=v, extent=(0, 1, 1, 0),
              interpolation="nearest", alpha=0.85 if overlay_rgb is not None else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/vit_teaser_pixelate")
    ap.add_argument("--out", default="paper/figs/vit_teaser.pdf")
    ap.add_argument("--granularity", default="patch", choices=["native", "patch"],
                    help="'patch' (default) pools every pixel-level map into the 14x14 grid "
                         "the mask-based methods score, so all panels are compared on one "
                         "unit set; 'native' shows pixel maps where a method produces them")
    ap.add_argument("--smoothgrad-variant", default="plain", choices=["plain", "xinput"])
    ap.add_argument("--clip", type=float, default=0.995,
                    help="quantile of |relevance| mapped to full saturation")
    ap.add_argument("--overlay", action="store_true",
                    help="draw the photo faintly under each heatmap")
    ap.add_argument("--width", type=float, default=5.5, help="figure width (inches)")
    ap.add_argument("--header", default=None,
                    help="text above the strip; default names both classes of the explained "
                         "logit difference")
    args = ap.parse_args()

    res = Path(args.results)
    data = np.load(res / "attributions.npz")
    meta = json.loads((res / "meta.json").read_text())
    rgb = data["input/rgb"]

    # --- layout, in inches, so the image strip and the table share one column grid
    n = len(COLUMNS)
    pad, gap = 0.02, 0.05
    panel = (args.width - 2 * pad - (n - 1) * gap) / n
    head, label_h, row_h = 0.15, 0.15, 0.24
    n_rows = len(RATINGS)
    height = pad + n_rows * row_h + label_h + panel + head + pad
    fig = plt.figure(figsize=(args.width, height))

    def col_x(i):
        return pad + i * (panel + gap)

    def add_axes(x, y, w, h):
        return fig.add_axes([x / args.width, y / height, w / args.width, h / height])

    y_panel = pad + n_rows * row_h + label_h

    for i, (name, key) in enumerate(COLUMNS):
        ax = add_axes(col_x(i), y_panel, panel, panel)
        if key == "input":
            ax.imshow(rgb, extent=(0, 1, 1, 0))
        else:
            draw_heatmap(ax, load_map(data, key, args.granularity, args.smoothgrad_variant),
                         args.clip, rgb if args.overlay else None)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
            sp.set_color("#000000")
        fig.text((col_x(i) + panel / 2) / args.width, (y_panel - 0.035) / height, name,
                 ha="center", va="top", size=7,
                 weight="bold" if key == "mattr" else "normal")

    header = args.header
    if header is None:
        header = f"explanation for “{meta['pos']['label']}”"
        if meta["target"] == "logit_diff":       # the explained scalar is a class contrast
            header += f" vs. “{meta['neg']['label']}”"
    fig.text(pad / args.width, (y_panel + panel + 0.035) / height, header,
             ha="left", va="bottom", size=7)

    # --- suitability table: row labels sit under the input column, marks under the methods
    tab = add_axes(pad, pad, args.width - 2 * pad, n_rows * row_h)
    tab.set_axis_off()
    tab.set_xlim(0, args.width - 2 * pad)
    tab.set_ylim(0, n_rows * row_h)
    for r, (prop, marks) in enumerate(RATINGS.items()):
        y = (n_rows - r - 0.5) * row_h
        tab.text(0, y, prop, ha="left", va="center", size=6.5, linespacing=1.15)
        for i, m in enumerate(marks):
            tab.plot(col_x(i + 1) - pad + panel / 2, y, color=RATING[m],
                     clip_on=False, **MARKER[m])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out} ({args.width:.2f} x {height:.2f} in)")
    print("  measured seconds: " +
          ", ".join(f"{k}={v:.2f}" for k, v in meta["seconds"].items()))


if __name__ == "__main__":
    main()
