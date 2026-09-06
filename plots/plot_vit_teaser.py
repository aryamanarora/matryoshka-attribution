"""Teaser figure in the style of Figure 1 of AttnLRP (Achtibat et al., ICML 2024),
with MAttr (soft top-k forward, log-k schedule, lr 0.05 -- the headline variant) as "ours".

A row of attribution heatmaps for one image containing a dog and a cat, explained for
"dog", and a second row below it explaining the same image for "cat" (the reversed
contrast, results/vit_teaser_cat). AttnLRP's Figure 1 follows the strip with a method x property table of qualitative
suitability marks (+ / o / -); that table is kept behind `--table` and off by default, since
on a single image the marks are opinion rather than measurement.

Data comes from `scripts/vit/vit_teaser_attr.py` (see that file for the substrate and the
explained scalar). Regenerate end to end with:

    sbatch -J teaser scripts/vit/launch/vit_teaser.sbatch --baseline pixelate --out results/vit_teaser_pixelate
    sbatch -J teaser scripts/vit/launch/vit_teaser.sbatch --baseline pixelate --explain cat --out results/vit_teaser_cat
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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
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


def signed_percentile(r, gamma):
    """Rank transform: each unit's percentile within its own map, centred at the median and
    mapped to [-1, 1], then |p|^gamma with the sign kept.

    Rank-based so that every method is drawn on the same footing whatever its score scale
    (a raw-score map is dominated by the method's own distribution: MAttr's logits skew
    negative and the whole panel reads blue). The power keeps the middle of the ranking white
    and colours only the tails -- at gamma 5 the central 60% of units sit below 0.03 of full
    saturation and the top/bottom 10% above 0.6 -- which is the reading the top-k sweep uses.
    (Gamma 3 still tinted ~half the cells; 8 loses the second tier of the ranking.)
    """
    flat = np.asarray(r, dtype=np.float64).ravel()
    ranks = np.empty_like(flat)
    ranks[np.argsort(flat, kind="stable")] = np.arange(flat.size)
    p = 2.0 * ranks / (flat.size - 1) - 1.0
    return (np.sign(p) * np.abs(p) ** gamma).reshape(np.shape(r))


def draw_heatmap(ax, r, norm, clip, gamma, overlay_rgb=None):
    """Symmetric blue-white-red heatmap. `norm="percentile"` draws the signed-percentile
    transform above on a fixed [-1, 1] scale; `norm="score"` draws the raw scores scaled by
    a high quantile of |r| so that a couple of extreme patches cannot wash the map out."""
    if norm == "percentile":
        r, v = signed_percentile(r, gamma), 1.0
    else:
        v = np.quantile(np.abs(r), clip)
    if overlay_rgb is not None:
        ax.imshow(overlay_rgb.mean(-1), cmap="gray", vmin=-0.2, vmax=1.6,
                  extent=(0, 1, 1, 0))
    ax.imshow(r, cmap="bwr", vmin=-v, vmax=v, extent=(0, 1, 1, 0),
              interpolation="nearest", alpha=0.85 if overlay_rgb is not None else 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+",
                    default=["results/vit_teaser_pixelate", "results/vit_teaser_cat"],
                    help="one strip per run, top to bottom; the default pair is the same "
                         "image explained for the dog and then for the cat")
    ap.add_argument("--out", default="paper/figs/vit_teaser.pdf")
    ap.add_argument("--granularity", default="patch", choices=["native", "patch"],
                    help="'patch' (default) pools every pixel-level map into the 14x14 grid "
                         "the mask-based methods score, so all panels are compared on one "
                         "unit set; 'native' shows pixel maps where a method produces them")
    ap.add_argument("--smoothgrad-variant", default="plain", choices=["plain", "xinput"])
    ap.add_argument("--norm", default="percentile", choices=["percentile", "score"],
                    help="colour by each unit's signed percentile rank within its method "
                         "(default; see signed_percentile) or by the raw score")
    ap.add_argument("--gamma", type=float, default=5.0,
                    help="power applied to the signed percentile: larger keeps more of the "
                         "middle of the ranking white")
    ap.add_argument("--clip", type=float, default=0.995,
                    help="--norm score only: quantile of |relevance| mapped to full saturation")
    ap.add_argument("--overlay", action="store_true",
                    help="draw the photo faintly under each heatmap")
    ap.add_argument("--width", type=float, default=5.5, help="figure width (inches)")
    ap.add_argument("--table", action="store_true",
                    help="draw AttnLRP's qualitative suitability table under the strip")
    ap.add_argument("--header", default=None,
                    help="text above the strip; default names both classes of the explained "
                         "logit difference")
    args = ap.parse_args()

    runs = [(np.load(Path(r) / "attributions.npz"), json.loads((Path(r) / "meta.json").read_text()))
            for r in args.results]
    data, meta = runs[0]
    rgb = data["input/rgb"]

    # --- layout, in inches, so the image strip(s) and the table share one column grid. One
    # strip per results dir, top to bottom, each under its own header line; the colour scale
    # sits in the top header band, the method names under the bottom strip.
    n = len(COLUMNS)
    pad, gap = 0.02, 0.05
    panel = (args.width - 2 * pad - (n - 1) * gap) / n
    head, head_top, label_h, row_h = 0.17, 0.24, 0.15, 0.24
    n_rows = len(RATINGS) if args.table else 0
    m = len(runs)
    height = pad + n_rows * row_h + label_h + m * panel + (m - 1) * head + head_top + pad
    fig = plt.figure(figsize=(args.width, height))

    def col_x(i):
        return pad + i * (panel + gap)

    def add_axes(x, y, w, h):
        return fig.add_axes([x / args.width, y / height, w / args.width, h / height])

    y_bottom = pad + n_rows * row_h + label_h
    for r, (d, mt) in enumerate(runs):
        y_panel = y_bottom + (m - 1 - r) * (panel + head)      # first results dir on top
        for i, (name, key) in enumerate(COLUMNS):
            ax = add_axes(col_x(i), y_panel, panel, panel)
            if key == "input":
                ax.imshow(d["input/rgb"], extent=(0, 1, 1, 0))
            else:
                draw_heatmap(ax, load_map(d, key, args.granularity, args.smoothgrad_variant),
                             args.norm, args.clip, args.gamma, rgb if args.overlay else None)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_color("#000000")
            if r == m - 1:
                fig.text((col_x(i) + panel / 2) / args.width, (y_panel - 0.035) / height, name,
                         ha="center", va="top", size=7,
                         weight="bold" if key == "mattr" else "normal")

        header = args.header if (args.header is not None and m == 1) else None
        if header is None:
            header = f"explanation for “{mt['pos']['label']}”"
            if mt["target"] == "logit_diff":       # the explained scalar is a class contrast
                header += f" vs. “{mt['neg']['label']}”"
            elif mt["target"] == "prob":
                header += f": p(“{mt['pos']['label']}”)"
            elif mt["target"] == "ce":
                header += f": cross-entropy of “{mt['pos']['label']}”"
        fig.text(pad / args.width, (y_panel + panel + 0.035) / height, header,
                 ha="left", va="bottom", size=7)

    y_top = y_bottom + (m - 1) * (panel + head) + panel        # top of the top strip
    # colour scale, right-aligned in the top header band. Only meaningful under the percentile
    # transform, where one scale serves every panel; raw scores have a scale per method.
    if args.norm == "percentile":
        cw, ch = 1.15, 0.055
        cax = add_axes(args.width - pad - cw, y_top + 0.145, cw, ch)
        cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap="bwr"), cax=cax,
                          orientation="horizontal")
        pct = [1, 10, 50, 90, 99]
        ticks = [np.sign(2 * q / 100 - 1) * abs(2 * q / 100 - 1) ** args.gamma for q in pct]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{q}" for q in pct])
        cax.tick_params(labelsize=5, length=1.5, pad=1)
        cb.outline.set_linewidth(0.4)
        fig.text((args.width - pad - cw - 0.04) / args.width,
                 (y_top + 0.145 + ch / 2) / height, "score percentile",
                 ha="right", va="center", size=5.5)

    # --- suitability table: row labels sit under the input column, marks under the methods
    if args.table:
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
