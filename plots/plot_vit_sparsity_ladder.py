"""What the ViT sees, and what it predicts, as an attribution method's circuit grows.

Top strip: the actual model input at each sparsity -- the top-k units of the method's
ranking keep their clean content, every other unit is corrupted. Bottom panel: the class
posterior over the same axis, with the sparsity at which the top-1 prediction flips marked.

Pass one `--ladder` for a single method (both classes of the contrast are drawn in colour,
and the other classes that reach the top of the posterior at some budget -- the rug the dog
lies on, at the smallest k -- as thin grey lines behind them), or several to compare methods
on one axis (one image strip each, and the probability of the
*explained* class only -- the contrast class sits at ~0 for every method until the last
rung, so plotting it N times only adds ink).

Data from `scripts/vit_teaser_ladder.py`. Regenerate with:

    .venv-vit/bin/python scripts/vit_teaser_ladder.py --method mattr   --granularity patch
    .venv-vit/bin/python scripts/vit_teaser_ladder.py --method attnlrp --granularity patch
    uv run python plots/plot_vit_sparsity_ladder.py \
        --ladder results/vit_teaser_cat/ladder_{mattr,attnlrp}.npz

Raw matplotlib: an image strip that has to align column-for-column with the x positions of
a line panel underneath it is a layout constraint, not a grammar-of-graphics one, so the
geometry is set in inches. The line panel keeps the theme_bw look by hand.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import CLASS, color  # noqa: E402

# npz `method` field -> display name (also the key into the shared method palette)
METHOD_NAME = {"mattr": "MAttr", "mattr_pixel": "MAttr", "attnlrp": "AttnLRP",
               "smoothgrad": "SmoothGrad", "gradattnroll": "Grad×AttnRoll",
               "kernelshap": "KernelSHAP"}
UNIT = {"patch": "patches", "pixel": "pixels"}

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", nargs="+",
                    default=["results/vit_teaser_pixelate/ladder_mattr.npz"],
                    help="one or more ladder npz files; several overlays them")
    ap.add_argument("--out", default="paper/figs/vit_sparsity_ladder.pdf")
    ap.add_argument("--width", type=float, default=5.5)
    ap.add_argument("--display", default="input", choices=["input", "selected"],
                    help="'input' shows what the model actually sees -- kept units clean, the "
                         "rest corrupted. 'selected' shows ONLY the kept content on a blank "
                         "ground; the model still sees the corruption, so this is a picture "
                         "of the mask, not of the input. Much more legible at pixel "
                         "granularity, where a 1%% selection vanishes into the mosaic.")
    ap.add_argument("--extra-top", type=int, default=2,
                    help="single-ladder mode: also draw every class that is in the top-N of "
                         "the posterior at some budget (0 = only the contrast pair)")
    ap.add_argument("--extra-min", type=float, default=0.03,
                    help="...and whose posterior reaches at least this somewhere; below it a "
                         "line is indistinguishable from the axis and its label is clutter")
    ap.add_argument("--blank", default="#eceff3",
                    help="ground colour for --display selected; deliberately a cool grey, so "
                         "it does not read as the cat's cream-white fur")
    args = ap.parse_args()

    runs = [np.load(p, allow_pickle=True) for p in args.ladder]
    d0 = runs[0]
    ks = d0["ks"]
    total = int(d0["total"]) if "total" in d0.files else int(ks[-1])
    unit = UNIT[str(d0["granularity"]) if "granularity" in d0.files else "patch"]
    for d in runs[1:]:
        if not np.array_equal(d["ks"], ks):
            raise SystemExit("ladders disagree on the sparsity grid; regenerate them together")
    names = [METHOD_NAME.get(str(d["method"]), str(d["method"])) for d in runs]
    multi = len(runs) > 1
    n, m = len(ks), len(runs)

    # left margin holds the y axis (and the per-strip method labels when comparing), right
    # margin the direct series labels; the image strips span exactly the line panel's x range
    # in compare mode the coloured strip labels on the left ARE the legend, so no right-hand
    # series labels are needed -- and they would collide anyway, since every method's curve
    # ends at the same point (the unmasked image)
    left, right, gap = (0.72 if multi else 0.46), (0.10 if multi else 0.62), 0.03
    panel = (args.width - left - right - (n - 1) * gap) / n
    plot_h, xlab_h, head_h, strip_gap = 0.95, 0.42, 0.28, 0.05
    height = (0.02 + xlab_h + plot_h + strip_gap
              + m * panel + (m - 1) * gap + head_h + 0.02)
    fig = plt.figure(figsize=(args.width, height))

    col_x = lambda i: left + i * (panel + gap)                         # noqa: E731
    add = lambda x, y, w, h: fig.add_axes(                             # noqa: E731
        [x / args.width, y / height, w / args.width, h / height])

    blank = np.array(to_rgb(args.blank))

    def strip(d):
        """The images to draw for one ladder: the model input, or the kept content alone."""
        if args.display == "input":
            return d["images"]
        mk = d["masks"][..., None]
        return d["clean"][None] * mk + blank * (1 - mk)

    y0 = 0.02 + xlab_h + plot_h + strip_gap
    for r, (d, name) in enumerate(zip(runs, names)):
        y_img = y0 + (m - 1 - r) * (panel + gap)          # first ladder on top
        for i, im in enumerate(strip(d)):
            ax = add(col_x(i), y_img, panel, panel)
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4), sp.set_color("#000000")
        if multi:
            fig.text((left - 0.05) / args.width, (y_img + panel / 2) / height, name,
                     ha="right", va="center", size=7, color=color(name))

    who = "each method’s" if multi else f"{names[0]}’s {unit[:-1]}-level"
    # two lines: the wording alone overruns 5.5in at 7pt once the method count is in it
    if args.display == "input":
        head_txt = (f"model input, top-k {unit} of {who} ranking left clean",
                    f"k out of {total:,}")
    else:
        head_txt = (f"top-k {unit} of {who} ranking, shown alone",
                    f"k out of {total:,} — the model still sees the rest corrupted, not blank")
    y_head = (y0 + m * panel + (m - 1) * gap + 0.03) / height
    fig.text(left / args.width, y_head + 0.055, head_txt[0], ha="left", va="bottom", size=7)
    fig.text(left / args.width, y_head, head_txt[1], ha="left", va="bottom", size=6,
             color="#444444")

    ax = add(col_x(0), 0.02 + xlab_h, col_x(n - 1) + panel - col_x(0), plot_h)
    x = np.arange(n)
    if multi:
        series = [(d["p_pos"], color(nm), nm) for d, nm in zip(runs, names)]
    else:
        series = [(d0["p_pos"], CLASS["pos"], str(d0["pos_label"])),
                  (d0["p_neg"], CLASS["neg"], str(d0["neg_label"]))]
    # the competing classes: whatever else reaches the top-N at some budget, drawn thin and
    # grey so the contrast pair stays the figure and these are its context. Labelled at
    # their own peak, since most of them matter only at one end of the ladder.
    if not multi and args.extra_top > 0 and "probs" in d0.files:
        P, cats = d0["probs"], d0["categories"]
        cat_names = [str(c) for c in cats]
        pos_i, neg_i = cat_names.index(str(d0["pos_label"])), cat_names.index(str(d0["neg_label"]))
        extra = sorted({int(j) for row in P for j in np.argsort(-row)[:args.extra_top]}
                       - {pos_i, neg_i}, key=lambda j: -P[:, j].max())
        for j in extra:
            y = P[:, j]
            if y.max() < args.extra_min:
                continue
            ax.plot(x, y, color="#999999", lw=0.7, marker="o", ms=1.8, mew=0, clip_on=False,
                    zorder=2)
            # label at the highest INTERIOR rung: the right margin belongs to the contrast
            # pair's labels, and a class that peaks at the full image (dingo) has a second,
            # clearer bump earlier in the ladder
            i = int(np.argmax(y[:-1]))
            ax.text(i + 0.12, y[i] + 0.025, str(cats[j]), color="#666666", size=5.5,
                    ha="left", va="bottom", clip_on=False)
    for y, c, name in series:
        ax.plot(x, y, color=c, lw=1.3, marker="o", ms=2.8, mew=0, clip_on=False, zorder=3)
        if not multi:
            ax.text(n - 1 + 0.15, y[-1], f" {name}", color=c, size=6.5,
                    va="center", ha="left", clip_on=False)

    # where the first method's top-1 prediction flips to the explained class
    flip = next((i for i, s in enumerate(d0["labels"]) if str(s) == str(d0["pos_label"])), None)
    if flip is not None:
        ax.axvline(flip, color="#000000", lw=0.4, ls="dashed", zorder=1)
        ax.text(flip + 0.15, 0.95, f"top-1 becomes “{d0['pos_label']}”", size=6,
                ha="left", va="top")

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k:,}\n{k / total:.1%}" for k in ks], size=6)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], size=6)
    ax.set_xlabel(f"{unit.capitalize()} kept clean", size=7, labelpad=1)
    ax.set_ylabel(f"p({d0['pos_label']})" if multi else "Class probability",
                  size=7, labelpad=2)
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out} ({args.width:.2f} x {height:.2f} in)")
    head = "".join(f"{nm:>14}" for nm in names)
    print(f"  {'k':>7} {'frac':>7}   p({d0['pos_label']}) per method:{head}")
    for i, k in enumerate(ks):
        print(f"  {k:>7,} {k / total:>7.1%}   " + " " * len(f"p({d0['pos_label']}) per method:")
              + "".join(f"{float(d['p_pos'][i]):>14.3f}" for d in runs))


if __name__ == "__main__":
    main()
