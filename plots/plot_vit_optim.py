"""MAttr's optimizer, learning rate and Adam-eps on the ViT teaser, against Expected Gradients.

Data from `scripts/vit/vit_teaser_optim.py`, one directory per seed (`results/vit_optim/seed*`).
Every number is the held-out hard-top-k sufficiency AUC of a patch ranking (log-spaced area of
the explained logit difference over MIB's sparsity grid, `vit_teaser_faith.py`'s protocol),
averaged over seeds; ribbons/whiskers are the seed range.

Layout (raw matplotlib: a tile grid, a line panel with reference rules, a trace panel and a strip
of 14x14 score maps do not share one grammar):

  (a) Adam, AUC over learning rate x eps -- the eps question at 196 units.
  (b) AUC over learning rate: SGD, Adam at torch's default eps, Adam at its best eps; Expected
      Gradients, the teaser's KernelSHAP / AttnLRP, and a random ranking as horizontal rules.
  (c) Probe AUC during training for the best cell of each arm.
  (d) The rankings themselves, as the teaser draws them.

    uv run python plots/plot_vit_optim.py
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import METHOD, RC, furnish  # noqa: E402
from plot_vit_teaser import signed_percentile  # noqa: E402  (one colouring for both figures)

plt.rcParams.update(RC)

SERIES = {   # arm -> (label, colour); per palette's optimizer rule
    "sgd": ("MAttr (SGD)", METHOD["MAttr (SGD)"]),
    "adam_best": ("MAttr (Adam, best ε)", METHOD["MAttr"]),
    "adam_default": ("MAttr (Adam, ε = 10⁻⁸)", METHOD["MAttr (Adam, default eps)"]),
    "ig": ("Expected Gradients", METHOD["Expected Gradients"]),
    "ref_kernelshap": ("KernelSHAP", METHOD["Node Pruning"]),
    "ref_attnlrp": ("AttnLRP", METHOD["AttnLRP"]),
    "random": ("random", "#888888"),
}
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def sci(x):
    """10⁻⁸-style label for a power of ten."""
    return "10" + f"{int(round(np.log10(x)))}".translate(SUP)


def logticks(ax, xs):
    """Decade ticks with Unicode-superscript labels (mathtext would leave the font)."""
    lo_, hi_ = int(np.floor(np.log10(min(xs)))), int(np.ceil(np.log10(max(xs))))
    ticks = [10.0 ** e for e in range(lo_, hi_ + 1)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([sci(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())


def load(runs):
    """{arm name: {"auc": [per seed], "trace": [per seed [n,2]], "scores": [per seed 14x14],
    "meta": record}} over every seed directory."""
    out = {}
    for d in runs:
        j = json.loads((Path(d) / "grid.json").read_text())
        z = np.load(Path(d) / "grid.npz")
        for name, r in j["arms"].items():
            e = out.setdefault(name, {"auc": [], "trace": [], "scores": [], "meta": r})
            e["auc"].append(r["auc"])
            if f"{name}/auc_trace" in z.files:
                e["trace"].append(z[f"{name}/auc_trace"])
                e["scores"].append(z[f"{name}/scores"])
    return out, j


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=sorted(glob.glob("results/vit_optim/seed*")))
    ap.add_argument("--out", default="paper/figs/vit_optim.pdf")
    ap.add_argument("--width", type=float, default=5.5)
    ap.add_argument("--gamma", type=float, default=5.0,
                    help="power on the signed percentile rank in panel (d); the teaser's")
    args = ap.parse_args()

    arms, meta = load(args.runs)
    n_seed = len(args.runs)
    mean = {k: float(np.mean(v["auc"])) for k, v in arms.items()}
    lo = {k: float(np.min(v["auc"])) for k, v in arms.items()}
    hi = {k: float(np.max(v["auc"])) for k, v in arms.items()}

    adam = {k: v["meta"] for k, v in arms.items() if v["meta"].get("arm") == "adam"}
    sgd = {k: v["meta"] for k, v in arms.items() if v["meta"].get("arm") == "sgd"}
    lrs = sorted({m["lr"] for m in adam.values()})
    epss = sorted({m["eps"] for m in adam.values()})
    sgd_lrs = sorted(m["lr"] for m in sgd.values())
    key_a = {(m["lr"], m["eps"]): k for k, m in adam.items()}
    key_s = {m["lr"]: k for k, m in sgd.items()}

    best_adam = max(adam, key=mean.get)
    best_eps = adam[best_adam]["eps"]
    default_eps = min(epss)
    best_default = max((k for k, m in adam.items() if m["eps"] == default_eps), key=mean.get)
    best_sgd = max(sgd, key=mean.get)

    # ------------------------------------------------------------------ layout (inches)
    W = args.width
    top_h, gap, lm, rm = 1.45, 0.62, 0.38, 0.06
    pw = (W - lm - rm - 2 * gap) / 3
    strip = [("ref_mattr", "MAttr (teaser)"), (best_adam, "Adam, best"),
             (best_default, "Adam, ε = 10⁻⁸"), (best_sgd, "SGD"), ("ig", "Expected Gradients"),
             ("ref_kernelshap", "KernelSHAP")]
    strip = [(k, lab) for k, lab in strip if k in arms and arms[k]["scores"]]
    n_s = len(strip)
    sgap, spad = 0.05, 0.02
    sp = (W - 2 * spad - (n_s - 1) * sgap) / n_s
    H = 0.02 + 0.40 + sp + 0.36 + 0.34 + top_h + 0.20
    fig = plt.figure(figsize=(W, H))
    add = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])   # noqa: E731
    y_top = H - 0.20 - top_h

    # ---- (a) Adam lr x eps tiles
    ax = add(lm, y_top, pw, top_h)
    grid = np.array([[mean.get(key_a.get((lr, e)), np.nan) for lr in lrs] for e in epss])
    cmap = LinearSegmentedColormap.from_list("mattr", ["#ffffff", METHOD["MAttr"]])
    vmin = min(mean["random"], np.nanmin(grid))
    vmax = np.nanmax(grid)
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower",
              interpolation="nearest")
    for i, e in enumerate(epss):
        for j, lr in enumerate(lrs):
            v = grid[i, j]
            if np.isfinite(v):
                dark = (v - vmin) / (vmax - vmin + 1e-9) > 0.6
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", size=5.5,
                        color="#ffffff" if dark else "#000000")
    bi, bj = epss.index(best_eps), lrs.index(adam[best_adam]["lr"])
    ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, lw=1.0,
                               ec="#000000", zorder=3))
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels([f"{lr:g}" for lr in lrs], size=6, rotation=45)
    ax.set_yticks(range(len(epss))), ax.set_yticklabels([sci(e) for e in epss], size=6)
    ax.set_xlabel("Learning rate", size=7, labelpad=1)
    ax.set_ylabel("Adam ε", size=7, labelpad=2)
    ax.tick_params(length=2, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.text(0.0, 1.03, "(a) MAttr + Adam: sufficiency AUC", transform=ax.transAxes,
            size=7, ha="left", va="bottom")

    # ---- (b) AUC vs lr, three MAttr series + rules
    ax = add(lm + pw + gap, y_top, pw, top_h)
    furnish(ax)

    def series(keys, xs, lab, col, ls="-"):
        m = np.array([mean[k] for k in keys])
        l_ = np.array([lo[k] for k in keys])
        h_ = np.array([hi[k] for k in keys])
        ax.fill_between(xs, l_, h_, color=col, alpha=0.15, lw=0)
        ax.plot(xs, m, color=col, lw=1.0, ls=ls, marker="o", ms=2.4, mew=0, label=lab,
                clip_on=False, zorder=3)

    series([key_s[lr] for lr in sgd_lrs], sgd_lrs, *SERIES["sgd"])
    series([key_a[(lr, best_eps)] for lr in lrs], lrs,
           f"MAttr (Adam, ε = {sci(best_eps)})", SERIES["adam_best"][1])
    if default_eps != best_eps:
        series([key_a[(lr, default_eps)] for lr in lrs], lrs,
               f"MAttr (Adam, ε = {sci(default_eps)})", SERIES["adam_default"][1])
    # reference rules, labelled in the clear region right of the MAttr optima. The random
    # floor (~0.6) is left off the axis: drawing it would spend half the panel on empty space
    # below every real method; it is reported in the printed table instead.
    # "Expected Gradients" is too wide for that region (it ran off the frame at lr 8 and
    # collides with SGD's collapse at lr 100 if right-anchored), so its rule is labelled at the
    # LEFT edge, where nothing sits below AUC ~4.3.
    xlo, xhi = min(sgd_lrs + lrs) / 1.6, max(sgd_lrs + lrs) * 1.6
    for k in ("ref_kernelshap", "ref_attnlrp", "ig"):
        if k not in mean:
            continue
        lab, col = SERIES[k]
        ax.axhline(mean[k], color=col, lw=0.8, ls="dashed", zorder=2)
        x = xlo * 1.15 if k == "ig" else 8.0
        ax.text(x, mean[k] + 0.04, lab, size=5.5, color=col, ha="left", va="bottom")
    ax.set_xscale("log")
    logticks(ax, sgd_lrs + lrs)
    ax.set_xlim(xlo, xhi)
    floor = min(mean[k] for k in ("ig", "ref_attnlrp") if k in mean)
    ax.set_ylim(min(floor, min(lo[k] for k in list(key_s.values()) + list(key_a.values()))) - 0.15,
                max(hi[k] for k in list(key_s.values()) + list(key_a.values())) + 0.1)
    ax.set_xlabel("Learning rate", size=7, labelpad=1)
    ax.set_ylabel("Sufficiency AUC", size=7, labelpad=2)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower left", handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.3)
    ax.text(0.0, 1.03, "(b) by learning rate", transform=ax.transAxes, size=7,
            ha="left", va="bottom")

    # ---- (c) convergence of the best cell per arm
    ax = add(lm + 2 * (pw + gap), y_top, pw, top_h)
    furnish(ax)
    for k, (lab, col) in [(best_adam, SERIES["adam_best"]), (best_default, SERIES["adam_default"]),
                          (best_sgd, SERIES["sgd"]), ("ig", SERIES["ig"])]:
        if k not in arms or not arms[k]["trace"]:
            continue
        tr = np.stack(arms[k]["trace"])              # [seed, n, 2]
        x = np.maximum(tr[0, :, 0], 1)
        y = tr[:, :, 1]
        ax.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(x, y.mean(0), color=col, lw=0.9, label=lab, clip_on=False, zorder=3)
    for k in ("ref_kernelshap", "ref_attnlrp"):
        if k in mean:
            ax.axhline(mean[k], color=SERIES[k][1], lw=0.8, ls="dashed", zorder=2)
    ax.set_xscale("log")
    logticks(ax, [1, x.max()])
    ax.set_xlabel("Training step", size=7, labelpad=1)
    ax.set_ylabel("Probe sufficiency AUC", size=7, labelpad=2)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.legend(fontsize=5.5, frameon=False, loc="lower right", handlelength=1.4,
              borderaxespad=0.2, labelspacing=0.3)
    ax.text(0.0, 1.03, "(c) in training", transform=ax.transAxes,
            size=7, ha="left", va="bottom")

    # ---- (d) score maps
    y_s = 0.02 + 0.40
    for i, (k, lab) in enumerate(strip):
        ax = add(spad + i * (sp + sgap), y_s, sp, sp)
        r = signed_percentile(arms[k]["scores"][0], args.gamma)
        ax.imshow(r, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.5)
        m = arms[k]["meta"]
        sub = lab
        if m.get("arm") == "adam":
            sub += f"\nlr {m['lr']:g}, ε = {sci(m['eps'])}"
        elif m.get("arm") == "sgd":
            sub += f"\nlr {m['lr']:g}"
        else:
            sub += "\n"
        fig.text((spad + i * (sp + sgap) + sp / 2) / W, (y_s - 0.03) / H,
                 f"{sub}\nAUC {mean[k]:.2f}", ha="center", va="top", size=6, linespacing=1.15)
    fig.text(spad / W, (y_s + sp + 0.04) / H,
             f"(d) the rankings (seed {json.loads((Path(args.runs[0]) / 'grid.json').read_text())['args']['seed']}), "
             f"explaining “{meta['pos']['label']}” vs. “{meta['neg']['label']}”",
             ha="left", va="bottom", size=7)
    # the teaser's colour scale: signed percentile rank within each map, |p|^gamma
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cw, ch = 1.15, 0.055
    cax = add(W - spad - cw, y_s + sp + 0.17, cw, ch)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(-1, 1), cmap="bwr"), cax=cax,
                      orientation="horizontal")
    pct = [1, 10, 50, 90, 99]
    cb.set_ticks([np.sign(2 * q / 100 - 1) * abs(2 * q / 100 - 1) ** args.gamma for q in pct])
    cb.set_ticklabels([str(q) for q in pct])
    cax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.outline.set_linewidth(0.4)
    fig.text((W - spad - cw - 0.04) / W, (y_s + sp + 0.17 + ch / 2) / H, "score percentile",
             ha="right", va="center", size=5.5)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"wrote {out} ({W:.2f} x {H:.2f} in), {n_seed} seed(s)")

    # ------------------------------------------------------------------ tables
    print("\nAdam: mean AUC over seeds, eps (rows) x lr (cols)")
    print(f"{'eps':>8} " + "".join(f"{lr:>8g}" for lr in lrs))
    for i, e in enumerate(epss):
        print(f"{e:>8.0e} " + "".join(f"{grid[i, j]:>8.2f}" for j in range(len(lrs))))
    print("\nSGD: " + "  ".join(f"lr {lr:g}: {mean[key_s[lr]]:.2f}" for lr in sgd_lrs))
    print("\nbest per arm (mean [min, max] over seeds):")
    for lab, k in [("Adam best", best_adam), ("Adam default-eps best", best_default),
                   ("SGD best", best_sgd), ("Expected Gradients", "ig"), ("teaser MAttr", "ref_mattr"),
                   ("KernelSHAP", "ref_kernelshap"), ("AttnLRP", "ref_attnlrp"),
                   ("random", "random")]:
        if k in mean:
            print(f"  {lab:<22} {k:<24} {mean[k]:.3f} [{lo[k]:.3f}, {hi[k]:.3f}]")
    # the eps mechanism: does the score distribution flatten at small eps, as at neuron scale?
    print("\nAdam score-distribution diagnostics at the best-eps lr, by eps "
          "(|s| p99/p50, top-20 overlap with Expected Gradients):")
    if "ig" in arms:
        ig_top = [set(np.argsort(-s.ravel())[:20]) for s in arms["ig"]["scores"]]
    for e in epss:
        k = key_a[(adam[best_adam]["lr"], e)]
        ratio = np.mean([json.loads((Path(d) / "grid.json").read_text())["arms"][k]["abs_p99_over_p50"]
                         for d in args.runs])
        ov = (np.mean([len(set(np.argsort(-s.ravel())[:20]) & t) / 20
                       for s, t in zip(arms[k]["scores"], ig_top)]) if "ig" in arms else np.nan)
        print(f"  eps {e:>6.0e}  AUC {mean[k]:.2f}  p99/p50 {ratio:6.1f}  IG overlap {ov:.2f}")


if __name__ == "__main__":
    main()
