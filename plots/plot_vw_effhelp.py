"""Helpfulness against each ranking: the note's "The most effective weights are helpful",
generalised so that every attribution method gets the same panel.

Turner, Wu & Batson (2026) draw this for Fisher effectiveness only, over a 1,111-weight
log-uniform sample of the effectiveness rank, because helpfulness costs them 1B tokens per
weight. Restricted to Tokens->Logits we have EXACT helpfulness for all 20,971,520 weights,
so the same panel can be drawn for any score at no extra cost -- which is the whole point:
the note's three-regime reading (ineffective ~ 0, a mixed middle, a helpful-only tail) is a
claim about a ranking, and it is worth asking which rankings have it.

  top row     x = rank under that score (1 = kept first), log; y = mean helpfulness, symlog.
              Points are a log-uniform sample of the rank axis, as in the note. Colour is the
              note's significance convention: grey where the 95% CI includes zero.
  bottom      cumulative positive-helpfulness mass captured by the top-n of each ranking,
              against density -- the note's "helpfulness mass" analysis as a curve.

    uv run python plots/plot_vw_effhelp.py --run results/vw/base
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vw"))
from palette import METHOD, OTHER, RC, furnish  # noqa: E402
from plot_vw_prune import parse_mattr  # noqa: E402
from vw_prune import rankings  # noqa: E402

plt.rcParams.update(RC)
HELPFUL, HARMFUL, NOTSIG = "#0072b2", "#cc3311", "#cccccc"


# The default panel set. vw_mattr.py writes ~15 score tensors and one panel each would be a
# 32-inch figure, so the figure shows a curated row and `--methods` overrides it.
DEFAULT = ("fisher", "weight_abs", "mattr_adam_lr0.05_eps1e-08_kuniform",
           "mattr_adam_lr0.05_eps0.01", "ig_long")


def panels(curves, keys=None):
    keys = set(keys) if keys else None
    out = [("fisher", "Fisher effectiveness", METHOD["IG"]),
           ("weight_abs", "Virtual weight magnitude", METHOD["I×G"])]
    for k in curves:
        if keys is not None and k not in keys:
            continue
        if k.startswith("mattr_adam"):
            lr, eps, extra, default = parse_mattr(k)
            out.append((k, f"MAttr (Adam, ε={eps}{extra})",
                        METHOD["MAttr (Adam, default eps)"] if default else METHOD["MAttr"]))
        elif k.startswith("mattr_sgd"):
            lr, _, extra, _ = parse_mattr(k)
            out.append((k, f"MAttr (SGD{extra})", METHOD["MAttr (SGD)"]))
        elif k.startswith("ig"):
            out.append((k, "Expected Gradients", METHOD["Expected Gradients"]))
    if keys is None or "random_s0" in keys:
        out.append(("random_s0", "Random", OTHER))
    seen, uniq = set(), []
    for t in out:                     # de-duplicate, preserve order
        if t[0] not in seen and (keys is None or t[0] in keys):
            seen.add(t[0])
            uniq.append(t)
    return uniq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/vw/base"))
    ap.add_argument("--sample", type=int, default=4000, help="points per panel")
    ap.add_argument("--sig", type=float, default=1.96)
    ap.add_argument("--methods", nargs="*", default=list(DEFAULT),
                    help="score keys to draw, in order; default is a curated row")
    ap.add_argument("--out", type=Path, default=Path("paper/figs/vw_effhelp.pdf"))
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sc = torch.load(args.run / "scores" / "tl.pt", map_location=dev, weights_only=False)
    H = sc["helpfulness"].to(dev).flatten().float()
    SE = sc["helpfulness_se"].to(dev).flatten().float()
    lo, hi = H - args.sig * SE, H + args.sig * SE
    ranks = rankings(args.run, dev)
    P = [p for p in panels(ranks, args.methods) if p[0] in ranks]
    assert P, f"none of {args.methods} are in {sorted(ranks)}"
    N = H.numel()

    # the note samples log-uniformly across the rank axis to resolve the extremes
    idx = np.unique(np.round(np.exp(np.linspace(0, np.log(N - 1), args.sample))).astype(np.int64))
    dens = np.geomspace(1 / N, 1.0, 160)

    fig, axes = plt.subplots(2, len(P), figsize=(2.15 * len(P), 4.3),
                             gridspec_kw={"height_ratios": [2.0, 1.25]})
    axes = np.atleast_2d(axes)
    pos_total = float(H[H > 0].sum())
    lim = float(H.abs().max()) * 1.5

    for c, (key, lab, col) in enumerate(P):
        order = ranks[key].argsort(descending=True)
        h = H[order].cpu().numpy()
        sig_p = (lo[order] > 0).cpu().numpy()
        sig_n = (hi[order] < 0).cpu().numpy()
        ax = axes[0, c]
        furnish(ax)
        y, x = h[idx], idx + 1
        colr = np.where(sig_p[idx], HELPFUL, np.where(sig_n[idx], HARMFUL, NOTSIG))
        ax.scatter(x, y, s=1.6, c=colr, lw=0, rasterized=True)
        ax.axhline(0, color="#999999", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_ylim(-lim, lim)
        ax.set_title(lab, fontsize=7.5, color=col)
        ax.set_xlabel("rank (1 = kept first)", fontsize=7)
        if c == 0:
            ax.set_ylabel("mean helpfulness (nats/token)", fontsize=7)
        else:
            ax.set_yticklabels([])

        ax2 = axes[1, c]
        furnish(ax2)
        cum = np.cumsum(np.clip(h, 0, None)) / pos_total
        ks = np.clip((dens * N).astype(np.int64), 1, N) - 1
        ax2.plot(dens, cum[ks], color=col, lw=1.4)
        ax2.plot(dens, dens, color="#bbbbbb", lw=0.7, ls="--")
        ax2.set_xscale("log")
        ax2.set_ylim(0, 1.02)
        ax2.set_xlabel("density", fontsize=7)
        if c == 0:
            ax2.set_ylabel("positive helpfulness\nmass captured", fontsize=7)
        else:
            ax2.set_yticklabels([])
        # the note's two reference densities for helpfulness mass
        for f, m in ((0.90, "90%"), (0.99, "99%")):
            j = int(np.searchsorted(cum, f * cum[-1])) + 1
            ax2.plot(j / N, f, marker="o", ms=2.5, color=col)
        ax2.tick_params(labelsize=6.5)
        ax.tick_params(labelsize=6.5)

    for lbl, c in (("helpful (p<.05)", HELPFUL), ("harmful (p<.05)", HARMFUL),
                   ("not significant", NOTSIG)):
        axes[0, 0].scatter([], [], s=8, c=c, lw=0, label=lbl)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="lower left")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=600)
    fig.savefig(args.out.with_suffix(".png"), dpi=200)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
