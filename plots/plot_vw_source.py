"""One source token's Tokens->Logits weights, scored every way: the note's per-node figure.

Redraws "Effectiveness concentrates helpful and harmful weights (IN Tokens → Logits)" from
Turner, Wu & Batson (2026). Their version has three panels -- a virtual-weight histogram, a
virtual-weight-vs-helpfulness scatter and a Fisher-vs-helpfulness scatter -- and the reading
is that

  "Virtual weight shows a weak relationship to helpfulness (panels 1 and 2) ... Fisher
   effectiveness has a much clearer relationship with helpfulness (panel 3). Weights with
   increasing effectiveness bifurcate into helpful and harmful weights."

We keep that layout and append one panel per attribution method, so the question "does this
score bifurcate the way effectiveness does" is asked of MAttr and the gradient baselines on
exactly the same 4,096 weights. Colour is the note's significance convention: grey where
the 95% CI on helpfulness includes zero.

    uv run python plots/plot_vw_source.py --run results/vw/base --source IN
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "vw"))
from palette import METHOD, RC, furnish  # noqa: E402
from plot_vw_effhelp import DEFAULT, HARMFUL, HELPFUL, NOTSIG, panels  # noqa: E402
from vw_model import VOCAB  # noqa: E402
from vw_prune import rankings  # noqa: E402
from vw_scores import D_VP  # noqa: E402

plt.rcParams.update(RC)
ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/vw/base"))
    ap.add_argument("--source", default="IN")
    ap.add_argument("--sig", type=float, default=1.96)
    ap.add_argument("--annotate", type=int, default=3, help="label the top-n of each panel")
    ap.add_argument("--methods", nargs="*", default=list(DEFAULT))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or Path(f"paper/figs/vw_source_{args.source.strip()}.pdf")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tk = Tokenizer.from_file(str(ROOT / "data" / "vw" / "tok4096.json"))
    ids = tk.encode(args.source, add_special_tokens=False).ids
    assert len(ids) == 1, f"{args.source!r} is {len(ids)} tokens"
    i = ids[0]

    sc = torch.load(args.run / "scores" / "tl.pt", map_location=dev, weights_only=False)
    H = sc["helpfulness"][i].float().cpu().numpy()
    SE = sc["helpfulness_se"][i].float().cpu().numpy()
    col = np.where(H - args.sig * SE > 0, HELPFUL, np.where(H + args.sig * SE < 0, HARMFUL, NOTSIG))
    ranks = rankings(args.run, dev)
    P = [("weight", "Virtual weight (signed)", METHOD["I×G"])] + \
        [p for p in panels(ranks, args.methods) if p[0] in ranks and p[0] != "random_s0"]

    fig, axes = plt.subplots(1, len(P) + 1, figsize=(1.95 * (len(P) + 1), 2.5))
    ax0 = axes[0]
    furnish(ax0)
    W = sc["W"][i].float().cpu().numpy()
    ax0.hist(W, bins=80, color="#999999", lw=0)
    ax0.set_xlabel("virtual weight", fontsize=7)
    ax0.set_ylabel("count", fontsize=7)
    ax0.set_title(f"{tk.decode([i])!r} → logits", fontsize=7.5)
    ax0.tick_params(labelsize=6.5)

    lim = np.abs(H).max() * 1.4
    for c, (key, lab, colr) in enumerate(P):
        ax = axes[c + 1]
        furnish(ax)
        s = ranks[key].view(D_VP, VOCAB)[i].cpu().numpy()
        x = np.abs(s) if key in ("fisher", "era") else s
        ax.scatter(x, H, s=3.0, c=col, lw=0, rasterized=True)
        ax.axhline(0, color="#999999", lw=0.5)
        if key in ("fisher", "era"):
            # Fisher and ERA are non-negative, so a plain log axis -- symlog here produced an
            # unreadable pile of decade labels either side of a meaningless linear region.
            pos = x[x > 0]
            ax.set_xscale("log")
            if pos.size:
                ax.set_xlim(np.quantile(pos, 0.01), pos.max() * 2)
            ax.xaxis.set_major_locator(plt.LogLocator(numticks=5))
            ax.xaxis.set_minor_locator(plt.NullLocator())
        elif key not in ("weight", "weight_abs"):
            # Learned scores pile up at zero with a long tail, so a linear axis compresses
            # every point into one column. symlog at the median |score| spreads the bulk
            # while keeping the sign, which is what the panel is about.
            lt = float(np.median(np.abs(x[x != 0]))) if (x != 0).any() else 1e-6
            ax.set_xscale("symlog", linthresh=max(lt, 1e-12))
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(lab, fontsize=7, color=colr)
        ax.set_yticklabels([]) if c else ax.set_ylabel("helpfulness", fontsize=7)
        ax.tick_params(labelsize=6.5)
        for j in np.argsort(-s)[:args.annotate]:
            ax.annotate(tk.decode([int(j)]), (x[j], H[j]), fontsize=5.5,
                        xytext=(2, 2), textcoords="offset points")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
