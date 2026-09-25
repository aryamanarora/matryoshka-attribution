"""Loss vs density for the Tokens->Logits virtual weights: the note's pruning figure, plus us.

Redraws the figure "Fisher effectiveness can cheaply filter tens of percent" from Turner,
Wu & Batson, "Characterizing interference weights in a tiny language model" (Transformer
Circuits, 2026), restricted to the Tokens->Logits family of our replication, with the
helpfulness oracle and MAttr added as series the note does not have.

  x  density = fraction of the family's 20,971,520 virtual weights kept
  y  dL = L(pruned) - L(full model), on the held-out shard, log scale

The note's own annotations (dL = 0.000146 / 0.0107 / 0.0702 at density 0.55 / 0.30 / 0.15,
over all six families) are drawn as reference rules so the two are comparable at a glance.

    uv run python plots/plot_vw_prune.py --run results/vw/base
"""
import argparse
import json
import sys
from pathlib import Path

import re

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from palette import METHOD, OTHER, RC, furnish  # noqa: E402

plt.rcParams.update(RC)

# series key on disk -> (label, colour, linestyle, z)
SERIES = [
    ("helpfulness", "Helpfulness (oracle)", "#000000", (0, (1, 1)), 6),
    ("fisher", "Fisher effectiveness", METHOD["IG"], "-", 5),
    ("weight_abs", "Virtual weight magnitude", METHOD["I×G"], "-", 4),
    ("era", "Expected attribution", METHOD["AttnLRP"], "--", 3),
    ("random_s0", "Random", OTHER, "-", 1),
]
NOTE_POINTS = {0.55: 0.000146, 0.30: 0.0107, 0.15: 0.0702}

def parse_mattr(k):
    """tag -> (label suffix, is_default_eps). Tags look like
    mattr_adam_lr0.05_eps0.01[_row|_col] / mattr_sgd_lr1.0_eps1e-08 / *_long."""
    gran = ""
    for g in ("_row", "_col"):
        if k.endswith(g):
            gran, k = f", {g[1:]}", k[: -len(g)]
    long = ", long" if k.endswith("_long") else ""
    k = k[:-5] if k.endswith("_long") else k
    # strip the k-schedule suffix BEFORE reading eps, or eps parses as "1e-08_kuniform"
    # and the arm loses its default-eps colour
    for ks in ("_klog_both", "_kuniform", "_klogit", "_kitem"):
        if k.endswith(ks):
            gran, k = f"{gran}, {ks[2:]}", k[: -len(ks)]
    eps = k.split("eps")[-1] if "eps" in k else ""
    lr = k.split("_lr")[-1].split("_eps")[0] if "_lr" in k else ""
    default = eps in ("1e-08", "1e-8")
    return lr, eps, gran + long, default



# vw_mattr.py writes ~30 score tensors; drawing all of them makes an unreadable figure with
# duplicate legend entries. This is the curated default; `--methods` overrides it and
# `--methods all` draws everything.
# The curated row shows BOTH ends of the MAttr tuning range, because the difference between
# them is the largest single effect in the study (260x at density 0.55): the tuned arm
# (eps 1e-8 + uniform k) and the untuned default (lr 0.05, eps 1e-2) that Result 6 quoted.
DEFAULT = ("helpfulness", "fisher", "weight_abs", "era",
           "J_eps1e-8_lr0.01_s24000_b64",          # the tuned, batch-64 arm -- the best one (stage J)
           "mattr_adam_lr0.05_eps0.01",            # the untuned default, for contrast
           "mattr_adam_lr0.05_eps0.01_row", "ig_long", "random_s0")


# Explicit --tag runs from the budget/batch stages look like
# `H_eps1e-8_lr0.01_s12000_b32[_kitem]`; without this they match no prefix below and are
# silently dropped from the figure -- which is how the winning arm went missing once.
STAGE_TAG = re.compile(r"^[A-Z]_eps([\d.e+-]+)_lr([\d.]+)_s(\d+)_b(\d+)(.*)$")


def mattr_series(curves, keys=None):
    """Whatever vw_mattr.py left on disk, coloured by the palette's optimizer rule."""
    out = []
    for k in curves:
        if keys is not None and k not in keys:
            continue
        m = STAGE_TAG.match(k)
        if m:
            eps, lr, steps, batch, extra = m.groups()
            default = eps in ("1e-08", "1e-8")
            lab = (f"MAttr (Adam, lr {lr}, ε={eps}, b{batch}, {int(steps)//1000}k"
                   + (", per-item k" if "kitem" in extra else "") + ")")
            out.append((k, lab, METHOD["MAttr (Adam, default eps)"] if default
                        else METHOD["MAttr"], "--", 8))
            continue
        if k.startswith("mattr_adam"):
            lr, eps, extra, default = parse_mattr(k)
            out.append((k, f"MAttr (Adam, lr {lr}, ε={eps}{extra})",
                        METHOD["MAttr (Adam, default eps)"] if default else METHOD["MAttr"],
                        ":" if extra else "-", 7))
        elif k.startswith("mattr_sgd"):
            lr, _, extra, _ = parse_mattr(k)
            out.append((k, f"MAttr (SGD, lr {lr}{extra})", METHOD["MAttr (SGD)"],
                        ":" if extra else "-", 7))
        elif k.startswith("ig"):
            out.append((k, "Expected Gradients" + (" (long)" if "long" in k else ""),
                        METHOD["Expected Gradients"], ":" if "long" in k else "-", 2))
        elif k.startswith("ixg"):
            out.append((k, "I×G", METHOD["I×G"], ":", 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/vw/base"))
    ap.add_argument("--prune", default="prune.json")
    # The note's own figure uses a LINEAR density axis from 0 to 1; ours default to log
    # because the interesting behaviour here spans four decades of density. `--linear-x`
    # redraws it in the note's coordinates for a like-for-like visual comparison.
    ap.add_argument("--linear-x", action="store_true")
    ap.add_argument("--methods", nargs="*", default=list(DEFAULT),
                    help='score keys to draw; "all" draws every series on disk')
    ap.add_argument("--out", type=Path, default=Path("paper/figs/vw_prune.pdf"))
    args = ap.parse_args()
    d = json.loads((args.run / args.prune).read_text())
    x = np.array(d["densities"])
    full, curves = d["full_loss"], d["curves"]

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    furnish(ax)
    # The note's three annotated points are for ALL SIX families under ZERO ablation, so they
    # are only a fair reference on that panel; on the token-rows cut or the mean-ablation
    # control they would invite a comparison that is not like for like.
    if d.get("ablation", "zero") == "zero" and d.get("rows", "all") == "all":
        for dens, dl in NOTE_POINTS.items():
            ax.plot(dens, dl, marker="*", ms=9, color="#888888", zorder=2, ls="none")
        ax.plot([], [], marker="*", ms=7, color="#888888", ls="none",
                label="note, all six families")

    keys = None if args.methods == ["all"] else set(args.methods)
    for key, lab, col, ls, z in SERIES + mattr_series(curves, keys):
        if key not in curves or (keys is not None and key not in keys):
            continue
        y = np.array(curves[key]) - full
        # linestyle as a KEYWORD: the oracle series uses a dash tuple (0, (1, 1)), and a
        # third positional argument to ax.plot is parsed as a format string, not a style.
        # NOT clipped: the y axis is symlog precisely so that NEGATIVE dL -- a ranking whose
        # pruned model beats the full one -- is visible. Clipping to a positive floor here
        # silently drew every such point on the zero line, which is the single most
        # interesting thing any of these curves does.
        ax.plot(x, y, linestyle=ls, color=col, lw=1.4, zorder=z, label=lab)

    ax.axhline(d["empty_loss"] - full, color="#bbbbbb", lw=0.8, ls="-.", zorder=0)
    ax.text(0.985 if not args.linear_x else 0.98,
            (d["empty_loss"] - full) * 1.25, "no direct path at all", ha="right",
            fontsize=6, color="#777777")
    ax.axhline(0, color="#999999", lw=0.6, zorder=0)
    ax.set_xscale("linear" if args.linear_x else "log")
    # symlog, NOT log: the helpfulness oracle's dL goes NEGATIVE (pruning the 70% of weights
    # whose individual ablation helps is an improvement to the model), and a log axis would
    # silently clip that to the floor -- which is the single most striking number here.
    ax.set_yscale("symlog", linthresh=1e-4, linscale=0.4)
    fam = {"token": "token→logit", "position": "position→logit"}.get(d.get("rows"), "Tokens→Logits")
    # the token-rows cut is ALSO 4096 x 4096 (16.8M), so this must key on the family, not
    # the count -- or the controlled Tokens->Logits figure gets labelled Features->Logits
    if d["n_weights"] == 4096 * 4096 and d.get("rows", "all") == "all":
        fam = "Features→Logits"
    ax.set_xlabel(f"density (fraction of the {d['n_weights']/1e6:.1f}M {fam} weights kept)")
    ax.set_ylabel("Δ loss on held-out text (nats)")
    if d.get("ablation") == "mean":
        ax.set_title("mean ablation (the constant the family supplies is preserved)",
                     fontsize=7.5)
    ax.set_xlim(0.0, 1.0) if args.linear_x else ax.set_xlim(min(x) * 0.8, 1.15)
    # keep the negative half of the symlog axis tight to the data -- the oracle bottoms out
    # around -0.02 and an unbounded negative range wastes half the panel
    lo = min(min(np.array(v) - full) for v in curves.values())
    ax.set_ylim(min(lo * 1.4, -3e-4), max(max(np.array(v) - full) for v in curves.values()) * 2)
    ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=600, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
