"""MAttr's top MLP neurons and top MLP-output SAE latents per SVA+ task, with descriptions.

One panel per task (the 4 SVA subtasks and the 4 arithmetic-wild tasks, Llama 3.1 8B), two
blocks per panel: the five distinct MLP neurons the headline MAttr run ranks first at the `mlp`
substrate (results/sva_sweep_5k, uniform k, Adam eps=1e-2, 5k steps) and the five distinct Llama
Scope MLP-output latents it ranks first at the `mlp_sae_span` substrate (results/sva_sweep_ferr5k,
frozen error, same optimiser). Bars are the unit's score relative to the block's top score, so the
bar says how peaked the ranking is; the label is the unit and its published description --
Transluce for neurons (make_sva_neuron_table.describe, cached in results/.transluce_cache.json),
Neuronpedia (llama3.1-8b / <layer>-llamascope-mlp-32k) for latents (results/neuronpedia_cache.json).

Neurons are deduplicated across positions exactly as the table is (make_sva_neuron_table.top_units:
best-scoring position per (layer, neuron)); latents across spans the same way, and the per-span
reconstruction-error nodes are skipped (their count in the raw top-5 is printed). Neurons that
Feucht et al. (2026, arithmetic-wild) publish for the arithmetic tasks get a star.

Caveats that belong in the caption: Transluce descriptions are for Llama-3.1-8B-INSTRUCT while
the runs use the base model (same neuron indexing; indicative, not proof); Neuronpedia's Llama
Scope descriptions are auto-interp (GPT-4o-mini) on the base model.

    uv run python plots/plot_sva_top_units.py            -> paper/figs/sva_top_units.pdf
    uv run python plots/plot_sva_top_units.py --no-fetch    (cached descriptions only)
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "scripts" / "sva"))
import make_sva_neuron_table as NT                    # noqa: E402  (decode/top_units/describe)
from neuronpedia import fetch_feature, load_cache  # noqa: E402  (scripts/sva)
from palette import RC, METHOD, furnish               # noqa: E402  (plots/ is on sys.path)

plt.rcParams.update(RC)
plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 5.5, "ytick.labelsize": 6})
OUT = ROOT / "paper" / "figs"

TASKS = NT.TASKS                                       # (task, display), SVA then arithmetic
TOPN = 5
MLP_RES = ROOT / "results" / "sva_sweep_5k"
MLP_TAG = "iso_topk_adam_eps1e-2_uniformk_bs1_s5000"
SAE_RES = ROOT / "results" / "sva_sweep_ferr5k"
SAE_TAG = "iso_topk_adam_eps1e-2_uniformk_ferr_bs1_s5000"
D_SAE = 32768
NP_MODEL, NP_SET = "llama3.1-8b", "{layer}-llamascope-mlp-32k"
NP_CACHE = ROOT / "results" / "neuronpedia_cache.json"
COL = {"neuron": METHOD["MAttr"], "latent": "#d55e00"}      # Wong blue / Wong vermillion
DESC_CHARS = 56


def top_neurons(task):
    scores, meta = NT.load_run(MLP_RES, task, "mlp", MLP_TAG)
    if scores is None:
        return None
    return NT.top_units(scores, meta, "mlp", NT.layout(meta, "mlp", False), False, n=TOPN)


def top_latents(task):
    """Top-n distinct (layer, latent) at the SAE substrate; error nodes skipped and counted."""
    stem = SAE_RES / f"{task}_llama3_mlp_sae_span_{SAE_TAG}"
    pt, js = Path(str(stem) + ".scores.pt"), Path(str(stem) + ".json")
    if not pt.exists():
        return None, 0
    meta = json.load(open(js)); S = meta["seq_len"]; W = D_SAE + 1
    scores = torch.load(pt, map_location="cpu", weights_only=False)
    assert scores.numel() == meta["num_layers"] * S * W, (scores.numel(), meta["num_layers"], S, W)
    order = torch.argsort(scores, descending=True)
    out, seen, err_in_top = [], set(), 0
    for i, idx in enumerate(order.tolist()):
        layer, rem = divmod(idx, S * W); span, feat = divmod(rem, W)
        if feat == D_SAE:                                 # the (layer, span) error node
            err_in_top += i < TOPN
            continue
        if (layer, feat) in seen:
            continue
        seen.add((layer, feat))
        out.append(dict(layer=layer, latent=feat, span=span, score=float(scores[idx])))
        if len(out) == TOPN:
            break
    return out, err_in_top


def clip(s, n=DESC_CHARS):
    """One line, Transluce's {{token}} markers unwrapped to quotes, word-boundary truncated."""
    s = " ".join((s or "(no description)").split()).replace("{{", "\u2018").replace("}}", "\u2019")
    return s if len(s) <= n else textwrap.shorten(s, n, placeholder="\u2026")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-fetch", action="store_true", help="cached descriptions only")
    a = ap.parse_args()
    np_cache = load_cache(NP_CACHE)

    fig, axes = plt.subplots(4, 2, figsize=(5.5, 7.6))
    for ax, (task, disp) in zip(axes.ravel(), TASKS):
        neurons, (latents, err) = top_neurons(task), top_latents(task)
        rows = []            # (label, description, rel_score, kind)
        for u in neurons or []:
            d = NT.describe(u["layer"], u["neuron"], "+", fetch=not a.no_fetch)
            star = "★ " if NT.gf_mark(u, task, "mlp") else ""
            rows.append((f"{star}L{u['layer']} N{u['neuron']}", clip(d), u["score"], "neuron"))
        rows.append(None)
        for u in latents or []:
            info = fetch_feature(NP_MODEL, NP_SET.format(layer=u["layer"]), u["latent"], np_cache) \
                if not a.no_fetch else np_cache.get(
                    f"{NP_MODEL}/{NP_SET.format(layer=u['layer'])}/{u['latent']}",
                    {"desc": ""})
            rows.append((f"L{u['layer']} F{u['latent']}", clip(info.get("desc", "")), u["score"],
                         "latent"))
        print(f"{task}: {len(neurons or [])} neurons, {len(latents or [])} latents, "
              f"{err} error nodes in the raw SAE top-{TOPN}")
        for r in rows:
            if r: print("   ", r[0], "|", r[1])
        # relative score within each block
        y, top = [], {}
        for r in rows:
            if r: top[r[3]] = max(top.get(r[3], 0), r[2])
        ys = np.arange(len(rows))[::-1]
        for yy, r in zip(ys, rows):
            if r is None:
                continue
            lab, desc, sc, kind = r
            rel = sc / top[kind] if top[kind] > 0 else 0
            ax.barh(yy, rel, height=0.72, color=COL[kind], alpha=0.28, edgecolor="none")
            ax.text(-0.02, yy, lab, ha="right", va="center", fontsize=5.6,
                    color=COL[kind], fontweight="bold")
            ax.text(0.015, yy, desc, ha="left", va="center", fontsize=5.0, color="#000",
                    clip_on=True)
        ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks([]); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", "½", "1"])
        ax.set_title(disp, pad=3)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Score relative to the block's top unit", labelpad=2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=COL["neuron"], label="MLP neurons (mlp substrate)"),
                        Patch(color=COL["latent"], label="MLP-output SAE latents (Llama Scope 32k)")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0.11, 0, 1, 0.975), h_pad=1.0, w_pad=4.2)
    NT.save_cache()
    json.dump(np_cache, open(NP_CACHE, "w"))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "sva_top_units.pdf"); fig.savefig(OUT / "sva_top_units.png", dpi=200)
    print("wrote", OUT / "sva_top_units.pdf")


if __name__ == "__main__":
    main()
