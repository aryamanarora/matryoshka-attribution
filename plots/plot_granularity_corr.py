"""Substrate x substrate agreement: do different GRANULARITIES put the circuit in the same layers?

THE SIBLING is plots/plot_task_corr_substrates.py, which holds the substrate fixed and varies the
task. This transposes it: hold the task fixed, vary the substrate, and ask whether attributing the
same behaviour over nodes / MLP neurons / SAE latents finds the same part of the model.

*** THE COMMON SPACE IS THE LAYER, AND IT HAS TO BE. *** These substrates do not share an index
space in any other way -- `node` has 1,056 units (32L x 32 heads + 32 MLPs), `mlp` has 32 x
seq_len x 14,336, the SAE spans have 32 x seq_len x 32,769 -- and unit i means a different object
in each. There is no map from an SAE latent to a neuron, so a unit-level correlation across
granularities is not a hard measurement, it is an undefined one. Every substrate does have a
layer axis, so that is where they can be compared.

WHAT IS CORRELATED is each substrate's TOP-K LAYER DISTRIBUTION: take the top `--frac` of units by
score, count how many fall in each of the 32 layers, normalise to a distribution, and correlate
those 32-vectors between substrates. Deliberately NOT the per-layer sum of raw scores, for two
reasons: the scores are signed (a layer with large opposing contributions would cancel to look
unimportant) and their SCALES differ by orders of magnitude across substrates (MAttr's SAE scores
reach 1e4 while node scores span ~13), so a sum would compare units of different currency. A
top-k membership count is scale-free and sign-free and is also what the eval actually does -- it
keeps a top-k.

READ THIS AS A LAYER-PROFILE AGREEMENT, NOT A CIRCUIT AGREEMENT. Two substrates can put their
top-k in exactly the same layers while selecting entirely unrelated units inside them, and this
figure would show 1.0. It is an upper bound on agreement, and the honest use is the negative
direction: a LOW value here means the granularities genuinely disagree, because they cannot even
agree on which layers matter.

METHOD is MAttr+Adam (eps=1e-2), tag `sufficient_topk_adam_eps1e-2_bs1`, -input runs, llama3 only
-- same cut as the task sibling so the two figures can be read together.

Run:  uv run python plots/plot_granularity_corr.py [--frac 0.01]
Out:  plots/granularity_corr.pdf  (+ .png)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from plotnine import (ggplot, aes, geom_tile, geom_text, labs, facet_wrap, theme, theme_set,
                      theme_bw, element_text, element_blank, scale_fill_gradient2,
                      scale_x_discrete, scale_y_discrete)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES = "results/sva_sweep"
TAG = "sufficient_topk_adam_eps1e-2_bs1"
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
N_LAYERS, N_HEADS = 32, 32
# (disk key, label, per-layer width). None width = `node`, which has its own two-block layout.
SUBS = [("node", "Node", None),
        ("mlp", "MLP", 14336),
        ("mlp-attn_head", "MLP+Attn", 14368),
        ("resid_sae_span", "SAE resid", 32769),
        ("mlp_sae_span", "SAE MLPout", 32769)]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.2),
        axis_title=element_blank(),
        axis_text=element_text(size=5),
        axis_text_x=element_text(size=5, rotation=45, hjust=1),
        legend_text=element_text(size=5.5), legend_title=element_text(size=6),
        legend_key_size=8,
        panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
        panel_spacing_x=0.02, panel_spacing_y=0.03,
        strip_background=element_blank(), strip_text=element_text(size=7),
    )
)


def layer_of(n, width):
    """Layer index for every unit of a score vector of length n."""
    if width is None:                               # node: [32L x 32H attn][32 mlp]
        assert n == N_LAYERS * N_HEADS + N_LAYERS, f"node len {n}"
        return np.concatenate([np.repeat(np.arange(N_LAYERS), N_HEADS), np.arange(N_LAYERS)])
    assert n % (N_LAYERS * width) == 0, f"{n} not 32*{width}*seq_len"
    per = n // N_LAYERS                             # [layer][span][unit], layer-major
    return np.repeat(np.arange(N_LAYERS), per)


def layer_profile(task, sub, width, frac):
    """Normalised top-k-per-layer distribution, or None if the run is missing."""
    p = f"{RES}/{task}_llama3_{sub}_{TAG}.scores.pt"
    if not os.path.exists(p):
        return None
    s = torch.load(p, map_location="cpu").float().numpy()
    k = max(N_LAYERS, int(round(frac * len(s))))    # >= 32 so the profile is not all-zero
    top = np.argpartition(s, -k)[-k:]
    lay = layer_of(len(s), width)[top]
    c = np.bincount(lay, minlength=N_LAYERS).astype(float)
    return c / c.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.01, help="top fraction of units kept")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or "plots/granularity_corr.pdf"

    rows, missing = [], []
    for task in TASKS:
        prof = {}
        for sub, lab, w in SUBS:
            v = layer_profile(task, sub, w, a.frac)
            if v is None:
                missing.append(f"{task}/{lab}")
            else:
                prof[lab] = v
        for la in [l for _, l, _ in SUBS]:
            for lb in [l for _, l, _ in SUBS]:
                if la not in prof or lb not in prof:
                    continue
                r = float(np.corrcoef(prof[la], prof[lb])[0, 1])
                rows.append({"task": task.replace("_", " "), "a": la, "b": lb, "r": r,
                             "lab": f"{r:.2f}".replace("0.", ".").replace("-.", "−.")})
    if not rows:
        raise SystemExit(f"no {TAG} runs under {RES}")
    df = pd.DataFrame(rows)
    order = [l for _, l, _ in SUBS]
    df["task"] = pd.Categorical(df["task"], [t.replace("_", " ") for t in TASKS])

    p = (ggplot(df, aes("a", "b", fill="r")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="lab"), size=3.4)
         + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                                midpoint=0, limits=[-1, 1], na_value="#eeeeee")
         + scale_x_discrete(limits=order, expand=(0, 0))
         + scale_y_discrete(limits=order[::-1], expand=(0, 0))
         + facet_wrap("~task", ncol=4)
         + labs(fill="Pearson $r$"))
    p.save(out, dpi=300, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print(f"wrote {out}  (top {a.frac:.1%} of units, layer profiles)")
    if missing:
        print(f"missing: {', '.join(missing[:8])}")

    print(f"\nmean r per substrate PAIR, averaged over {df['task'].nunique()} tasks:")
    m = (df[df["a"] != df["b"]].groupby(["a", "b"])["r"].mean().reset_index()
         .sort_values("r", ascending=False))
    seen = set()
    for _, x in m.iterrows():
        key = frozenset((x["a"], x["b"]))
        if key in seen:
            continue
        seen.add(key)
        print(f"  {x['a']:<11} vs {x['b']:<11} {x['r']:+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
