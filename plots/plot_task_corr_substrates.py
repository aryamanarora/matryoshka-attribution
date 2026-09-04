"""Task x task PEARSON correlation of MAttr+Adam's attribution scores, faceted by SUBSTRATE.

THE QUESTION. plots/plot_task_corr_heatmap.py asks "how much does the circuit depend on the
task?" for one method on the NODE substrate. This asks whether the answer to that question is a
property of the model or a property of the VARIABLE SET you attribute over: the same method, the
same eight llama3 tasks, five different substrates side by side.

*** THE POSITION AXIS IS COLLAPSED, AND IT HAS TO BE. *** Only `node` is position-agnostic (1,056
units for every task). Every other substrate here is a PER-POSITION layout whose score vector is
[layer][position][unit], so its length is 32 * seq_len * width and seq_len is the modal prompt
length of that task -- 3 for `simple`, 38 for `hours`. Two tasks therefore have different-length
vectors indexing different objects, and correlating them raw is not merely noisy, it is
undefined. This script reshapes to (32, seq_len, width) and reduces over the position axis, which
puts every task on the common (layer, unit) index space the correlation needs.

WHAT THAT COSTS, stated plainly: the figure answers "do these tasks rely on the same UNITS", not
"...at the same POSITIONS". A method that used identical units at systematically different
positions would look perfectly correlated here. `node` is the one facet with no such caveat,
which makes it the reference the other four should be read against rather than just another
panel. --agg max is provided because mean and max disagree about a unit that matters enormously
at one position and not at all elsewhere; if the two give different pictures, that difference is
itself the finding and neither should be quoted alone.

PEARSON, not Spearman, per the request -- so this measures agreement on score MAGNITUDES, where
the sibling figure measures agreement on RANKS. On the SAE substrates that distinction is sharp:
those score distributions are dominated by a small number of extreme values (on resid, the top
units reach 1e4 while typical live units sit near 1), so Pearson there is largely a statement
about whether the same few outliers appear in both tasks. Read it with that in mind, and prefer
the rank sibling if the question is about the circuit's ordering.

METHOD IS MAttr+Adam (eps=1e-2), tag `sufficient_topk_adam_eps1e-2_bs1` -- one method on purpose,
because overlaying methods folds method-disagreement back into a panel meant to isolate
task-disagreement.

llama3 ONLY (all eight SVA + arithmetic tasks are llama3; IOI is qwen2.5 and is excluded, since
correlating across models compares different objects that share a naming scheme).

Run:  uv run python plots/plot_task_corr_substrates.py [--agg mean|max]
Out:  plots/task_corr_substrates.pdf  (+ .png)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from plotnine import (ggplot, aes, geom_tile, geom_text, labs, facet_wrap, theme, theme_set,
                      theme_bw, element_text, element_line, element_blank,
                      scale_fill_gradient2, scale_x_discrete, scale_y_discrete)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                            # noqa: E402

# (results dir, label). The +input dir holds runs where the input-embedding node is scored and
# ablated as an extra unit at index 0. It exists for `node` and -- since 2026-08-30 -- for the two
# SAE layouts; `mlp` / `mlp+attn_head` have no +input arm at all (the hooker silently drops the
# flag there), so those facets simply do not appear in the +input half.
SOURCES = [("results/sva_sweep", "−input"), ("results/sva_sweep_input", "+input")]
RES = "results/sva_sweep"
# --method picks the attribution whose task-agreement is being measured. One method per figure on
# purpose: overlaying them folds method-disagreement back into a panel meant to isolate
# task-disagreement (the sibling plot_method_corr_heatmap.py is the figure for that question).
METHOD_TAG = {"mattr-adam": "sufficient_topk_adam_eps1e-2_bs1",
              "mattr-sgd": "sufficient_topk_sgd_bs1",
              "ig": "ig", "ixg": "ixg", "random": "random_s42"}
TAG = METHOD_TAG["mattr-adam"]
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
N_LAYERS = 32
# (on-disk substrate key, display label, per-layer WIDTH). Width is what lets seq_len be derived
# as n / (32 * width) and is asserted below -- llama3 intermediate_size 14336; +32 attn heads for
# the joint substrate; Llama-Scope 8x is 32768 latents + 1 error node per (layer, position).
# `node` is position-agnostic (32 layers x (32 heads + 1 MLP) = 1,056) so it takes no reshape.
SUBS = [("node", "Node", None),
        ("mlp", "MLP neuron", 14336),
        ("mlp-attn_head", "MLP+Attn", 14368),
        ("resid_sae_span", "SAE (resid)", 32769),
        ("mlp_sae_span", "SAE (MLP out)", 32769)]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 3.4),
        axis_title=element_blank(),
        axis_text=element_text(size=5),
        axis_text_x=element_text(size=5, rotation=45, hjust=1),
        legend_text=element_text(size=5.5),
        legend_title=element_text(size=6),
        legend_key_size=8,
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_spacing_x=0.02,
        panel_spacing_y=0.03,
        strip_background=element_blank(),
        strip_text=element_text(size=7),
    )
)


def profile(task, sub, width, agg, res):
    """(layer, unit) score profile for one (task, substrate, source), or None if missing."""
    p = f"{res}/{task}_llama3_{sub}_{TAG}.scores.pt"
    if not os.path.exists(p):
        return None
    v = torch.load(p, map_location="cpu").float()
    # +input runs carry the input-embedding node at index 0. It is DROPPED, not appended.
    #
    # *** KEEPING IT DESTROYS THE FIGURE. *** Its score is orders of magnitude above the bulk --
    # on mlp_sae_span/addition IG scores it 13.41 against a max of 0.56 over the other 5,243,040
    # units and a 99.99th percentile of 0.027. Two vectors that share one spike 500x larger than
    # everything else correlate at ~1 whatever the rest does, and that is exactly what happened:
    # the +input facets read 0.945 / 0.995 with it in, against 0.29 / 0.25 with it out.
    #
    # Consequence worth knowing before reading the +input column: for GRADIENT methods the
    # non-input scores of a +input run are BIT-IDENTICAL to the -input run (verified on node,
    # resid_sae_span and mlp_sae_span), so IG/IxG's two facets are the same figure twice. Only
    # MAttr differs -- there the input node competes for the top-k budget during training, which
    # moves 100% of the learned scores.
    #
    # Detected by divisibility rather than by directory name, so a mislabelled dir cannot
    # silently shift every unit by one index.
    if width is not None and v.numel() % (N_LAYERS * width) == 1:
        v = v[1:]
    elif width is None and v.numel() == N_LAYERS * 33 + 1:
        v = v[1:]
    if width is None:                       # node: already position-agnostic
        return v.numpy()
    n = v.numel()
    assert n % (N_LAYERS * width) == 0, f"{task}/{sub}: {n} not 32*{width}*seq_len"
    seq = n // (N_LAYERS * width)
    v = v.view(N_LAYERS, seq, width)
    # Reduce over POSITION (axis 1). max is on |score| then re-signed, so a large NEGATIVE score
    # is not silently replaced by a small positive one at another position.
    if agg == "max":
        idx = v.abs().argmax(dim=1, keepdim=True)
        v = v.gather(1, idx).squeeze(1)
    else:
        v = v.mean(dim=1)
    return v.reshape(-1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="mean", choices=["mean", "max"],
                    help="how the position axis is collapsed (node is unaffected)")
    ap.add_argument("--method", default="mattr-adam", choices=list(METHOD_TAG))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global TAG
    TAG = METHOD_TAG[a.method]
    msuf = "" if a.method == "mattr-adam" else f"_{a.method}"
    out = a.out or (f"plots/task_corr_substrates{msuf}"
                    f"{'' if a.agg == 'mean' else '_max'}.pdf")

    rows, missing = [], []
    for res, ilabel in SOURCES:
        for sub, slabel, width in SUBS:
            prof = {}
            for t in TASKS:
                v = profile(t, sub, width, a.agg, res)
                if v is not None:
                    prof[t] = v
            if len(prof) < 2:               # no +input arm for this substrate: skip the facet
                if prof:
                    missing.append(f"{slabel} {ilabel}")
                continue
            facet = f"{slabel}, {ilabel}"
            for ta in TASKS:
                for tb in TASKS:
                    if ta not in prof or tb not in prof:
                        continue
                    r = float(np.corrcoef(prof[ta], prof[tb])[0, 1])
                    rows.append({"sub": facet, "a": ta.replace("_", " "),
                                 "b": tb.replace("_", " "), "r": r,
                                 "lab": f"{r:.2f}".replace("0.", ".").replace("-.", "−.")})
            del prof
    if not rows:
        raise SystemExit(f"no {TAG} runs found under {RES}")
    df = pd.DataFrame(rows)
    order = [t.replace("_", " ") for t in TASKS]
    facet_order = [f"{sl}, {il}" for _, il in SOURCES for _, sl, _ in SUBS]
    df["sub"] = pd.Categorical(df["sub"], [f for f in facet_order if f in set(df["sub"])])

    p = (ggplot(df, aes("a", "b", fill="r")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="lab"), size=3.6)
         # Same diverging ramp and orientation as plot_method_corr_heatmap.py / the task sibling,
         # so a reader moving between the three figures does not have to relearn the colour.
         + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                                midpoint=0, limits=[-1, 1], na_value="#eeeeee")
         + scale_x_discrete(limits=order, expand=(0, 0))
         + scale_y_discrete(limits=order[::-1], expand=(0, 0))
         + facet_wrap("~sub", ncol=4)
         + labs(fill="Pearson $r$"))
    p.save(out, dpi=300, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print(f"wrote {out}  (method={a.method}, tag={TAG}, agg={a.agg})")
    if missing:
        print(f"missing cells ({len(missing)}): {', '.join(missing[:8])}")

    print(f"\nmean off-diagonal r per substrate ({a.method}, agg={a.agg}) -- how "
          f"task-INDEPENDENT the attribution is:")
    for s in df["sub"].cat.categories:
        d = df[(df["sub"] == s) & (df["a"] != df["b"])]
        print(f"  {s:<16} mean={d['r'].mean():+.3f}   min={d['r'].min():+.3f}   "
              f"max={d['r'].max():+.3f}   n_pairs={len(d) // 2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
