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
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from plotnine import (ggplot, aes, geom_tile, geom_text, labs, facet_wrap, theme, theme_set,
                      theme_bw, element_text, element_line, element_blank,
                      scale_fill_gradient2, scale_x_discrete, scale_y_discrete)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                            # noqa: E402
import plot_accauc_vs_faithauc as V   # trees, parse_method, on_model  # noqa: E402

# (results dir, label). The +input dir holds runs where the input-embedding node is scored and
# ablated as an extra unit at index 0. It exists for `node` and -- since 2026-08-30 -- for the two
# SAE layouts; `mlp` / `mlp+attn_head` have no +input arm at all (the hooker silently drops the
# flag there), so those facets simply do not appear in the +input half.
# +input DROPPED from this figure (2026-09-04) along with the substrates that no longer appear in
# the paper's cut; it lives on in git history if the contrast is wanted again.
LOSS = "logit_diff"
RES = "results/sva_sweep"
# --method picks the attribution whose task-agreement is being measured. One method per figure on
# purpose: overlaying them folds method-disagreement back into a panel meant to isolate
# task-disagreement (the sibling plot_method_corr_heatmap.py is the figure for that question).
# *** RESOLVED THROUGH parse_method, NOT A HARDCODED TAG. *** The tags moved out from under this
# figure twice: the SAE runs gained `_ferr` (--sae-error frozen) and the MLP / SAE MAttr runs
# gained `_s5000`, so a literal "sufficient_topk_adam_eps1e-2_bs1" now names a run no published
# figure draws. Scanning the run json and asking V.parse_method for the key is what the scatter
# and curve figures do, and it means this one cannot drift from them again.
METHOD_KEY = {"mattr-adam": "stopk-log-eps1e-2", "mattr-sgd": "softsgd-log",
              "ig": "IG", "ixg": "IxG", "random": "Random"}
KEY = METHOD_KEY["mattr-adam"]
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
N_LAYERS = 32
# (on-disk substrate key, display label, per-layer WIDTH). Width is what lets seq_len be derived
# as n / (32 * width) and is asserted below -- llama3 intermediate_size 14336; +32 attn heads for
# the joint substrate; Llama-Scope 8x is 32768 latents + 1 error node per (layer, position).
# `node` is position-agnostic (32 layers x (32 heads + 1 MLP) = 1,056) so it takes no reshape.
# (substrate key, label, per-layer WIDTH). Width lets seq_len be derived as n / (32 * width) and
# is asserted in profile(); `node` is position-agnostic so it takes no reshape. Cut to the three
# the paper's figures show, ordered by unit count.
SUBS = [("node", "Node", None),
        ("mlp", "MLP neuron", 14336),
        ("mlp_sae_span", "SAE (MLP out)", 32769)]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.1),
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


def find_scores(res, sub, task, key):
    """Path to the (tree, substrate, task, method) score tensor, or None.

    Matches on the RUN JSON -- d["nodes"], d["loss"], V.on_model, V.parse_method -- never on the
    filename. `_llama3_mlp_` is a substring of `_llama3_mlp_sae_span_`, and parse_method's key
    does not encode the loss, so filename matching silently picks up the wrong substrate or the
    ACC-loss twin of a logit-diff run.
    """
    if not os.path.isdir(res):
        return None
    for f in sorted(os.listdir(res)):
        if not f.endswith(".json") or not f.startswith(f"{task}_llama3_"):
            continue
        try:
            d = json.load(open(os.path.join(res, f)))
        except Exception:
            continue
        if d.get("nodes") != sub or d.get("loss") != LOSS or not V.on_model(d):
            continue
        if V.parse_method(f, d) != key:
            continue
        sp = os.path.join(res, f[:-len(".json")] + ".scores.pt")
        return sp if os.path.exists(sp) else None
    return None


def profile(task, sub, width, agg, res, key, corr):
    """(layer, unit) profile for one (task, substrate), or None if the run is missing.

    Ranked here when corr == "spearman", i.e. WITHIN a task and over the full (layer, unit)
    profile, so the later corrcoef is Spearman. Ranking before the position collapse would be a
    different statistic and is not what is wanted -- the collapse defines the units being ranked.
    """
    p = find_scores(res, sub, task, key)
    if p is None:
        return None
    v = torch.load(p, map_location="cpu").float()
    # +input runs carry the input-embedding node at index 0. It is DROPPED, not appended.
    #
    # *** KEEPING IT DESTROYS THE FIGURE. *** Its score is orders of magnitude above the bulk --
    # on mlp_sae_span/addition IG scores it 13.41 against a max of 0.56 over the other 5,243,040
    # units. Two vectors sharing one spike 500x larger than everything else correlate at ~1
    # whatever the rest does. Detected by divisibility rather than by directory name, so a
    # mislabelled dir cannot silently shift every unit by one index.
    if width is not None and v.numel() % (N_LAYERS * width) == 1:
        v = v[1:]
    elif width is None and v.numel() == N_LAYERS * 33 + 1:
        v = v[1:]
    if width is not None:
        n = v.numel()
        assert n % (N_LAYERS * width) == 0, f"{task}/{sub}: {n} not 32*{width}*seq_len"
        seq = n // (N_LAYERS * width)
        v = v.view(N_LAYERS, seq, width)
        # Reduce over POSITION (axis 1). max is on |score| then re-signed, so a large NEGATIVE
        # score is not silently replaced by a small positive one at another position.
        if agg == "max":
            v = v.gather(1, v.abs().argmax(dim=1, keepdim=True)).squeeze(1)
        else:
            v = v.mean(dim=1)
    a = v.reshape(-1).numpy()
    return rankdata(a) if corr == "spearman" else a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", default="mean", choices=["mean", "max"],
                    help="how the position axis is collapsed (node is unaffected)")
    ap.add_argument("--method", default="mattr-adam", choices=list(METHOD_KEY))
    ap.add_argument("--corr", default="spearman", choices=["spearman", "pearson"],
                    help="spearman ranks each task's profile first, so the figure reports "
                         "agreement on the ORDER of units; pearson reports agreement on their "
                         "MAGNITUDES, which on the SAE basis is mostly a statement about whether "
                         "the same few extreme values appear in both tasks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    key = METHOD_KEY[a.method]
    msuf = "" if a.method == "mattr-adam" else f"_{a.method}"
    out = a.out or (f"plots/task_corr_substrates{msuf}"
                    f"{'' if a.corr == 'spearman' else '_pearson'}"
                    f"{'' if a.agg == 'mean' else '_max'}.pdf")

    rows, missing = [], []
    for sub, slabel, width in SUBS:
        # Each substrate reads the tree the published figures read: MLP and SAE at 5,000 MAttr
        # steps, SAE under --sae-error frozen, node at 2,000. A ranking is a property of its RUN,
        # so correlating a wave nobody plots would describe circuits nobody reports.
        res = V.SUBSTRATE_RES.get(sub, RES)
        prof = {}
        for t in TASKS:
            v = profile(t, sub, width, a.agg, res, key, a.corr)
            if v is not None:
                prof[t] = v
        if len(prof) < 2:
            missing.append(slabel)
            continue
        for ta in TASKS:
            for tb in TASKS:
                if ta not in prof or tb not in prof:
                    continue
                r = float(np.corrcoef(prof[ta], prof[tb])[0, 1])
                rows.append({"sub": slabel, "a": ta.replace("_", " "),
                             "b": tb.replace("_", " "), "r": r,
                             "lab": f"{r:.2f}".replace("0.", ".").replace("-.", "−.")})
        del prof
    if not rows:
        raise SystemExit(f"no {key} runs found")
    df = pd.DataFrame(rows)
    order = [t.replace("_", " ") for t in TASKS]
    df["sub"] = pd.Categorical(df["sub"], [sl for _, sl, _ in SUBS if sl in set(df["sub"])])
    lab = "Spearman $\\rho$" if a.corr == "spearman" else "Pearson $r$"

    p = (ggplot(df, aes("a", "b", fill="r")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="lab"), size=3.6)
         # Same diverging ramp and orientation as the other correlation heatmaps in this paper,
         # so a reader moving between them does not have to relearn the colour.
         + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                                midpoint=0, limits=[-1, 1], na_value="#eeeeee")
         + scale_x_discrete(limits=order, expand=(0, 0))
         + scale_y_discrete(limits=order[::-1], expand=(0, 0))
         + facet_wrap("~sub", ncol=len(SUBS))
         + labs(fill=lab))
    p.save(out, dpi=300, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print(f"wrote {out}  (method={a.method}, key={key}, corr={a.corr}, agg={a.agg})")
    if missing:
        print(f"missing facets: {', '.join(missing)}")

    print(f"\nmean off-diagonal {a.corr} per substrate ({a.method}, agg={a.agg}) -- how "
          f"task-INDEPENDENT the attribution is:")
    for sname in df["sub"].cat.categories:
        d = df[(df["sub"] == sname) & (df["a"] != df["b"])]
        print(f"  {sname:<16} mean={d['r'].mean():+.3f}   min={d['r'].min():+.3f}   "
              f"max={d['r'].max():+.3f}   n_pairs={len(d) // 2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
