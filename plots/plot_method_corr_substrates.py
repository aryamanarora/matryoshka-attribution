"""Method x method Spearman rho between circuit RANKINGS, faceted by substrate.

THE QUESTION. plots/plot_method_corr_heatmap.py asks "do these methods find the same circuit?"
at MIB node level. This asks whether the answer depends on the VARIABLE SET: the same eight
SVA+ tasks, the same methods, on three substrates spanning 1,056 to 39.8M units.

*** RHO IS COMPUTED WITHIN A (TASK, SUBSTRATE) CELL AND ONLY THEN AVERAGED OVER TASKS. ***
That is what makes the per-position substrates usable at all. Two methods scored on the same
task share a layout exactly -- [layer][position][unit], same seq_len, same length -- so their
full score vectors are directly comparable and nothing has to be collapsed. (Contrast
plot_task_corr_substrates.py, which correlates ACROSS tasks and therefore must reduce the
position axis first, because two tasks have different seq_len and their vectors index different
objects.) Averaging rho over tasks rather than pooling units keeps each task's cell equally
weighted; pooling would let `hours` (39.8M units) outvote `simple` (3.1M) 13:1.

TREES ARE THE PUBLISHED ONES, imported from plot_accauc_vs_faithauc (SUBSTRATE_RES,
parse_method, METHODS), so this figure describes the runs the paper's scatter and curve figures
draw -- MLP and SAE at 5,000 MAttr steps, the SAE column under --sae-error frozen, node at
2,000. A method's ranking is a property of its RUN, so reading rho off a different wave than
the one plotted would compare circuits nobody reports.

WHAT THE SAE COLUMN IS MISSING, and it is coverage rather than a result: Expected Gradients, Node
Pruning and DBM have no SAE runs at any setting, so that facet is 5x5 where the others are 8x8.
main() prints the per-substrate method list every run.

SPEARMAN, NOT PEARSON. These score vectors have wildly different scales across methods (MAttr's
learned logits against IG's gradient-times-delta) and, on the SAE basis, distributions dominated
by a handful of extreme values. Pearson there would mostly report whether the same few outliers
appear in both. Rank correlation is the question actually being asked: do the two methods ORDER
the units the same way, which is what the top-k eval consumes.

llama3 only (every SVA+ task is llama3).

Run:  uv run python plots/plot_method_corr_substrates.py
Out:  plots/method_corr_substrates.pdf  (+ .png)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from plotnine import (ggplot, aes, geom_tile, geom_text, labs, facet_wrap, theme, theme_set,
                      theme_bw, element_text, element_blank, scale_fill_gradient2,
                      scale_x_discrete, scale_y_discrete)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_accauc_vs_faithauc as V                    # trees, parse_method, METHODS  # noqa: E402

LOSS = "logit_diff"
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
# (substrate key, facet label). Ordered by unit count, the axis the figure is read along.
SUBS = [("node", "Node"), ("mlp", "MLP neuron"), ("mlp_sae_span", "SAE (MLP out)")]
# Draw order = the curve figure's, so a reader moving between them meets the methods in one
# sequence. Random is kept: a method that correlates with it is reporting noise, which is the
# most useful single reference on the SAE facet.
ORDER = ["Random", "IxG", "IG", "mc_ig", "eprun-s090", "sig_lr0.3_l16.0",
         "stopk-log-eps1e-2", "softsgd-log"]

theme_set(
    theme_bw(base_size=8)
    + theme(
        text=element_text(color="#000", family="Inter"),
        figure_size=(5.5, 2.1),
        axis_title=element_blank(),
        axis_text=element_text(size=5),
        axis_text_x=element_text(size=5, rotation=45, hjust=1),
        legend_text=element_text(size=5.5), legend_title=element_text(size=6),
        legend_key_size=8,
        panel_grid_major=element_blank(), panel_grid_minor=element_blank(),
        panel_spacing_x=0.03, strip_background=element_blank(),
        strip_text=element_text(size=7),
    )
)


def scores_for(res, sub, task):
    """{method key: rank vector} for one (tree, substrate, task).

    *** THE RUN JSON IS THE SOURCE OF TRUTH, NOT THE FILENAME. *** Two bugs came from matching on
    the filename instead, and neither raised -- both showed up as blank cells:
      LOSS. parse_method's key does not encode the training loss, so `..._acc_bs1` and
        `..._ce_bs1` collide with the logit-diff run. Taking the first alphabetical match gave
        MAttr+SGD its ACC-loss circuit (`sgd_acc_bs1` sorts before `sgd_bs1_s5000`).
      SUBSTRATE. `_llama3_mlp_` is a SUBSTRING of `_llama3_mlp_sae_span_`, so the MLP facet was
        picking up SAE files -- 6,291,648 units against MLP's 2,752,512, which the length guard
        below then skipped, blanking every MAttr+SGD pair in that facet.
    Reading d off the json and testing d["nodes"] / d["loss"] exactly is what V.load does.

    Ranked ONCE per method rather than inside the pair loop: rankdata on `hours`'s 39.8M elements
    is the expensive step, and the pair loop would otherwise redo it 4-7 times per method.
    Pearson-on-ranks is Spearman, so the pairs are then a cheap corrcoef.
    """
    out = {}
    for f in sorted(os.listdir(res)):
        if not f.endswith(".json") or not f.startswith(f"{task}_llama3_"):
            continue
        try:
            d = json.load(open(os.path.join(res, f)))
        except Exception:
            continue
        if d.get("nodes") != sub or d.get("loss") != LOSS or not V.on_model(d):
            continue
        m = V.parse_method(f, d)
        if m is None or m not in ORDER or m in out:
            continue
        sp = os.path.join(res, f[:-len(".json")] + ".scores.pt")
        if not os.path.exists(sp):
            continue
        out[m] = rankdata(torch.load(sp, map_location="cpu").float().numpy())
    return out


def main():
    rows, cover = [], {}
    for sub, slabel in SUBS:
        res = V.SUBSTRATE_RES.get(sub, "results/sva_sweep")
        acc = {}
        for task in TASKS:
            R = scores_for(res, sub, task)
            for a in R:
                for b in R:
                    if len(R[a]) != len(R[b]):      # different seq_len: not comparable
                        continue
                    acc.setdefault((a, b), []).append(float(np.corrcoef(R[a], R[b])[0, 1]))
            del R
        cover[slabel] = sorted({a for a, _ in acc}, key=ORDER.index)
        for (a, b), vs in acc.items():
            rows.append({"sub": slabel, "a": V.METHODS[a][0], "b": V.METHODS[b][0],
                         "r": float(np.mean(vs)), "n": len(vs)})
        print(f"  {slabel:<15}{len(cover[slabel])} methods, "
              f"{', '.join(V.METHODS[m][0] for m in cover[slabel])}")
    if not rows:
        raise SystemExit("no score tensors found")

    df = pd.DataFrame(rows)
    df["lab"] = df["r"].map(lambda x: f"{x:.2f}".replace("0.", ".").replace("-.", "−."))
    order = [V.METHODS[m][0] for m in ORDER]
    df["sub"] = pd.Categorical(df["sub"], [s for _, s in SUBS])

    p = (ggplot(df, aes("a", "b", fill="r")) + geom_tile(color="white", size=0.3)
         + geom_text(aes(label="lab"), size=3.4)
         # Same diverging ramp and orientation as the other correlation heatmaps in this paper,
         # so a reader moving between them does not have to relearn the colour.
         + scale_fill_gradient2(low="#b2182b", mid="#f7f7f7", high="#2166ac",
                                midpoint=0, limits=[-1, 1], na_value="#eeeeee")
         + scale_x_discrete(limits=order, expand=(0, 0))
         + scale_y_discrete(limits=order[::-1], expand=(0, 0))
         + facet_wrap("~sub", ncol=len(SUBS))
         + labs(fill="Spearman $\\rho$"))
    out = "plots/method_corr_substrates.pdf"
    p.save(out, dpi=300, verbose=False)
    p.save(out.replace(".pdf", ".png"), dpi=200, verbose=False)
    print(f"\nwrote {out}  ({df['n'].max()} tasks averaged)")

    print("\nmean off-diagonal rho per substrate -- how much the methods agree at all:")
    for s in df["sub"].cat.categories:
        d = df[(df["sub"] == s) & (df["a"] != df["b"])]
        if len(d):
            print(f"  {s:<15}mean={d['r'].mean():+.3f}  min={d['r'].min():+.3f}  "
                  f"max={d['r'].max():+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
