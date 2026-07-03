"""Quick check: eval denoising loss vs training step for the variable-level quadratic toy.

Sidesteps the (ill-defined) ground-truth ranking by measuring the thing MAttr actually
optimizes: how well the current variable ranking denoises. At each checkpoint we take the
method's current scores, and for a fixed held-out set of CF pairs compute the hard top-k
denoising MSE (y_topk_clean - y_clean)^2 averaged over ALL sparsity levels k=1..n
(an AUC-style scalar; lower = better). Works identically for MAttr (uniform/log) and IxG.
A random-ranking baseline is shown for reference.

n up to 16. Saves results/toy_var_evalloss.pkl, plots paper/figs/toy_var_evalloss.pdf.
  uv run python scripts/toy_var_evalloss.py
"""
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from plotnine import (
    ggplot, aes, geom_line, labs, facet_wrap, scale_color_brewer,
    scale_linetype_manual, scale_x_log10, scale_y_log10, theme_bw, theme_set, theme,
    element_text, element_line, element_blank,
)
from mizani.formatters import label_log

from learning_to_attribute import learn_scores
from toy_var_mattr import build_model, forward

R = Path("results"); OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
R.mkdir(exist_ok=True)


def make_eval(n, a, cq, qii, qjj, n_eval=512, seed=0, base=10_000):
    """Fixed CF set; returns a fn rank_order -> mean-over-k denoising MSE.

    base=10_000 -> held-out eval subset; base=20_000 -> a (disjoint) train subset.
    Both are iid samples from the training distribution; the same all-k metric is used so
    the two curves isolate the generalization gap.
    """
    g = torch.Generator().manual_seed(base + seed)
    Xc = torch.randn(n_eval, n, generator=g)
    Xcf = torch.randn(n_eval, n, generator=g)
    y_clean = forward(Xc, a, cq, qii, qjj)                 # [n_eval]

    def eval_loss(order):
        losses = []
        for k in range(1, n + 1):
            m = torch.zeros(n)
            m[order[:k]] = 1.0
            x_eff = m * Xc + (1 - m) * Xcf
            y = forward(x_eff, a, cq, qii, qjj)
            losses.append(((y - y_clean) ** 2).mean().item())
        return float(np.mean(losses))                       # AUC over sparsity

    return eval_loss


def run_mattr(n, k_schedule, lr, num_steps, eval_every, seed):
    a, cq, qii, qjj, _ = build_model(n, seed)
    eval_loss = make_eval(n, a, cq, qii, qjj, seed=seed, base=10_000)   # held-out
    train_loss = make_eval(n, a, cq, qii, qjj, seed=seed, base=20_000)  # train subset
    steps, evals, trains = [], [], []

    def loss_fn(mask):
        x, x_cf = torch.randn(1, n), torch.randn(1, n)
        y_clean = forward(x, a, cq, qii, qjj)
        x_eff = mask * x + (1 - mask) * x_cf
        return ((forward(x_eff, a, cq, qii, qjj) - y_clean) ** 2).mean()

    def on_step(step, k, lv, sc):
        if (step + 1) % eval_every == 0 or step == 0:
            order = sc.data.argsort(descending=True)
            steps.append(step + 1)
            evals.append(eval_loss(order)); trains.append(train_loss(order))

    learn_scores(n, loss_fn, steps=num_steps, variant="hard_topk",
                 k_schedule=k_schedule, T=0.5, n_iters=50, lr=lr, on_step=on_step)
    return steps, evals, trains


def run_ixg(n, num_steps, eval_every, seed):
    a, cq, qii, qjj, _ = build_model(n, seed)
    eval_loss = make_eval(n, a, cq, qii, qjj, seed=seed, base=10_000)
    train_loss = make_eval(n, a, cq, qii, qjj, seed=seed, base=20_000)
    ap_sum = torch.zeros(n)
    steps, evals, trains = [], [], []
    cps = set(range(eval_every, num_steps + 1, eval_every)) | {1, num_steps}
    for s in range(1, num_steps + 1):
        Xc = torch.randn(1, n); Xcf = torch.randn(1, n, requires_grad=True)
        y_cf = forward(Xcf, a, cq, qii, qjj)
        grad = torch.autograd.grad(y_cf.sum(), Xcf)[0]
        with torch.no_grad():
            y_clean = forward(Xc, a, cq, qii, qjj)
            ap_sum += ((Xc - Xcf) * grad * (2.0 * (y_cf.detach() - y_clean))[:, None])[0]
        if s in cps:
            order = (-ap_sum / s).argsort(descending=True)
            steps.append(s); evals.append(eval_loss(order)); trains.append(train_loss(order))
    return steps, evals, trains


def random_baseline(n, seed, reps=20):
    a, cq, qii, qjj, _ = build_model(n, seed)
    eval_loss = make_eval(n, a, cq, qii, qjj, seed=seed)
    g = torch.Generator().manual_seed(seed)
    return float(np.mean([eval_loss(torch.randperm(n, generator=g)) for _ in range(reps)]))


def main():
    ns = [4, 8, 16]
    seeds = 3
    num_steps = 3000
    eval_every = 50
    methods = [("uniform", "MAttr (uniform $k$)"), ("log", "MAttr (log $k$)"),
               ("ixg", "IxG"), ("random", "Random")]

    rows = []
    for n in ns:
        for seed in range(seeds):
            rb = random_baseline(n, seed)
            for key, label in methods:
                if key in ("uniform", "log"):
                    st, ev, tr = run_mattr(n, key, 0.05, num_steps, eval_every, seed)
                elif key == "ixg":
                    st, ev, tr = run_ixg(n, num_steps, eval_every, seed)
                else:  # random: flat reference line (no train/eval distinction)
                    st, ev, tr = [1, num_steps], [rb, rb], [rb, rb]
                for s, e, t in zip(st, ev, tr):
                    rows.append({"n": n, "seed": seed, "method": label, "step": s,
                                 "eval": e, "train": t})
        print(f"n={n} done")

    df = pd.DataFrame(rows)
    with open(R / "toy_var_evalloss.pkl", "wb") as f:
        pickle.dump(df, f)

    # long form: one row per (split, ...) with split in {eval (held-out), train (subset)}
    longdf = df.melt(id_vars=["n", "seed", "method", "step"],
                     value_vars=["eval", "train"], var_name="split", value_name="loss")
    longdf["split"] = longdf["split"].map({"eval": "Held-out", "train": "Train subset"})
    agg = longdf.groupby(["n", "method", "split", "step"])["loss"].mean().reset_index()
    order = [l for _, l in methods]
    agg["method"] = pd.Categorical(agg["method"], categories=order, ordered=True)
    agg["split"] = pd.Categorical(agg["split"], categories=["Held-out", "Train subset"],
                                  ordered=True)
    agg["facet"] = pd.Categorical("$n = " + agg["n"].astype(str) + "$",
                                  categories=[f"$n = {n}$" for n in ns], ordered=True)

    theme_set(
        theme_bw(base_size=8)
        + theme(
            text=element_text(color="#000", family="Inter"),
            figure_size=(5.5, 1.9),
            axis_title=element_text(size=7), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(), panel_spacing_x=0.04,
            strip_background=element_blank(), strip_text=element_text(size=7),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=6, legend_position="top", legend_direction="horizontal",
            legend_box_margin=0,
        )
    )
    p = (
        ggplot(agg, aes("step", "loss", color="method", linetype="split"))
        + geom_line(size=0.5)
        + facet_wrap("facet", nrow=1, scales="free_y")
        + scale_color_brewer(type="qual", palette="Set1")
        + scale_linetype_manual(values={"Held-out": "solid", "Train subset": "dashed"})
        + scale_x_log10(labels=label_log(base=10))
        + scale_y_log10(labels=label_log(base=10))
        + labs(x="Training Step", y="Denoising Loss", color="", linetype="")
    )
    p.save(OUT / "toy_var_evalloss.pdf", verbose=False)
    print(f"Saved {OUT / 'toy_var_evalloss.pdf'}")


if __name__ == "__main__":
    main()
