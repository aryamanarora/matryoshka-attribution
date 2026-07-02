"""Eval loss-AUC over training (counterfactual pairs seen), faceted by task, one line per
method, at a fixed n. Learned methods (MAttr, id-STE, MAttr+IG, id-STE+IG) are snapshotted
during training; gradient methods (IxG, IG-5 global) are the running attribution vs #samples.
All x = pairs seen. Real eval-set loss-AUC (lower=better). Full-width -> paper/figs/.
Regenerate (~minutes): uv run python scripts/plot_bilinear_lossauc_curves.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotnine import (
    ggplot, aes, geom_line, labs, facet_wrap, scale_color_manual, scale_x_log10,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)
from mizani.formatters import label_log

sys.path.insert(0, "scripts")
from compare_bilinear_lossauc import fwd, eval_loss_auc
from toy_bilinear import make_instance
from learning_to_attribute import learn_scores, build_mask

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
N = 128
SEEDS = 3
STEPS = 3000
CKPTS = [50, 100, 200, 350, 600, 1000, 1600, 2200, 3000]
B = 2048
TASKS = [("Bilinear", 0.0, 0.0), ("Saturated gate ($\\beta$=8)", 8.0, 0.5)]


def curve_learn(seed, beta, tau, variant, opt, lr):
    a, c, n_lin, _ = make_instance(N, seed); ck = set(CKPTS); snaps = {}
    def loss_fn(mask):
        x = torch.randn(1, N)
        return ((fwd(a, c, n_lin, mask * x, beta, tau) - fwd(a, c, n_lin, x, beta, tau)) ** 2).mean()
    def on_step(step, k, lv, sc):
        if (step + 1) in ck:
            snaps[step + 1] = sc.detach().clone()
    learn_scores(N, loss_fn, steps=STEPS, variant=variant, k_schedule="uniform", lr=lr,
                 optimizer=opt, on_step=on_step)
    return [(s, eval_loss_auc(snaps[s].numpy(), a, c, n_lin, beta, tau, B=B)) for s in sorted(snaps)]


def curve_ig_intervene(seed, beta, tau, sigmoid, ig_steps=5, lr=0.05, T=0.5):
    torch.manual_seed(seed); a, c, n_lin, _ = make_instance(N, seed); ck = set(CKPTS); out = []
    scores = torch.nn.Parameter(torch.zeros(N))
    opt = (torch.optim.Adam if sigmoid else torch.optim.SGD)([scores], lr=lr if sigmoid else 0.05)
    al = torch.arange(1, ig_steps + 1).float() / ig_steps
    for step in range(STEPS):
        k = max(1, min(N, int(round(1 + (N - 1) * torch.rand(1).item()))))
        x = torch.randn(1, N); yc = fwd(a, c, n_lin, x, beta, tau).detach()
        if sigmoid:
            mask = build_mask(scores, float(k), "hard_topk", T=T).mask; mval = mask.detach()
        else:
            with torch.no_grad():
                mval = torch.zeros(N); mval[torch.topk(scores, k).indices] = 1.0
        Z = (al[None, :, None] * (mval[None, None, :] * x[:, None, :])).reshape(-1, N).detach().requires_grad_(True)
        L = (fwd(a, c, n_lin, Z, beta, tau) - yc.repeat_interleave(ig_steps)) ** 2
        g = torch.autograd.grad(L.sum(), Z)[0].reshape(1, ig_steps, N)
        a_ig = (x * g.mean(1)).mean(0).detach()
        opt.zero_grad()
        if sigmoid:
            (a_ig * mask).sum().backward()
        else:
            scores.grad = a_ig
        opt.step()
        if (step + 1) in ck:
            out.append((step + 1, eval_loss_auc(scores.detach().numpy(), a, c, n_lin, beta, tau, B=B)))
    return out


def curve_grad(seed, beta, tau, m):  # m=1 -> IxG at corrupt baseline; m>=1 -> IG-m global
    a, c, n_lin, _ = make_instance(N, seed)
    Nmax = STEPS; X = torch.randn(Nmax, N); Xcf = torch.zeros(Nmax, N); d = X - Xcf
    yc = fwd(a, c, n_lin, X, beta, tau).detach()
    al = torch.zeros(1) if m == 1 else torch.arange(1, m + 1).float() / m
    Z = (Xcf[:, None, :] + al[None, :, None] * d[:, None, :]).reshape(-1, N).clone().requires_grad_(True)
    M = (fwd(a, c, n_lin, Z, beta, tau) - yc.repeat_interleave(len(al))) ** 2
    g = torch.autograd.grad(M.sum(), Z)[0].reshape(Nmax, len(al), N)
    cum = (-(d * g.mean(1))).cumsum(0)
    return [(s, eval_loss_auc((cum[s - 1] / s).numpy(), a, c, n_lin, beta, tau, B=B)) for s in CKPTS]


METHODS = [
    ("IxG (1-step)",  "#ff7f00", lambda s, b, t: curve_grad(s, b, t, 1)),
    ("IG-5 (global)", "#4daf4a", lambda s, b, t: curve_grad(s, b, t, 5)),
    ("id-STE+SGD",    "#377eb8", lambda s, b, t: curve_learn(s, b, t, "hard_topk_identity", "sgd", 0.01)),
    ("id-STE $+$ IG", "#984ea3", lambda s, b, t: curve_ig_intervene(s, b, t, sigmoid=False)),
    ("MAttr $+$ IG",  "#17becf", lambda s, b, t: curve_ig_intervene(s, b, t, sigmoid=True)),
    ("MAttr",         "#e41a1c", lambda s, b, t: curve_learn(s, b, t, "hard_topk", "adam", 0.05)),
]

rows = []
for task, beta, tau in TASKS:
    for name, _, fn in METHODS:
        agg = {}
        for seed in range(SEEDS):
            for step, auc in fn(seed, beta, tau):
                agg.setdefault(step, []).append(auc)
        for step, vals in agg.items():
            rows.append({"task": task, "method": name, "step": step, "auc": float(np.mean(vals))})
        print(f"[{task}] {name} done")

df = pd.DataFrame(rows)
labels = [m[0] for m in METHODS]; colors = [m[1] for m in METHODS]
df["method"] = pd.Categorical(df["method"], categories=labels, ordered=True)
df["task"] = pd.Categorical(df["task"], categories=[t[0] for t in TASKS], ordered=True)

theme_set(
    theme_bw(base_size=8)
    + theme(text=element_text(color="#000", family="Inter"), figure_size=(5.5, 2.2),
            axis_title=element_text(size=7), axis_text=element_text(size=6),
            axis_text_x=element_text(size=6, rotation=45, hjust=0.5, vjust=1.0),
            panel_grid_major=element_line(size=0.25, color="#dddddd"),
            panel_grid_minor=element_blank(), panel_spacing_x=0.05,
            strip_background=element_blank(), strip_text=element_text(size=7),
            legend_title=element_text(size=7), legend_text=element_text(size=6),
            legend_key_size=6, legend_position="top", legend_direction="horizontal",
            legend_box_margin=0))
p = (ggplot(df, aes("step", "auc", color="method"))
     + geom_line(size=0.5)
     + facet_wrap("task", nrow=1, scales="free_y")
     + scale_color_manual(values=colors)
     + scale_x_log10(labels=label_log(base=10))
     + labs(x=f"Steps (counterfactual pairs seen), $n$={N}",
            y="Eval loss-AUC (lower is better)", color=""))
p.save(OUT / "bilinear_lossauc_curves.pdf", verbose=False)
print(f"Saved {OUT / 'bilinear_lossauc_curves.pdf'}")
