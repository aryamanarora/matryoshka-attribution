"""Final eval-set loss-AUC vs n, faceted by task (plain bilinear | saturated gate), one
Set1 line per method. Loss-AUC = area under the (true eval-set denoising loss vs top-k-kept-
clean) curve for each method's learned/attributed ranking; lower = better. Uses a real MC
eval batch (NOT the analytic shortcut, which is only exact for the mean-zero unsaturated
task) so it matches what MAttr trains on and is correct under the (non-mean-zero) gate.

Methods:
  IxG (1-step)        gradient*delta at the all-corrupt baseline (blind to products)
  IG-5 (global)       5-step integrated gradients, single global baseline
  id-STE+SGD          MAttr with identity STE = per-step IxG accumulated under interventions
  IG-5 (intervene)    NEW: per step integrate the selected nodes CF->clean (5 pts), delta*
                      path-avg-grad as the score gradient -> IG accumulated under interventions
  MAttr               soft sigmoid-topk forward (the learned mask baseline)

Regenerate (trains everything; ~minutes on CPU): uv run python scripts/compare_bilinear_lossauc.py
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from plotnine import (
    ggplot, aes, geom_line, geom_point, labs, facet_wrap,
    scale_color_manual, scale_x_log10,
    theme_bw, theme_set, theme, element_text, element_line, element_blank,
)

import sys
sys.path.insert(0, "scripts")
from toy_bilinear import make_instance
from learning_to_attribute import learn_scores, build_mask

OUT = Path("paper/figs"); OUT.mkdir(parents=True, exist_ok=True)
RES = Path("results"); RES.mkdir(parents=True, exist_ok=True)


def fwd(a, c, n_lin, xeff, beta, tau):
    lin = (a * xeff[:, :n_lin]).sum(-1)
    p = xeff[:, n_lin:]; xx = p[:, 0::2] * p[:, 1::2]
    gate = torch.sigmoid(beta * (xx - tau)) if beta > 0 else xx
    return lin + (c * gate).sum(-1)


def eval_loss_auc(scores, a, c, n_lin, beta, tau, B=8192, seed=999):
    """Real eval-set loss-AUC: fixed MC batch, true masked forward (CF=0), integrate over k."""
    n = n_lin + 2 * len(c); g = torch.Generator().manual_seed(seed)
    X = torch.randn(B, n, generator=g); yc = fwd(a, c, n_lin, X, beta, tau)
    order = np.argsort(-scores); curve = []; m = torch.zeros(n)
    for k in range(n + 1):
        if k > 0:
            m[order[k - 1]] = 1.0
        curve.append(float(((fwd(a, c, n_lin, X * m, beta, tau) - yc) ** 2).mean()))
    y = np.array(curve); y = y / y[0]
    return float((y[:-1] + y[1:]).sum() / 2 / n)


def train_mattr(n, seed, beta, tau, variant="hard_topk", opt="adam", lr=0.05, steps=3000):
    a, c, n_lin, _ = make_instance(n, seed)
    def loss_fn(mask):
        x = torch.randn(1, n)
        return ((fwd(a, c, n_lin, mask * x, beta, tau) - fwd(a, c, n_lin, x, beta, tau)) ** 2).mean()
    return learn_scores(n, loss_fn, steps=steps, variant=variant, k_schedule="uniform",
                        lr=lr, optimizer=opt).scores.numpy()


def attr_ixg(n, seed, beta, tau, N=8000):
    a, c, n_lin, _ = make_instance(n, seed)
    X = torch.randn(N, n); Xcf = torch.zeros(N, n)
    yc = fwd(a, c, n_lin, X, beta, tau).detach()
    Xc = Xcf.clone().requires_grad_(True)          # gradient at the all-corrupt baseline
    M = (fwd(a, c, n_lin, Xc, beta, tau) - yc) ** 2
    g = torch.autograd.grad(M.sum(), Xc)[0]
    return (-(X - Xcf) * g).mean(0).numpy()


def attr_ig_global(n, seed, beta, tau, m=5, N=8000):
    a, c, n_lin, _ = make_instance(n, seed)
    X = torch.randn(N, n); Xcf = torch.zeros(N, n); d = X - Xcf
    yc = fwd(a, c, n_lin, X, beta, tau).detach()
    al = torch.arange(1, m + 1).float() / m
    Z = (Xcf[:, None, :] + al[None, :, None] * d[:, None, :]).reshape(-1, n).clone().requires_grad_(True)
    M = (fwd(a, c, n_lin, Z, beta, tau) - yc.repeat_interleave(m)) ** 2
    gg = torch.autograd.grad(M.sum(), Z)[0].reshape(N, m, n)
    return (-(d * gg.mean(1)).mean(0)).numpy()


def train_ig_intervene(n, seed, beta, tau, ig_steps=5, steps=3000, lr=0.05):
    """Per step: sample top-k mask, integrate the SELECTED nodes CF->clean (ig_steps pts),
    use delta * path-avg grad as the score gradient (SGD). ig_steps=1 -> id-STE+SGD."""
    torch.manual_seed(seed); a, c, n_lin, _ = make_instance(n, seed)
    scores = torch.zeros(n, requires_grad=True); opt = torch.optim.SGD([scores], lr=lr)
    al = torch.arange(1, ig_steps + 1).float() / ig_steps
    for _ in range(steps):
        k = max(1, min(n, int(round(1 + (n - 1) * torch.rand(1).item()))))
        with torch.no_grad():
            m = torch.zeros(n); m[torch.topk(scores, k).indices] = 1.0
        x = torch.randn(1, n)                          # CF=0 -> delta = x
        yc = fwd(a, c, n_lin, x, beta, tau).detach()
        Z = (al[None, :, None] * (m[None, None, :] * x[:, None, :])).reshape(-1, n).detach().requires_grad_(True)
        L = (fwd(a, c, n_lin, Z, beta, tau) - yc.repeat_interleave(ig_steps)) ** 2
        g = torch.autograd.grad(L.sum(), Z)[0].reshape(1, ig_steps, n)
        opt.zero_grad(); scores.grad = (x * g.mean(1)).mean(0).detach(); opt.step()
    return scores.detach().numpy()


def train_ig_intervene_sigmoid(n, seed, beta, tau, ig_steps=5, steps=3000, lr=0.05, T=0.5):
    """IG-under-intervention routed through the SIGMOID STE (sigma'/T envelope + coupling)
    with Adam, instead of the raw identity STE with SGD. Same IG-integrated per-node effect
    a_j = delta_j * path-avg grad, but fed to scores via the differentiable sigmoid_topk mask."""
    torch.manual_seed(seed); a, c, n_lin, _ = make_instance(n, seed)
    scores = torch.nn.Parameter(torch.zeros(n)); opt = torch.optim.Adam([scores], lr=lr)
    al = torch.arange(1, ig_steps + 1).float() / ig_steps
    for _ in range(steps):
        k = max(1, min(n, int(round(1 + (n - 1) * torch.rand(1).item()))))
        mask = build_mask(scores, float(k), "hard_topk", T=T).mask   # hard fwd, sigmoid bwd
        x = torch.randn(1, n)
        yc = fwd(a, c, n_lin, x, beta, tau).detach()
        Z = (al[None, :, None] * (mask.detach()[None, None, :] * x[:, None, :])).reshape(-1, n).detach().requires_grad_(True)
        L = (fwd(a, c, n_lin, Z, beta, tau) - yc.repeat_interleave(ig_steps)) ** 2
        g = torch.autograd.grad(L.sum(), Z)[0].reshape(1, ig_steps, n)
        a_ig = (x * g.mean(1)).mean(0).detach()                     # IG-integrated dL/dm_j
        opt.zero_grad(); (a_ig * mask).sum().backward(); opt.step()  # -> sigmoid STE grad to scores
    return scores.detach().numpy()


METHODS = [
    ("IxG (1-step)",         "#ff7f00", lambda n, s, b, t: attr_ixg(n, s, b, t)),
    ("IG-5 (global)",        "#4daf4a", lambda n, s, b, t: attr_ig_global(n, s, b, t, 5)),
    ("id-STE+SGD",           "#377eb8", lambda n, s, b, t: train_mattr(n, s, b, t, "hard_topk_identity", "sgd", 0.01)),
    ("id-STE $+$ IG",        "#984ea3", lambda n, s, b, t: train_ig_intervene(n, s, b, t, 5)),
    ("MAttr $+$ IG",         "#17becf", lambda n, s, b, t: train_ig_intervene_sigmoid(n, s, b, t, 5)),
    ("MAttr",                "#e41a1c", lambda n, s, b, t: train_mattr(n, s, b, t)),
]
TASKS = [("Bilinear", 0.0, 0.0), ("Saturated gate ($\\beta$=8)", 8.0, 0.5)]
NS = [8, 16, 32, 64, 128]
SEEDS = 3


def main():
    rows = []
    for task, beta, tau in TASKS:
        for name, _, fn in METHODS:
            for n in NS:
                aucs = []
                for seed in range(SEEDS):
                    scores = fn(n, seed, beta, tau)
                    a, c, n_lin, _ = make_instance(n, seed)
                    aucs.append(eval_loss_auc(scores, a, c, n_lin, beta, tau))
                rows.append({"task": task, "method": name, "n": n,
                             "auc": float(np.mean(aucs)), "sd": float(np.std(aucs))})
                print(f"[{task}] {name:16s} n={n:4d}  loss-AUC={np.mean(aucs):.3f}")
    df = pd.DataFrame(rows)
    with open(RES / "compare_bilinear_lossauc.pkl", "wb") as f:
        pickle.dump(df, f)

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
    p = (ggplot(df, aes("n", "auc", color="method"))
         + geom_line(size=0.5) + geom_point(size=1.1)
         + facet_wrap("task", nrow=1, scales="free_y")
         + scale_color_manual(values=colors)
         + scale_x_log10(breaks=NS, labels=[str(n) for n in NS])
         + labs(x="Number of Terms $n$", y="Eval loss-AUC (lower is better)", color=""))
    p.save(OUT / "compare_bilinear_lossauc.pdf", verbose=False)
    print(f"Saved {OUT / 'compare_bilinear_lossauc.pdf'}")


if __name__ == "__main__":
    main()
