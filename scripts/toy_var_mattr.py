"""Toy quadratic model with VARIABLE-level scores (n scores total, not per-term).

Same model as toy_quadratic_mattr.py:  y = sum_i a_i x_i + sum_{i<=j} b_ij x_i x_j,
but now we learn ONE MAttr score per input variable x_i (n scores), and the mask acts at
the INPUT level: a kept variable uses its clean value EVERYWHERE it appears (linear term +
every quadratic term involving it); a masked variable uses its CF value everywhere.

    x_eff_i = m_i * x_i^clean + (1 - m_i) * x_i^cf          (m: soft top-k over n vars)
    y_mask  = a . x_eff + sum_{i<=j} b_ij x_eff_i x_eff_j
    loss    = (y_mask - y_clean)^2                          (denoising / sufficient)

This is the realistic interventional setting: corrupting a variable propagates through all
terms that use it. Notably the output is NONLINEAR in x, so IxG's gradient dy/dx_i is NOT
constant (unlike the per-term toys) -- its first-order estimate can be biased, which is the
regime where the interventional MAttr objective may beat gradient attribution.

Ground-truth importance of variable i = expected squared output change from corrupting ONLY
x_i (single-variable noising), E[(y(x) - y(x with x_i->x_i'))^2]. Analytically (x ~ N(0,1)):
    imp_i = 2 a_i^2  +  2 * sum_{cross pairs (i,j), j!=i} b_ij^2  +  4 b_ii^2
(linear 2a^2; each cross term i is in contributes 2b^2; the square x_i^2 contributes 4b^2.)

Methods: MAttr uniform-k, MAttr log-k, IxG (attribution patching, gradient at corrupt run).
n up to 64. Same convergence test as the other toys.

  uv run python scripts/toy_var_mattr.py --method uniform
  uv run python scripts/toy_var_mattr.py --method log
  uv run python scripts/toy_var_mattr.py --method ixg
"""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from learning_to_attribute import learn_scores, build_mask


def build_model(n, seed):
    """Return (a [n], cq [num_quad], quad_ii, quad_jj, importance [n])."""
    torch.manual_seed(seed)
    a = torch.randn(n)
    quad_ii, quad_jj = torch.triu_indices(n, n, offset=0)   # i<=j incl. diagonal
    cq = torch.randn(quad_ii.numel())
    importance = 2.0 * a ** 2
    for c, i, j in zip(cq.tolist(), quad_ii.tolist(), quad_jj.tolist()):
        if i == j:
            importance[i] += 4.0 * c * c                    # square x_i^2
        else:
            importance[i] += 2.0 * c * c                    # cross terms
            importance[j] += 2.0 * c * c
    return a, cq, quad_ii, quad_jj, importance


def forward(x, a, cq, quad_ii, quad_jj):
    """x: [B, n] -> y: [B]."""
    lin = (x * a).sum(dim=-1)
    quad = (x[:, quad_ii] * x[:, quad_jj] * cq).sum(dim=-1)
    return lin + quad


def _rank_metrics(scores, importance, true_order, true_top, m_top,
                  ut, true_diff, n_pairs):
    corr, _ = spearmanr(scores.numpy(), importance.numpy())
    lo = scores.argsort(descending=True)
    n = scores.numel()
    learned_ranks = torch.zeros(n)
    learned_ranks[lo] = torch.arange(n, dtype=torch.float)
    ld = learned_ranks.unsqueeze(1) - learned_ranks.unsqueeze(0)
    pa = ((true_diff[ut] * ld[ut]) > 0).sum().item() / n_pairs
    ov = len(set(lo[:m_top].tolist()) & true_top) / m_top
    return float(corr), ov, pa


def _setup_truth(importance, topk_frac):
    n = importance.numel()
    true_order = importance.argsort(descending=True)
    m_top = max(1, round(topk_frac * n))
    true_top = set(true_order[:m_top].tolist())
    true_ranks = torch.zeros(n)
    true_ranks[true_order] = torch.arange(n, dtype=torch.float)
    ut = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)
    return true_order, true_top, m_top, ut, true_diff, n * (n - 1) // 2


def train_one(n, lr=0.05, T=0.5, n_iters=50, num_steps=4000, batch=1,
              eval_every=20, topk_frac=0.25, seed=0, k_schedule="uniform"):
    """MAttr with variable-level scores. k_schedule in {uniform, log, sum_pow2}."""
    a, cq, qii, qjj, importance = build_model(n, seed)
    true_order, true_top, m_top, ut, true_diff, n_pairs = _setup_truth(importance, topk_frac)

    pow2_ks, v = [], 1
    while v < n:
        pow2_ks.append(v); v *= 2
    if n - 1 >= 1 and (n - 1) not in pow2_ks:
        pow2_ks.append(n - 1)

    steps, spearman, topk_overlap, pairwise = [], [], [], []

    def record(step, s):
        if (step + 1) % eval_every != 0 and step != 0:
            return
        with torch.no_grad():
            corr, ov, pa = _rank_metrics(s.detach().cpu(), importance, true_order,
                                         true_top, m_top, ut, true_diff, n_pairs)
        steps.append(step + 1)
        spearman.append(corr); topk_overlap.append(ov); pairwise.append(pa)

    def denoise(mask, x, x_cf, y_clean):
        x_eff = mask * x + (1 - mask) * x_cf            # variable-level intervention
        return ((forward(x_eff, a, cq, qii, qjj) - y_clean) ** 2).mean()

    if k_schedule == "sum_pow2":  # seam-bend: one data draw, loss summed over k-grid (caller-side)
        scores = nn.Parameter(torch.zeros(n))
        optimizer = torch.optim.Adam([scores], lr=lr)
        for step in range(num_steps):
            x, x_cf = torch.randn(batch, n), torch.randn(batch, n)
            y_clean = forward(x, a, cq, qii, qjj)
            loss = sum(denoise(build_mask(scores, float(kv), "hard_topk", T=T, n_iters=n_iters).mask,
                               x, x_cf, y_clean) for kv in pow2_ks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            record(step, scores.data)
        final_scores = scores.data
    else:
        def loss_fn(mask):
            x, x_cf = torch.randn(batch, n), torch.randn(batch, n)
            return denoise(mask, x, x_cf, forward(x, a, cq, qii, qjj))

        res = learn_scores(n, loss_fn, steps=num_steps, variant="hard_topk",
                           k_schedule=k_schedule, T=T, n_iters=n_iters, lr=lr,
                           on_step=lambda step, k, lv, sc: record(step, sc.data))
        final_scores = res.scores

    return {"n": n, "seed": seed, "N": n, "steps": steps, "spearman": spearman,
            "topk_overlap": topk_overlap, "pairwise": pairwise, "m_top": m_top,
            "scores": final_scores.numpy() if hasattr(final_scores, "numpy") else final_scores}


def ixg_one(n, num_samples=4000, eval_every=20, topk_frac=0.25, seed=0, chunk=2000):
    """Attribution patching with variable-level nodes; gradient at the corrupt run.

    AP_i = (x_i^clean - x_i^cf) * dM/dx_i|_cf,  M = (y - y_clean)^2,
    dM/dx_i = 2(y_cf - y_clean) * dy/dx_i.  Here dy/dx_i is NOT constant (quadratic model),
    so this is a genuine first-order (linearized) estimate. importance := -mean(AP).
    """
    a, cq, qii, qjj, importance = build_model(n, seed)
    true_order, true_top, m_top, ut, true_diff, n_pairs = _setup_truth(importance, topk_frac)

    # accumulate AP over samples in chunks (autograd grad wrt corrupt inputs)
    ap_sum = torch.zeros(n)
    traces = {"steps": [], "spearman": [], "topk_overlap": [], "pairwise": []}
    done = 0
    checkpoints = set(range(eval_every, num_samples + 1, eval_every)) | {1, num_samples}
    next_cp = sorted(c for c in checkpoints if c <= num_samples)

    while done < num_samples:
        b = min(chunk, num_samples - done)
        Xc = torch.randn(b, n)
        Xcf = torch.randn(b, n, requires_grad=True)
        y_cf = forward(Xcf, a, cq, qii, qjj)
        grad = torch.autograd.grad(y_cf.sum(), Xcf)[0]            # [b, n] dy/dx_i at cf
        with torch.no_grad():
            y_clean = forward(Xc, a, cq, qii, qjj)
            ap = (Xc - Xcf) * grad * (2.0 * (y_cf.detach() - y_clean))[:, None]
        # running cumulative metrics at checkpoints inside this chunk
        ap_cum = ap.cumsum(0)
        for local in range(b):
            s = done + local + 1
            if s in checkpoints:
                imp_hat = -(ap_sum + ap_cum[local]) / s
                corr, ov, pa = _rank_metrics(imp_hat, importance, true_order,
                                             true_top, m_top, ut, true_diff, n_pairs)
                traces["steps"].append(s); traces["spearman"].append(corr)
                traces["topk_overlap"].append(ov); traces["pairwise"].append(pa)
        ap_sum += ap_cum[-1]
        done += b

    return {"n": n, "seed": seed, "N": n, **traces, "m_top": m_top}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["uniform", "log", "sum_pow2", "ixg"])
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--topk_frac", type=float, default=0.25)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out = args.output or f"results/toy_var_{args.method}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    runs = []
    for n in args.n:
        for seed in range(args.seeds):
            if args.method == "ixg":
                r = ixg_one(n, num_samples=args.steps, eval_every=args.eval_every,
                            topk_frac=args.topk_frac, seed=seed)
            else:
                r = train_one(n, lr=args.lr, num_steps=args.steps,
                              eval_every=args.eval_every, topk_frac=args.topk_frac,
                              seed=seed, k_schedule=args.method)
            runs.append(r)
        fin = np.mean([rr["spearman"][-1] for rr in runs if rr["n"] == n])
        print(f"{args.method:8s} n={n:3d}  final Spearman={fin:.3f}")

    with open(f"{out}.pkl", "wb") as f:
        pickle.dump({"runs": runs, "args": vars(args)}, f)
    print(f"Saved {out}.pkl")


if __name__ == "__main__":
    main()
