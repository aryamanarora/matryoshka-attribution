"""Toy QUADRATIC model: y = sum_i a_i x_i  +  sum_{i<=j} b_ij x_i x_j.

Every linear monomial (a_i x_i) AND every quadratic monomial (b_ij x_i x_j, including the
squares x_i^2) is its own node in the computation graph. We learn one MAttr score per node,
exactly as in the linear toy (toy_linear_mattr.py), with the same denoising / sufficient
counterfactual: top-k nodes keep their CLEAN value, the complement gets the CF value.

    N = n + n(n+1)/2  nodes   (n linear + n(n+1)/2 quadratic incl. squares)
    y_mask = sum_node [ m * clean_val + (1-m) * cf_val ];  loss = (y_mask - y_clean)^2

Ground-truth importance of a node = expected squared denoising error from dropping it,
E[(cf_val - clean_val)^2], with x, x' ~ N(0,1) iid. Analytically:
    linear  a_i x_i        : 2 a_i^2
    cross   b_ij x_i x_j    : 2 b_ij^2   (i<j)
    square  b_ii x_i^2      : 4 b_ii^2
i.e. coeff^2 * factor, factor=4 for squares else 2. (Squares get a boost from E[x^4]=3.)
Recovery = Spearman / top-k overlap / pairwise of learned scores vs this importance.

Methods: MAttr uniform-k, MAttr log-k, IxG (attribution patching on the same MSE metric).
n up to 64 only (N up to 2144). Same convergence test as the linear toy.

  uv run python scripts/toy_quadratic_mattr.py --method uniform
  uv run python scripts/toy_quadratic_mattr.py --method log
  uv run python scripts/toy_quadratic_mattr.py --method ixg
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
    """Return (coeffs [N], importance [N], quad_ii, quad_jj) for the quadratic toy."""
    torch.manual_seed(seed)
    quad_ii, quad_jj = torch.triu_indices(n, n, offset=0)   # i<=j incl. diagonal
    num_quad = quad_ii.numel()
    N = n + num_quad
    coeffs = torch.randn(N)
    factor = torch.full((N,), 2.0)
    is_square = (quad_ii == quad_jj)                        # within quad block
    factor[n:][is_square] = 4.0                             # square terms x_i^2
    importance = coeffs ** 2 * factor                       # analytic E[delta^2]
    return coeffs, importance, quad_ii, quad_jj, N


def node_values(x, coeffs, quad_ii, quad_jj, n):
    """x: [B, n] -> node values [B, N] (linear block then quadratic block)."""
    lin = x * coeffs[:n]                                    # [B, n]
    quad = x[:, quad_ii] * x[:, quad_jj] * coeffs[n:]       # [B, num_quad]
    return torch.cat([lin, quad], dim=-1)


def _rank_metrics(scores, importance, true_order, true_top, m_top,
                  ut, true_diff, n_pairs):
    corr, _ = spearmanr(scores.numpy(), importance.numpy())
    lo = scores.argsort(descending=True)
    N = scores.numel()
    learned_ranks = torch.zeros(N)
    learned_ranks[lo] = torch.arange(N, dtype=torch.float)
    ld = learned_ranks.unsqueeze(1) - learned_ranks.unsqueeze(0)
    pa = ((true_diff[ut] * ld[ut]) > 0).sum().item() / n_pairs
    ov = len(set(lo[:m_top].tolist()) & true_top) / m_top
    return float(corr), ov, pa


def train_one(n, lr=0.05, T=0.5, n_iters=50, num_steps=6000, batch=1,
              eval_every=40, topk_frac=0.25, seed=0, k_schedule="uniform"):
    """MAttr on the quadratic toy. k_schedule in {uniform, log, sum_pow2}."""
    coeffs, importance, quad_ii, quad_jj, N = build_model(n, seed)
    true_order = importance.argsort(descending=True)
    m_top = max(1, round(topk_frac * N))
    true_top = set(true_order[:m_top].tolist())
    true_ranks = torch.zeros(N)
    true_ranks[true_order] = torch.arange(N, dtype=torch.float)
    ut = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)
    n_pairs = N * (N - 1) // 2

    pow2_ks, v = [], 1
    while v < N:
        pow2_ks.append(v); v *= 2
    if N - 1 >= 1 and (N - 1) not in pow2_ks:
        pow2_ks.append(N - 1)

    steps, spearman, topk_overlap, pairwise = [], [], [], []

    def record(step, s):
        if (step + 1) % eval_every != 0 and step != 0:
            return
        with torch.no_grad():
            corr, ov, pa = _rank_metrics(s.detach().cpu(), importance, true_order,
                                         true_top, m_top, ut, true_diff, n_pairs)
        steps.append(step + 1)
        spearman.append(corr); topk_overlap.append(ov); pairwise.append(pa)

    def denoise(mask, clean_nodes, cf_nodes, y_clean):
        y_mask = (mask * clean_nodes + (1 - mask) * cf_nodes).sum(dim=-1)
        return ((y_mask - y_clean) ** 2).mean()

    if k_schedule == "sum_pow2":  # seam-bend: one data draw, loss summed over k-grid (caller-side)
        scores = nn.Parameter(torch.zeros(N))
        optimizer = torch.optim.Adam([scores], lr=lr)
        for step in range(num_steps):
            x, x_cf = torch.randn(batch, n), torch.randn(batch, n)
            clean_nodes = node_values(x, coeffs, quad_ii, quad_jj, n)
            cf_nodes = node_values(x_cf, coeffs, quad_ii, quad_jj, n)
            y_clean = clean_nodes.sum(dim=-1)
            loss = sum(denoise(build_mask(scores, float(kv), "hard_topk", T=T, n_iters=n_iters).mask,
                               clean_nodes, cf_nodes, y_clean) for kv in pow2_ks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            record(step, scores.data)
        final_scores = scores.data
    else:
        def loss_fn(mask):
            x, x_cf = torch.randn(batch, n), torch.randn(batch, n)
            clean_nodes = node_values(x, coeffs, quad_ii, quad_jj, n)
            cf_nodes = node_values(x_cf, coeffs, quad_ii, quad_jj, n)
            return denoise(mask, clean_nodes, cf_nodes, clean_nodes.sum(dim=-1))

        res = learn_scores(N, loss_fn, steps=num_steps, variant="hard_topk",
                           k_schedule=k_schedule, T=T, n_iters=n_iters, lr=lr,
                           on_step=lambda step, k, lv, sc: record(step, sc.data))
        final_scores = res.scores

    return {"n": n, "seed": seed, "N": N, "steps": steps, "spearman": spearman,
            "topk_overlap": topk_overlap, "pairwise": pairwise, "m_top": m_top,
            "scores": final_scores.numpy() if hasattr(final_scores, "numpy") else final_scores}


def ixg_one(n, num_samples=6000, eval_every=40, topk_frac=0.25, seed=0):
    """Attribution patching (IxG) on the same MSE metric, gradient at corrupt run.

    M = (y - y_clean)^2, dM/dz_node = 2(y_cf - y_clean) (same for all nodes since
    y = sum of node values). attr_node = (clean_val - cf_val) * dM/dz; importance := -attr.
    """
    coeffs, importance, quad_ii, quad_jj, N = build_model(n, seed)
    true_order = importance.argsort(descending=True)
    m_top = max(1, round(topk_frac * N))
    true_top = set(true_order[:m_top].tolist())
    true_ranks = torch.zeros(N)
    true_ranks[true_order] = torch.arange(N, dtype=torch.float)
    ut = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)
    n_pairs = N * (N - 1) // 2

    X = torch.randn(num_samples, n)
    Xcf = torch.randn(num_samples, n)
    Zc = node_values(X, coeffs, quad_ii, quad_jj, n)        # [S, N]
    Zcf = node_values(Xcf, coeffs, quad_ii, quad_jj, n)
    g = 2.0 * (Zcf.sum(1) - Zc.sum(1))                      # [S]
    attr = (Zc - Zcf) * g[:, None]                          # [S, N]
    cum = attr.cumsum(0)

    checkpoints = list(range(eval_every, num_samples + 1, eval_every))
    if checkpoints[0] != 1:
        checkpoints = [1] + checkpoints

    steps, spearman, topk_overlap, pairwise = [], [], [], []
    for s in checkpoints:
        imp_hat = -cum[s - 1] / s
        corr, ov, pa = _rank_metrics(
            imp_hat, importance, true_order, true_top, m_top, ut, true_diff, n_pairs)
        steps.append(s); spearman.append(corr)
        topk_overlap.append(ov); pairwise.append(pa)

    return {"n": n, "seed": seed, "N": N, "steps": steps, "spearman": spearman,
            "topk_overlap": topk_overlap, "pairwise": pairwise, "m_top": m_top}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["uniform", "log", "sum_pow2", "ixg"])
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=40)
    p.add_argument("--topk_frac", type=float, default=0.25)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    out = args.output or f"results/toy_quadratic_{args.method}"
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
        print(f"{args.method:8s} n={n:3d} (N={runs[-1]['N']:4d})  final Spearman={fin:.3f}")

    with open(f"{out}.pkl", "wb") as f:
        pickle.dump({"runs": runs, "args": vars(args)}, f)
    print(f"Saved {out}.pkl")


if __name__ == "__main__":
    main()
