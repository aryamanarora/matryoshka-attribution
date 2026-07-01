"""Toy linear task for MAttr: y = sum_i a_i * x_i.

Each product node a_i*x_i is a node in the computation graph; we learn one MAttr
score per node using the *exact* MIB MAttr setup (hard sigmoid top-k forward with
straight-through gradient, uniform-k sampling, denoising/sufficient counterfactual).

Setup
-----
- a_i ~ N(0, 1), fixed at init (the "model weights").
- x_i ~ N(0, 1), resampled per example. Counterfactual pairs (x clean, x' cf) are
  both drawn from this distribution.
- Denoising / sufficient intervention (matches all our MIB runs, see CLAUDE.md):
  the top-k nodes keep their CLEAN value a_i*x_i, the complement is patched with the
  CF value a_i*x'_i. Masked output:
      y_mask = sum_i [ m_i * a_i*x_i + (1 - m_i) * a_i*x'_i ]
  Target = clean output y_clean = sum_i a_i*x_i. Loss = (y_mask - y_clean)^2.
  => error = sum_{i not selected} a_i * (x'_i - x_i), so keeping the largest-|a_i|
     nodes clean minimizes expected error.

Ground truth
------------
Causal importance of node i is its expected squared contribution, ∝ a_i^2, i.e.
ordering nodes by |a_i| (sign is irrelevant to the denoising error). Recovery =
how well the learned scores reproduce that |a| ordering (Spearman / top-k / pairwise).

We sweep the number of terms n and track the recovery metric over training steps to
see (a) final recovery vs n and (b) how n affects convergence rate. MAttr hard-fwd +
uniform-k only, fixed LR.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from learning_to_attribute import learn_scores, build_mask


def train_one(n, lr=0.05, T=0.5, n_iters=50, num_steps=2000, batch=1,
              eval_every=20, topk_frac=0.25, seed=0, k_schedule="uniform",
              k_eval_levels=None, ste="sigmoid", opt="adam"):
    """Run MAttr on one linear toy instance; return convergence trace + final scores.

    k_schedule: "uniform" samples k ~ U(1, n); "log" samples k ~ exp(U(0, log n))
    (log-uniform), so 1-10 is as likely as 10-100 etc. Matches attribute.py:sample_k.
    k_eval_levels: optional list of k; if given, also record overlap@k vs the true
    top-k for each level over training (returned under "topk_multi": {k: [trace]}).
    ste: straight-through estimator for the hard top-k mask.
      "sigmoid"  -> backward through the soft sigmoid_topk gate (default MAttr).
      "identity" -> dm/ds = 1 for every node (hard_topk_identity from eval_mib): the
                    score gradient is purely g*delta_i per node, no sigmoid gate-slope.
    """
    import math
    torch.manual_seed(seed)

    # powers-of-two k grid {1,2,4,...} up to n-1, for k_schedule="sum_pow2"
    pow2_ks = []
    v = 1
    while v < n:
        pow2_ks.append(v)
        v *= 2
    if n - 1 >= 1 and (n - 1) not in pow2_ks:
        pow2_ks.append(n - 1)

    a = torch.randn(n)                       # fixed model weights
    abs_a = a.abs()
    true_order = abs_a.argsort(descending=True)
    m_top = max(1, round(topk_frac * n))
    true_top = set(true_order[:m_top].tolist())
    k_eval_levels = [k for k in (k_eval_levels or []) if k <= n]
    true_top_k = {k: set(true_order[:k].tolist()) for k in k_eval_levels}
    topk_multi = {k: [] for k in k_eval_levels}

    # true ranks (0 = most important) for pairwise accuracy
    true_ranks = torch.zeros(n)
    true_ranks[true_order] = torch.arange(n, dtype=torch.float)
    ut = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)
    n_pairs = n * (n - 1) // 2

    steps, spearman, topk_overlap, pairwise, losses = [], [], [], [], []

    def record(step, s, loss_val):
        """Append recovery metrics (Spearman / overlap@k / pairwise) at eval checkpoints."""
        if (step + 1) % eval_every != 0 and step != 0:
            return
        with torch.no_grad():
            s = s.detach().cpu()
            corr, _ = spearmanr(s.numpy(), abs_a.numpy())
            lo = s.argsort(descending=True)
            learned_ranks = torch.zeros(n)
            learned_ranks[lo] = torch.arange(n, dtype=torch.float)
            ld = learned_ranks.unsqueeze(1) - learned_ranks.unsqueeze(0)
            pa = ((true_diff[ut] * ld[ut]) > 0).sum().item() / n_pairs
            ov = len(set(lo[:m_top].tolist()) & true_top) / m_top
            lo_list = lo.tolist()
            for kk in k_eval_levels:
                topk_multi[kk].append(len(set(lo_list[:kk]) & true_top_k[kk]) / kk)
        steps.append(step + 1)
        spearman.append(float(corr))
        topk_overlap.append(ov)
        pairwise.append(pa)
        losses.append(loss_val)

    def denoise_mse(mask, clean_nodes, cf_nodes, y_clean):
        y_mask = (mask * clean_nodes + (1 - mask) * cf_nodes).sum(dim=-1)
        return ((y_mask - y_clean) ** 2).mean()

    if k_schedule == "sum_pow2":
        # Seam-bend (kept caller-side): ONE data draw per step, loss summed over the whole
        # k-grid {1,2,4,...,n-1}. Doesn't fit learn_scores's one-mask-per-step contract, so
        # run a thin dedicated loop here, still using the shared build_mask for each k.
        scores = nn.Parameter(torch.zeros(n))
        optimizer = (torch.optim.SGD([scores], lr=lr) if opt == "sgd"
                     else torch.optim.Adam([scores], lr=lr))
        variant = "hard_topk_identity" if ste == "identity" else "hard_topk"
        for step in range(num_steps):
            x, x_cf = torch.randn(batch, n), torch.randn(batch, n)
            clean_nodes, cf_nodes = a * x, a * x_cf
            y_clean = clean_nodes.sum(dim=-1)
            loss = sum(denoise_mse(build_mask(scores, float(kv), variant, T=T, n_iters=n_iters).mask,
                                   clean_nodes, cf_nodes, y_clean) for kv in pow2_ks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            record(step, scores.data, loss.item())
        final_scores = scores.data.cpu().numpy()
    else:
        variant = "hard_topk_identity" if ste == "identity" else "hard_topk"

        def loss_fn(mask):
            x, x_cf = torch.randn(batch, n), torch.randn(batch, n)
            clean_nodes, cf_nodes = a * x, a * x_cf
            return denoise_mse(mask, clean_nodes, cf_nodes, clean_nodes.sum(dim=-1))

        res = learn_scores(
            n, loss_fn, steps=num_steps, variant=variant, k_schedule=k_schedule,
            T=T, n_iters=n_iters, lr=lr, optimizer=opt,
            on_step=lambda step, k, lv, sc: record(step, sc.data, lv),
        )
        final_scores = res.scores.numpy()

    return {
        "n": n, "seed": seed, "lr": lr, "a": a.numpy(), "scores": final_scores,
        "steps": steps, "spearman": spearman, "topk_overlap": topk_overlap,
        "pairwise": pairwise, "loss": losses, "m_top": m_top,
        "topk_multi": topk_multi,
    }


def steps_to_threshold(steps, metric, thresh):
    """First step at which metric >= thresh (else None)."""
    for st, v in zip(steps, metric):
        if v >= thresh:
            return st
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256],
                   help="term counts to sweep")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch", type=int, default=1,
                   help="counterfactual pairs per step (1 = exact MIB setup)")
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--topk_frac", type=float, default=0.25,
                   help="fraction of nodes defining the top-k overlap metric")
    p.add_argument("--thresh", type=float, default=0.9,
                   help="Spearman threshold for convergence-rate measurement")
    p.add_argument("--k_schedule", default="uniform",
                   choices=["uniform", "log", "sum_pow2"],
                   help="k schedule: uniform (headline MAttr), log (+log k), or sum_pow2 "
                        "(sum loss over k=1,2,4,...,n-1 each step)")
    p.add_argument("--ste", default="sigmoid", choices=["sigmoid", "identity"],
                   help="straight-through estimator: sigmoid gate (default) or identity "
                        "(hard_topk_identity, g*delta gradient to every node)")
    p.add_argument("--opt", default="adam", choices=["adam", "sgd"],
                   help="optimizer for the mask scores; id-STE needs sgd (Adam normalizes "
                        "out the g*delta magnitude signal)")
    p.add_argument("--output", default="results/toy_linear_mattr")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for n in args.n:
        for seed in range(args.seeds):
            r = train_one(n, lr=args.lr, T=args.T, n_iters=args.n_iters,
                          num_steps=args.steps, batch=args.batch,
                          eval_every=args.eval_every, topk_frac=args.topk_frac,
                          seed=seed, k_schedule=args.k_schedule, ste=args.ste, opt=args.opt)
            r["tts"] = steps_to_threshold(r["steps"], r["spearman"], args.thresh)
            all_runs.append(r)
            print(f"n={n:4d} seed={seed}  final Spearman={r['spearman'][-1]:.3f}  "
                  f"top{r['m_top']}_overlap={r['topk_overlap'][-1]:.3f}  "
                  f"pairwise={r['pairwise'][-1]:.3f}  "
                  f"steps_to_{args.thresh}={r['tts']}")

    with open(f"{args.output}.pkl", "wb") as f:
        pickle.dump({"runs": all_runs, "args": vars(args)}, f)
    print(f"Saved raw results to {args.output}.pkl")

    # ---- aggregate by n ----
    ns = sorted(set(r["n"] for r in all_runs))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(ns)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: convergence curves (Spearman vs step), mean +/- band over seeds
    for color, n in zip(cmap, ns):
        runs = [r for r in all_runs if r["n"] == n]
        steps = runs[0]["steps"]
        sp = np.array([r["spearman"] for r in runs])
        mean, std = sp.mean(0), sp.std(0)
        axes[0].plot(steps, mean, color=color, label=f"n={n}", linewidth=1.5)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
    axes[0].axhline(args.thresh, color="gray", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel(r"Spearman(scores, $|a|$)")
    axes[0].set_title("Convergence of recovery")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)

    # Panel 2: final recovery metrics vs n
    final_sp = [np.mean([r["spearman"][-1] for r in all_runs if r["n"] == n]) for n in ns]
    final_sp_sd = [np.std([r["spearman"][-1] for r in all_runs if r["n"] == n]) for n in ns]
    final_ov = [np.mean([r["topk_overlap"][-1] for r in all_runs if r["n"] == n]) for n in ns]
    final_pw = [np.mean([r["pairwise"][-1] for r in all_runs if r["n"] == n]) for n in ns]
    axes[1].errorbar(ns, final_sp, yerr=final_sp_sd, marker="o", label="Spearman", capsize=3)
    axes[1].plot(ns, final_ov, marker="s", label=f"top-k overlap")
    axes[1].plot(ns, final_pw, marker="^", label="pairwise acc")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("n (number of terms)")
    axes[1].set_ylabel("Final recovery")
    axes[1].set_title(f"Recovery vs n (after {args.steps} steps)")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)

    # Panel 3: convergence rate (steps to reach threshold) vs n
    tts_mean, tts_sd, ns_conv = [], [], []
    for n in ns:
        tts = [r["tts"] for r in all_runs if r["n"] == n and r["tts"] is not None]
        if tts:
            ns_conv.append(n)
            tts_mean.append(np.mean(tts))
            tts_sd.append(np.std(tts))
    if ns_conv:
        axes[2].errorbar(ns_conv, tts_mean, yerr=tts_sd, marker="o", capsize=3)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xlabel("n (number of terms)")
    axes[2].set_ylabel(f"Steps to Spearman >= {args.thresh}")
    axes[2].set_title("Convergence rate vs n")

    fig.suptitle(f"MAttr (hard-fwd, uniform-k) on linear toy  y=Σ a_i x_i   "
                 f"lr={args.lr} T={args.T} batch={args.batch}", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{args.output}.png", dpi=150)
    print(f"Saved plot to {args.output}.png")


if __name__ == "__main__":
    main()
