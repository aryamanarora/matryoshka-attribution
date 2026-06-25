"""Input x Gradient (attribution patching) baseline on the linear toy y = sum_i a_i x_i.

Instead of learning a mask (MAttr), score each node by the classic attribution-patching
estimate, averaged over counterfactual pairs:

    attr_i = E_pairs[ (z_i^clean - z_i^cf) * dM/dz_i |_corrupt ]

where z_i = a_i x_i is the node activation, M = (y - y_clean)^2 is the same denoising
squared-error metric MAttr minimizes, and the gradient is taken at the fully-corrupted
run (all nodes = cf). For this linear net dM/dz_i = 2(y_cf - y_clean) is the same for
every i, so attr_i = (z_i^clean - z_i^cf) * 2(y_cf - y_clean).

Expected value (x, x' ~ N(0,1) iid, Delta = x - x'):
    E[attr_i] = -4 a_i^2   (cross terms vanish)
so importance := -attr_i has E = 4 a_i^2 -> recovers the |a| ordering. Variance of each
single-pair estimate grows ~ sum_j a_j^2 ~ n, so more terms need more samples to converge.

NB on the metric: with the *raw linear output* as the metric (dM/dz_i = 1) attr_i = delta_i
averages to 0 -- the same zero-mean failure as "just maximize the output". The nonlinear
(squared-error) metric is what makes delta*grad informative.

We track the SAME metrics as MAttr (Spearman / top-k overlap / pairwise acc) as a function
of the number of averaged samples, to compare convergence to the true ranking.

Saves results/toy_linear_ixg.pkl. Plot/compare with plot_toy_linear_ixg.py.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from toy_linear_mattr import steps_to_threshold


def ixg_one(n, num_samples=2000, eval_every=20, topk_frac=0.25, seed=0):
    """Accumulate the IxG/attribution-patching estimate; trace recovery vs #samples."""
    torch.manual_seed(seed)

    a = torch.randn(n)
    abs_a = a.abs()
    true_order = abs_a.argsort(descending=True)
    m_top = max(1, round(topk_frac * n))
    true_top = set(true_order[:m_top].tolist())
    true_ranks = torch.zeros(n)
    true_ranks[true_order] = torch.arange(n, dtype=torch.float)
    ut = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)
    n_pairs = n * (n - 1) // 2

    # vectorized over samples
    X = torch.randn(num_samples, n)
    Xcf = torch.randn(num_samples, n)
    Zc = a * X                              # [N, n] clean node activations
    Zcf = a * Xcf                           # [N, n] corrupted node activations
    g = 2.0 * (Zcf.sum(1) - Zc.sum(1))      # [N] dM/dz_i at corrupt (same for all i)
    attr = (Zc - Zcf) * g[:, None]          # [N, n] delta * grad
    cum = attr.cumsum(0)                    # running sum over samples

    checkpoints = list(range(eval_every, num_samples + 1, eval_every))
    if checkpoints[0] != 1:
        checkpoints = [1] + checkpoints

    samples, spearman, topk_overlap, pairwise = [], [], [], []
    for s in checkpoints:
        importance = -cum[s - 1] / s        # E[importance] = 4 a^2
        corr, _ = spearmanr(importance.numpy(), abs_a.numpy())
        lo = importance.argsort(descending=True)
        learned_ranks = torch.zeros(n)
        learned_ranks[lo] = torch.arange(n, dtype=torch.float)
        ld = learned_ranks.unsqueeze(1) - learned_ranks.unsqueeze(0)
        pa = ((true_diff[ut] * ld[ut]) > 0).sum().item() / n_pairs
        ov = len(set(lo[:m_top].tolist()) & true_top) / m_top
        samples.append(s)
        spearman.append(float(corr))
        topk_overlap.append(ov)
        pairwise.append(pa)

    return {
        "n": n, "seed": seed, "method": "IxG",
        "steps": samples, "spearman": spearman, "topk_overlap": topk_overlap,
        "pairwise": pairwise, "m_top": m_top,
        "scores": (-cum[-1] / num_samples).numpy(), "a": a.numpy(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--samples", type=int, default=2000)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--topk_frac", type=float, default=0.25)
    p.add_argument("--thresh", type=float, default=0.9)
    p.add_argument("--output", default="results/toy_linear_ixg")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for n in args.n:
        for seed in range(args.seeds):
            r = ixg_one(n, num_samples=args.samples, eval_every=args.eval_every,
                        topk_frac=args.topk_frac, seed=seed)
            r["tts"] = steps_to_threshold(r["steps"], r["spearman"], args.thresh)
            r.pop("a", None); r.pop("scores", None)
            all_runs.append(r)
        fin = np.mean([rr["spearman"][-1] for rr in all_runs if rr["n"] == n])
        tts = [rr["tts"] for rr in all_runs if rr["n"] == n and rr["tts"] is not None]
        print(f"n={n:4d}  final Spearman={fin:.3f}  "
              f"mean samples_to_{args.thresh}={np.mean(tts):.0f}" if tts
              else f"n={n:4d}  final Spearman={fin:.3f}  (thresh not reached)")

    with open(f"{args.output}.pkl", "wb") as f:
        pickle.dump({"runs": all_runs, "args": vars(args)}, f)
    print(f"Saved raw results to {args.output}.pkl")


if __name__ == "__main__":
    main()
