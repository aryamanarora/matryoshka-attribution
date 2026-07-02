"""Bilinear toy where IxG is *guaranteed* suboptimal (unlike the linear toy y=Σa_i x_i,
on which averaged IxG converges to the true |a| ordering).

    y = Σ_{i∈L} a_i x_i  +  Σ_{(p,q)∈Pairs} c_j · x_p x_q          x_i ~ N(0,1)

with a ZERO / mean-ablation counterfactual (x' = 0). At the zero baseline the
attribution-patching gradient through a product node is  dy/dx_p = c_j x_q = 0  (the
partner is also zeroed), so IxG assigns *exactly zero* importance to every product node
no matter how many samples it averages. The linear anchors (n//4 of the nodes) are the
only nodes IxG scores; it misses the entire product circuit. MAttr optimizes the actual
top-k denoising loss and recovers the pair-strength ordering.

Ground-truth importance = marginal denoising error (corrupt one node, rest clean):
  linear node i -> a_i^2 ;  product node in pair j -> c_j^2.
Node-node products are supermodular (a node is worthless unless its partner is also kept
clean), so even MAttr's ceiling is < 1 here -- but it beats IxG's ~0 at every n.

Records Spearman(scores, true importance) vs steps (MAttr) / #samples (IxG), same schema
as toy_linear_mattr, so scripts/plot_toy_linear_convrate_facet.py bilinear reads it.
Regenerate: uv run python scripts/toy_bilinear.py --method mattr --k_schedule uniform ...
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from learning_to_attribute import learn_scores
from toy_linear_mattr import steps_to_threshold


def make_instance(n, seed):
    """Split n nodes into n//4 linear anchors + product pairs; return coeffs + GT imp."""
    torch.manual_seed(seed)
    n_lin = max(2, n // 4)
    if (n - n_lin) % 2:                      # keep the product block even
        n_lin += 1
    n_pairs = (n - n_lin) // 2
    a_lin = torch.randn(n_lin)
    c_pair = torch.randn(n_pairs)
    imp = torch.empty(n)
    imp[:n_lin] = a_lin ** 2                  # marginal denoising importance
    imp[n_lin:] = c_pair.repeat_interleave(2) ** 2
    return a_lin, c_pair, n_lin, imp


def mean_pct(scores, idx):
    """Mean percentile rank (0-100) of nodes `idx`: fraction of all nodes they score above."""
    s = np.asarray(scores, dtype=float)
    return float(np.mean([(s < s[i]).mean() for i in idx]) * 100.0)


def forward(a_lin, c_pair, n_lin, xeff):
    """y = Σ a_i x_i + Σ c_j x_p x_q on effective (masked) node values xeff [batch, n]."""
    lin = (a_lin * xeff[:, :n_lin]).sum(-1)
    p = xeff[:, n_lin:]
    prod = (c_pair * (p[:, 0::2] * p[:, 1::2])).sum(-1)
    return lin + prod


def train_one(n, lr=0.05, num_steps=3000, batch=1, eval_every=20, seed=0,
              k_schedule="uniform", ste="sigmoid", opt="adam", cf="zero"):
    """MAttr on one bilinear instance; trace Spearman(scores, true imp) over steps."""
    a_lin, c_pair, n_lin, imp = make_instance(n, seed)
    imp_np = imp.numpy()
    prod_idx, lin_idx = range(n_lin, n), range(n_lin)
    steps, spearman, prod_pct, lin_pct = [], [], [], []

    def record(step, sc):
        if (step + 1) % eval_every != 0 and step != 0:
            return
        s = sc.detach().cpu().numpy()
        corr, _ = spearmanr(s, imp_np)
        steps.append(step + 1)
        spearman.append(float(corr) if not np.isnan(corr) else 0.0)
        prod_pct.append(mean_pct(s, prod_idx))
        lin_pct.append(mean_pct(s, lin_idx))

    def loss_fn(mask):
        x = torch.randn(batch, n)
        xcf = torch.zeros(batch, n) if cf == "zero" else torch.randn(batch, n)
        xeff = mask * x + (1 - mask) * xcf
        return ((forward(a_lin, c_pair, n_lin, xeff)
                 - forward(a_lin, c_pair, n_lin, x)) ** 2).mean()

    variant = "hard_topk_identity" if ste == "identity" else "hard_topk"
    learn_scores(n, loss_fn, steps=num_steps, variant=variant, k_schedule=k_schedule,
                 lr=lr, optimizer=opt, on_step=lambda step, k, lv, sc: record(step, sc))
    return {"n": n, "seed": seed, "steps": steps, "spearman": spearman,
            "prod_pct": prod_pct, "lin_pct": lin_pct}


def ixg_one(n, num_samples=4000, eval_every=40, seed=0, cf="zero"):
    """Attribution-patching (delta*grad at the corrupt baseline), accumulated over samples."""
    a_lin, c_pair, n_lin, imp = make_instance(n, seed)
    imp_np = imp.numpy()
    prod_idx, lin_idx = range(n_lin, n), range(n_lin)
    X = torch.randn(num_samples, n)
    Xcf = torch.zeros(num_samples, n) if cf == "zero" else torch.randn(num_samples, n)
    y_clean = forward(a_lin, c_pair, n_lin, X).detach()
    Xc = Xcf.clone().requires_grad_(True)     # gradient taken at the corrupt run
    M = (forward(a_lin, c_pair, n_lin, Xc) - y_clean) ** 2
    g = torch.autograd.grad(M.sum(), Xc)[0]   # [N, n] dM/dx_i at corrupt
    attr = (X - Xcf) * g                       # delta * grad
    cum = attr.cumsum(0)

    checkpoints = [1] + list(range(eval_every, num_samples + 1, eval_every))
    steps, spearman, prod_pct, lin_pct = [], [], [], []
    for s in checkpoints:
        importance = (-cum[s - 1] / s).numpy()
        corr, _ = spearmanr(importance, imp_np)
        steps.append(s)
        spearman.append(float(corr) if not np.isnan(corr) else 0.0)
        prod_pct.append(mean_pct(importance, prod_idx))
        lin_pct.append(mean_pct(importance, lin_idx))
    return {"n": n, "seed": seed, "steps": steps, "spearman": spearman,
            "prod_pct": prod_pct, "lin_pct": lin_pct}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="mattr", choices=["mattr", "ixg"])
    p.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128, 256])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--steps", type=int, default=3000, help="MAttr steps / IxG samples")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--thresh", type=float, default=0.6)
    p.add_argument("--k_schedule", default="uniform", choices=["uniform", "log"])
    p.add_argument("--ste", default="sigmoid", choices=["sigmoid", "identity"])
    p.add_argument("--opt", default="adam", choices=["adam", "sgd"])
    p.add_argument("--cf", default="zero", choices=["zero", "resample"],
                   help="counterfactual: zero (mean-ablation; kills IxG on products) or "
                        "resample (~N(0,1); IxG then only mis-scales, stays competitive)")
    p.add_argument("--output", default="results/toy_bilinear_mattr")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for n in args.n:
        for seed in range(args.seeds):
            if args.method == "mattr":
                r = train_one(n, lr=args.lr, num_steps=args.steps, batch=args.batch,
                              eval_every=args.eval_every, seed=seed,
                              k_schedule=args.k_schedule, ste=args.ste, opt=args.opt, cf=args.cf)
            else:
                r = ixg_one(n, num_samples=args.steps, eval_every=max(1, args.steps // 100),
                            seed=seed, cf=args.cf)
            r["tts"] = steps_to_threshold(r["steps"], r["spearman"], args.thresh)
            all_runs.append(r)
        fin = np.mean([rr["spearman"][-1] for rr in all_runs if rr["n"] == n])
        print(f"n={n:4d}  final Spearman={fin:.3f}")

    with open(f"{args.output}.pkl", "wb") as f:
        pickle.dump({"runs": all_runs, "args": vars(args)}, f)
    print(f"Saved raw results to {args.output}.pkl")


if __name__ == "__main__":
    main()
