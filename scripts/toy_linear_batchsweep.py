"""Batch-size sweep for MAttr on the linear toy (y = sum_i a_i x_i), LR fixed at 0.05.

Reuses train_one from toy_linear_mattr.py. For each (n, batch, seed) we run MAttr
(hard-fwd, uniform-k) and record convergence + final recovery. Larger batch = more
counterfactual pairs averaged per step = lower-variance gradient (at higher per-step
cost). NB: convergence is measured in *steps*; one step at batch B costs B forwards,
so steps-to-threshold * B = counterfactual pairs ("examples") seen.

Saves results/toy_linear_mattr_batchsweep.pkl. Plot with plot_toy_linear_batchsweep.py.
"""

import argparse
import pickle
from pathlib import Path

from toy_linear_mattr import train_one, steps_to_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--topk_frac", type=float, default=0.25)
    p.add_argument("--thresh", type=float, default=0.9)
    p.add_argument("--k_schedule", default="uniform",
                   choices=["uniform", "log", "sum_pow2"])
    p.add_argument("--output", default="results/toy_linear_mattr_batchsweep")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    import numpy as np
    all_runs = []
    for n in args.n:
        for batch in args.batch:
            for seed in range(args.seeds):
                r = train_one(n, lr=args.lr, T=args.T, n_iters=args.n_iters,
                              num_steps=args.steps, batch=batch,
                              eval_every=args.eval_every, topk_frac=args.topk_frac,
                              seed=seed, k_schedule=args.k_schedule)
                r["batch"] = batch
                r["tts"] = steps_to_threshold(r["steps"], r["spearman"], args.thresh)
                r.pop("a", None); r.pop("scores", None)
                all_runs.append(r)
            fin = np.mean([rr["spearman"][-1] for rr in all_runs
                           if rr["n"] == n and rr["batch"] == batch])
            print(f"n={n:4d} batch={batch:<3d}  final Spearman={fin:.3f}")

    with open(f"{args.output}.pkl", "wb") as f:
        pickle.dump({"runs": all_runs, "args": vars(args)}, f)
    print(f"Saved raw results to {args.output}.pkl")


if __name__ == "__main__":
    main()
