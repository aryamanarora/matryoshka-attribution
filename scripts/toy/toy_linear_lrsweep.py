"""LR sweep for MAttr on the linear toy (y = sum_i a_i x_i).

Reuses train_one from toy_linear_mattr.py. For each (n, lr, seed) we run MAttr
(hard-fwd, uniform-k) and record the convergence trace + final recovery, to see how
the learning rate trades off speed vs stability and how that interacts with n.

Saves results/toy_linear_mattr_lrsweep.pkl. Plot with plot_toy_linear_lrsweep.py.
"""

import argparse
import pickle
from pathlib import Path

from toy_linear_mattr import train_one, steps_to_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--lr", type=float, nargs="+",
                   default=[0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--n_iters", type=int, default=50)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--topk_frac", type=float, default=0.25)
    p.add_argument("--thresh", type=float, default=0.9)
    p.add_argument("--k_schedule", default="uniform",
                   choices=["uniform", "log", "sum_pow2"])
    p.add_argument("--output", default="results/toy_linear_mattr_lrsweep")
    args = p.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for n in args.n:
        for lr in args.lr:
            for seed in range(args.seeds):
                r = train_one(n, lr=lr, T=args.T, n_iters=args.n_iters,
                              num_steps=args.steps, batch=args.batch,
                              eval_every=args.eval_every, topk_frac=args.topk_frac,
                              seed=seed, k_schedule=args.k_schedule)
                r["tts"] = steps_to_threshold(r["steps"], r["spearman"], args.thresh)
                # drop bulky fields we don't need for LR plots
                r.pop("a", None); r.pop("scores", None)
                all_runs.append(r)
            import numpy as np
            fin = np.mean([rr["spearman"][-1] for rr in all_runs
                           if rr["n"] == n and rr["lr"] == lr])
            print(f"n={n:4d} lr={lr:<6g}  final Spearman={fin:.3f}")

    with open(f"{args.output}.pkl", "wb") as f:
        pickle.dump({"runs": all_runs, "args": vars(args)}, f)
    print(f"Saved raw results to {args.output}.pkl")


if __name__ == "__main__":
    main()
