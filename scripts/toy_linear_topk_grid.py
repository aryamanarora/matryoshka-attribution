"""Multi-k overlap convergence traces for the method x k grid plot, at a fixed n.

For each k-schedule (uniform / log / sum_pow2) and each swept hyperparam value, runs MAttr
and records overlap@k (fraction of the true top-k recovered) over training steps, for
several k levels. Two sweeps:
  - LR sweep  (lines = learning rate, batch fixed = 1)
  - batch sweep (lines = batch size, lr fixed = 0.05)

Saves results/toy_linear_topk_grid_{lr,batch}.pkl. Plot with plot_toy_linear_topk_grid.py.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from toy_linear_mattr import train_one

METHODS = ["uniform", "log", "sum_pow2"]


def run_sweep(which, n, k_levels, lrs, batches, seeds, steps, eval_every):
    runs = []
    if which == "lr":
        combos = [(m, lr, 1) for m in METHODS for lr in lrs]
        xkey = "lr"
    else:
        combos = [(m, 0.05, b) for m in METHODS for b in batches]
        xkey = "batch"
    for method, lr, batch in combos:
        for seed in range(seeds):
            r = train_one(n, lr=lr, num_steps=steps, batch=batch, seed=seed,
                          eval_every=eval_every, k_schedule=method,
                          k_eval_levels=k_levels)
            runs.append({"method": method, "lr": lr, "batch": batch, "seed": seed,
                         "steps": r["steps"], "topk_multi": r["topk_multi"]})
        x = lr if xkey == "lr" else batch
        fin = np.mean([runs[-1 - s]["topk_multi"][k_levels[-1]][-1] for s in range(seeds)])
        print(f"{which} {method:8s} {xkey}={x:<6g}  final overlap@{k_levels[-1]}={fin:.3f}")
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--k_levels", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--lrs", type=float, nargs="+",
                   default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--output", default="results/toy_linear_topk_grid")
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    for which in ["lr", "batch"]:
        runs = run_sweep(which, args.n, args.k_levels, args.lrs, args.batches,
                         args.seeds, args.steps, args.eval_every)
        out = f"{args.output}_{which}.pkl"
        with open(out, "wb") as f:
            pickle.dump({"runs": runs, "args": vars(args), "which": which}, f)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
