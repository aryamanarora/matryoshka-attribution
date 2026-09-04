"""Benchmark MAttr on the two runnable end-tasks of Bilodeau et al. (2022),
"Impossibility Theorems for Feature Attribution" (arXiv:2212.11870).

The paper proves that any COMPLETE and LINEAR attribution method (IG, SHAP) can be no better
than random guessing at these tasks, and shows empirically that non-complete local methods
(Gradient, SmoothGrad) are not so limited. MAttr is neither complete nor linear, so the
theorems do not cover it; the question this script answers is where it actually lands, and
at what QUERY BUDGET -- the axis the paper's Theorem 5.1 lower bound (~20k model evaluations
for perfect specificity at 90% sensitivity) is stated on.

Both tasks are hypothesis tests on the per-example min-max normalized attribution phi:

  recourse  oracle 1[ mean f over the UPPER half of the +-10%-range grid > the lower half ],
            test 1[ phi_j > alpha ]        (a SIGNED question)
  spurious  oracle 1[ var f over the whole grid > model's own 80th-quantile variance ],
            test 1[ |phi_j| > alpha ]      (a MAGNITUDE question)

MAttr arms, both using the headline variant (soft top-k forward, log-k schedule) over the p
input features, with the counterfactual distribution set to the paper's perturbation
distribution nu -- which is the whole reason the method has any purchase here:

  spurious  cf = x + delta, delta_j ~ U(-0.1 R_j, +0.1 R_j) resampled EVERY step;
            loss = (f(x_masked)_c - f(x)_c)^2, i.e. denoising/sufficiency exactly as in
            `toy_quadratic_mattr.py`. Score_j = "keeping j clean matters" = j's sensitivity
            under nu, which is the quantity the spurious oracle thresholds.
  recourse  a SIGNED two-sided mask: x_masked = m * x_plus + (1-m) * x_minus with
            x_plus = x + U(0, 0.1R), x_minus = x + U(-0.1R, 0) (the two halves of the
            oracle's grid), and loss = -f(x_masked)_c. Now m_j=1 means "take the up-shifted
            value", so score_j > 0 <=> shifting j up raises the output <=> oracle = 1. This
            is a genuinely new mask parameterisation, not the denoising one.

Handing MAttr nu is strictly more information than IG/SHAP get from a baseline point, so
"MAttr beats SHAP" is NOT the claim to make from this. The honest reference is `bruteforce`,
which spends its budget sampling nu directly and computing the oracle statistic -- that is
the Theorem 5.1 method. Both are traced against forward-passed ROWS so the comparison is a
query-efficiency one. Every method's budget is measured, not assumed (`CountingModel.rows`).

    uv run python scripts/impossibility_bench.py --datasets wine --n-models 2   # pilot
    uv run python scripts/impossibility_bench.py                                # full
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import impossibility_tasks as T                                    # noqa: E402
from learning_to_attribute.trainer import learn_scores             # noqa: E402

DATASETS = ["wine", "ecoli", "credit", "chess", "abalone"]
BUDGETS = [8, 16, 32, 64, 128, 256, 512, 1024]        # MAttr score snapshots (= rows used)
BF_BUDGETS = [2, 4, 8, 16, 32, 64]                    # brute-force queries PER FEATURE


# ---------------------------------------------------------------- MAttr

def half_widths_std(cm, x_raw, ranges):
    """The raw +-0.1*range half-width expressed in STANDARDIZED units. Standardization is
    affine and per-feature, so a per-feature mask commutes with it and we can build the
    intervention directly in standardized space."""
    hi = cm.to_std(x_raw + T.PERCENT_PERTURB * ranges)
    return hi - cm.to_std(x_raw)


def m_mattr(cm, x_raw, c, *, ranges, mode, steps, lr, optimizer, T_temp=0.5,
            adam_eps=1e-8, budgets=BUDGETS, seed=0, **_):
    """-> {budget: phi [p]}. One run; the budget curve comes from score snapshots, so the
    whole curve costs what the longest run costs."""
    torch.manual_seed(seed)
    p = x_raw.shape[-1]
    x_std = cm.to_std(x_raw)
    hw = half_widths_std(cm, x_raw, ranges)

    if mode == "spurious":
        with torch.no_grad():
            y_clean = cm.torch_std(x_std)[0, c].detach()

        def loss_fn(mask):
            cf = x_std + (2 * torch.rand_like(x_std) - 1) * hw
            xm = mask.unsqueeze(0) * x_std + (1 - mask.unsqueeze(0)) * cf
            return (cm.torch_std(xm)[0, c] - y_clean) ** 2
    elif mode == "recourse":
        def loss_fn(mask):
            up = x_std + torch.rand_like(x_std) * hw
            dn = x_std - torch.rand_like(x_std) * hw
            xm = mask.unsqueeze(0) * up + (1 - mask.unsqueeze(0)) * dn
            return -cm.torch_std(xm)[0, c]
    else:
        raise ValueError(mode)

    snaps = {}

    def on_step(step, k, loss, scores):
        if step + 1 in budgets:
            snaps[step + 1] = scores.detach().cpu().numpy().copy()

    learn_scores(p, loss_fn, steps=steps, variant="topk", k_schedule="log", lr=lr,
                 optimizer=optimizer, adam_eps=adam_eps, T=T_temp, device=cm.device,
                 on_step=on_step)
    return snaps


def m_bruteforce(cm, x_raw, c, *, ranges, mode, rng, budgets=BF_BUDGETS, **_):
    """The Theorem 5.1 method: spend the budget sampling nu and compute the oracle's own
    statistic. `n` queries PER FEATURE, so total rows = n * p -- charged honestly."""
    p = x_raw.shape[-1]
    out = {}
    for n in budgets:
        phi = np.zeros(p)
        for j in range(p):
            hw = T.PERCENT_PERTURB * ranges[j]
            if mode == "spurious":
                d = rng.uniform(-hw, hw, n)
            else:
                d = np.concatenate([rng.uniform(0, hw, n // 2), rng.uniform(-hw, 0, n // 2)])
            X = np.tile(x_raw, (n, 1))
            X[:, j] = x_raw[j] + d
            y = cm(X)[:, c]
            phi[j] = np.var(y) if mode == "spurious" else y[:n // 2].mean() - y[n // 2:].mean()
        out[n * p] = phi                                   # key = total rows spent
    return out


# ---------------------------------------------------------------- driver

def run_dataset(name, args, device="cpu"):
    feats, labels, ordered, standardize, task = T.load_tabular(name, root=args.data_root)
    ranges = T.feature_ranges(feats)
    p = feats.shape[1]
    rng_global = np.random.default_rng(args.seed)
    # each model sees the same examples, as in tabular_experiment.run_experiment
    ex_idx = rng_global.integers(0, len(feats), size=args.n_examples)
    sp_idx = rng_global.integers(0, len(feats), size=100)
    examples, sp_examples = feats[ex_idx], feats[sp_idx]

    ig_base_zero = np.zeros(p)
    ig_base_min = standardize(feats).min(0)
    records, model_scores = [], []

    for mi in range(args.n_models):
        t0 = time.time()
        model, qual = T.train_model(name, feats, labels, standardize, task, seed=1000 + mi,
                                    device=device)
        model_scores.append(qual)
        cm = T.CountingModel(model, standardize, device=device)
        rng = np.random.default_rng(args.seed * 977 + mi)
        bg = feats[rng.integers(0, len(feats), size=100)]      # SHAP background

        cm.reset()
        sp_q = T.spurious_quantile(cm, ranges, sp_examples, ordered)
        oracle_rows = cm.rows                                  # not charged to any method

        for ei, ex in enumerate(examples):
            c = int(np.argmax(cm(ex.reshape(1, -1))))
            # oracles
            orc = {"recourse": np.zeros(p), "spurious": np.zeros(p)}
            for j in range(p):
                y = cm(T.perturbation(ranges, ex, j))[:, c]
                orc["recourse"][j] = T.recourse_oracle(y)
                orc["spurious"][j] = float(np.var(y) > sp_q)

            phis, raws, budget = {}, {}, {}
            baselines = {} if args.no_baselines else {
                "grad": (T.m_gradient, {}),
                "smoothgrad": (T.m_smoothgrad, {"rng": rng}),
                "ig_zero": (T.m_integrated_gradient, {"baseline": ig_base_zero}),
                "ig_min": (T.m_integrated_gradient, {"baseline": ig_base_min}),
                "lime": (T.m_lime, {"rng": rng}),
                "shap": (T.m_kernel_shap, {"background": bg, "rng": rng}),
                "random": (T.m_random, {"rng": rng}),
            }
            for mname, (fn, kw) in baselines.items():
                cm.reset()
                raws[mname] = np.asarray(fn(cm, ex, c, **kw), dtype=float)
                phis[mname] = T.normalize(raws[mname])
                budget[mname] = cm.rows

            arms = [("mattr_adam", "adam", args.lr_adam, args.adam_eps),
                    ("mattr_sgd", "sgd", args.lr_sgd, 1e-8)]
            if args.eps_sweep:
                # Adam's eps is a real knob wherever the per-step |g| falls below it (the score
                # degenerates to a signed step COUNT). These models emit SATURATED softmax
                # probabilities, so |g| is small and the question is live -- but it has to be
                # asked on the same examples/models as the main run, not a side sample.
                arms = [(f"mattr_adam_eps{e:g}", "adam", args.lr_adam, e)
                        for e in args.eps_sweep] + [("mattr_sgd", "sgd", args.lr_sgd, 1e-8)]
            if args.lr_sweep:
                # block-argmax discipline: bracket each optimizer's LR on this substrate
                # rather than importing the MIB / ViT-teaser cell and hoping it transfers.
                arms = [(f"mattr_{o}_lr{lr:g}", o, lr, args.adam_eps)
                        for o in ("adam", "sgd") for lr in args.lr_sweep]
            for mode in ("recourse", "spurious"):
                for tag, opt, lr, eps in arms:
                    cm.reset()
                    snaps = m_mattr(cm, ex, c, ranges=ranges, mode=mode, steps=args.steps,
                                    lr=lr, optimizer=opt, adam_eps=eps, seed=mi * 131 + ei)
                    for b, s in snaps.items():
                        raws[f"{tag}@{b}|{mode}"] = s
                        phis[f"{tag}@{b}|{mode}"] = T.normalize(s)
                        budget[f"{tag}@{b}|{mode}"] = b
                if args.lr_sweep or args.eps_sweep:
                    continue
                cm.reset()
                for b, s in m_bruteforce(cm, ex, c, ranges=ranges, mode=mode,
                                         rng=rng).items():
                    raws[f"bruteforce@{b}|{mode}"] = s
                    phis[f"bruteforce@{b}|{mode}"] = T.normalize(s)
                    budget[f"bruteforce@{b}|{mode}"] = b

            records.append({"model": mi, "example": ei, "class": c, "ordered": ordered,
                            "oracle": {k: v.tolist() for k, v in orc.items()},
                            "phi": {k: v.tolist() for k, v in phis.items()},
                            # UNNORMALIZED scores too. The paper's `_normalize` is a per-example
                            # MIN-MAX onto [-1,1], which moves the zero point; the recourse test
                            # (a threshold on phi) is a rank test and so is invariant to it, but
                            # the spurious test (|phi| > alpha) is NOT. For a one-sided
                            # importance score like MAttr's -- most features ~0, a few large
                            # positive -- min-max sends the UNIMPORTANT features to -1 and |-1|
                            # is then the largest magnitude in the example, inverting the test.
                            # Keeping the raw scores lets the report separate that artifact from
                            # the method's own ranking quality instead of re-running.
                            "phi_raw": {k: np.asarray(v).tolist() for k, v in raws.items()},
                            "budget": budget})
        print(f"  [{name}] model {mi} quality={qual:.3f} oracle_rows={oracle_rows} "
              f"({time.time() - t0:.0f}s)", flush=True)

    return {"dataset": name, "p": p, "task": task, "ordered": ordered,
            "model_quality": model_scores, "records": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--n-models", type=int, default=10)
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--steps", type=int, default=max(BUDGETS))
    ap.add_argument("--lr-adam", type=float, default=0.05)   # ViT-teaser cell
    ap.add_argument("--lr-sgd", type=float, default=1.0)     # MIB node headline
    ap.add_argument("--lr-sweep", type=float, nargs="+", default=None,
                    help="run BOTH optimizers at each of these LRs (arm names carry the "
                         "LR) and skip the baselines/brute force; for picking lr-adam/lr-sgd")
    ap.add_argument("--eps-sweep", type=float, nargs="+", default=None,
                    help="run MAttr+Adam at each of these eps (plus the SGD arm) and skip the "
                         "baselines/brute force; arm names carry the eps")
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="data/impossibility")
    ap.add_argument("--out", default="results/impossibility")
    args = ap.parse_args()
    if args.lr_sweep or args.eps_sweep:
        args.no_baselines = True

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        print(f"=== {name} ===", flush=True)
        res = run_dataset(name, args)
        res["args"] = vars(args)
        with open(out / f"{name}.json", "w") as f:
            json.dump(res, f)
        print(f"  -> {out / f'{name}.json'}", flush=True)


if __name__ == "__main__":
    main()
