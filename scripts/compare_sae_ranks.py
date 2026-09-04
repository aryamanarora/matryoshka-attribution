"""Do MAttr and IG select the SAME SAE latents? Rank agreement on the residual-SAE basis.

WHY THIS EXISTS. plots/plot_sae_train_probe.py established that MAttr's poor IIA on the
Llama-Scope residual SAE is not undertraining -- both arms plateau by ~step 500 far below IG.
This script asks the next question: are the two methods ranking the same 6.29M units differently,
or ranking DIFFERENT UNIVERSES of units? The score vectors say the latter, and it is structural.

*** THE SUPPORT IS NOT SHARED, SO A GLOBAL SPEARMAN IS NEARLY MEANINGLESS HERE. *** SAE latents
are sparse: a latent that never fires on any example has zero activation and zero grad-dot-delta,
so IG and I x G score it EXACTLY 0. Measured on disk that is 95.6% of units on rc, 99.1% on
months. Those units are one giant tie at the median rank, so a Spearman over all units is
dominated by an arbitrary tie-break between two methods' orderings of things neither can
distinguish. MAttr, by contrast, scores every unit densely (100% nonzero) because a mask logit
exists whether or not the latent ever fires.

So this reports THREE views and they answer different questions:
  rho_all      Spearman over all units. Reported for completeness; read it knowing that ~96-99%
               of one side is tied.
  rho_supp     Spearman restricted to IG's nonzero support -- the latents that actually fire.
               This is the honest "do they agree about live units" number.
  overlap@k    |top-k(A) & top-k(B)| / k at the k the eval grid actually uses. This is the only
               view that matches what the CPR/IIA curves do, since those only ever KEEP a top-k.

AND ONE DIAGNOSTIC THAT IS NOT A CORRELATION: `dead@k`, the fraction of a method's own top-k that
falls OUTSIDE IG's nonzero support -- i.e. latents that never fire on this task's data. Patching
a dead latent is a no-op in the clean->patch direction, so budget spent there is budget wasted. If
MAttr's dead@k is high, that alone explains the curves and no amount of further training fixes it.

ERROR NODES are tracked separately (index d_sae of each of the (layer x span) blocks, block size
d_sae+1). They are the one unit per block that is never "dead", so a method that learns to hoard
them looks different from one that spreads over latents.

Run:  uv run python scripts/compare_sae_ranks.py            # all tasks
      uv run python scripts/compare_sae_ranks.py --tasks rc # subset
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata

RES = "results/sva_sweep"
SUB = "resid_sae_span"
D_SAE = 32768                     # Llama-Scope 8x; block = D_SAE latents + 1 error node
TASKS = ["nounpp", "rc", "simple", "within_rc", "addition", "months", "weekdays", "hours"]
# (tag on disk, display name). Order is the report order.
METHODS = [("ig", "IG"),
           ("ixg", "I×G"),
           ("sufficient_topk_sgd_bs1", "MAttr^S"),
           ("sufficient_topk_adam_eps1e-2_bs1", "MAttr^A"),
           ("random_s42", "Random")]
KS = [10, 100, 1000, 10_000, 100_000]


def load(task, tag):
    p = f"{RES}/{task}_llama3_{SUB}_{tag}.scores.pt"
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu").float().numpy()


def topk(v, k):
    """Indices of the k largest scores. Selection is by RAW score, not |score|, because the
    eval keeps the top-k by score -- mirroring the eval is the point of this statistic."""
    return np.argpartition(v, -k)[-k:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    a = ap.parse_args()

    agg_rho, agg_ov, agg_dead, agg_err = {}, {}, {}, {}
    for task in a.tasks:
        S = {tag: load(task, tag) for tag, _ in METHODS}
        S = {k: v for k, v in S.items() if v is not None}
        if "ig" not in S or len(S) < 2:
            print(f"{task}: skipped ({len(S)} score vectors, need ig + 1)")
            continue
        n = len(next(iter(S.values())))
        nblk = n // (D_SAE + 1)
        assert nblk * (D_SAE + 1) == n, f"{task}: {n} not a multiple of {D_SAE + 1}"
        err_idx = np.arange(nblk) * (D_SAE + 1) + D_SAE
        supp = S["ig"] != 0                       # latents that fire at least once
        print(f"\n=== {task}  ({n:,} units = {nblk} blocks x {D_SAE + 1};  "
              f"live support {supp.sum():,} = {supp.mean():.2%}) ===")

        R = {k: rankdata(v) for k, v in S.items()}
        Rs = {k: rankdata(v[supp]) for k, v in S.items()}
        print(f"{'pair':<20}{'rho_all':>9}{'rho_supp':>10}" +
              "".join(f"{'ov@' + f'{k:,}':>12}" for k in KS))
        for (ta, na), (tb, nb) in itertools.combinations(METHODS, 2):
            if ta not in S or tb not in S:
                continue
            ra = np.corrcoef(R[ta], R[tb])[0, 1]
            rs = np.corrcoef(Rs[ta], Rs[tb])[0, 1]
            ovs = []
            for k in KS:
                o = len(np.intersect1d(topk(S[ta], k), topk(S[tb], k), assume_unique=True)) / k
                ovs.append(o)
                agg_ov.setdefault((na, nb, k), []).append(o)
            agg_rho.setdefault((na, nb), []).append((ra, rs))
            print(f"{na + ' / ' + nb:<20}{ra:>9.3f}{rs:>10.3f}" +
                  "".join(f"{o:>12.3f}" for o in ovs))

        print(f"\n{'method':<12}{'dead@1k':>9}{'dead@10k':>10}{'dead@100k':>11}"
              f"{'err@1k':>9}{'err@10k':>9}{'(of ' + str(nblk) + ')':>10}")
        for tag, name in METHODS:
            if tag not in S:
                continue
            row, d = [], {}
            for k in (1000, 10_000, 100_000):
                tk = topk(S[tag], k)
                d[k] = 1.0 - supp[tk].mean()
                row.append(d[k])
            agg_dead.setdefault(name, []).append(d)
            e = {}
            for k in (1000, 10_000):
                e[k] = int(np.isin(topk(S[tag], k), err_idx).sum())
            agg_err.setdefault(name, []).append(e)
            print(f"{name:<12}{row[0]:>9.3f}{row[1]:>10.3f}{row[2]:>11.3f}"
                  f"{e[1000]:>9}{e[10_000]:>9}")

    if not agg_rho:
        return 1
    print(f"\n{'=' * 78}\nMEAN OVER {len(set(t for t in a.tasks))} TASKS (cells present)\n{'=' * 78}")
    print(f"{'pair':<20}{'rho_all':>9}{'rho_supp':>10}" +
          "".join(f"{'ov@' + f'{k:,}':>12}" for k in KS))
    for (na, nb), vs in agg_rho.items():
        ra = np.mean([v[0] for v in vs]); rs = np.mean([v[1] for v in vs])
        print(f"{na + ' / ' + nb:<20}{ra:>9.3f}{rs:>10.3f}" +
              "".join(f"{np.mean(agg_ov[(na, nb, k)]):>12.3f}" for k in KS))
    print(f"\n{'method':<12}{'dead@1k':>9}{'dead@10k':>10}{'dead@100k':>11}"
          f"{'err@1k':>9}{'err@10k':>9}")
    for _, name in METHODS:
        if name not in agg_dead:
            continue
        d = agg_dead[name]; e = agg_err[name]
        print(f"{name:<12}" + "".join(f"{np.mean([x[k] for x in d]):>{w}.3f}"
                                      for k, w in ((1000, 9), (10_000, 10), (100_000, 11)))
              + "".join(f"{np.mean([x[k] for x in e]):>9.1f}" for k in (1000, 10_000)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
