"""Compare the global-circuit arms: sweep table + between-method rank agreement.

    python scripts/global_kl/collect_global_kl.py                       # results/global_kl/*.json
    python scripts/global_kl/collect_global_kl.py --dir results/_smoke_gkl

Two blocks. The first is the sparsity sweep read at a few readable budgets -- KL to the
unintervened model, and top-1 agreement with it -- plus the log-sparsity AUCs. Read KL DOWN
(0 = the circuit reproduces the model) and agreement UP.

The second is Spearman between the arms' score vectors, and top-k Jaccard, which is the
question the sweep cannot answer: whether two methods that score similarly are actually
naming the same neurons and heads. Rank correlation over ~5e5 mostly-irrelevant nodes is
dominated by the tail, so the top-k overlap is the one to read for "same circuit?".
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/global_kl")
    p.add_argument("--budgets", type=float, nargs="+", default=[0.001, 0.1, 0.9, 0.99],
                   help="sparsity fractions to print (nearest grid point is used). Defaults "
                        "straddle both ends: under zero ablation the sparse end is a destroyed "
                        "model for every arm, and the arms only separate near 1.")
    p.add_argument("--topk", type=int, default=1000, help="k for the Jaccard overlap")
    p.add_argument("--skip-agreement", action="store_true",
                   help="table only; the pairwise block is O(runs^2) Spearman over ~5e5 nodes "
                        "and is noise for a one-knob sweep where every run is the same method")
    a = p.parse_args()

    runs = []
    for f in sorted(Path(a.dir).glob("*.json")):
        d = json.load(open(f))
        d["_stem"] = f.stem
        d["_scores"] = f.with_name(f.stem + "_scores.pt")
        runs.append(d)
    if not runs:
        raise SystemExit(f"no runs in {a.dir}")

    total = runs[0]["total"]
    sp = np.array(runs[0]["sweep"]["sparsities"])
    # dedupe: a coarse --n-sparsities grid can snap two requested budgets to the same column
    cols = sorted({int(np.abs(sp - b).argmin()) for b in a.budgets})
    print(f"{len(runs)} runs, clean KL={runs[0]['sweep']['clean']['kl']:.4f}, "
          f"all-ablated KL={runs[0]['sweep']['ablated']['kl']:.2f}. The `nodes` column is the "
          f"layout size -- rows with different node counts are NOT comparable.\n")

    head = f"{'run':<52}{'nodes':>9}" + "".join(f"{'KL@' + f'{sp[c]:.2g}':>11}" for c in cols) \
        + "".join(f"{'agr@' + f'{sp[c]:.2g}':>11}" for c in cols) + f"{'KL-AUC':>10}{'agr-AUC':>10}"
    print(head); print("-" * len(head))
    for d in runs:
        s = d["sweep"]["learned"]
        row = f"{d['_stem'][:52]:<52}{d['total']:>9,}" + "".join(f"{s['kl'][c]:>11.3f}" for c in cols) \
            + "".join(f"{s['top1_agree'][c]:>11.3f}" for c in cols) \
            + f"{d['aucs']['learned_kl_auc']:>10.3f}{d['aucs']['learned_top1_agree_auc']:>10.3f}"
        print(row)
    r = runs[0]["sweep"]["random"]
    print(f"{'(random ordering)':<52}{total:>9,}" + "".join(f"{r['kl'][c]:>11.3f}" for c in cols)
          + "".join(f"{r['top1_agree'][c]:>11.3f}" for c in cols)
          + f"{runs[0]['aucs']['random_kl_auc']:>10.3f}"
          f"{runs[0]['aucs']['random_top1_agree_auc']:>10.3f}")

    if all("thresholds" in d for d in runs):
        print("\n smallest circuit reaching a tolerance (nodes kept; '-' = never, on this grid):")
        klv, agv = ["1.0", "0.5", "0.1"], ["0.9", "0.95", "0.99"]
        h = f"{'run':<52}" + "".join(f"{'KL<=' + v:>12}" for v in klv) \
            + "".join(f"{'agr>=' + v:>12}" for v in agv)
        print(h); print("-" * len(h))
        fmt = lambda v: "-" if v is None else f"{v:,}"
        for d in runs:
            t = d["thresholds"]
            print(f"{d['_stem'][:52]:<52}"
                  + "".join(f"{fmt(t['kl'][v]):>12}" for v in klv)
                  + "".join(f"{fmt(t['top1_agree'][v]):>12}" for v in agv))

    have = [] if a.skip_agreement else [d for d in runs if d["_scores"].exists()]
    if len(have) < 2:
        return
    print(f"\nrank agreement (Spearman / top-{a.topk} Jaccard):")
    sc = {d["_stem"]: torch.load(d["_scores"]).float().numpy() for d in have}
    names = list(sc)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = sc[names[i]], sc[names[j]]
            if x.shape != y.shape:
                continue    # different node layouts are not comparable
            rho = spearmanr(x, y).statistic
            tx = set(np.argsort(-x)[:a.topk]); ty = set(np.argsort(-y)[:a.topk])
            jac = len(tx & ty) / len(tx | ty)
            print(f"  {names[i][:40]:<40} vs {names[j][:40]:<40} rho={rho:+.3f}  J={jac:.3f}")


if __name__ == "__main__":
    main()
