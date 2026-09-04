"""Anatomy of the global-circuit rankings: what is in the top of each arm, and do the arms
agree? Reads the saved ``*_scores.pt`` from eval_global_kl.py runs -- CPU only, no model.

    python scripts/analyse_global_ranks.py --arms results/global_kl_lr/*adam_lr0.2*_scores.pt ...

Four blocks:
  composition  -- MLP vs attention share of the top-k, against the 0.22% base rate at which
                  heads occur at all. A circuit that is "all neurons" at every k is not
                  selecting heads; one that is head-heavy at small k is.
  layers       -- where in depth the top-k lives, as a per-layer count.
  heads        -- the actual (layer, head) identities in the top of each arm, which is the
                  only block you can eyeball against known head roles.
  agreement    -- Spearman over all nodes and Jaccard at several k. Full-vector Spearman on
                  ~5e5 mostly-irrelevant nodes is dominated by the tail; the top-k Jaccard is
                  the one that answers "are these the same circuit?".
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

L, N, NH = 32, 14336, 32          # llama3-8B: layers, intermediate, heads
MLP_TOTAL = L * N


def decode(idx):
    """flat index -> ('mlp', layer, neuron) or ('attn', layer, head)."""
    if idx < MLP_TOTAL:
        return "mlp", idx // N, idx % N
    r = idx - MLP_TOTAL
    return "attn", r // NH, r % NH


def label(p):
    """Readable arm name from a scores-file path."""
    s = Path(p).stem.replace("_scores", "")
    return s.replace("llama3_mlp_tied-attn_head_tied_", "").replace("_seed42", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True, metavar="SCORES_PT")
    ap.add_argument("--ks", type=int, nargs="+", default=[100, 1000, 10000, 100000])
    ap.add_argument("--top-heads", type=int, default=15)
    ap.add_argument("--consensus", nargs="*", default=None, metavar="ARM",
                    help="arm labels to intersect for the consensus block (default: all given). "
                         "Pass only the arms you believe -- including a null like `random` or "
                         "`magnitude` makes the intersection meaningless, not more conservative.")
    a = ap.parse_args()

    arms = {label(p): torch.load(p, map_location="cpu").float().numpy() for p in a.arms}
    total = next(iter(arms.values())).shape[0]
    order = {n: np.argsort(-s) for n, s in arms.items()}
    base = 100 * (total - MLP_TOTAL) / total

    print(f"{total:,} nodes = {MLP_TOTAL:,} MLP neurons + {total-MLP_TOTAL:,} attn heads "
          f"(heads are {base:.2f}% of all nodes -- that is the null share below)\n")

    print("=== composition: % of top-k that are ATTENTION HEADS ===")
    hdr = f"{'arm':<26}" + "".join(f"{'k=' + str(k):>10}" for k in a.ks)
    print(hdr); print("-" * len(hdr))
    for n, o in order.items():
        row = f"{n:<26}"
        for k in a.ks:
            share = 100 * (o[:k] >= MLP_TOTAL).mean()
            row += f"{share:>9.2f}%"
        print(row)
    print(f"{'(null / random)':<26}" + "".join(f"{base:>9.2f}%" for _ in a.ks))

    print("\n=== layer profile: count of top-1000 nodes per layer (MLP | attn) ===")
    print(f"{'arm':<26} " + " ".join(f"{i:>3}" for i in range(L)))
    for n, o in order.items():
        top = o[:1000]
        cm = np.zeros(L, int); ca = np.zeros(L, int)
        for i in top:
            kind, li, _ = decode(int(i))
            (cm if kind == "mlp" else ca)[li] += 1
        print(f"{n[:24]:<24}M " + " ".join(f"{c:>3}" for c in cm))
        print(f"{'':<24}A " + " ".join(f"{c:>3}" for c in ca))

    print(f"\n=== top-{a.top_heads} attention heads per arm (layer.head, best first) ===")
    for n, s in arms.items():
        hs = s[MLP_TOTAL:]
        top = np.argsort(-hs)[:a.top_heads]
        # rank of each head within the FULL node ranking, so "head #1" can be read against the
        # neurons it is competing with rather than only against other heads
        rank_of = {int(v): r for r, v in enumerate(order[n])}
        txt = " ".join(f"{int(h)//NH}.{int(h)%NH}(#{rank_of[int(h)+MLP_TOTAL]:,})" for h in top)
        print(f"  {n}\n    {txt}")

    print("\n=== depth concentration: % of top-1000 in layers 0-1 and 30-31 ===")
    for n, o in order.items():
        top = [decode(int(i)) for i in o[:1000]]
        early = 100 * sum(1 for k, li, _ in top if li <= 1) / len(top)
        late = 100 * sum(1 for k, li, _ in top if li >= 30) / len(top)
        print(f"  {n:<26} early(L0-1)={early:>5.1f}%   late(L30-31)={late:>5.1f}%   "
              f"middle={100 - early - late:>5.1f}%")
    print(f"  {'(null / random)':<26} early(L0-1)={2/L*100:>5.1f}%   late(L30-31)={2/L*100:>5.1f}%"
          f"   middle={100 - 4/L*100:>5.1f}%")

    if len(arms) >= 2 and a.consensus:
        cons = [n for n in arms if n in a.consensus] or list(arms)
        print(f"\n=== CONSENSUS across {len(cons)} arms: {', '.join(cons)} ===")
        # rank[n][i] = position of node i in arm n's ranking (0 = most important)
        rank = {}
        for n in cons:
            r = np.empty(total, dtype=np.int64); r[order[n]] = np.arange(total)
            rank[n] = r
        worst = np.max([rank[n] for n in cons], axis=0)   # top-k in ALL arms <=> worst < k
        best_of_worst = np.min([rank[n] for n in cons], axis=0)
        m = len(cons)

        print("  consistently IMPORTANT: nodes in the top-k of EVERY arm")
        print(f"  {'k':>8} {'observed':>10} {'chance':>10} {'enrich':>9}")
        for k in a.ks:
            obs = int((worst < k).sum())
            exp = total * (k / total) ** m
            print(f"  {k:>8,} {obs:>10,} {exp:>10.2f} {(obs / exp if exp else float('nan')):>8.1f}x")

        print("  consistently UNIMPORTANT: nodes in the bottom-k of EVERY arm")
        print(f"  {'k':>8} {'observed':>10} {'chance':>10} {'enrich':>9}")
        for k in a.ks:
            obs = int((best_of_worst >= total - k).sum())
            exp = total * (k / total) ** m
            print(f"  {k:>8,} {obs:>10,} {exp:>10.2f} {(obs / exp if exp else float('nan')):>8.1f}x")

        top = np.argsort(worst)[:a.top_heads]
        print(f"  the {a.top_heads} most consistently important nodes (by WORST rank across arms):")
        for i in top:
            kind, li, u = decode(int(i))
            rs = " ".join(f"{n.split('_')[1][:4]}=#{rank[n][i]:,}" for n in cons)
            name = f"L{li}.h{u}" if kind == "attn" else f"L{li}.n{u}"
            print(f"    {kind:<4} {name:<12} worst=#{worst[i]:,}   {rs}")

        # heads on their own: 1024 items, so a consensus head is a much weaker claim than a
        # consensus neuron and the chance rate must be computed against 1024, not `total`.
        hr = {n: np.empty(1024, dtype=np.int64) for n in cons}
        for n in cons:
            ho = np.argsort(-arms[n][MLP_TOTAL:]); hr[n][ho] = np.arange(1024)
        hworst = np.max([hr[n] for n in cons], axis=0)
        print("  HEADS only (of 1024): in the top-k head of every arm")
        for k in (16, 64, 256):
            obs = int((hworst < k).sum()); exp = 1024 * (k / 1024) ** m
            print(f"    top{k:>4}: {obs:>4} observed, {exp:>6.2f} chance, "
                  f"{(obs / exp if exp else float('nan')):.1f}x")
        hb = np.argsort(hworst)[:8]
        print("    most consistent heads: " + " ".join(
            f"L{int(h)//NH}.h{int(h)%NH}(worst #{hworst[h]})" for h in hb))

    if len(arms) < 2:
        return

    print("\n=== HEAD-ONLY agreement (1024 heads; the sub-ranking where structure lives) ===")
    # Over all 459,776 nodes a Spearman is dominated by the ~458k neurons, most of which no arm
    # has an opinion about. Restricted to heads it is a real comparison of 1024 ranked items.
    names = list(arms)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            hx, hy = arms[names[i]][MLP_TOTAL:], arms[names[j]][MLP_TOTAL:]
            rho = spearmanr(hx, hy).statistic
            ov = []
            for k in (16, 64, 256):
                tx = set(np.argsort(-hx)[:k].tolist()); ty = set(np.argsort(-hy)[:k].tolist())
                ov.append(f"top{k}={len(tx & ty)}/{k}")
            print(f"  {names[i][:22]:<22} vs {names[j][:22]:<22} rho={rho:+.3f}  " + "  ".join(ov))

    print("\n=== agreement between arms (all nodes) ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = arms[names[i]], arms[names[j]]
            rho = spearmanr(x, y).statistic
            js = []
            for k in a.ks:
                tx, ty = set(order[names[i]][:k].tolist()), set(order[names[j]][:k].tolist())
                js.append(len(tx & ty) / len(tx | ty))
            jtxt = "  ".join(f"J@{k}={v:.3f}" for k, v in zip(a.ks, js))
            print(f"  {names[i][:22]:<22} vs {names[j][:22]:<22} rho={rho:+.3f}  {jtxt}")


if __name__ == "__main__":
    main()
