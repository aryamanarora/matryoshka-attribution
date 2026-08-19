"""Ranking stability of SAE-latent attribution across training objectives.

Zero new model compute: the runs in ``results/sae_variant_pilot/<task>/n200/<cell>/`` already
store the full attribution-score vector (``scores.pt``), so the ranking each method would hand
the evaluator can be recovered exactly.

Ranking convention is the evaluator's, verbatim (``attribute_sae.evaluate`` ->
``sigmoid_topk_hard`` -> ``scores.topk(int(k))``): the RAW SIGNED score, descending. No abs(),
no normalisation, no re-centring. Ties fall to the lower flat index, as in ``torch.topk``.

A unit is a flat index ``i`` into ``[num_spans * (d_sae + 1)]``:
    span = i // width,  slot = i % width;  slot < d_sae -> SAE latent,  slot == d_sae -> the
per-span reconstruction-error node. The error node is KEPT by default because the evaluator
keeps it; ``--exclude-error-node`` runs the sensitivity check.

Full-vector rank correlation is deliberately not reported as a headline: I x G leaves ~116k of
~131k coordinates at exactly zero (JumpReLU sparsity), so a whole-vector coefficient is dominated
by an arbitrary tie ordering over units no method would ever select.

Usage:
    python sae_pilot/analyze_ranking_stability.py --root results/sae_variant_pilot
    python sae_pilot/analyze_ranking_stability.py --objectives ce acc ld --exclude-error-node
"""
import argparse, json, os
import torch

TASKS = ["npi_ever_subj-relc", "npi_any_obj-relc", "agr_sv_num_subj-relc",
         "garden_npz_v-trans", "garden_npz_obj_mod", "filler_gap_pp"]
KS = [8, 16, 32, 64, 128]
# (label, (method_a, objective_a), (method_b, objective_b))
CONTRASTS = [("MAttr CE<->LD",   ("MAttr", "ce"), ("MAttr", "ld")),
             ("IxG   CE<->LD",   ("IxG",   "ce"), ("IxG",   "ld")),
             ("MAttr<->IxG @LD", ("MAttr", "ld"), ("IxG",   "ld")),
             ("MAttr<->IxG @CE", ("MAttr", "ce"), ("IxG",   "ce"))]
# soft accuracy is a monotone reparametrisation of the logit-difference margin, so these are a
# compatibility check against the paper's grid rather than an independent objective direction.
ACC_CONTRASTS = [("MAttr CE<->acc", ("MAttr", "ce"),  ("MAttr", "acc")),
                 ("MAttr acc<->LD", ("MAttr", "acc"), ("MAttr", "ld")),
                 ("IxG   CE<->acc", ("IxG",   "ce"),  ("IxG",   "acc")),
                 ("IxG   acc<->LD", ("IxG",   "acc"), ("IxG",   "ld")),
                 ("MAttr<->IxG @acc", ("MAttr", "acc"), ("IxG", "acc"))]


def load_scores(root, task, method, objective, seed):
    """`ld` runs were written under two directory spellings; accept both."""
    names = ["logitdiff", "ld"] if objective == "ld" else [objective]
    for n in names:
        p = os.path.join(root, task, "n200", f"{method}_{n}_s{seed}", "scores.pt")
        if os.path.exists(p):
            return torch.load(p, map_location="cpu", weights_only=True)
    return None


def width_of(root, task, seed=0):
    for cell in ("MAttr_ce", "IxG_ce"):
        p = os.path.join(root, task, "n200", f"{cell}_s{seed}", "results.json")
        if os.path.exists(p):
            return json.load(open(p))["provenance"]["width"]
    raise FileNotFoundError(f"no results.json for {task}")


def drop_error_node(v, width):
    keep = (torch.arange(v.numel()) % width) != (width - 1)
    return v[torch.nonzero(keep).flatten()]


def topk_agreement(a, b, k):
    """(intersection size, overlap fraction |A n B|/k, Jaccard |A n B|/|A u B|)."""
    A = set(a.topk(k).indices.tolist())
    B = set(b.topk(k).indices.tolist())
    inter = len(A & B)
    return inter, inter / k, inter / len(A | B)


def rbo(a, b, p, depth):
    """Rank-biased overlap truncated at `depth`: (1-p) * sum_d p^(d-1) * |A_d n B_d| / d.

    Top-weighted by construction; `p` sets the expected inspection depth 1/(1-p). Truncation
    makes this a lower bound on the untruncated RBO, with the omitted tail bounded by p^depth.
    """
    ia, ib = a.topk(depth).indices.tolist(), b.topk(depth).indices.tolist()
    A, B, s = set(), set(), 0.0
    for d in range(1, depth + 1):
        A.add(ia[d - 1]); B.add(ib[d - 1])
        s += (p ** (d - 1)) * (len(A & B) / d)
    return (1 - p) * s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sae_variant_pilot")
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--ks", nargs="+", type=int, default=KS)
    ap.add_argument("--objectives", nargs="+", default=["ce", "ld"],
                    help="include 'acc' to add the soft-accuracy compatibility contrasts")
    ap.add_argument("--rbo-depth", type=int, default=256)
    ap.add_argument("--exclude-error-node", action="store_true",
                    help="sensitivity check: drop the per-span reconstruction-error node")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    contrasts = list(CONTRASTS) + (ACC_CONTRASTS if "acc" in args.objectives else [])
    rows, missing = {}, []
    for task in args.tasks:
        w = width_of(args.root, task)
        for seed in args.seeds:
            cache = {}
            for _, x, y in contrasts:
                for m, o in (x, y):
                    if (m, o) in cache:
                        continue
                    v = load_scores(args.root, task, m, o, seed)
                    if v is not None and args.exclude_error_node:
                        v = drop_error_node(v, w)
                    cache[(m, o)] = v
            for name, x, y in contrasts:
                a, b = cache[x], cache[y]
                if a is None or b is None:
                    missing.append((task, seed, name)); continue
                rows[(task, seed, name)] = {
                    "overlap": {k: topk_agreement(a, b, k)[1] for k in args.ks},
                    "jaccard": {k: topk_agreement(a, b, k)[2] for k in args.ks},
                    "intersection": {k: topk_agreement(a, b, k)[0] for k in args.ks},
                    "rbo_0.90": rbo(a, b, 0.90, args.rbo_depth),
                    "rbo_0.98": rbo(a, b, 0.98, args.rbo_depth)}

    def mean(vals):
        vals = list(vals)
        return sum(vals) / len(vals) if vals else float("nan")

    present = [n for n, _, _ in contrasts if any((t, s, n) in rows for t in args.tasks for s in args.seeds)]
    tag = " (reconstruction-error node EXCLUDED)" if args.exclude_error_node else ""
    print(f"top-k overlap fraction |A_k n B_k| / k   mean over {len(args.tasks)} tasks x "
          f"{len(args.seeds)} seed(s){tag}")
    hdr = f"{'contrast':18}" + "".join(f"{k:>8}" for k in args.ks) + f"{'RBO .9':>9}{'RBO .98':>9}"
    print(hdr); print("-" * len(hdr))
    for name in present:
        cells = [(t, s) for t in args.tasks for s in args.seeds if (t, s, name) in rows]
        ov = [mean(rows[(t, s, name)]["overlap"][k] for t, s in cells) for k in args.ks]
        r9 = mean(rows[(t, s, name)]["rbo_0.90"] for t, s in cells)
        r98 = mean(rows[(t, s, name)]["rbo_0.98"] for t, s in cells)
        print(f"{name:18}" + "".join(f"{x:>8.3f}" for x in ov) + f"{r9:>9.3f}{r98:>9.3f}")

    print(f"\nJaccard |A_k n B_k| / |A_k u B_k|")
    print(hdr[:len(f"{'contrast':18}") + 8 * len(args.ks)])
    for name in present:
        cells = [(t, s) for t in args.tasks for s in args.seeds if (t, s, name) in rows]
        ja = [mean(rows[(t, s, name)]["jaccard"][k] for t, s in cells) for k in args.ks]
        print(f"{name:18}" + "".join(f"{x:>8.3f}" for x in ja))

    k_ref = 32 if 32 in args.ks else args.ks[len(args.ks) // 2]
    print(f"\nper task, top-{k_ref} overlap (mean over seeds {args.seeds})")
    print(f"{'task':24}" + "".join(f"{n:>18}" for n in present))
    for t in args.tasks:
        cells = [s for s in args.seeds if (t, s, present[0]) in rows]
        print(f"{t:24}" + "".join(
            f"{mean(rows[(t, s, n)]['overlap'][k_ref] for s in cells):>18.3f}" for n in present))

    if missing:
        print(f"\n[warn] {len(missing)} contrast(s) skipped for missing runs, e.g. {missing[:3]}")
    if args.json_out:
        json.dump({f"{t}|{s}|{n}": v for (t, s, n), v in rows.items()},
                  open(args.json_out, "w"), indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
