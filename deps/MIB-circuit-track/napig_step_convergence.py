"""Is NAP-IG converged at 30 steps? Rank stability across the 5 / 10 / 30 step ladder.

Reads importances.json (ATTRIBUTION output), not the eval pkls, so it runs as soon as the
attribution half of each job finishes -- no need to wait for the sparsity sweep.

Why this and not CPR-AUC: CPR-AUC is a noisy downstream readout of the ranking, and it is
probed at 0.1-1% sparsity, so it depends almost entirely on the top few nodes. The quantity
that has to converge is the ranking itself, and specifically its head.

Reading it: convergence looks like 10->30 agreement being markedly TIGHTER than 5->10 across
cells. If 10->30 is as loose as 5->10, 30 is not converged either and the ladder must continue.
The sign-flip column tracks nodes whose score changes sign between rungs, restricted to each
rung's top-10 by |score| -- that is the failure mode actually observed at 5->10 (m0 on
mcqa/qwen2.5 went -1.28 to +1.14 while being the largest-magnitude node in the circuit).
"""
import json, os, sys
import numpy as np

CELLS = [
    ("gpt2","ioi"),("qwen2.5","ioi"),("qwen2.5","mcqa"),
    ("gemma2","ioi"),("gemma2","mcqa"),("gemma2","arc_easy"),
    ("llama3","ioi"),("llama3","mcqa"),("llama3","arithmetic_addition"),
    ("llama3","arithmetic_subtraction"),("llama3","arc_easy"),("llama3","arc_challenge"),
]
RUNGS = [("5", "napig_ref"), ("10", "napig10"), ("30", "napig30")]
MDIR = "EAP-IG-inputs_patching_node"


def rankdata(x):
    o = np.argsort(x, kind="mergesort"); r = np.empty(len(x), float); r[o] = np.arange(len(x))
    xs = x[o]; i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d else float("nan")


def nodes(dirname, task, model):
    p = f"results/{dirname}/{MDIR}/{task.replace('_','-')}_{model}/importances.json"
    if not os.path.exists(p):
        return None
    ns = json.load(open(p)).get("nodes", {})
    ks = sorted(ns)
    v = np.array([ns[k] if not isinstance(ns[k], dict) else ns[k].get("score", np.nan)
                  for k in ks], float)
    return np.array(ks), v


def compare(lo, hi):
    (ks, a), (_, b) = lo, hi
    fin = np.isfinite(a) & np.isfinite(b)
    ks, a, b = ks[fin], a[fin], b[fin]
    out = {"rho": spearman(a, b)}
    for K in (5, 20, 100):
        ta = set(ks[np.argsort(-np.abs(a))[:K]]); tb = set(ks[np.argsort(-np.abs(b))[:K]])
        out[f"top{K}"] = len(ta & tb) / K
    # sign flips among nodes that are top-10 by |score| in EITHER rung
    idx = sorted(set(np.argsort(-np.abs(a))[:10]) | set(np.argsort(-np.abs(b))[:10]))
    out["flips"] = sum(1 for i in idx if a[i] * b[i] < 0)
    out["flipped"] = [ks[i] for i in idx if a[i] * b[i] < 0]
    return out


pairs = [(RUNGS[i], RUNGS[i + 1]) for i in range(len(RUNGS) - 1)]
W = max(len(f"{t}/{m}") for m, t in CELLS) + 2
hdr = f"{'cell':<{W}}" + "".join(f"{f'{a[0]}->{b[0]}':>34}" for a, b in pairs)
sub = f"{'':<{W}}" + "".join(f"{'rho':>8}{'top5':>7}{'top20':>7}{'top100':>7}{'flip':>5}" for _ in pairs)
print(hdr); print(sub); print("-" * len(sub))

agg = {f"{a[0]}->{b[0]}": [] for a, b in pairs}
for model, task in CELLS:
    loaded = {tag: nodes(d, task, model) for tag, d in RUNGS}
    line = f"{task+'/'+model:<{W}}"
    for (ta, _), (tb, _) in pairs:
        if loaded[ta] is None or loaded[tb] is None:
            line += f"{'--':>34}"; continue
        c = compare(loaded[ta], loaded[tb])
        agg[f"{ta}->{tb}"].append((f"{task}/{model}", c))
        line += (f"{c['rho']:>8.3f}{c['top5']:>7.1%}{c['top20']:>7.1%}"
                 f"{c['top100']:>7.1%}{c['flips']:>5d}")
    print(line)

print("-" * len(sub))
foot = f"{'mean':<{W}}"
for (ta, _), (tb, _) in pairs:
    cs = [c for _, c in agg[f"{ta}->{tb}"]]
    if not cs:
        foot += f"{'--':>34}"; continue
    foot += (f"{np.mean([c['rho'] for c in cs]):>8.3f}"
             f"{np.mean([c['top5'] for c in cs]):>7.1%}"
             f"{np.mean([c['top20'] for c in cs]):>7.1%}"
             f"{np.mean([c['top100'] for c in cs]):>7.1%}"
             f"{np.sum([c['flips'] for c in cs]):>5d}")
print(foot)

for (ta, _), (tb, _) in pairs:
    entries = agg[f"{ta}->{tb}"]
    print(f"\n  {ta}->{tb}: {len(entries)}/{len(CELLS)} cells")
    for cell, c in entries:
        if c["flipped"]:
            print(f"      sign flip among top-10: {cell}: {', '.join(c['flipped'])}")
