"""Compare MLP-neuron MAttr importance scores against the paper's claimed layer-18
arithmetic neurons (goodfire-ai/arithmetic-wild, arXiv 2605.01148).

scores layout per task: [n_layers * num_spans * intermediate_size], reshaped to
[L, S, I]; last_token span index 2. The paper localises arithmetic to layer-18 MLP
neurons (indices into intermediate_size=14336)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
import torch

PKL = sys.argv[1] if len(sys.argv) > 1 else "results/arith_mlp.pkl"
PAPER = "/home/guests/aryaman/arithmetic-wild/src/neurons_per_task.json"
PAPER_LAYER = 18
SPAN_LAST = 2

paper = json.load(open(PAPER))
r = pickle.load(open(PKL, "rb"))
L = r["n_layers"]; I = r["intermediate_size"]
tasks = r["tasks"]
print(f"pkl={PKL}  mask={r['mask']}  tasks={tasks}  L={L} I={I}\n")


def layerwise_importance(sc3, span):
    """max-abs score per layer at given span -> [L]."""
    return sc3[:, span, :].abs().max(dim=1).values.numpy()


def topk_overlap(scores_vec, paper_neurons, k):
    top = set(np.argsort(-scores_vec)[:k].tolist())
    pn = set(paper_neurons)
    hit = top & pn
    return len(hit), sorted(hit)


for t in tasks:
    sc = r["scores"][t]                      # [L*S*I]
    S = r["num_spans"][t]
    sc3 = sc.view(L, S, I)
    pn = paper.get(t, [])
    print(f"=== {t}  (paper: {len(pn)} layer-{PAPER_LAYER} neurons) ===")

    # 1) Does MAttr concentrate importance at layer 18 (last_token span)?
    li = layerwise_importance(sc3, SPAN_LAST)
    order = np.argsort(-li)
    rank18 = int(np.where(order == PAPER_LAYER)[0][0]) + 1
    print(f"  layer importance (last_token, max|score|): "
          f"L18 rank {rank18}/{L} (val {li[PAPER_LAYER]:.3f}, "
          f"max L{int(order[0])}={li[order[0]]:.3f})")
    print(f"  top-5 layers: " +
          ", ".join(f"L{int(l)}={li[l]:.2f}" for l in order[:5]))

    # 2) Within layer 18 last_token, overlap of top-N neurons with paper list
    v18 = sc3[PAPER_LAYER, SPAN_LAST, :].numpy()
    N = len(pn)
    for k in sorted({N, 2 * N, 50}):
        h, _ = topk_overlap(v18, pn, k)
        exp = N * k / I
        print(f"  L18 top-{k:3d}: {h}/{N} paper neurons recovered "
              f"(chance {exp:.2f}, enrichment {h / max(exp,1e-9):.1f}x)")
    h, hits = topk_overlap(v18, pn, N)
    print(f"  L18 top-{N} hits: {hits}")

    # 3) also: where do the paper neurons rank within L18 last_token?
    ranks = {n: int(np.where(np.argsort(-v18) == n)[0][0]) + 1 for n in pn}
    med = int(np.median(list(ranks.values())))
    print(f"  median rank of paper neurons within L18 ({I} neurons): {med}")
    print()
