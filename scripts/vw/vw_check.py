"""Correctness test: the closed forms in vw_scores.py against brute-force ablation.

    uv run python scripts/vw/vw_check.py        # ~20s, CPU, no data or checkpoint needed

The two quantities vw_scores.py accumulates are analytic identities from the note's
appendix, not approximations, so they are checkable exactly. On a tiny random model we

  1. compute helpfulness for a sample of weights by the streaming closed form, and
  2. recompute it by literally zeroing that weight in W_TL, re-running the forward over the
     same tokens, and taking the mean per-position loss difference,

and require agreement to float precision. Same for Fisher effectiveness against a direct
evaluation of 1/2 E[a^T F a] with F = diag(p) - p p^T built explicitly.

This is worth its own script because both formulas are easy to get subtly wrong in ways
that still produce plausible-looking rankings: an off-by-one in which position predicts
which token, the missing `+ s w 1_{j=t}` term, or the sign of the ablation. Any of those
would leave every figure downstream looking fine and being wrong.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import TinyLM, loss_from_logits  # noqa: E402
from vw_scores import accumulate  # noqa: E402

torch.manual_seed(0)
V, T, N = 48, 12, 24
model = TinyLM(vocab=V, ctx=T, d_model=16, n_heads=2, d_head=8, d_mlp=24, seed=1)
with torch.no_grad():          # widen the weights so ablations are not all ~0
    for p in model.parameters():
        p.mul_(3.0)
toks = torch.randint(0, V, (N, T))
acc = accumulate(model, toks, 8, "cpu", log_every=10**9)
W, Hc, Fc = acc["W"], acc["H"], (0.5 * acc["W"].pow(2) * acc["C"])
d_vp = V + T


@torch.no_grad()
def brute_loss(Wm, m64, pos):
    """Held-out-free brute force, in FLOAT64. This matters: a single Tokens->Logits weight
    moves the mean loss by ~1e-9 nats, and a float32 mean of ~4 nats cannot resolve that --
    an earlier version of this test failed at 72% relative error purely on that account,
    with the closed form itself exact to 5 digits per position."""
    logits = Wm[toks] + Wm[pos] + sum(m64.paths(toks)[1:])
    return F.cross_entropy(logits[:, :-1].reshape(-1, V).double(),
                           toks[:, 1:].reshape(-1), reduction="mean")


m64 = TinyLM(vocab=V, ctx=T, d_model=16, n_heads=2, d_head=8, d_mlp=24, seed=1).double()
m64.load_state_dict({k: v.double() for k, v in model.state_dict().items()})
pos = V + torch.arange(T)
W64 = m64.W_TL()
base = brute_loss(W64, m64, pos)
print(f"model loss {base:.9f} | {d_vp}x{V} = {d_vp*V} weights | {N*(T-1)} positions")

# 20 random weights plus the 20 largest-|helpfulness| ones: a uniform sample of a 2880-weight
# family is almost all near-zero entries, which tests only the accumulator's zero.
g = torch.Generator().manual_seed(7)
idx = torch.cat([torch.randperm(d_vp * V, generator=g)[:20],
                 Hc.abs().flatten().argsort(descending=True)[:20]]).unique()
worst_abs, worst_rel, worst_f = 0.0, 0.0, 0.0
with torch.no_grad():
    p_all = model(toks)[:, :-1].float().softmax(-1)
for f in idx.tolist():
    i, j = f // V, f % V
    Wz = W64.clone()
    Wz[i, j] = 0.0
    got, want = float(brute_loss(Wz, m64, pos) - base), float(Hc[i, j])
    worst_abs = max(worst_abs, abs(got - want))
    if abs(want) > 1e-6:
        worst_rel = max(worst_rel, abs(got - want) / abs(want))
    act = (toks[:, :-1] == i) if i < V else (torch.arange(T - 1) == (i - V)).expand(N, T - 1)
    pj = p_all[..., j][act]
    want_f = float(Fc[i, j])
    got_f = float(0.5 * W[i, j] ** 2 * (pj * (1 - pj)).sum() / (N * (T - 1)))
    worst_f = max(worst_f, abs(got_f - want_f) / max(abs(want_f), 1e-12))

print(f"helpfulness: worst ABS error {worst_abs:.3e} (float32 accumulator floor ~1e-8)")
print(f"helpfulness: worst REL error {worst_rel:.3e} over weights with |h| > 1e-6")
print(f"fisher     : worst REL error {worst_f:.3e}")
assert worst_abs < 1e-7, worst_abs
assert worst_rel < 1e-3, worst_rel
assert worst_f < 1e-4, worst_f
print("OK - both closed forms match brute-force ablation")
