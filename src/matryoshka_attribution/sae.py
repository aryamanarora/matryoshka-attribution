"""TopK sparse autoencoder with an optional sigmoid top-k forward (MAttr's mask operator).

Deliberately minimal: the only loss is reconstruction MSE. The SAE is a TopK SAE (Gao et al.
2024) in SAEBench's parameter layout (``W_enc`` [d_in, d_sae], ``W_dec`` [d_sae, d_in],
``b_enc``, ``b_dec``), so a checkpoint loads straight into
``sae_bench.custom_saes.topk_sae.TopKSAE`` and is scored with a hard top-k at any target k
(scores=acts/pre; scores=sep adds ``W_score``/``b_score`` and is scored by
scripts/sae/eval_core.py's ScoredTopKSAE).

Latents are always ``a = relu((x - b_dec) @ W_enc + b_enc)``; what the top-k ranks (the SCORES)
is set by ``scores``:
  * ``acts``: a itself (a plain TopK SAE). Every relu-zeroed latent then ties at score 0, and
    under the sigmoid those ties soak up the sum-of-mask = k budget (2026-09-24, ste run: at
    k~470 only ~18 latents had mask > 0.5), pulling tau far above the k-th latent.
  * ``pre``: the pre-relu encoder output z (same matmul). Ranks positive latents as ``acts`` does,
    so the hard forward is unchanged, but the inactive latents spread out below 0.
  * ``sep``: a separate matmul ``s = (x - b_dec) @ W_score + b_score``, no relu -- free real-valued
    logits in the role of MAttr's scores. Selection by s, magnitude from a.

Three forwards, same parameters:
  * ``hard_encode``: keep the k top-scoring latents' activations (TopK SAE; also the eval forward).
  * ``soft_encode``: gate every latent by ``sigmoid_topk(scores)`` (bisection tau,
    implicit-diff backward, sum of mask = k) -- MAttr's operator, numerics unchanged. Loose as a
    relaxation with scores=acts: the latents rescale to absorb a flat mask (2026-09-24, soft_k20
    run: every latent parked in the sigmoid's exponential tail, i.e. f = k * a * softmax(a/T)).
  * ``ste_encode``: hard forward, sigmoid top-k backward (straight-through).

The decoder rows are kept at unit norm (the TopK SAE constraint, which SAEBench checks), with the
gradient component along each row projected out before the step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigmoid_topk import sigmoid_topk


SCORES = ("acts", "pre", "sep")


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, scores: str = "acts"):
        super().__init__()
        assert scores in SCORES, scores
        self.d_in, self.d_sae, self.scores = d_in, d_sae, scores
        # AutoEncoderTopK init: unit-norm random decoder rows, encoder = decoder transpose.
        W_dec = torch.randn(d_sae, d_in)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        if scores == "sep":                      # same init as the encoder, untied from here on
            self.W_score = nn.Parameter(W_dec.T.clone())
            self.b_score = nn.Parameter(torch.zeros(d_sae))

    def acts(self, x):
        """Post-ReLU latent activations -- the magnitudes every forward keeps."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def scores_and_acts(self, x):
        z = (x - self.b_dec) @ self.W_enc + self.b_enc
        a = F.relu(z)
        if self.scores == "acts":
            return a, a
        if self.scores == "pre":
            return z, a
        return (x - self.b_dec) @ self.W_score + self.b_score, a

    def hard_encode(self, x, k: int):
        s, a = self.scores_and_acts(x)
        idx = s.topk(k, dim=-1, sorted=False).indices
        return torch.zeros_like(a).scatter_(-1, idx, a.gather(-1, idx))

    def soft_encode(self, x, k: float, T=0.5, n_iters=50):
        """Returns (latents, mask)."""
        s, a = self.scores_and_acts(x)
        mask = sigmoid_topk(s, k, T=T, n_iters=n_iters)
        return a * mask, mask

    def ste_encode(self, x, k: float, T=0.5, n_iters=50):
        """Hard top-round(k) forward, sigmoid top-k backward: mask = hard - soft.detach() + soft,
        MAttr's ``hard_topk`` variant (masks.build_mask). The forward equals ``hard_encode``, so
        the magnitudes cannot absorb a fractional gate; the sigmoid only shapes the gradient,
        giving latents just outside the top-k a ranking signal. Returns (latents, soft mask)."""
        s, a = self.scores_and_acts(x)
        idx = s.topk(max(1, round(k)), dim=-1, sorted=False).indices
        hard = torch.zeros_like(s).scatter_(-1, idx, 1.0)
        soft = sigmoid_topk(s, k, T=T, n_iters=n_iters)
        return a * (hard - soft.detach() + soft), soft

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True) + torch.finfo(self.W_dec.dtype).eps

    @torch.no_grad()
    def remove_parallel_decoder_grad(self):
        """Project out the gradient along each unit decoder row, so the step only rotates it."""
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-6)
        self.W_dec.grad -= (self.W_dec.grad * w).sum(dim=1, keepdim=True) * w


@torch.no_grad()
def geometric_median(points, max_iter=100, tol=1e-5):
    """Weiszfeld iterations; the TopK recipe initialises b_dec with this on the first batch."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        w = 1 / torch.norm(points - guess, dim=1)
        w /= w.sum()
        guess = (w.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def fve(x, x_hat):
    """Batch fraction of variance explained -- a training diagnostic only; SAEBench's core eval
    computes its own ``explained_variance`` on openwebtext."""
    return float(1 - (x - x_hat).pow(2).sum() / (x - x.mean(dim=0, keepdim=True)).pow(2).sum())
