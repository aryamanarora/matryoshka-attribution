"""TopK sparse autoencoder with an optional sigmoid top-k forward (MAttr's mask operator).

Deliberately minimal: the only loss is reconstruction MSE. The SAE is a TopK SAE (Gao et al.
2024) in SAEBench's parameter layout (``W_enc`` [d_in, d_sae], ``W_dec`` [d_sae, d_in],
``b_enc``, ``b_dec``), so a checkpoint loads straight into
``sae_bench.custom_saes.topk_sae.TopKSAE`` and is scored with a hard top-k at any target k.

Three forwards, same parameters:
  * ``hard_encode``: keep the k largest post-ReLU latents (TopK SAE; also the eval forward).
  * ``soft_encode``: gate every post-ReLU latent by ``sigmoid_topk`` (bisection tau,
    implicit-diff backward, sum of mask = k) -- MAttr's operator, numerics unchanged. Loose as a
    relaxation here: the latents rescale to absorb a flat mask (2026-09-24, soft_k20 run: every
    latent parked in the sigmoid's exponential tail, i.e. f = k * a * softmax(a/T), dense).
  * ``ste_encode``: hard forward, sigmoid top-k backward (straight-through).

The decoder rows are kept at unit norm (the TopK SAE constraint, which SAEBench checks), with the
gradient component along each row projected out before the step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigmoid_topk import sigmoid_topk


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in, self.d_sae = d_in, d_sae
        # AutoEncoderTopK init: unit-norm random decoder rows, encoder = decoder transpose.
        W_dec = torch.randn(d_sae, d_in)
        W_dec /= W_dec.norm(dim=1, keepdim=True)
        self.W_dec = nn.Parameter(W_dec)
        self.W_enc = nn.Parameter(W_dec.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def acts(self, x):
        """Post-ReLU latent activations -- what both top-k operators rank."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def hard_encode(self, x, k: int):
        a = self.acts(x)
        top = a.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(a).scatter_(-1, top.indices, top.values)

    def soft_encode(self, x, k: float, T=0.5, n_iters=50):
        """Returns (latents, mask)."""
        a = self.acts(x)
        mask = sigmoid_topk(a, k, T=T, n_iters=n_iters)
        return a * mask, mask

    def ste_encode(self, x, k: float, T=0.5, n_iters=50):
        """Hard top-round(k) forward, sigmoid top-k backward: mask = hard - soft.detach() + soft,
        MAttr's ``hard_topk`` variant (masks.build_mask). The forward equals ``hard_encode``, so
        the magnitudes cannot absorb a fractional gate; the sigmoid only shapes the gradient,
        giving latents just outside the top-k a ranking signal. Returns (latents, soft mask)."""
        a = self.acts(x)
        idx = a.topk(max(1, round(k)), dim=-1, sorted=False).indices
        hard = torch.zeros_like(a).scatter_(-1, idx, 1.0)
        soft = sigmoid_topk(a, k, T=T, n_iters=n_iters)
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
