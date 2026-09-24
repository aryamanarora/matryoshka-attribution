"""Sigmoid top-k sparse autoencoder: MAttr's mask operator and k schedule inside an SAE.

A TopK SAE (Gao et al. 2024) trained with the recipe SAEBench's baselines were trained with
(dictionary_learning's ``AutoEncoderTopK`` / ``TopKTrainer``), with ONE change to the forward:
the hard top-k over post-ReLU latents is replaced by MAttr's ``sigmoid_topk`` mask (bisection
tau, implicit-diff backward, unchanged numerics), and ``k`` is redrawn every step from
``schedules.sample_k`` -- one k per batch, exactly as MAttr draws one k per step. Everything
else (init, AuxK, unit-norm decoder with parallel-gradient removal, grad clip, Adam) is the
TopKTrainer's, so SAEBench's released TopK SAEs isolate the operator + k schedule.

Parameters use SAEBench's layout (``W_enc`` [d_in, d_sae], ``W_dec`` [d_sae, d_in], ``b_enc``,
``b_dec``) so the state dict loads straight into ``sae_bench.custom_saes.topk_sae.TopKSAE``,
which applies a HARD top-k at whatever k it is built with: one trained SAE is scored at every
target L0. ``hard_encode`` below is that same encoder, for in-training diagnostics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sigmoid_topk import sigmoid_topk


class SigmoidTopKSAE(nn.Module):
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
        """Post-ReLU latent activations -- the scores both top-k operators rank."""
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def soft_encode(self, x, k, T=0.5, n_iters=50):
        """Training forward: latents gated by the sigmoid top-k mask (sum of mask = k).
        Returns (latents, scores, mask, magnitudes); here scores and magnitudes coincide."""
        a = self.acts(x)
        mask = sigmoid_topk(a, k, T=T, n_iters=n_iters)
        return a * mask, a, mask, a

    def hard_encode(self, x, k: int):
        """Inference forward (= SAEBench TopKSAE.encode): keep the k largest post-ReLU latents."""
        a = self.acts(x)
        top = a.topk(k, dim=-1, sorted=False)
        return torch.zeros_like(a).scatter_(-1, top.indices, top.values)

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def normalize_decoder(self):
        self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True) + torch.finfo(self.W_dec.dtype).eps

    @torch.no_grad()
    def remove_parallel_decoder_grad(self):
        """Project out the gradient component along each (unit) decoder row, so the step only
        rotates it -- dictionary_learning's remove_gradient_parallel_to_decoder_directions."""
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data / (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-6)
        self.W_dec.grad -= (self.W_dec.grad * w).sum(dim=1, keepdim=True) * w

    def state_dict_raw_space(self, scale: float):
        """State dict for activations in the ORIGINAL (un-normalized) space.

        Training sees x / scale. relu and top-k commute with a positive rescale, so the same
        W_enc / W_dec serve raw activations once both biases are multiplied by ``scale``
        (dictionary_learning's ``scale_biases``)."""
        sd = {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}
        sd["b_enc"] *= scale
        sd["b_dec"] *= scale
        return sd


class GatedSigmoidTopKSAE(SigmoidTopKSAE):
    """Scores decoupled from latent magnitudes (Gated SAE parameterisation, Rajamanoharan et al.
    2024), with the gate replaced by MAttr's sigmoid top-k:

        s = (x - b_dec) @ W_enc + gate_bias                      gate scores -> sigmoid_topk(s, k)
        m = relu((x - b_dec) @ (W_enc * exp(r_mag)) + b_mag)     magnitudes
        f = m * sigmoid_topk(s, k)

    Why: in SigmoidTopKSAE the ranked score IS the reconstruction magnitude, so it cannot grow
    to sharpen the sigmoid at fixed T -- the mask stays diffuse and a hard top-k at eval
    overshoots. Here the gate's scale is free, as MAttr's score logits are. Parameter names are
    sae_bench's GatedSAE's (b_enc unused, kept at 0 so both classes share the base init);
    evaluation takes the top-k latents by gate score and keeps their magnitudes.
    """

    def __init__(self, d_in: int, d_sae: int):
        super().__init__(d_in, d_sae)
        self.gate_bias = nn.Parameter(torch.zeros(d_sae))
        self.r_mag = nn.Parameter(torch.zeros(d_sae))
        self.b_mag = nn.Parameter(torch.zeros(d_sae))
        del self.b_enc

    def _enc(self, x):
        x_enc = (x - self.b_dec) @ self.W_enc
        return x_enc + self.gate_bias, F.relu(x_enc * self.r_mag.exp() + self.b_mag)

    def soft_encode(self, x, k, T=0.5, n_iters=50):
        s, m = self._enc(x)
        mask = sigmoid_topk(s, k, T=T, n_iters=n_iters)
        return m * mask, s, mask, m

    def hard_encode(self, x, k: int):
        s, m = self._enc(x)
        idx = s.topk(k, dim=-1, sorted=False).indices
        return torch.zeros_like(m).scatter_(-1, idx, m.gather(-1, idx))

    def state_dict_raw_space(self, scale: float):
        # Magnitudes rescale like b_dec; gate scores only get multiplied by `scale`, which
        # leaves their top-k (the only thing eval reads from them) unchanged.
        sd = {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}
        for b in ("b_dec", "gate_bias", "b_mag"):
            sd[b] *= scale
        return sd


ARCHS = {"sigtopk": SigmoidTopKSAE, "gated_sigtopk": GatedSigmoidTopKSAE}


@torch.no_grad()
def geometric_median(points, max_iter=100, tol=1e-5):
    """Weiszfeld iterations; TopKTrainer initialises b_dec with this on the first batch."""
    guess = points.mean(dim=0)
    for _ in range(max_iter):
        prev = guess
        w = 1 / torch.norm(points - guess, dim=1)
        w /= w.sum()
        guess = (w.unsqueeze(1) * points).sum(dim=0)
        if torch.norm(guess - prev) < tol:
            break
    return guess


def auxk_loss(sae, residual, acts, dead, k_aux):
    """TopKTrainer's AuxK: reconstruct the residual from the top-``k_aux`` DEAD latents,
    normalized by the residual's variance (OpenAI's normalization). 0 when nothing is dead."""
    n_dead = int(dead.sum())
    if n_dead == 0:
        return residual.new_zeros(())
    masked = torch.where(dead[None], acts, -torch.inf)
    top = masked.topk(min(k_aux, n_dead), dim=-1, sorted=False)
    f = torch.zeros_like(acts).scatter_(-1, top.indices, top.values)
    recon = f @ sae.W_dec                       # no b_dec: the residual already excludes it
    l2 = (residual.float() - recon.float()).pow(2).sum(dim=-1).mean()
    denom = (residual.float() - residual.float().mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean()
    return (l2 / denom).nan_to_num(0.0)


def fve(x, x_hat):
    """Batch fraction of variance explained -- a training diagnostic only; SAEBench's core eval
    computes its own ``explained_variance`` on openwebtext."""
    return float(1 - (x - x_hat).pow(2).sum() / (x - x.mean(dim=0, keepdim=True)).pow(2).sum())
