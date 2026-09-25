"""Mask-variant registry: the single place every MAttr masking ablation is defined.

The trainer (``trainer.learn_scores``) calls
``build_mask`` each step and dispatches the optimizer step on the returned
:class:`MaskResult` aux fields (REINFORCE manual gradient, L0 penalty).

All variants build on the frozen primitives in :mod:`matryoshka_attribution.sigmoid_topk`
(``sigmoid_topk``, ``sigmoid_topk_detached_tau``); their numerics must not change.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from .sigmoid_topk import sigmoid_topk, sigmoid_topk_detached_tau

# Variants that need a sampled ``k``.
VARIANTS = (
    "topk", "topk_detached", "topk_identity", "hard_topk", "hard_topk_identity",
    "hard_topk_identity_gumbel", "bernoulli_reinforce",
    "hard_concrete",
)


@dataclass
class MaskResult:
    """Output of a mask builder.

    ``mask`` is the differentiable tensor handed to the caller's ``loss_fn``. The aux
    fields tell the trainer how to take the gradient step:
      - ``reinforce`` present  -> manual REINFORCE grad (needs the scalar loss value too).
      - ``l0_scores`` present  -> add ``l0_lambda * l0_scores.sum()`` to the loss (L0 penalty).
    """
    mask: torch.Tensor
    reinforce: Optional[dict] = None       # {"tau", "sample", "T"}
    l0_scores: Optional[torch.Tensor] = None


def _hard_topk_indices(scores: torch.Tensor, k: float):
    ki = max(1, int(k))
    _, top_idx = scores.topk(ki)
    hard = torch.zeros_like(scores)
    hard[top_idx] = 1.0
    return hard


def build_mask(scores: torch.Tensor, k: float, variant: str = "topk",
               T: float = 0.5, n_iters: int = 50) -> MaskResult:
    """Construct the mask for ``variant`` at sparsity ``k``. See :data:`VARIANTS`."""
    if variant == "topk":
        return MaskResult(sigmoid_topk(scores, k=k, T=T, n_iters=n_iters))

    if variant == "topk_detached":
        return MaskResult(sigmoid_topk_detached_tau(scores, k=k, T=T, n_iters=n_iters))

    if variant == "topk_identity":
        # SOFT forward, IDENTITY backward (2026-09-18): the mask is sigmoid top-k, so at zero
        # scores the forward is the uniform interpolation t*1 of the gradient appendix, but the
        # Jacobian is I -- no gate slope sigma', no centring. With optimizer="none" the averaged
        # gradient is then exactly IG along the mask path (rho = the k-schedule's induced
        # density; uniform k -> plain IG), which is what the no-learning control should be.
        soft = sigmoid_topk(scores, k=k, T=T, n_iters=n_iters)
        return MaskResult(soft.detach() + (scores - scores.detach()))

    if variant == "hard_topk":
        hard = _hard_topk_indices(scores, k)
        soft = sigmoid_topk(scores, k=k, T=T, n_iters=n_iters)
        return MaskResult(hard - soft.detach() + soft)

    if variant == "hard_topk_identity":
        # hard top-k forward, IDENTITY straight-through backward (dm/ds = 1): score
        # gradient is purely g*delta per node (no sigmoid gate-slope, no temperature).
        hard = _hard_topk_indices(scores, k)
        return MaskResult(hard.detach() + (scores - scores.detach()))

    if variant == "hard_topk_identity_gumbel":
        # Gumbel(0,1) per score, hard top-k on the PERTURBED scores (stochastic selection),
        # but IDENTITY straight-through backward (dm/ds = 1) on the CLEAN scores — same
        # gradient as hard_topk_identity, just with noisy forward selection so always-on /
        # always-off nodes still flip in/out and get a gradient signal.
        gumbel = -torch.log(-torch.log(torch.rand_like(scores).clamp(1e-8, 1 - 1e-8)))
        perturbed = scores + gumbel
        hard = _hard_topk_indices(perturbed, k)
        return MaskResult(hard.detach() + (scores - scores.detach()))

    if variant == "bernoulli_reinforce":
        # Bernoulli with a k-adjusted threshold (bisection tau, same as sigmoid_topk) so
        # E[active] ~ k. Gradient handled by the trainer via the REINFORCE estimator.
        lo = scores.min() - 10 * T
        hi = scores.max() + 10 * T
        with torch.no_grad():
            for _ in range(n_iters):
                mid = (lo + hi) / 2
                f_mid = torch.sigmoid((scores - mid) / T).sum()
                if f_mid > k:
                    lo = mid
                else:
                    hi = mid
            tau = ((lo + hi) / 2).detach()
        probs = torch.sigmoid((scores - tau) / T)
        hard = torch.bernoulli(probs).detach()
        return MaskResult(hard, reinforce={"tau": tau, "sample": hard, "T": T})

    if variant == "hard_concrete":
        probs = torch.sigmoid(scores)
        hard = torch.bernoulli(probs)
        return MaskResult(hard - probs.detach() + probs, l0_scores=torch.sigmoid(scores))

    raise ValueError(f"Unknown mask variant: {variant!r}. Known: {VARIANTS}")
