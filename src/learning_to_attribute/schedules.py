"""Sampling schedules for the number of kept nodes ``k`` and related helpers.

Canonical home for the ``k``-sampling logic that was copy-pasted across
``scripts/eval_mib.py`` (lines ~265-269), ``scripts/eval_mib_edge.py`` (``sample_k``),
``scripts/attribute.py`` (``sample_k``), and the toy scripts. Each function makes exactly
one ``torch.rand(1).item()`` draw in the same place as the originals, so seeded runs stay
bit-identical after migration.
"""

import math

import torch


def sample_k(total: int, schedule: str = "uniform") -> float:
    """Sample a (float) number of kept nodes ``k``.

    Args:
        total: total number of score parameters.
        schedule: ``"uniform"`` samples ``k ~ Uniform(1, total)``; ``"log"`` samples
            ``k ~ exp(Uniform(log 1, log total))`` so 1-10 is as likely as 10-100.

    One ``torch.rand(1)`` draw, matching the inlined implementations.
    """
    if schedule == "log":
        log_k = math.log(1) + (math.log(total) - math.log(1)) * torch.rand(1).item()
        return math.exp(log_k)
    return 1.0 + (total - 1.0) * torch.rand(1).item()


def sample_k_sum_pow2(n: int) -> list[float]:
    """Powers-of-two ``k`` grid ``{1, 2, 4, ..., n-1}`` (deterministic, no RNG draw).

    Matches ``toy_linear_mattr.py``'s ``sum_pow2`` schedule: the masked loss is summed over
    every ``k`` in this grid each step. Returned as floats for use with ``build_mask``.
    """
    ks: list[float] = []
    v = 1
    while v < n:
        ks.append(float(v))
        v *= 2
    if n - 1 >= 1 and float(n - 1) not in ks:
        ks.append(float(n - 1))
    return ks


def natural_k(scores: torch.Tensor) -> float:
    """``k`` implied by the current threshold: number of non-negative scores (min 1).

    Matches ``attribute.py``'s ``natural_k_frac`` selection. Distinct from ``eval_mib.py``'s
    bias-step feature (which trains a global bias); this just reads off ``k`` from the scores.
    """
    return float(max(1, int((scores.detach() >= 0).sum().item())))
