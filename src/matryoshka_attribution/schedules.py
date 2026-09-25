"""Sampling schedules for the number of kept nodes ``k`` and related helpers.

Canonical home for the ``k``-sampling logic shared by every entry point. Each schedule makes
exactly one ``torch.rand(1).item()`` draw per call (``log_both`` two), so seeded runs are
reproducible across scripts.
"""

import math

import torch


def sample_k(total: int, schedule: str = "uniform") -> float:
    """Sample a (float) number of kept nodes ``k``.

    Args:
        total: total number of score parameters.
        schedule: ``"uniform"`` samples ``k ~ Uniform(1, total)``; ``"log"`` samples
            ``k ~ exp(Uniform(log 1, log total))`` so 1-10 is as likely as 10-100;
            ``"log_both"`` spends half its draws on ``total - k`` for a log-uniform ``k``.

    One ``torch.rand(1)`` draw, matching the inlined implementations.
    """
    if schedule == "log":
        log_k = math.log(1) + (math.log(total) - math.log(1)) * torch.rand(1).item()
        return math.exp(log_k)
    if schedule == "log_both":
        # 50/50: log-uniform k (small k -> supervises the HEAD of the ranking) OR
        # total - log-uniform (small complement -> supervises the TAIL: which nodes to
        # exclude last). Two-sided so both ends of the ordering get gradient.
        L = math.exp(math.log(total) * torch.rand(1).item())   # log-uniform in [1, total]
        if torch.rand(1).item() < 0.5:
            return L
        return max(1.0, total - L)
    return 1.0 + (total - 1.0) * torch.rand(1).item()


class FixedK:
    """k_sampler that returns a constant ``k`` every step -- train the mask at a single fixed
    sparsity instead of sampling k from a schedule. ``observe`` is a no-op."""

    def __init__(self, k):
        self.k = int(k)

    def sample(self):
        return self.k

    def observe(self, *args):
        pass


class AdaptiveLogK:
    """Frontier-restricted log-uniform ``k`` sampler (single-scalar Robbins-Monro).

    Tracks one number, ``kmax_log`` = log of the frontier ``k_max`` past which the circuit
    is already "decided" (acc >= ``target``). Most steps sample a training k log-uniformly
    in ``(1, k_max]``; a fraction ``probe_frac`` of steps instead probe exactly at ``k_max``
    and take a constant sign step toward ``acc = target``:

        kmax_log += lr        if acc <  target   (top not saturated -> frontier higher)
        kmax_log -= lr        if acc >= target   (top saturated     -> frontier lower)

    Sign (not proportional) steps so descent from the unrestricted start is fast even when
    acc is pinned at 1 (a proportional ``target-acc`` signal is only -0.05 there and crawls).
    Self-correcting (no separate exploration: if k_max overshoots low, probes go
    under-target and push it back up); equilibrium oscillates +-lr in log around the k where
    acc=target. The probe reuses the loss forward. Clamped to ``[log(floor), log total]``.

    Use with ``learn_scores(..., k_sampler=AdaptiveLogK(total))``: the trainer calls
    ``.sample()`` each step; ``loss_fn`` calls ``.observe(acc)`` after the masked forward
    (acc measured at the just-sampled k).
    """

    def __init__(self, total, target=0.95, lr=0.1, probe_frac=0.25, floor=10.0):
        self.total = float(total)
        self.target = target
        self.lr = lr
        self.probe_frac = probe_frac
        self.logmax = math.log(self.total)
        self.floor_log = math.log(max(1.0, min(floor, total)))
        self.kmax_log = self.logmax       # start unrestricted (= plain log-uniform)
        self._probe = False               # was the last draw a frontier probe?

    def sample(self):
        if torch.rand(1).item() < self.probe_frac:
            self._probe = True
            return math.exp(self.kmax_log)            # probe exactly at the frontier
        self._probe = False
        return math.exp(self.kmax_log * torch.rand(1).item())   # log-uniform in (1, k_max]

    def observe(self, acc):
        if not self._probe:
            return
        self.kmax_log += self.lr if float(acc) < self.target else -self.lr
        self.kmax_log = min(self.logmax, max(self.floor_log, self.kmax_log))
