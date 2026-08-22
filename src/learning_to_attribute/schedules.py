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
            ``k ~ exp(Uniform(log 1, log total))`` so 1-10 is as likely as 10-100;
            ``"logit"`` samples ``k/total`` logit-uniformly, the schedule under which a
            zero-init SGD run's expected score is exactly activation-path IG (see below).

    One ``torch.rand(1)`` draw, matching the inlined implementations.
    """
    if schedule == "log":
        log_k = math.log(1) + (math.log(total) - math.log(1)) * torch.rand(1).item()
        return math.exp(log_k)
    if schedule == "logit":
        # k = total * sigmoid(u), u ~ U(-ln(total-1), +ln(total-1)); i.e. alpha = k/total is
        # LOGIT-uniform on [1/total, 1-1/total]. Equivalently: sample sigmoid_topk's bisection
        # threshold tau uniformly instead of sampling the count.
        #
        # WHY THIS ONE: at zero init the mask is uniform at alpha = k/total, the intervention is
        # h = alpha*clean + (1-alpha)*cf, and sigmoid_topk's implicit-diff backward multiplies
        # dL/dm by the gate slope sp = alpha*(1-alpha). So the score a zero-init SGD run
        # accumulates is a PATH INTEGRAL of the attribution g.delta with weight
        #     w(alpha) ∝ alpha*(1-alpha) * p(alpha).
        # Choosing p ∝ 1/(alpha*(1-alpha)) cancels the gate slope exactly and leaves w flat --
        # so MAttr+SGD's expected update becomes activation-path integrated gradients, up to a
        # positive constant and a rank-irrelevant mean shift. It is the UNIQUE such p.
        # (For contrast: "log" gives w ∝ 1-alpha, "uniform" gives w ∝ alpha*(1-alpha).)
        #
        # The exactness is an argument about the near-zero-score regime only; once the scores
        # spread past T the mask is no longer uniform and there is no single alpha. Also note
        # this schedule spends ~half its draws at alpha -> 1, where the gate slope makes the
        # step tiny, so per-step signal is weaker than "log" at matched step count -- the
        # variance is the price of the unbiasedness.
        L = math.log(total - 1.0)
        u = -L + 2.0 * L * torch.rand(1).item()
        return total / (1.0 + math.exp(-u))
    if schedule == "log_both":
        # 50/50: log-uniform k (small k -> supervises the HEAD of the ranking) OR
        # total - log-uniform (small complement -> supervises the TAIL: which nodes to
        # exclude last). Two-sided so both ends of the ordering get gradient.
        L = math.exp(math.log(total) * torch.rand(1).item())   # log-uniform in [1, total]
        if torch.rand(1).item() < 0.5:
            return L
        return max(1.0, total - L)
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


def natural_k(scores: torch.Tensor) -> float:
    """``k`` implied by the current threshold: number of non-negative scores (min 1).

    Matches ``attribute.py``'s ``natural_k_frac`` selection. Distinct from ``eval_mib.py``'s
    bias-step feature (which trains a global bias); this just reads off ``k`` from the scores.
    """
    return float(max(1, int((scores.detach() >= 0).sum().item())))
