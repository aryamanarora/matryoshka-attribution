"""The shared MAttr learning algorithm: ``learn_scores``.

Owns the optimization (param init, k-sampling, mask construction, optimizer/loop, and the
REINFORCE / L0 special cases). The *environment* — how a mask is applied to a
model and turned into a scalar loss — is supplied by the caller as ``loss_fn(mask)``. Torch
hooks (or SAE masking, or an analytic product) live in the caller's closure; this module is
environment-agnostic and communicates only via the differentiable mask tensor, so gradients
flow trainer -> loss_fn -> (hooks) -> scores.

``loss_fn(mask) -> Tensor | None``: apply the mask, run the forward, return the scalar loss
(keep it attached to the graph). Return ``None`` to skip the step (e.g. an empty batch).
"""

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

from .masks import build_mask
from .schedules import sample_k


@dataclass
class TrainResult:
    scores: torch.Tensor                       # final scores, on CPU
    loss_log: list = field(default_factory=list)
    k_log: list = field(default_factory=list)
    train_log: list = field(default_factory=list)   # (step, k, k_frac, loss)
    train_time_s: float = 0.0


def learn_scores(
    total: int,
    loss_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
    *,
    steps: int,
    variant: str = "topk",
    k_schedule: str = "uniform",
    k_avg: int = 1,
    T: float = 0.5,
    n_iters: int = 50,
    lr: float = 0.01,
    optimizer: str = "adam",
    adam_eps: float = 1e-8,
    adam_betas: tuple = (0.9, 0.999),
    sgd_momentum: float = 0.0,
    sgd_dampening: float = 0.0,
    grad_norm: float = 0.0,
    l0_lambda: float = 0.0,
    device="cpu",
    on_step: Optional[Callable[[int, float, float, torch.Tensor], None]] = None,
    log_every: int = 0,
    logger=None,
    k_sampler=None,
) -> TrainResult:
    """Learn attribution ``scores`` over ``total`` nodes by minimizing ``loss_fn(mask)``.

    ``variant`` selects the masking ablation (see ``masks.VARIANTS``); ``k_schedule`` is a
    ``schedules.sample_k`` schedule; ``optimizer`` in ``{adam, sgd, none}``. (Dropped 2026-08-26: ``lr_schedule``, the ``use_bias``/``natural_k_frac``
    bias-step, and ``k_schedule="natural"`` -- no headline result used them.)

    ``optimizer="none"`` (2026-09-18) is MAttr WITHOUT LEARNING: the scores stay at their init
    (zero) for every forward, and the returned scores are the mean over steps of the NEGATED
    gradient at that point -- i.e. the first-step update of the paper's gradient appendix, estimated
    by Monte Carlo over the k-schedule and the data: centred, path-reweighted IG in mask space
    (rho = 6 t (1-t) under uniform k, 2t under log k, 1 under logit k), through the soft top-k
    Jacobian and with the recomputation of intervened parents attached. Same per-step cost as
    learning (one forward + backward); no learning rate.
    """
    frozen = optimizer == "none"
    scores = nn.Parameter(torch.zeros(total, device=device))
    main_params = [scores]
    opt_cls = torch.optim.SGD if optimizer in ("sgd", "none") else torch.optim.Adam
    # Adam's eps is a real knob at neuron scale, not a numerical guard. With ~2.3M mask logits
    # most per-step gradients are far below the default 1e-8, so m_hat/(sqrt(v_hat)+eps) ~
    # sign(g) and the learned score degenerates to a signed COUNT of steps -- every trace of
    # effect MAGNITUDE is divided out. Raising eps above the typical |g| restores magnitude
    # sensitivity and interpolates Adam -> SGD-with-momentum (at effective lr = lr/eps).
    # SGD momentum closes the eps story's loop: Adam at eps >> |g| is (lr/eps)*EMA(g), i.e.
    # SGD with EMA momentum (momentum=mu, dampening=mu in torch's parameterisation). Plain
    # heavy-ball (dampening=0) instead SUMS gradients with steady-state gain 1/(1-mu).
    adam_kw = ({"momentum": sgd_momentum, "dampening": sgd_dampening}
               if optimizer in ("sgd", "none") else {"eps": adam_eps, "betas": tuple(adam_betas)})
    # frozen: lr 0 keeps the forward at the init while the loop below still computes .grad.
    groups = [{"params": main_params, "lr": 0.0 if frozen else lr, **adam_kw}]
    frozen_acc, frozen_n = (torch.zeros(total, dtype=torch.float64, device=device), 0) if frozen else (None, 0)
    optimizer_ = opt_cls(groups)

    result = TrainResult(scores=scores)
    t0 = time.time()
    for step in range(steps):
        # k-averaging: average the gradient over `k_avg` independent k-draws per step. Since k
        # is fixed within a batch, this is the only knob that reduces *k-schedule* variance
        # (batching only averages example noise). k_avg=1 is the original single-draw step.
        optimizer_.zero_grad()
        ks, losses = [], []
        for _ka in range(k_avg):
            if k_sampler is not None:
                # adaptive sampler owns k; loss_fn feeds it back per-step via k_sampler.observe(acc)
                k = k_sampler.sample()
            else:
                k = sample_k(total, k_schedule)

            mr = build_mask(scores, k, variant, T=T, n_iters=n_iters)

            loss = loss_fn(mr.mask)
            if loss is None:                   # caller signalled skip (e.g. empty batch)
                continue
            if mr.l0_scores is not None:
                loss = loss + l0_lambda * mr.l0_scores.sum()
            if mr.reinforce is not None:
                # REINFORCE: grad = loss * d/ds log P(sample | scores); accumulate across draws
                with torch.no_grad():
                    r = mr.reinforce
                    p = torch.sigmoid((scores - r["tau"]) / r["T"])
                    g = loss.item() * ((r["sample"] - p) / r["T"])
                    scores.grad = g if scores.grad is None else scores.grad + g
            else:
                loss.backward()               # accumulates into .grad across draws
            lv = loss.item()
            ks.append(float(k)); losses.append(lv)
        if not losses:                             # every draw skipped
            continue
        if k_avg > 1:                              # mean gradient (keep effective lr ~ bs=1)
            for pgrp in optimizer_.param_groups:
                for prm in pgrp["params"]:
                    if prm.grad is not None:
                        prm.grad /= len(losses)
        # Grad norm BEFORE the step, for the log line only. Read here because the optimizer may
        # rewrite .grad in place. Diagnostic for the residual-SAE runaway: there the accumulated
        # update implies per-step gradients ~1e12 near init, which no learning rate can absorb --
        # `loss` alone cannot distinguish an exploding gradient from a merely large one.
        # float64: the fp32 norm silently overflows to inf at this width (see the clip below),
        # which would make the log read `inf` on the very steps worth reading.
        gnorm = (float(scores.grad.detach().double().norm())
                 if scores.grad is not None else float("nan"))
        # PER-STEP GRADIENT CLIPPING (grad_norm > 0): cap this step's score gradient at norm
        # `grad_norm`. SCALE DOWN ONLY -- min(1, cap/|grad|), never up. Motivated by the
        # residual-SAE failure, where |grad| swings ~10 orders of magnitude across k-draws (2086
        # at k=1.4M, 1e-14 at k=64) and one large-k step sets the whole score vector, freezing it
        # by step 3.
        #
        # *** DO NOT "IMPROVE" THIS INTO NORMALISATION (scaling every step TO the cap). *** That
        # was tried on 2026-08-29 and is catastrophically worse: a step with |grad|=1e-14 gets
        # multiplied by 1e14, so |s|max reached 8.3e18 (lr=0.1) / 1.2e21 (lr=1.0) by step 200 and
        # froze. On this substrate most steps are numerically-dead (saturated sigmoid at small
        # k), and their SMALL NORM is the only thing marking them as noise -- normalising erases
        # exactly that signal. It is the same failure as adam eps=1e-8 (sign(g)), which promotes
        # dead steps to full-weight votes and lands at chance, but unbounded instead of capped.
        #
        # Clipping keeps all three properties that matter: spikes cannot dominate ACROSS steps,
        # noise steps stay small, and relative magnitudes WITHIN a step are untouched.
        # *** THE NORM MUST BE TAKEN IN float64. *** scores.grad.norm() in fp32 over millions of
        # elements overflows to inf once elements reach ~1e20 (5e6 * (1e20)^2 = 5e46 > 3.4e38
        # fp32 max) even though every element is finite. A guard of the form
        # `math.isfinite(gnorm) and gnorm > cap` then SKIPS the clip on exactly the steps that
        # need it -- observed 2026-08-29: step 1 correctly capped at 0.0714, then one overflowing
        # step slipped through unclipped and |s|max hit 9.1e23 by step 200.
        #
        # A genuinely non-finite gradient (inf/nan element, not just an overflowing norm) carries
        # no usable direction, so that step is ZEROED rather than passed through or rescaled --
        # rescaling nan by anything is still nan.
        if grad_norm > 0 and scores.grad is not None:
            gn64 = float(scores.grad.detach().double().norm())
            if not math.isfinite(gn64):
                scores.grad.zero_()
            elif gn64 > grad_norm:
                scores.grad.mul_(grad_norm / gn64)
        optimizer_.step()
        if frozen and scores.grad is not None:
            frozen_acc -= scores.grad.detach().double()   # goodness = -loss, as SGD would subtract
            frozen_n += 1
        k = sum(ks) / len(ks)                      # mean k for logging
        loss_val = sum(losses) / len(losses)

        result.loss_log.append(loss_val)
        result.k_log.append(float(k))
        result.train_log.append((step, float(k), float(k) / total, loss_val))
        if on_step is not None:
            # live scores passed for callers that track recovery metrics during training
            # (toys); read-only — do not mutate. wandb-style loggers can ignore it.
            on_step(step, float(k), loss_val, scores)
        if log_every and logger is not None and ((step + 1) % log_every == 0 or step == 0):
            rate = (step + 1) / (time.time() - t0)
            logger.info("Step %4d/%d  loss=%.4f  |grad|=%.4g  |s|max=%.4g  k=%.0f/%d  "
                        "(%.1f step/s)", step + 1, steps, loss_val, gnorm,
                        float(scores.data.abs().max()), k, total, rate)

    result.train_time_s = time.time() - t0
    result.scores = ((frozen_acc / max(frozen_n, 1)).float().cpu() if frozen
                     else scores.data.cpu())
    return result


def expected_gradients(
    total: int,
    grad_fn: Callable,
    *,
    steps: int,
    k_schedule: str = "uniform",
    log_every: int = 0,
    logger=None,
) -> TrainResult:
    """Expected Gradients sharing MAttr's schedules and loop conventions: the gradient-only sibling
    of ``learn_scores``.

    ``grad_fn(draw_alphas) -> (dloss, loss_val) | None`` is the environment, mirroring
    ``loss_fn(mask)``: sample a batch, call ``draw_alphas(batch_size)`` for a float tensor of
    per-EXAMPLE interpolation points, run ONE forward with each example's embedding
    interpolated at its own alpha along the caller's path -- in this repo always the
    INPUT-EMBEDDING path, ``emb(alpha) = base + alpha * (emb_clean - base)`` -- backward the
    LOSS, and return the per-node contraction ``dL/da_j . delta_j`` as a [total] tensor plus
    the loss value. Return ``None`` to skip the step. Scores are the NEGATED mean of the
    returned vectors (goodness = -loss, matching ``loss_fn``'s minimize convention),
    accumulated in float64 on ``grad_fn``'s device and returned on CPU.

    Each alpha is an independent ``sample_k(total, k_schedule) / total`` draw from the shared
    schedule; ``"uniform"`` is canonical Expected Gradients (alpha ~ U(0,1) up to the 1/total floor
    ``sample_k`` puts on k -- negligible at real node counts, visible in small-``total``
    tests), anything else a p(alpha)-weighted path integral (e.g. "log" weights the
    near-baseline end by ~1/alpha).

    Per-example alpha is FREE variance reduction (batch rows are independent, so giving each
    its own interpolation point costs no extra compute) and matches the round-1 / eval_sva
    ``mc_ig`` convention. NOTE the deliberate asymmetry with MAttr: ``learn_scores`` fixes
    ONE ``k`` per forward (the mask is shared across the batch) and pays a full forward per
    extra draw (``k_avg``), so at equal steps x batch the IG arm sees batch-size x more
    alpha-draws than MAttr sees k-draws. Read comparisons as COMPUTE-matched, not
    draw-matched.

    Shared with a ``learn_scores`` run: ``steps``, ``k_schedule``, the batch handling inside
    the environment closure, the global-RNG seeding convention, and the
    :class:`TrainResult`/``train_log`` row schema ``(step, mean_k, mean_alpha, loss)``.
    Per-step cost is one forward + one backward, so equal ``steps`` is compute-comparable
    to MAttr.
    """
    acc = None
    result = TrainResult(scores=torch.zeros(total))
    t0 = time.time()
    n_used = 0
    for step in range(steps):
        drawn = []

        def draw_alphas(n):
            a = torch.tensor([sample_k(total, k_schedule) for _ in range(n)],
                             dtype=torch.float32) / total
            drawn.append(a)
            return a

        out = grad_fn(draw_alphas)
        if out is None:                        # caller signalled skip (e.g. empty batch)
            continue
        dloss, lv = out
        if acc is None:
            acc = torch.zeros(total, dtype=torch.float64, device=dloss.device)
        acc -= dloss.double()                  # goodness = -loss
        n_used += 1
        lv = float(lv)
        alpha = float(torch.cat(drawn).mean()) if drawn else float("nan")
        result.loss_log.append(lv)
        result.k_log.append(alpha * total)
        result.train_log.append((step, alpha * total, alpha, lv))
        if log_every and logger is not None and ((step + 1) % log_every == 0 or step == 0):
            rate = (step + 1) / (time.time() - t0)
            logger.info("IG   %4d/%d  loss=%.4f  mean alpha=%.4f  k=%.0f/%d  (%.1f step/s)",
                        step + 1, steps, lv, alpha, alpha * total, total, rate)
    if acc is not None and n_used:
        result.scores = (acc / n_used).float().cpu()
    result.train_time_s = time.time() - t0
    return result
