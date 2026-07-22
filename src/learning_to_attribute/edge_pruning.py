"""Edge Pruning (Bhaskar et al., 2024) mask learning: hard-concrete gates + Lagrangian L0.

Faithful port of the optimization recipe from https://github.com/princeton-nlp/Edge-Pruning
(``src/modeling/l0.py`` for the hard-concrete math, ``src/prune/fpt2_*.py`` +
``src/modeling/modeling_fpt2.py`` for the training loop), adapted to this repo's
environment-agnostic trainer interface (see ``trainer.learn_scores``): the caller supplies
``loss_fn(z) -> Tensor | None`` which applies the sampled [0,1] gate vector ``z`` to the
model (edge patching, node patching, ...) and returns the task loss — Edge Pruning's task
loss is a KL to the unmasked model's output distribution, but that lives in the caller.

This module owns everything else from the original recipe:
  - per-unit latent ``log_alpha`` parameters, init N(10, 0.01) (start fully dense);
  - stretched hard-concrete sampling (T=2/3, stretch [-0.1, 1.1], hardtanh to [0,1]);
  - sparsity Lagrangian ``l1*(s - s_t) + l2*(s - s_t)^2`` where ``s = 1 - z.sum()/total``
    and the multipliers are trained by gradient ASCENT (AdamW ``maximize=True`` group);
  - the sparsity target ``s_t`` annealed from ``start_sparsity`` to ``target_sparsity``
    over the first ``sparsity_warmup_frac`` of steps (linear or logarithmic);
  - AdamW with linear lr warmup + linear decay (their default: lr 0.8, warmup ~7%).

The final ``log_alphas`` are returned as ``TrainResult.scores``: they are the latent
importance scores, ranked as-is by the MIB eval (higher = keep clean).
"""

import math
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .trainer import TrainResult

# Hard-concrete constants (Edge-Pruning src/modeling/l0.py; Louizos et al., 2018).
LIMIT_LEFT = -0.1
LIMIT_RIGHT = 1.1
EPS = 1e-6
TEMPERATURE = 2 / 3
FACTOR = 0.8


def cdf_stretched_concrete(x: float, log_alpha: torch.Tensor) -> torch.Tensor:
    """P(z <= x) under the stretched concrete; ``1 - cdf(0)`` is P(gate is nonzero)."""
    x_01 = (x - LIMIT_LEFT) / (LIMIT_RIGHT - LIMIT_LEFT)
    intermediate = math.log(x_01) - math.log(1 - x_01)
    prob_unclamped = torch.sigmoid(TEMPERATURE * intermediate - log_alpha)
    return torch.clamp(prob_unclamped, EPS, 1 - EPS)


def sample_z_from_log_alpha(log_alpha: torch.Tensor) -> torch.Tensor:
    """Reparameterized hard-concrete sample in [0, 1] (grad flows to ``log_alpha``)."""
    u = torch.empty_like(log_alpha).uniform_(EPS, 1 - EPS)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / TEMPERATURE)
    z = (LIMIT_RIGHT - LIMIT_LEFT) * s + LIMIT_LEFT
    return F.hardtanh(z, 0.0, 1.0)


@torch.no_grad()
def deterministic_z_from_log_alpha(log_alpha: torch.Tensor) -> torch.Tensor:
    """Eval-time mask: zero out the E[#zeros] lowest gates (Edge-Pruning's eval binarizer)."""
    size = log_alpha.numel()
    expected_num_nonzeros = torch.sum(1 - cdf_stretched_concrete(0, log_alpha))
    num_zeros = int(torch.round(size - expected_num_nonzeros).item())
    soft_mask = torch.sigmoid(log_alpha / TEMPERATURE * FACTOR).reshape(-1)
    if num_zeros > 0:
        _, indices = torch.topk(soft_mask, k=num_zeros, largest=False)
        soft_mask[indices] = 0
    return soft_mask.reshape(log_alpha.shape)


def _target_sparsity_at(step: int, warmup_steps: int, start: float, target: float,
                        warmup_type: str) -> float:
    """Annealed sparsity target (get_current_edge_target_sparsity in fpt2_*.py)."""
    if step >= warmup_steps or warmup_steps == 0:
        return target
    if warmup_type == "linear":
        return start + (target - start) * step / warmup_steps
    # logarithmic: anneal log(1 - sparsity) linearly (spends more steps near the target)
    log_one_minus = math.log(1 - start) + (math.log(1 - target) - math.log(1 - start)) \
        * step / warmup_steps
    return 1 - math.exp(log_one_minus)


def learn_scores_edge_pruning(
    total: int,
    loss_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
    *,
    steps: int,
    target_sparsity: float,
    start_sparsity: float = 0.0,
    lr: float = 0.8,
    reg_lr: float = 0.8,
    lr_warmup_frac: float = 0.07,
    sparsity_warmup_frac: float = 0.83,
    warmup_type: str = "linear",
    init_mean: float = 10.0,
    init_std: float = 0.01,
    device="cpu",
    on_step: Optional[Callable[[int, float, float, torch.Tensor], None]] = None,
    log_every: int = 0,
    logger=None,
) -> TrainResult:
    """Learn ``log_alphas`` over ``total`` units by minimizing ``loss_fn(z) + Lagrangian``.

    Fraction-of-steps defaults mirror the Edge-Pruning IOI recipe (3000 steps, 200 lr-warmup,
    2500 sparsity-warmup) so they scale with ``steps``. Returns a :class:`TrainResult` whose
    ``scores`` are the final log-alphas; ``train_log`` rows are ``(step, kept, kept_frac,
    loss, 0)`` matching the eval_mib train-log schema (kept = sampled ``z.sum()``).
    """
    log_alphas = nn.Parameter(
        torch.empty(total, device=device).normal_(mean=init_mean, std=init_std))
    lambda_1 = nn.Parameter(torch.zeros(1, device=device))
    lambda_2 = nn.Parameter(torch.zeros(1, device=device))

    # Multipliers are ASCENDED on the same objective (their get_optimizers: maximize=True).
    optimizer = torch.optim.AdamW([
        {"params": [log_alphas], "lr": lr},
        {"params": [lambda_1, lambda_2], "lr": reg_lr, "maximize": True},
    ], lr=lr)
    lr_warmup_steps = int(round(lr_warmup_frac * steps))

    def lr_lambda(step):  # linear warmup then linear decay to 0 (their linear schedule)
        if step < lr_warmup_steps:
            return step / max(1, lr_warmup_steps)
        return max(0.0, (steps - step) / max(1, steps - lr_warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    sparsity_warmup_steps = int(round(sparsity_warmup_frac * steps))

    result = TrainResult(scores=log_alphas)
    t0 = time.time()
    for step in range(steps):
        target = _target_sparsity_at(step, sparsity_warmup_steps, start_sparsity,
                                     target_sparsity, warmup_type)
        optimizer.zero_grad()
        z = sample_z_from_log_alpha(log_alphas)
        task_loss = loss_fn(z)
        if task_loss is None:              # caller signalled skip (e.g. length mismatch)
            scheduler.step()               # keep lr/target schedules aligned with `step`
            continue
        model_sparsity = 1 - z.sum() / total
        reg_loss = lambda_1.reshape([]) * (model_sparsity - target) \
            + lambda_2.reshape([]) * (model_sparsity - target) ** 2
        loss = task_loss + reg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        kept = float(z.detach().sum().item())
        loss_val = float(task_loss.item())
        result.loss_log.append(loss_val)
        result.k_log.append(kept)
        result.train_log.append((step, kept, kept / total, loss_val, 0))
        if on_step is not None:
            on_step(step, kept, loss_val, log_alphas)
        if log_every and logger is not None and ((step + 1) % log_every == 0 or step == 0):
            rate = (step + 1) / (time.time() - t0)
            logger.info(
                "Step %4d/%d  task=%.4f  sparsity=%.4f (target %.4f)  l1=%.3f l2=%.3f  (%.1f step/s)",
                step + 1, steps, loss_val, float(model_sparsity.item()), target,
                float(lambda_1.item()), float(lambda_2.item()), rate)

    result.train_time_s = time.time() - t0
    if logger is not None:
        det_kept = int((deterministic_z_from_log_alpha(log_alphas) > 0).sum().item())
        logger.info("Final deterministic mask keeps %d/%d units (sparsity %.4f)",
                    det_kept, total, 1 - det_kept / total)
    result.scores = log_alphas.data.cpu()
    return result
