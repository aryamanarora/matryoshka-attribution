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

``learn_scores_sigmoid_mask`` is a second, unrelated mask parameterization living in the
same file because it plugs into the same ``loss_fn(z)`` interface: pyvene's
``SigmoidMaskIntervention`` + its ``train_alignment`` temperature anneal (see that
function's docstring).
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


def learn_scores_sigmoid_mask(
    total: int,
    loss_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
    *,
    steps: int,
    lr: float = 1e-3,
    warmup_frac: float = 0.1,
    init_value: float = 0.0,
    init_temperature: float = 0.01,
    temperature_start: float = 50.0,
    temperature_end: float = 0.1,
    l1_coeff: float = 0.0,
    l1_target: str = "gate",
    device="cpu",
    on_step: Optional[Callable[[int, float, float, torch.Tensor], None]] = None,
    log_every: int = 0,
    logger=None,
) -> TrainResult:
    """pyvene's ``SigmoidMaskIntervention`` recipe, verbatim, over ``total`` units.

    A deliberately literal port of https://github.com/stanfordnlp/pyvene — the point of this
    baseline is that it is the *other* published way to learn a binary mask over model
    components, so the knobs are theirs, not ours:

      - ``mask = nn.Parameter(torch.zeros(embed_dim))`` and gate ``z = sigmoid(mask / temp)``
        (``models/interventions.py:SigmoidMaskIntervention``). Deterministic — no concrete
        noise — and no L0/Lagrangian term in the *library*: with ``l1_coeff=0`` (the default)
        the only loss is ``loss_fn(z)``. See the ``l1_coeff`` note below before repeating the
        claim that "pyvene has no sparsity term" — the library doesn't, but pyvene's own
        tutorial for this class does.
      - temperature annealed ``torch.linspace(50.0, 0.1, steps).to(torch.bfloat16)``, read
        AFTER each ``optimizer.step()`` (``models/intervenable_base.py:train_alignment``), so
        step 0 runs at the intervention's own ``__init__`` temperature of 0.01 and step ``t``
        runs at ``schedule[t-1]``. The bf16 cast is theirs and is kept: it quantizes the
        schedule to ~3 significant bits.
      - ``Adam(lr=1e-3)`` + HF ``get_linear_schedule_with_warmup`` with 10% warmup.

    Note what the anneal does and does not do. It anneals how *binary* the gate is (a mask
    of 1.0 gives z=0.51 at temp 50 and z≈1 at temp 0.1); it does not anneal, or in any way
    constrain, how *sparse* the mask is — that is the axis Edge Pruning's Lagrangian owns and
    this method simply has no opinion about. With a task loss alone the optimum is the dense
    mask; the baseline is usable in the MIB eval because the eval ranks units by score and
    sweeps sparsity itself.

    ``l1_coeff > 0`` adds a penalty, which is what pyvene's *tutorial* for this class does
    (``tutorials/advanced_tutorials/IOI_with_Mask_Intervention.ipynb``:
    ``loss + coeff * torch.norm(v.mask, 1)`` at ``coeff=1``; Boundless DAS gets the same
    treatment via ``2.0 * v.intervention_boundaries.sum()``, which is the L1 the paper
    describes). Two targets, because they are not the same thing:

      - ``l1_target="logit"`` is the tutorial's term verbatim, ``coeff * mask.abs().sum()``.
        Beware: our logits init at 0 and the gate is ``sigmoid(mask/temp)``, so shrinking
        ``|mask|`` drives every gate toward ``z=0.5``, i.e. toward ~50% density — the exact
        band the unpenalised runs already sit in. It is a magnitude regulariser, not a
        sparsity one. Kept for the "we ran pyvene's own term" row, not because it can sparsify.
      - ``l1_target="gate"`` (default) is ``coeff * z.mean()``, the standard L1 relaxation of
        L0 and the term that actually pushes density down. Normalised by ``total`` so one
        coefficient means the same thing across the 156--1056 gates of different MIB models;
        pyvene does not normalise because it masks one 768-dim site.

    Either way the penalty is excluded from ``loss_log``/the logged ``task=`` figure, so those
    stay comparable with the ``l1_coeff=0`` runs.

    In pyvene ``temperature`` is an ``nn.Parameter``, but the forward reads it through
    ``torch.tensor(self.temperature)``, which detaches — so it never receives gradient and is
    only ever written by the scheduler. It is a plain buffer here, which is what it is there.

    Returns a :class:`TrainResult` whose ``scores`` are the final mask logits (higher = keep
    clean, matching the log-alpha convention above).
    """
    from transformers import get_linear_schedule_with_warmup

    if l1_target not in ("gate", "logit"):
        raise ValueError(f"l1_target must be 'gate' or 'logit', got {l1_target!r}")

    mask = nn.Parameter(torch.full((total,), init_value, device=device))
    optimizer = torch.optim.Adam([mask], lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_frac * steps, num_training_steps=steps)
    temperature_schedule = torch.linspace(
        temperature_start, temperature_end, steps).to(torch.bfloat16).to(device)
    temperature = torch.tensor(init_temperature, device=device)

    result = TrainResult(scores=mask)
    t0 = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        z = torch.sigmoid(mask / temperature)
        task_loss = loss_fn(z)
        if task_loss is None:              # caller signalled skip (e.g. length mismatch)
            scheduler.step()               # keep the lr schedule aligned with `step`
            temperature = temperature_schedule[step]
            continue
        if l1_coeff:
            penalty = l1_coeff * (z.mean() if l1_target == "gate" else mask.abs().sum())
            pen_val = float(penalty.item())
            (task_loss + penalty).backward()
        else:
            pen_val = 0.0
            task_loss.backward()
        optimizer.step()
        scheduler.step()
        temperature = temperature_schedule[step]   # theirs: set after the step, index total_step

        kept = float(z.detach().sum().item())
        loss_val = float(task_loss.item())      # task only -- comparable across l1_coeff
        result.loss_log.append(loss_val)
        result.k_log.append(kept)
        result.train_log.append((step, kept, kept / total, loss_val, 0))
        if on_step is not None:
            on_step(step, kept, loss_val, mask)
        if log_every and logger is not None and ((step + 1) % log_every == 0 or step == 0):
            rate = (step + 1) / (time.time() - t0)
            logger.info(
                "Step %4d/%d  task=%.4f  l1=%.4f  soft-kept=%.1f/%d (density %.3f)  temp=%.3g  "
                "mask[min/mean/max]=%.3f/%.3f/%.3f  (%.1f step/s)",
                step + 1, steps, loss_val, pen_val, kept, total, kept / total,
                float(temperature),
                float(mask.min()), float(mask.mean()), float(mask.max()), rate)

    result.train_time_s = time.time() - t0
    if logger is not None:
        hard_kept = int((mask.data > 0).sum().item())
        logger.info("Final mask has %d/%d units with logit > 0 (sparsity %.4f)",
                    hard_kept, total, 1 - hard_kept / total)
    result.scores = mask.data.cpu()
    return result
