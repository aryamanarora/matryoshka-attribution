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

``learn_scores_dcm`` is a third: DCM's raw box-constrained coefficient plus the PID
controller that drives it to a pinned density (see that function's docstring).
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


class PIDController:
    """Drives a sparsity coefficient along a target trajectory, in log space.

    Verbatim in structure from ``roonbug/belief_dynamics``
    (``causal-experiments/exps/patching/dcm/run.py:PIDController``, commit fb3ddd3):

        u = kp*rate_error + ki*count_error + kd*d(rate_error)
        log_mult <- clamp(log_mult + u)

    ``rate_error`` is (target - actual) pruned-units-per-update, which reacts immediately
    when pruning plateaus; ``count_error`` is (target - actual) cumulative pruned count,
    which is the integral of the rate error and is bounded by ``total`` so it cannot wind
    up. Updates are in log space so the gains mean a *relative* change in the coefficient
    regardless of its magnitude. Their defaults ship kd=0, i.e. a PI controller.
    """

    def __init__(self, kp: float, ki: float, kd: float, init_mult: float,
                 mult_min: float = 1e-8, mult_max: float = 1e8, d_clip: float = 5.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.log_mult = math.log(max(init_mult, mult_min))
        self.log_mult_min = math.log(mult_min)
        self.log_mult_max = math.log(mult_max)
        self.d_clip = d_clip
        self._prev_rate_error = 0.0
        self._prev_n_pruned = None

    @property
    def mult(self) -> float:
        return math.exp(self.log_mult)

    def step(self, actual_rate: float, target_rate: float, count_error: float) -> float:
        rate_error = target_rate - actual_rate
        derivative = max(-self.d_clip, min(self.d_clip, rate_error - self._prev_rate_error))
        self._prev_rate_error = rate_error
        u = self.kp * rate_error + self.ki * count_error + self.kd * derivative
        self.log_mult = max(self.log_mult_min, min(self.log_mult_max, self.log_mult + u))
        return self.mult


def learn_scores_dcm(
    total: int,
    loss_fn: Callable[[torch.Tensor], Optional[torch.Tensor]],
    *,
    steps: int,
    lr: float = 1e-1,
    target_density: float = 0.05,
    penalty_mode: str = "additive",
    init_mult: float = 0.025,
    ramp_frac: float = 0.8,
    pid_kp: float = 0.1,
    pid_ki: float = 1e-3,
    pid_kd: float = 0.0,
    tie_eps: float = 1e-4,
    device="cpu",
    on_step: Optional[Callable[[int, float, float, torch.Tensor], None]] = None,
    log_every: int = 0,
    logger=None,
) -> TrainResult:
    """DCM's box-constrained mask + a PID controller that lands it on a pinned density.

    The mask parameterization is Prakash et al. (ICLR 2024) ``experiment_2/DCM.py`` as
    re-implemented in ``roonbug/belief_dynamics``
    (``causal-experiments/exps/patching/dcm/run.py``), which agree on all of it:

      - ``mask = nn.Parameter(torch.ones(total))`` — a RAW coefficient, not a logit. No
        sigmoid, no temperature anneal, no lr schedule.
      - the box constraint is enforced by ``mask.clamp_(0, 1)`` after each optimizer step.
      - ``Adam(lr=1e-1)``, constant.
      - the circuit is ``round(mask)`` at 0.5.
      - sparsity term on the mask, weighted by ``mult`` (see ``penalty_mode``).

    ``penalty_mode`` picks which of the two published sparsity terms is used, and the
    default is NOT belief_dynamics':

      - ``"additive"`` (default) is Prakash et al.'s: ``mult * mask.mean()``, no task-loss
        factor (``DCM.py:189``, ``lamb*torch.sum(1-mask)`` at lambda=0.01, modulo the
        normalisation and this repo's sign convention). Gradient ``mult/total`` per unit,
        constant, and defined for any task loss.
      - ``"multiplicative"`` is belief_dynamics' ``mult * task_loss.detach() * mask.mean()``
        (``run.py:222``), i.e. total ``= task_loss * (1 + mult*mean(mask))``. Two properties
        follow, and they are the method's, not a porting artifact: pruning pressure vanishes
        exactly when the circuit is working (which is most of why the controller has to
        exist), and **the sign of the penalty is the sign of the task loss**. That is safe
        for a divergence (their JS, our ``--loss kl``) and WRONG for a loss that goes
        negative: under ``--loss logit_diff`` the term becomes a reward for keeping the mask
        dense, the controller escalates ``mult`` to fight a gap it is widening, and the run
        rails at ``mult=1e8`` with density 1.0 against any target. Guarded below rather than
        silently corrected — an inverted penalty is not a variant of DCM, it is a broken
        run.

    Sign convention: this repo's ``loss_fn(z)`` takes z=1 to mean "keep clean", so the
    circuit is the set held at 1 and the penalty shrinks it — structurally identical to
    DCM, where the circuit is the set held at 1 and patched from the source. What differs
    is sufficiency vs necessity, which is the caller's business (``--mode``), not the
    parameterization's.

    Three deliberate deviations, all forced:

      - **The pruning-order tie-break is OURS, not theirs** — see ``Returns`` below for the
        mechanism. Upstream never sorts: no ``argsort``/``topk``/``sort`` in ``DCM.py``, which
        rounds and evaluates the resulting SET, and gets its multi-sparsity story by RETRAINING
        per ``lamb`` (``DCM.py:144``), not by ranking one run.

        The mask is nonetheless a continuous latent in [0,1], so it is sortable in principle —
        it just isn't, in practice. Measured at lr 0.1 on the three cells we ran, the fraction
        sitting at *exactly* 0.0 or 1.0 is 91-100% (100%, i.e. fully degenerate, on 2 of 3
        cells at the 1% pin). The additive penalty's per-unit gradient is the constant
        ``mult/total``, so each unit is pushed monotonically until it hits a boundary and
        ``clamp_`` holds it; nothing restores it to the interior. Contrast the other two gates,
        whose latent is an UNBOUNDED logit — the gate saturates but the parameter keeps moving
        and stays sortable. Here the parameter IS the gate.

        So MIB's rank-then-top-k harness needs an order for the two blocks, and this is the
        least arbitrary one available. ``tie_eps=1e-4`` is chosen below the smallest observed
        interior gap (3.5e-4), so the tie-break orders WITHIN the 0- and 1-blocks and never
        reorders a genuine interior value. It is
        the widest of the three deviations in effect, not the narrowest: it is inert at the
        pinned density but decides every other point of MIB's sweep, and since ``area_under``
        is a LINEAR trapezoid over 0.001..1.0 (evaluation.py:63), ~90% of which comes from
        k >= 20%, a DCM CPR-AUC is mostly a statement about pruning order rather than about
        the circuit. Runs whose mask collapsed to zero units still post competitive AUC on
        it. Report CPR at the pin (``collect_dcm_sweep.py``), never the AUC, and see
        ``scripts/dcm_rank_agreement.py`` for how much of a rho-vs-other-methods this term
        generates on its own.
      - **The setpoint.** Their controller ramps the pruned count to *all* units and runs
        until the circuit is empty (``pid_target_frac=1.0`` + early stop), because they
        want the whole sparsity trajectory. We are pinning one density, so the ramp goes
        to ``(1 - target_density) * total`` over the first ``ramp_frac`` of training and
        then HOLDS. Past the ramp their target rate stays positive and unsatisfiable, so
        their coefficient escalates without bound; holding the setpoint makes this an
        actual regulator, which is what "train at a given sparsity" requires.
      - **float32, not bfloat16.** Their mask is bf16 (``run.py:144``), whose resolution
        near 0.5 is ~0.004 — and 0.5 is where ``round()`` decides the circuit. Nothing
        else in this repo trains a mask in bf16.

    Batch size is the caller's (1 here, resampled per step), matching every other mask
    baseline in this repo; see ``docs/`` on the deliberate 32->1 deviation in the Edge
    Pruning port, which applies verbatim to DCM's batch of 32.

    Returns a :class:`TrainResult` whose ``scores`` are ``mask + tie_eps * (last step at
    which the unit was still in the circuit) / steps``. DCM produces a SET, not a ranking:
    ``clamp_`` piles values onto exactly 0.0 and 1.0, so a raw-mask ranking would hand the
    MIB harness large blocks of exact ties and let tie order decide the interior of the
    sparsity curve. The tie-break is the unit's pruning order along the training
    trajectory, which is the one extra ordering the method does generate. It is inert at
    the pinned density — there ``round(mask)`` already has k units and top-k selects
    exactly them — and only does anything at the other points of MIB's sweep.
    """
    if not 0.0 < target_density <= 1.0:
        raise ValueError(f"target_density must be in (0, 1], got {target_density}")
    if penalty_mode not in ("additive", "multiplicative"):
        raise ValueError(f"penalty_mode must be 'additive' or 'multiplicative', "
                         f"got {penalty_mode!r}")

    mask = nn.Parameter(torch.ones(total, device=device))
    optimizer = torch.optim.Adam([mask], lr=lr)          # no scheduler: theirs has none
    pid = PIDController(pid_kp, pid_ki, pid_kd, init_mult=init_mult)
    mult = pid.mult

    target_pruned = (1.0 - target_density) * total
    ramp_steps = max(1, int(ramp_frac * steps))
    # last_active[i] = last step at which unit i was still in round(mask); pruning order.
    last_active = torch.zeros(total, device=device)

    result = TrainResult(scores=mask)
    t0 = time.time()
    for step in range(steps):
        optimizer.zero_grad()
        task_loss = loss_fn(mask)
        if task_loss is None:                 # caller signalled skip (e.g. length mismatch)
            continue
        if penalty_mode == "multiplicative":
            scale = task_loss.detach()
            if float(scale) < 0.0:
                raise ValueError(
                    f"penalty_mode='multiplicative' needs a non-negative task loss (DCM's is "
                    f"a JS divergence); got {float(scale):.4g} at step {step}. The penalty's "
                    f"sign follows the loss's, so this would reward density. Use "
                    f"penalty_mode='additive' (Prakash et al.'s form) or a divergence loss.")
            penalty = mult * scale * mask.mean()
        else:
            penalty = mult * mask.mean()
        pen_val = float(penalty.item())
        (task_loss + penalty).backward()
        optimizer.step()
        with torch.no_grad():
            mask.clamp_(0, 1)
            rounded = torch.round(mask.data)
            last_active[rounded > 0] = step + 1
            n_pruned = float((rounded == 0).sum().item())

        # PID: setpoint is a linear ramp of the pruned COUNT, held after ramp_steps.
        prev = pid._prev_n_pruned
        actual_rate = 0.0 if prev is None else n_pruned - prev
        pid._prev_n_pruned = n_pruned
        progress = min(1.0, (step + 1) / ramp_steps)
        target_n_pruned = target_pruned * progress
        target_rate = target_pruned / ramp_steps if step + 1 <= ramp_steps else 0.0
        mult = pid.step(actual_rate, target_rate, target_n_pruned - n_pruned)

        kept = total - n_pruned
        loss_val = float(task_loss.item())    # task only -- comparable across recipes
        result.loss_log.append(loss_val)
        result.k_log.append(kept)
        result.train_log.append((step, kept, kept / total, loss_val, 0))
        if on_step is not None:
            on_step(step, kept, loss_val, mask)
        if log_every and logger is not None and ((step + 1) % log_every == 0 or step == 0):
            rate = (step + 1) / (time.time() - t0)
            logger.info(
                "Step %4d/%d  task=%.4f  pen=%.4f  kept=%d/%d (density %.4f, target %.4f)  "
                "mult=%.4g  mask[min/mean/max]=%.3f/%.3f/%.3f  (%.1f step/s)",
                step + 1, steps, loss_val, pen_val, int(kept), total, kept / total,
                1.0 - target_n_pruned / total, mult,
                float(mask.min()), float(mask.mean()), float(mask.max()), rate)

    result.train_time_s = time.time() - t0
    final_kept = int((torch.round(mask.data) > 0).sum().item())
    if logger is not None:
        logger.info(
            "Final circuit has %d/%d units (density %.4f; pinned target %.4f, %+d units), "
            "final mult=%.4g",
            final_kept, total, final_kept / total, target_density,
            final_kept - round(target_density * total), mult)
    result.scores = (mask.data + tie_eps * (last_active / max(1, steps))).cpu()
    return result
