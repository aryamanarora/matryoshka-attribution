"""The shared MAttr learning algorithm: ``learn_scores``.

Owns the optimization (param init, k-sampling, mask construction, optimizer/loop, and the
REINFORCE / L0 / bias-step special cases). The *environment* — how a mask is applied to a
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

from .masks import build_mask, build_bias_mask
from .schedules import sample_k, natural_k


def _lr_at(lr0: float, schedule: str, step: int, steps: int) -> float:
    """Per-step learning rate. ``const`` (default), ``cosine`` (lr0->0), ``sqrt`` (1/sqrt(t))."""
    if schedule == "cosine":
        return lr0 * 0.5 * (1 + math.cos(math.pi * step / steps))
    if schedule == "sqrt":
        return lr0 / math.sqrt(1 + step / 200)
    return lr0


@dataclass
class TrainResult:
    scores: torch.Tensor                       # final scores, on CPU
    bias: Optional[torch.Tensor] = None        # final bias (if use_bias), on CPU
    loss_log: list = field(default_factory=list)
    k_log: list = field(default_factory=list)
    # (step, k, k_frac, loss, bias_step) -- matches eval_mib.py train_log schema
    train_log: list = field(default_factory=list)
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
    lr_schedule: str = "const",
    optimizer: str = "adam",
    l0_lambda: float = 0.0,
    natural_k_frac: float = 0.0,
    use_bias: bool = False,
    manual_backward: bool = False,
    extra_params: Optional[list] = None,
    lr_extra: Optional[float] = None,
    extra_optimizer: Optional[str] = None,
    device="cpu",
    init_scores: Optional[torch.Tensor] = None,
    on_step: Optional[Callable[[int, float, float, torch.Tensor], None]] = None,
    log_every: int = 0,
    logger=None,
    k_sampler=None,
) -> TrainResult:
    """Learn attribution ``scores`` over ``total`` nodes by minimizing ``loss_fn(mask)``.

    Args mirror the knobs previously inlined in eval_mib.py / attribute.py / the toys:
    ``variant`` selects the masking ablation (see ``masks.VARIANTS``); ``k_schedule`` in
    ``{uniform, log, natural}``; ``optimizer`` in ``{adam, sgd}``; ``use_bias`` +
    ``natural_k_frac`` enable the bias-step; ``extra_params``/``lr_extra`` add a second
    param group (e.g. DAS rotations). ``init_scores`` overrides the zero init.
    """
    if init_scores is not None:
        scores = nn.Parameter(init_scores.to(device).clone())
    else:
        scores = nn.Parameter(torch.zeros(total, device=device))
    bias = nn.Parameter(torch.zeros(1, device=device)) if use_bias else None

    main_params = [scores] + ([bias] if use_bias else [])
    opt_cls = torch.optim.SGD if optimizer == "sgd" else torch.optim.Adam
    groups = [{"params": main_params, "lr": lr}]
    extra_optimizer_ = None
    if extra_params:
        elr = lr_extra if lr_extra is not None else lr
        if extra_optimizer is not None and extra_optimizer != optimizer:
            # extra params (e.g. DAS rotation) get their OWN optimizer -- lets the scores use
            # SGD (id-STE magnitude signal) while the rotation uses Adam (manifold param).
            ecls = torch.optim.SGD if extra_optimizer == "sgd" else torch.optim.Adam
            extra_optimizer_ = ecls([{"params": list(extra_params), "lr": elr}])
        else:
            groups.append({"params": list(extra_params), "lr": elr})
    optimizer_ = opt_cls(groups)

    result = TrainResult(scores=scores, bias=bias)
    t0 = time.time()
    for step in range(steps):
        if lr_schedule != "const":
            for g in optimizer_.param_groups:
                base = lr_extra if (extra_params and g is optimizer_.param_groups[-1]
                                    and lr_extra is not None) else lr
                g["lr"] = _lr_at(base, lr_schedule, step, steps)
        # k-averaging: average the gradient over `k_avg` independent k-draws per step. Since k
        # is fixed within a batch, this is the only knob that reduces *k-schedule* variance
        # (batching only averages example noise). k_avg=1 is the original single-draw step.
        optimizer_.zero_grad()
        if extra_optimizer_ is not None:
            extra_optimizer_.zero_grad()
        ks, losses = [], []
        for _ka in range(k_avg):
            # natural_k_frac coin (one draw when >0). With use_bias it selects a bias-step
            # (eval_mib); without, it selects natural-k for this step's k (attribute.py).
            hit = (torch.rand(1).item() < natural_k_frac) if natural_k_frac > 0 else False
            bias_step = use_bias and hit
            if hit and not use_bias:
                k = natural_k(scores)
            elif k_schedule == "natural":
                k = natural_k(scores)
            elif k_sampler is not None:
                # adaptive sampler owns k; loss_fn feeds it back per-step via k_sampler.observe(acc)
                k = k_sampler.sample()
            else:
                k = sample_k(total, k_schedule)

            mr = build_bias_mask(scores, bias, T) if bias_step else \
                build_mask(scores, k, variant, T=T, n_iters=n_iters)

            if manual_backward:
                assert mr.reinforce is None and mr.l0_scores is None, \
                    "manual_backward is incompatible with REINFORCE / hard_concrete variants"
                loss = loss_fn(mr.mask)
                lv = float(loss) if loss is not None else float("nan")
            else:
                loss = loss_fn(mr.mask)
                if loss is None:                   # caller signalled skip (e.g. empty batch)
                    continue
                if mr.l0_scores is not None:
                    loss = loss + l0_lambda * mr.l0_scores.sum()
                if mr.reinforce is not None and not bias_step:
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
            for opt in (optimizer_, extra_optimizer_):
                if opt is None:
                    continue
                for pgrp in opt.param_groups:
                    for prm in pgrp["params"]:
                        if prm.grad is not None:
                            prm.grad /= len(losses)
        optimizer_.step()
        if extra_optimizer_ is not None:
            extra_optimizer_.step()
        k = sum(ks) / len(ks)                      # mean k for logging
        loss_val = sum(losses) / len(losses)

        result.loss_log.append(loss_val)
        result.k_log.append(float(k))
        result.train_log.append((step, float(k), float(k) / total, loss_val, int(bias_step)))
        if on_step is not None:
            # live scores passed for callers that track recovery metrics during training
            # (toys); read-only — do not mutate. wandb-style loggers can ignore it.
            on_step(step, float(k), loss_val, scores)
        if log_every and logger is not None and ((step + 1) % log_every == 0 or step == 0):
            rate = (step + 1) / (time.time() - t0)
            logger.info("Step %4d/%d  loss=%.4f  k=%.0f/%d  bias=%d  (%.1f step/s)",
                        step + 1, steps, loss_val, k, total, int(bias_step), rate)

    result.train_time_s = time.time() - t0
    result.scores = scores.data.cpu()
    if bias is not None:
        result.bias = bias.data.cpu()
    return result
