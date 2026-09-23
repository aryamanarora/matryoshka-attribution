"""Sparsity sweep for learned attribution scores (the hand-rolled CPR-style eval).

Generalizes the original CausalGym ``_eval_sparsity`` routine: rank nodes by the
learned scores, and at each sparsity level keep the top-k as a hard mask and measure how
well the masked model matches the reference. The metric computation (KL / CE / logit-diff /
MSE / ...) is the caller's via ``apply_and_eval(hard_mask) -> {metric_name: value}``, so this
function is environment-agnostic. A fixed-seed random ordering is included as a baseline.

NOTE: this is the toy / CausalGym sparsity sweep. MIB's official CPR (``evaluate_area_under_
curve`` in eval_mib.py) is separate and untouched.
"""

from typing import Callable

import torch


def sparsity_sweep(
    scores: torch.Tensor,
    total: int,
    sparsities: list,
    apply_and_eval: Callable[[torch.Tensor], dict],
    *,
    seed: int = 0,
    include_random: bool = True,
    device="cpu",
) -> dict:
    """Evaluate the learned ranking across ``sparsities``.

    Args:
        scores: learned scores ``[total]`` (higher = more important).
        total: number of nodes.
        sparsities: fractions of nodes to keep, e.g. ``[0.001, ..., 1.0]``.
        apply_and_eval: callback that takes a hard 0/1 mask ``[total]`` and returns a dict
            ``{metric_name: float}`` (sets the mask, runs the forward, computes metrics).
        seed: RNG seed for the random-ordering baseline (matches the original ``manual_seed(0)``).
        include_random: also evaluate a random ordering for reference.

    Returns:
        ``{"sparsities": [...], "learned": {metric: [val per sparsity]},
           "random": {metric: [...]}}`` (``"random"`` omitted if ``include_random`` is False).
    """
    flat = scores.detach().clone().to(device)
    sorted_idx = flat.argsort(descending=True)
    orderings = [("learned", sorted_idx)]
    if include_random:
        g_prev = torch.random.get_rng_state()
        torch.manual_seed(seed)
        random_idx = torch.randperm(total, device=device)
        torch.random.set_rng_state(g_prev)        # don't perturb caller's RNG stream
        orderings.append(("random", random_idx))

    out = {"sparsities": list(sparsities)}
    for name, ordering in orderings:
        acc: dict = {}
        for frac in sparsities:
            k = max(1, int(frac * total))
            hard_mask = torch.zeros(total, device=device)
            hard_mask[ordering[:k]] = 1.0
            metrics = apply_and_eval(hard_mask)
            for mname, val in metrics.items():
                acc.setdefault(mname, []).append(val)
        out[name] = acc
    return out
