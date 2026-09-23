"""Pure-torch checks of the mask builders, k-schedules and losses (no model, CPU)."""
import math

import torch

from matryoshka_attribution import build_mask, VARIANTS, sample_k, learn_scores
from matryoshka_attribution.losses import attribution_loss, LOSS_CHOICES
from matryoshka_attribution.modes import normalize_mode


def test_build_mask_every_variant():
    torch.manual_seed(0)
    s = torch.randn(50, requires_grad=True)
    for v in VARIANTS:
        m = build_mask(s, 7.0, v).mask
        assert m.shape == s.shape and torch.isfinite(m).all(), v


def test_sample_k_ranges():
    torch.manual_seed(0)
    for sched in ("uniform", "log", "log_both"):
        ks = [sample_k(1000, sched) for _ in range(200)]
        assert all(1.0 <= k <= 1000.0 for k in ks), sched


def test_losses_run_both_directions():
    torch.manual_seed(0)
    logits = torch.randn(4, 11, requires_grad=True)
    b, s = torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8])
    d = torch.ones(4)
    for name in LOSS_CHOICES:
        for corrupt_topk in (False, True):
            if name in ("kl", "cmd") and corrupt_topk:
                continue
            attribution_loss(name, logits, b, s, corrupt_topk=corrupt_topk, target_d=d,
                             corrupt_d=-d, clean_logp=logits.detach().log_softmax(-1)).backward()


def test_learn_scores_ranks_planted_units():
    torch.manual_seed(0)
    total, planted = 40, {3, 17, 29}
    target = torch.zeros(total); target[list(planted)] = 1.0

    def loss_fn(mask):          # keep the planted units: loss is the mass NOT on them
        return ((mask - target) ** 2).sum()

    res = learn_scores(total, loss_fn, steps=200, lr=0.1)
    top = set(res.scores.topk(3).indices.tolist())
    assert top == planted, top


def test_normalize_mode_folds_legacy_names():
    assert normalize_mode("sufficient") == "iso" and normalize_mode("cause") == "cause"
