"""Attribution training losses, shared across eval_sva / eval_mib / attribute.

A circuit is scored by an environment that applies the mask and produces last-token logits;
the loss turns those logits into a scalar to MINIMIZE. The intervention *direction* is encoded
by ``corrupt_topk``:

  - iso / sufficiency (corrupt_topk=False): the top-k circuit is kept CLEAN and should RETAIN
    the base behaviour -> reward predicting the BASE token / maximizing base-source margin.
  - cause / necessity (corrupt_topk=True): the top-k circuit is CORRUPTED to the source and
    should FORCE the SOURCE behaviour (the interchange counterfactual) -> reward predicting
    the SOURCE token / minimizing base-source margin.

Base-only losses (ce/logit/prob) therefore swap their target token between base and source;
the symmetric base-source losses (logit_diff/hinge/acc) just flip a sign.
"""

import torch
import torch.nn.functional as F

LOSS_CHOICES = ("logit_diff", "ce", "logit", "prob", "hinge", "acc")


def attribution_loss(name, logits, base_id, source_id, *, corrupt_topk,
                     hinge_margin=2.0, acc_temp=1.0):
    """Scalar loss to minimize.

    Args:
        name: one of LOSS_CHOICES.
        logits: [B, vocab] last-token logits from the masked forward.
        base_id, source_id: [B] long tensors -- the correct (base) and counterfactual (source)
            answer token ids.
        corrupt_topk: False = iso/sufficiency (target base), True = cause/necessity (force source).
        hinge_margin: margin (logits) for --loss hinge.
        acc_temp: temperature for --loss acc (smaller = sharper soft-0-1).
    """
    ar = torch.arange(logits.shape[0], device=logits.device)
    # base-only losses target the BASE token for iso, the SOURCE token for cause (force source)
    tgt = source_id if corrupt_topk else base_id
    if name == "ce":
        return F.cross_entropy(logits, tgt)
    if name == "logit":
        return -logits[ar, tgt].mean()
    if name == "prob":
        # softmax p(target): bounded in [0,1], self-saturates as p->1 (no gap-padding).
        return -logits.softmax(-1)[ar, tgt].mean()
    d = logits[ar, base_id] - logits[ar, source_id]          # base-source margin
    if name == "acc":
        # soft 0-1 / sigmoid surrogate for the accuracy metric 1[d>0]: gradient peaks at the
        # decision boundary and vanishes for confidently-correct AND hopeless examples.
        return torch.sigmoid((-d if not corrupt_topk else d) / acc_temp).mean()
    if name == "hinge":
        # margin hinge: flip the decision by `hinge_margin`, then saturate (no gradient once
        # decided). iso wants d>=margin; cause wants -d>=margin.
        return F.relu(hinge_margin - (d if not corrupt_topk else -d)).mean()
    if name == "logit_diff":
        return d.mean() if corrupt_topk else -d.mean()
    raise ValueError(f"unknown loss {name!r}; known: {LOSS_CHOICES}")


def resolve_direction(mode, corrupt_topk):
    """Per-step intervention direction. joint mode flips a coin between iso and cause each step
    (one circuit trained to both recover base and force source); else the fixed direction."""
    if mode == "joint":
        return torch.rand(1).item() < 0.5
    return corrupt_topk
