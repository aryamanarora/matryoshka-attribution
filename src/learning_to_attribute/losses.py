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

LOSS_CHOICES = ("logit_diff", "ce", "logit", "prob", "hinge", "acc", "ld_tanh",
                "ld_match", "ld_match_rel", "ld_norm", "kl", "cmd")

# Losses that need the per-example CLEAN-model margin passed as ``target_d``. Callers use this
# to decide whether to pay the (cached) clean forward.
CLEAN_TARGET_LOSSES = ("ld_match", "ld_match_rel", "ld_norm", "cmd")
# ...the per-example FULLY-PATCHED margin (mask = 0, every variable at its source value) as
# ``corrupt_d`` -- the "corrupted" reference of MIB's faithfulness normalisation.
CORRUPT_TARGET_LOSSES = ("cmd",)
# ...the clean model's last-token LOG-PROBABILITIES over the vocabulary as ``clean_logp`` [B, V].
CLEAN_LOGITS_LOSSES = ("kl",)


def attribution_loss(name, logits, base_id, source_id, *, corrupt_topk,
                     hinge_margin=2.0, acc_temp=1.0, ld_scale=2.0, target_d=None,
                     corrupt_d=None, clean_logp=None):
    """Scalar loss to minimize.

    Args:
        name: one of LOSS_CHOICES.
        logits: [B, vocab] last-token logits from the masked forward.
        base_id, source_id: [B] long tensors -- the correct (base) and counterfactual (source)
            answer token ids.
        corrupt_topk: False = iso/sufficiency (target base), True = cause/necessity (force source).
        hinge_margin: margin (logits) for --loss hinge.
        acc_temp: temperature for --loss acc (smaller = sharper soft-0-1).
        ld_scale: margin scale (logits) for --loss ld_tanh.
        target_d: [B] per-example CLEAN-model margins, required by the ld_match* and cmd losses.
        corrupt_d: [B] per-example FULLY-PATCHED margins (mask = 0), required by the cmd loss.
        clean_logp: [B, vocab] clean-model last-token log-probabilities, required by the kl loss.
    """
    ar = torch.arange(logits.shape[0], device=logits.device)
    if name == "kl":
        # KL(p_clean || p_masked) over the whole vocabulary at the last position (2026-09-21):
        # the mask-learning objective of Edge Pruning (Bhaskar et al.), matched to the CLEAN
        # model's actual prediction rather than to a label. Sufficiency direction only: the
        # circuit is asked to reproduce the full model's distribution, not just its argmax.
        assert not corrupt_topk, "kl is defined for the sufficient/iso direction only"
        assert clean_logp is not None, "kl needs the clean model's log-probs (clean_logp)"
        return F.kl_div(logits.log_softmax(-1), clean_logp, log_target=True, reduction="batchmean")
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
    if name == "ld_tanh":
        # PER-SAMPLE BOUNDED margin. logit_diff averages the RAW margin, so one example driven
        # to d=+40 outweighs ten examples sitting at d=-1: the optimizer can lower the loss by
        # padding an already-decided example instead of flipping an undecided one ("gap
        # padding"). tanh(d/scale) is in (-1,1), so every example contributes at most 1 and the
        # gradient decays once |d| >> scale -- the accuracy-shaped part of logit_diff, kept
        # smooth (unlike hinge, which is exactly 0 past the margin) and equal to
        # logit_diff/scale for small |d|.
        t = torch.tanh((d if corrupt_topk else -d) / ld_scale)
        return t.mean()
    if name == "ld_norm":
        # PER-EXAMPLE NORMALISED, still a MAXIMISATION. Completes the 2x2 with ld_tanh
        # (bounded, global scale) and ld_match_rel (normalised, overshoot-penalised): each
        # example's margin is divided by its OWN clean margin (floored at 1 logit), then
        # averaged -- so an example the clean model decides by 20 logits earns 10x less per
        # logit of padding than one it decides by 2, but the loss stays monotone in d and
        # never penalises overshoot. Isolates "normalise across samples" from "penalise
        # overshoot" as candidate fixes.
        assert target_d is not None, "ld_norm needs per-example clean margins (target_d)"
        dn = d / target_d.abs().clamp_min(1.0)
        return dn.mean() if corrupt_topk else -dn.mean()
    if name == "cmd":
        # TRAIN-TIME CMD (2026-09-21): MIB's faithfulness of THIS example at THIS budget,
        # f = (d - d_corr) / (d_clean - d_corr), penalised by |1 - f| -- the integrand of the
        # CMD metric (area between the faithfulness curve and 1). Unlike every margin loss it
        # penalises overshoot (f > 1) exactly as it penalises shortfall, and it is normalised
        # per example by the clean-minus-patched gap, floored at 1 logit (sign kept) so that
        # near-tie examples do not blow up. Sufficiency direction only, like ld_match.
        assert not corrupt_topk, "cmd is defined for the sufficient/iso direction only"
        assert target_d is not None and corrupt_d is not None, \
            "cmd needs per-example clean (target_d) and fully-patched (corrupt_d) margins"
        gap = target_d - corrupt_d
        gap = torch.where(gap >= 0, gap.clamp_min(1.0), gap.clamp_max(-1.0))
        f = (d - corrupt_d) / gap
        return (1.0 - f).abs().mean()
    if name in ("ld_match", "ld_match_rel"):
        # MATCHING loss, not a maximisation. Every other margin loss here is monotone in d, so
        # none of them ever PENALISES overshoot -- ld_tanh makes padding a decided example
        # worthless, but nothing makes it costly. This one drives each example's margin to the
        # CLEAN model's margin on that same example: the training-time analogue of
        # "faithfulness clipped at 1", with per-example normalisation in the _rel variant
        # (divide the error by |d_clean|, floored at 1 logit so near-tie examples don't blow
        # up). Iso/sufficient only: matching-to-clean is the wrong target when the top-k is
        # corrupted (that direction would have to match the PATCHED margin instead).
        assert not corrupt_topk, "ld_match* losses are defined for the sufficient/iso direction only"
        assert target_d is not None, "ld_match* needs per-example clean margins (target_d)"
        err = d - target_d
        if name == "ld_match_rel":
            err = err / target_d.abs().clamp_min(1.0)
        return (err ** 2).mean()
    raise ValueError(f"unknown loss {name!r}; known: {LOSS_CHOICES}")


def resolve_direction(mode, corrupt_topk):
    """Per-step intervention direction. joint mode flips a coin between iso and cause each step
    (one circuit trained to both recover base and force source); else the fixed direction."""
    if mode == "joint":
        return torch.rand(1).item() < 0.5
    return corrupt_topk
