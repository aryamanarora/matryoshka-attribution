"""Unit-test the HF-side AttnLRP attention rule against the vanilla eager attention.

Two checks:
  (1) the half-rule is EXACT -- grad wrt q, k scaled by 1/4 and wrt v by 1/2, with the forward
      value bit-identical (this is the whole claim of attnlrp_attention_forward);
  (2) install_attnlrp on a real model leaves the logits unchanged and produces a gradient that
      is neither the vanilla one nor RelP's.

Run: uv run python scripts/test_attnlrp_hf.py [--model Qwen/Qwen2.5-0.5B]
"""
import argparse

import torch
from transformers.models.llama.modeling_llama import eager_attention_forward

from learning_to_attribute.grad_attribution import (
    attnlrp_attention_forward, install_attnlrp, install_relp, revert_relp)


class _Mod:
    """Minimal stand-in for an HF attention module (what the attn interface actually reads)."""
    num_key_value_groups = 2
    training = False


def test_half_rule():
    torch.manual_seed(0)
    B, Hq, Hkv, P, D = 2, 4, 2, 7, 8
    scaling = D ** -0.5
    mask = torch.full((B, 1, P, P), 0.0)
    mask = mask.masked_fill(torch.triu(torch.ones(P, P, dtype=torch.bool), 1), float("-inf"))

    def run(fn):
        torch.manual_seed(1)
        q = torch.randn(B, Hq, P, D, requires_grad=True)
        k = torch.randn(B, Hkv, P, D, requires_grad=True)
        v = torch.randn(B, Hkv, P, D, requires_grad=True)
        out, _ = fn(_Mod(), q, k, v, mask, scaling, dropout=0.0)
        # a non-symmetric scalar so the q/k gradients do not cancel
        (out * torch.arange(out.numel(), dtype=out.dtype).view(out.shape)).sum().backward()
        return out.detach(), q.grad, k.grad, v.grad

    o_ref, gq, gk, gv = run(eager_attention_forward)
    o_lrp, hq, hk, hv = run(attnlrp_attention_forward)

    assert torch.equal(o_ref, o_lrp), "forward value changed"
    for name, ref, got, want in [("q", gq, hq, 0.25), ("k", gk, hk, 0.25), ("v", gv, hv, 0.5)]:
        err = (got - want * ref).abs().max().item()
        scale = (got.norm() / ref.norm()).item()
        print(f"  grad {name}: ratio {scale:.6f} (want {want}), max |g - {want}*g_ref| = {err:.2e}")
        assert err < 1e-5, f"{name} gradient is not {want}x the vanilla gradient"
    print("  half-rule OK (forward unchanged, q/k x1/4, v x1/2)")


def test_model(name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32,
                                              attn_implementation="eager")
    hf.eval()
    ids = tok("The keys to the cabinet", return_tensors="pt").input_ids

    def logits_and_grad(install):
        if install is not None:
            install(hf)
        emb = hf.model.embed_tokens(ids).detach().requires_grad_(True)
        h = hf.model.embed_tokens.register_forward_hook(lambda m, i, o: emb)
        logits = hf(ids).logits.float()
        h.remove()
        logits[0, -1].max().backward()
        g = emb.grad.clone()
        if install is not None:
            revert_relp(hf)
        return logits.detach(), g

    base_logits, g_base = logits_and_grad(None)
    lrp_logits, g_lrp = logits_and_grad(install_attnlrp)
    relp_logits, g_relp = logits_and_grad(install_relp)

    for lbl, lg in [("attnlrp", lrp_logits), ("relp", relp_logits)]:
        d = (lg - base_logits).abs().max().item()
        print(f"  {lbl}: max |logit - vanilla| = {d:.2e}")
        assert d < 1e-4, f"{lbl} changed the forward pass"

    def cos(a, b):
        return float((a.flatten() @ b.flatten()) / (a.norm() * b.norm()))
    print(f"  cos(grad_attnlrp, grad_vanilla) = {cos(g_lrp, g_base):.4f}")
    print(f"  cos(grad_attnlrp, grad_relp)    = {cos(g_lrp, g_relp):.4f}")
    assert cos(g_lrp, g_base) < 0.999, "attnlrp gradient is indistinguishable from vanilla"
    assert cos(g_lrp, g_relp) < 0.999, "attnlrp gradient is indistinguishable from relp"
    # the attn implementation must be back to what it was, or a later non-relp run in the same
    # process silently keeps the modified attention
    assert hf.model.config._attn_implementation == "eager", "attn implementation not restored"
    print("  model-level OK (forward preserved, gradient distinct, impl restored)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    a = p.parse_args()
    print("half-rule unit test:")
    test_half_rule()
    print(f"model test ({a.model}):")
    test_model(a.model)
