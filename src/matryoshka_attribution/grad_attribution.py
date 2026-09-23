"""RelP / AttnLRP modified-backward modules for HF Llama (ported from circuits/tracing/grad —
pure PyTorch/transformers, no nnsight). install_relp(model) / install_attnlrp(model) replace
norms/attn/mlp in place so a single forward+backward yields the modified gradient;
revert_relp(model) restores the originals (it reverts either variant).

RelP rules: (1) RMSNorm linearized (identity backward through the normalization, weight frozen);
(2) gated-MLP activation gate secant-linearized + half-rule on gate*up; (3) attention pattern
detached so gradient flows only through the OV (V) path.

AttnLRP shares (1) and (2) and replaces (3) with the uniform (half) rule for bilinear matmuls
(Achtibat et al. 2024, Eq. 14-15): the QK and OV matmul outputs each get a x0.5 backward, so
composed the Q/K gradients carry 1/4 and the V gradient 1/2 -- exactly LXT's
divide_gradient(q,4)/(k,4)/(v,2). The softmax keeps its ordinary Jacobian (LXT patches nothing
there). This mirrors the TransformerLens implementation used for MIB
(EAP-IG/src/eap/attribute_node.py, shapley_attn=True + softmax_rule=False + linearize_act=True).
"""
import torch
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import LlamaRMSNorm, repeat_kv


def _rms_layernorm_fn(x_X1X2D, estimator_X1D, norm_w_D, eps):
    device = x_X1X2D.device
    return (norm_w_D[None, None, :].to(device) * x_X1X2D
            * torch.rsqrt(estimator_X1D.to(device).pow(2).mean(dim=1) + eps)[:, None, None])


class _StopGrad(nn.Module):
    _stop_gradient = True


class StraightThroughRMSNorm(_StopGrad):
    """forward = real RMSNorm; backward = identity wrt input (frozen weight)."""
    def __init__(self, norm):
        super().__init__()
        self.norm = norm
        self.norm.weight.requires_grad_(False)
        self.weight = self.norm.weight

    def forward(self, x):
        B, L, D = x.shape
        coeff = _rms_layernorm_fn(x.new_ones(B * L, 1, 1), x.view(B * L, D),
                                  self.norm.weight, self.norm.variance_epsilon).detach()
        return x * coeff.permute(1, 0, 2).view(B, L, D)


def noqk_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Attention forward that detaches the softmax pattern: gradient flows only through OV."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype).detach()
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
    return attn_output, attn_weights


class _HalfGrad(torch.autograd.Function):
    """Identity forward, 0.5x gradient backward. Applied to a bilinear matmul's OUTPUT this is
    the uniform (half) rule on both its inputs: for z = x @ y, halving grad_z halves grad_x and
    grad_y alike."""
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return 0.5 * grad_output


def attnlrp_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Attention forward with AttnLRP's half-rule on the QK and OV matmuls (softmax untouched).

    The half is applied to the QK product BEFORE the additive mask: the mask is a constant with
    no gradient, so this is gradient-equivalent to halving after it and keeps the -inf entries
    out of the scaled tensor."""
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_scores = _HalfGrad.apply(torch.matmul(query, key_states.transpose(2, 3)) * scaling)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = _HalfGrad.apply(torch.matmul(attn_weights, value_states)).transpose(1, 2).contiguous()
    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS["noqk"] = noqk_attention_forward
ALL_MASK_ATTENTION_FUNCTIONS.register("noqk", eager_mask)
ALL_ATTENTION_FUNCTIONS["attnlrp"] = attnlrp_attention_forward
ALL_MASK_ATTENTION_FUNCTIONS.register("attnlrp", eager_mask)


class _WrappedAttention(_StopGrad):
    """Wrap attention so it dispatches to one of our modified attention implementations."""
    IMPL = None

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.q_proj = attn.q_proj; self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj; self.o_proj = attn.o_proj
        self.attn.config._attn_implementation = self.IMPL

    def forward(self, *a, **k):
        return self.attn(*a, **k)


class NoQKGradAttention(_WrappedAttention):
    """RelP: the softmax map gets no gradient (OV-only)."""
    IMPL = "noqk"


class AttnLRPAttention(_WrappedAttention):
    """AttnLRP: half-rule on the QK and OV matmuls, softmax gradient kept."""
    IMPL = "attnlrp"


def _shapley_mult(x, y, half=True):
    raw = x * y
    return 0.5 * raw + 0.5 * raw.detach() if half else raw


class RelPGradMLP(_StopGrad):
    """Gated-MLP RelP: secant-linearize the act gate (detached coeff) + half-rule on gate*up."""
    def __init__(self, mlp, half=True):
        super().__init__()
        self.mlp = mlp
        self.down_proj = mlp.down_proj; self.gate_proj = mlp.gate_proj; self.up_proj = mlp.up_proj
        self.half = half

    def forward(self, x):
        gp = self.mlp.gate_proj(x)
        coeff = (self.mlp.act_fn(gp) / (gp + 1e-10)).detach()
        gate_act = gp * coeff
        return self.mlp.down_proj(_shapley_mult(gate_act, self.mlp.up_proj(x), self.half))


def _install(model, attn_cls):
    """Replace norms/attn/mlp in the HF model with modified-backward variants (in place)."""
    m = model.model
    # the wrappers rewrite config._attn_implementation, which is the SHARED model config, so
    # stash the real one once here rather than per layer (per layer, the second install would
    # record "noqk"/"attnlrp" as the original and revert would leave it installed).
    if not hasattr(model, "_mattr_orig_attn_impl"):
        model._mattr_orig_attn_impl = m.config._attn_implementation
    m.norm = StraightThroughRMSNorm(m.norm)
    for layer in m.layers:
        layer.input_layernorm = StraightThroughRMSNorm(layer.input_layernorm)
        layer.post_attention_layernorm = StraightThroughRMSNorm(layer.post_attention_layernorm)
        layer.self_attn = attn_cls(layer.self_attn)
        layer.mlp = RelPGradMLP(layer.mlp)
    return model


def install_relp(model):
    return _install(model, NoQKGradAttention)


def install_attnlrp(model):
    return _install(model, AttnLRPAttention)


def revert_relp(model):
    """Undo install_relp / install_attnlrp."""
    m = model.model
    m.norm = m.norm.norm
    for layer in m.layers:
        layer.input_layernorm = layer.input_layernorm.norm
        layer.post_attention_layernorm = layer.post_attention_layernorm.norm
        if isinstance(layer.self_attn, _WrappedAttention):
            layer.self_attn = layer.self_attn.attn
        if isinstance(layer.mlp, RelPGradMLP):
            layer.mlp = layer.mlp.mlp
    if hasattr(model, "_mattr_orig_attn_impl"):
        m.config._attn_implementation = model._mattr_orig_attn_impl
        del model._mattr_orig_attn_impl
    return model
