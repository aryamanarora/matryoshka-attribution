"""Llama-specific hook management for sigmoid top-k attribution."""

from __future__ import annotations

import torch


class LlamaAttributionHooks:
    """Manages sigmoid top-k masking hooks for Llama-architecture HF models.

    Mask types (the substrate one score covers):
      - "node":          per-(layer, head) attention head + per-layer MLP block, tied over
                         positions (MIB's node granularity), plus the input embedding when
                         ``include_input``
      - "mlp":           per-(layer, pos, neuron) masking at the MLP down_proj input
      - "mlp+attn_head": that plus per-(layer, pos, head) masking of the o_proj input
      - "mlp+attn_dim":  that plus per-(layer, pos, dim) masking of the o_proj input
      - "mlp_sae_span" / "resid_sae_span": per-(layer, span, SAE latent) + one
                         reconstruction-error node per (layer, span), on the MLP output or the
                         layer output, intervened at each span's last token (see set_saes)

    Score layout (flat vector):
      - node:          [input?][attn: L*H][mlp: L]
      - mlp:           [L * seq_len * intermediate]
      - mlp+attn_head: [mlp | L * seq_len * H]
      - mlp+attn_dim:  [mlp | L * seq_len * d_model]
      - *_sae_span:    [input?][L * S * (d_sae + 1)]

    ``corrupt_topk`` is the intervention direction: False = iso (top-k stays clean, the
    complement is patched to the source), True = cause (top-k patched, complement clean).
    Callers flip it per forward for the two sweep directions.
    """

    MASK_TYPES = {"node", "mlp", "mlp+attn_head", "mlp+attn_dim", "mlp_sae_span", "resid_sae_span"}

    def __init__(self, model, mask_type, seq_len, corrupt_topk=False, include_input=False,
                 num_spans=None, zero_ablation=False, sae_error="absorb"):
        assert mask_type in self.MASK_TYPES, f"Unknown mask type: {mask_type}"
        self.model = model
        self.mask_type = mask_type
        self.seq_len = seq_len
        self.corrupt_topk = corrupt_topk
        #: ablate to ZERO instead of to the cached source activation. Changes what "corrupted"
        #: means everywhere the mask is applied -- training, evaluation and the faithfulness
        #: endpoints alike -- so it is a property of the run, not of a single call.
        self.zero_ablation = zero_ablation
        # Scoring/ablating the input-embedding node. Supported for `node` and for the two
        # *_sae_span substrates; the per-token layouts silently drop it, which is why this is an
        # explicit allowlist rather than a plain assignment -- a caller passing --include-input
        # to `mlp` would get a run byte-identical to the -input one under a +input label.
        self.include_input = include_input and (
            mask_type in ("node", "mlp_sae_span", "resid_sae_span"))
        self.num_spans = num_spans          # *_sae_span: number of spans (content spans or positions)
        self.span_last = None               # per-batch [B, num_spans] long: base last-token pos/span
        self.span_last_src = None           # per-batch [B, num_spans] long: source last-token pos/span
        self.saes = None                    # {layer: LlamaScopeSAE} for *_sae_span
        self.d_sae = None; self.sae_width = None
        # HOW THE SAE RECONSTRUCTION ERROR IS INTERVENED ON. Three settings, and the choice
        # decides what an "error node" even is -- see _sae_interchange for the algebra.
        #   "absorb" (legacy default) clean side is (b - sae_out), measured against the BLENDED
        #            reconstruction, so keeping the error node restores the whole site.
        #   "frozen" both sides frozen at their own reconstructions, so the error node carries
        #            only the genuinely unexplained residual and cannot override the latents.
        #   "none"   no error node at all: sae_width is d_sae, every offset shifts, and the
        #            error is pinned at the frozen CLEAN residual.
        assert sae_error in ("absorb", "frozen", "none"), sae_error
        self.sae_error = sae_error

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size

        self.mlp_total = self.num_layers * seq_len * self.intermediate_size
        self.attn_head_total = self.num_layers * seq_len * self.num_heads
        self.attn_dim_total = self.num_layers * seq_len * self.hidden_size  # per-dim o_proj input
        self.node_total = self.num_layers * self.num_heads + self.num_layers + (1 if self.include_input else 0)

        if mask_type == "mlp":
            self.total = self.mlp_total
        elif mask_type == "mlp+attn_head":
            self.total = self.mlp_total + self.attn_head_total
        elif mask_type == "mlp+attn_dim":
            self.total = self.mlp_total + self.attn_dim_total
        elif mask_type == "node":
            self.total = self.node_total
        else:
            assert num_spans, "*_sae_span requires num_spans"
            self.total = 0   # set by set_saes() once SAE width is known

        self.mask = None
        self.cf_acts_mlp = {}       # source activations at the down_proj input, per layer
        self.cf_acts_attn = {}      # source activations at the o_proj input, per layer
        self.cf_acts_site = {}      # *_sae_span: source activation at the SAE site, per layer
        self.cf_acts_embed = None
        self._hooks = []

    @property
    def has_mlp(self):
        return self.mask_type in ("mlp", "mlp+attn_head", "mlp+attn_dim", "node")

    @property
    def has_attn(self):
        return self.mask_type in ("mlp+attn_head", "mlp+attn_dim", "node")

    @property
    def is_node(self):
        return self.mask_type == "node"

    @property
    def is_sae(self):
        return self.mask_type in ("mlp_sae_span", "resid_sae_span")

    def set_saes(self, saes):
        """Provide per-layer frozen SAEs (.encode/.decode/.d_sae). Sets the node layout:
        per-(layer, span, d_sae feature) + 1 per-(layer,span) reconstruction-error node,
        or without that error node when sae_error == "none"."""
        assert self.is_sae, "set_saes only for *_sae_span"
        self.saes = saes
        self.d_sae = next(iter(saes.values())).d_sae
        self.sae_width = self.d_sae + (0 if self.sae_error == "none" else 1)
        # `_node_offset` reserves index 0 for the input-embedding node when include_input, exactly
        # as the `node` layout does. Every consumer of this layout must add the same offset --
        # _sae_interchange below, and eval_sva.gradient_scores' SAE branch.
        self.total = (self._node_offset
                      + self.num_layers * self.num_spans * self.sae_width)

    @property
    def _node_offset(self):
        """Offset into mask for attn/mlp scores (1 if include_input, else 0)."""
        return 1 if self.include_input else 0

    # Override these in subclasses for different model architectures
    def _get_layer(self, li):
        return self.model.model.layers[li]

    def _get_mlp_module(self, layer):
        return layer.mlp.down_proj

    def _get_attn_module(self, layer):
        return layer.self_attn.o_proj

    def _get_embed_module(self):
        return self.model.model.embed_tokens

    def _sae_site(self, layer):
        """The module whose OUTPUT the SAE reads: the MLP (down_proj) for mlp_sae_span, the whole
        decoder layer (residual stream) for resid_sae_span."""
        return self._get_mlp_module(layer) if self.mask_type == "mlp_sae_span" else layer

    def describe(self):
        if self.is_sae:
            site = "MLP-out" if self.mask_type == "mlp_sae_span" else "resid"
            return (f"SAE({site}): {self.num_layers}L x {self.num_spans}span x "
                    f"({self.d_sae}feat + 1err) = {self.total:,} total")
        if self.is_node:
            inp = "+input" if self.include_input else ""
            return (f"Node: {self.num_layers}L x ({self.num_heads}h + 1mlp){inp} = "
                    f"{self.node_total:,} = {self.total:,} total")
        parts = [f"MLP: {self.num_layers}L x {self.seq_len}pos x "
                 f"{self.intermediate_size}n = {self.mlp_total:,}"]
        if self.mask_type == "mlp+attn_dim":
            parts.append(f"Attn(pre-out per-dim): {self.num_layers}L x {self.seq_len}pos x "
                         f"{self.hidden_size}d = {self.attn_dim_total:,}")
        elif self.mask_type == "mlp+attn_head":
            parts.append(f"Attn: {self.num_layers}L x {self.seq_len}pos x "
                         f"{self.num_heads}h = {self.attn_head_total:,}")
        return " + ".join(parts) + f" = {self.total:,} total"

    def cache_cf_activations(self, cf_input_ids):
        """Run the source (CF) input through the model and cache activations at hook points."""
        self.mask = None  # disable masking hooks during CF forward
        hooks = []
        if self.include_input:
            def _embed_hook(mod, input, output):
                self.cf_acts_embed = output.detach()
            hooks.append(self._get_embed_module().register_forward_hook(_embed_hook))
        for li in range(self.num_layers):
            layer = self._get_layer(li)
            if self.is_sae:
                def _post(idx):
                    def hook(mod, inp, output):
                        self.cf_acts_site[idx] = (output[0] if isinstance(output, tuple) else output).detach()
                    return hook
                hooks.append(self._sae_site(layer).register_forward_hook(_post(li)))
                continue
            if self.has_mlp:
                def _mlp(idx):
                    def hook(mod, args):
                        self.cf_acts_mlp[idx] = args[0].detach()
                    return hook
                hooks.append(self._get_mlp_module(layer).register_forward_pre_hook(_mlp(li)))
            if self.has_attn:
                def _attn(idx):
                    def hook(mod, args):
                        self.cf_acts_attn[idx] = args[0].detach()
                    return hook
                hooks.append(self._get_attn_module(layer).register_forward_pre_hook(_attn(li)))

        with torch.no_grad():
            cf_logits = self.model(cf_input_ids).logits[0, -1].float()

        for h in hooks:
            h.remove()
        return cf_logits

    def _interpolate(self, x, m, cf_act):
        """Mix clean activation `x` with the ablated value under mask `m` (1 = top-k).

        iso (corrupt_topk=False): the top-k stays clean, the rest goes to the source value.
        cause (corrupt_topk=True): the top-k goes to the source value, the rest stays clean.
        Under zero ablation the corrupted side goes to 0 rather than to the source activation.
        """
        m = m.to(x.dtype)
        if self.zero_ablation or cf_act is None:
            # cf_act is None means "nothing cached"; in practice it is always cached by callers.
            return (x * (1 - m),) if self.corrupt_topk else (x * m,)
        if self.corrupt_topk:
            return (x * (1 - m) + cf_act * m,)
        return (x * m + cf_act * (1 - m),)

    def _ke(self, me):
        """Weight on the CLEAN side of the error term, matching km's own convention."""
        return (1.0 - me) if self.corrupt_topk else me

    def _sae_interchange(self, out, cf, layer_idx):
        """SAE feature interchange at each span's LAST token, cross-aligned base<-src.
        Node = per (span, feature) + 1 per-span reconstruction-error node. The update is a
        CONVEX BLEND of the clean and source reconstructions plus a blended error term (see
        below), exact at both endpoints and bounded in between.
        cause: mask=1 (top-k) -> source feature; iso: mask=1 -> base."""
        sae = self.saes.get(layer_idx)
        if sae is None or cf is None or self.mask is None or self.span_last is None:
            return out
        S, dm, dsae, W = self.num_spans, out.shape[-1], self.d_sae, self.sae_width
        B = self.span_last.shape[0]
        off = self._node_offset + layer_idx * S * W   # index 0 is the input node when +input
        per_span = self.mask[off:off + S * W].view(S, W)
        wdt = sae.W_enc.dtype
        mf = per_span[:, :dsae].to(wdt)[None]      # [1,S,dsae]
        me = None if self.sae_error == "none" else per_span[:, dsae].to(wdt)[None, :, None]
        bidx = self.span_last[:, :, None].expand(B, S, dm)
        sidx = self.span_last_src[:, :, None].expand(B, S, dm)
        b = out.gather(1, bidx).to(wdt)            # clean act at base span-last  [B,S,dm]
        c = cf.gather(1, sidx).to(wdt)             # source act at src span-last
        fb, fc = sae.encode(b), sae.encode(c)
        # ZERO ABLATION: the ablated side of every latent is 0 instead of the source latent, and
        # the ablated side of the error node is 0 instead of the source error -- the per-latent
        # analogue of the neuron path's x*m. With everything ablated the site reads
        # decode(0) = b_dec (the SAE's own zero), not a zeroed residual; that is the definition,
        # since the substrate is the latents, not the activation they reconstruct. Implemented
        # by substituting the source-side quantities, so every blend below is unchanged and
        # both endpoints stay exact (km=ke=1 -> b; km=ke=0 -> decode(0)).
        if self.zero_ablation:
            fc = torch.zeros_like(fb)

        # CONVEX-BLEND FORM, not base-plus-delta. `km`/`ke` are the weights on the CLEAN side,
        # so km=1 keeps the clean latent and km=0 takes the source one.
        #
        #     sae_out = decode( f_clean*km + f_source*(1-km) )
        #     err     = (b - sae_out)*ke + (c - decode(f_source))*(1-ke)
        #     new     = sae_out + err
        #
        # EXACT AT BOTH ENDPOINTS:
        #   km=ke=1 -> sae_out = decode(fb), err = b - decode(fb), new = b   (clean)
        #   km=ke=0 -> sae_out = decode(fc), err = c - decode(fc), new = c   (source)
        # so no metric changes meaning and F_clean / F_patch are untouched.
        #
        # WHY IT REPLACED `new = b + decode_delta((1-mf)*fd) + (1-me)*err_diff` (2026-08-29).
        # That form's exactness relied on the leading +b cancelling against the -b buried in
        # err_diff = (c-b) - decode_delta(fc-fb). The cancellation is exact in real arithmetic
        # but only holds numerically while decode(encode(x)) ~ x -- and this hook rewrites the
        # layer OUTPUT, so a perturbed `b` feeds the next layer, leaves the SAE's training
        # distribution, and the identity fails. Measured: ~1000x growth per layer from layer 19,
        # fp32 overflow by layer 24, NaN by 30; 35 of 102 runs finished with a non-finite AUC,
        # concentrated on the arithmetic tasks (IG 3/3 on all four) while Random never diverged
        # at all -- divergence was a property of CONCENTRATING the top-k, not of any estimator.
        #
        # The blend has no such dependency: with km, ke in [0,1] the result is a mix of two
        # reconstructions plus a mixed error, and there is no term whose coefficient on the live
        # activation exceeds 1. It is the same shape as this class's own non-SAE intervention
        # (`_interpolate`: x*m + cf_act*(1-m)).
        km = (1.0 - mf) if self.corrupt_topk else mf
        sae_out = sae.decode(fb * km + fc * (1.0 - km))
        # eb / ec are the two RECONSTRUCTION ERRORS, each measured against its OWN
        # reconstruction. Both are constants of the mask: neither depends on km.
        eb, ec = b - sae.decode(fb), c - sae.decode(fc)
        if self.zero_ablation:
            ec = torch.zeros_like(eb)
        if self.sae_error == "none":
            # No error node in the substrate; the error is pinned at the clean residual. Same as
            # "frozen" with ke == 1, which is the point -- "none" is not a fourth semantics.
            # km=1 -> b exactly. km=0 -> decode(f_c) + eb, i.e. the source circuit carrying the
            # clean error, NOT c. So F_patch (the k=0 endpoint) is a weaker corruption here and
            # the faithfulness denominator F_clean - F_patch is a smaller interval than under
            # the other two settings. eval_sva computes both endpoints through this same hook so
            # each run is internally consistent, but do not read a "none" AUC against an
            # "absorb"/"frozen" one.
            new = sae_out + eb
        elif self.sae_error == "frozen":
            new = sae_out + eb * self._ke(me) + ec * (1.0 - self._ke(me))
        else:
            # *** LEGACY "absorb", AND IT MAKES THE ERROR NODE A MASTER SWITCH. *** The clean
            # side is (b - sae_out), measured against the BLENDED reconstruction rather than
            # against decode(f_b) -- while the source side uses decode(f_c). That asymmetry
            # means ke=1 collapses the whole line to
            #     sae_out + (b - sae_out) = b
            # FOR ANY km: keeping one error node restores its entire (layer, position) site to
            # clean, overriding all d_sae latents there. Measured consequence on
            # addition/mlp_sae_span: there are 32*5 = 160 error nodes in a 5,243,040-unit
            # substrate, and MAttr's faithfulness reaches 0.981 by k=57 (1.1e-5 of the
            # substrate) because it selects them first. "MAttr picks 90-100% error nodes" is
            # therefore not an optimiser pathology -- it is the cheapest way to denoise under
            # THIS intervention, and the intervention is what should change.
            #
            # Kept as the default only so existing results/sva_sweep SAE runs remain
            # reproducible. Prefer "frozen" for anything new.
            ke = self._ke(me)
            new = sae_out + (b - sae_out) * ke + ec * (1.0 - ke)
        return out.scatter(1, bidx, new.to(out.dtype))

    def register_hooks(self):
        self.remove_hooks()

        # The input-embedding node is masked the SAME way for every substrate that supports it:
        # a plain convex interpolation of the embedding toward the cached source embedding. It is
        # registered BEFORE the substrate branches because the SAE branch returns early.
        if self.include_input:
            def make_embed_hook():
                def hook(mod, input, output):
                    if self.mask is None:
                        return
                    m = self.mask[0].view(1, 1, 1)
                    return self._interpolate(output, m, self.cf_acts_embed)[0]
                return hook
            self._hooks.append(
                self._get_embed_module().register_forward_hook(make_embed_hook()))

        if self.is_sae:
            for layer_idx in range(self.num_layers):
                def make_post_hook(li):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        tup = isinstance(output, tuple)
                        out = output[0] if tup else output
                        new = self._sae_interchange(out, self.cf_acts_site.get(li), li)
                        return ((new,) + tuple(output[1:])) if tup else new
                    return hook
                self._hooks.append(
                    self._sae_site(self._get_layer(layer_idx)).register_forward_hook(make_post_hook(layer_idx)))
            return

        for layer_idx in range(self.num_layers):
            layer = self._get_layer(layer_idx)

            def make_mlp_hook(li):
                def hook(mod, hook_args):
                    if self.mask is None:
                        return
                    x = hook_args[0]  # [B, seq_len, intermediate_size]
                    if self.is_node:
                        # Node: one scalar per MLP per layer, broadcast
                        off = self._node_offset
                        attn_count = self.num_layers * self.num_heads
                        m = self.mask[off + attn_count + li].view(1, 1, 1)
                    else:
                        start = li * self.seq_len * self.intermediate_size
                        end = start + self.seq_len * self.intermediate_size
                        m = self.mask[start:end].view(1, self.seq_len, self.intermediate_size)
                    return self._interpolate(x, m, self.cf_acts_mlp.get(li))
                return hook
            self._hooks.append(
                self._get_mlp_module(layer).register_forward_pre_hook(make_mlp_hook(layer_idx)))

            if self.has_attn:
                def make_attn_hook(li):
                    def hook(mod, hook_args):
                        if self.mask is None:
                            return
                        x = hook_args[0]  # [B, seq_len, hidden_size]
                        cf = self.cf_acts_attn.get(li)
                        B, seq = x.shape[0], x.shape[1]

                        if self.mask_type == "mlp+attn_dim":
                            # per-(pos, dim) over the o_proj input (pre-out, hidden_size)
                            off = self.mlp_total + li * self.seq_len * self.hidden_size
                            end = off + self.seq_len * self.hidden_size
                            m = self.mask[off:end].view(1, self.seq_len, self.hidden_size)
                            return self._interpolate(x, m, cf)

                        if self.is_node:
                            # Node: [n_heads] per layer, broadcast over batch & positions
                            off = self._node_offset + li * self.num_heads
                            m = self.mask[off:off + self.num_heads].view(1, 1, self.num_heads, 1)
                        else:  # mlp+attn_head: per-(pos, head)
                            off = self.mlp_total + li * self.seq_len * self.num_heads
                            end = off + self.seq_len * self.num_heads
                            m = self.mask[off:end].view(1, self.seq_len, self.num_heads, 1)
                        x4d = x.view(B, seq, self.num_heads, self.head_dim)
                        cf4d = (cf.view(cf.shape[0], cf.shape[1], self.num_heads, self.head_dim)
                                if cf is not None else None)
                        out = self._interpolate(x4d, m, cf4d)
                        return (out[0].reshape(B, seq, -1),)
                    return hook
                self._hooks.append(
                    self._get_attn_module(layer).register_forward_pre_hook(make_attn_hook(layer_idx)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def decode_index(self, flat_idx):
        """Convert a flat score index -> dict with layer, pos, and component info."""
        flat_idx = int(flat_idx)

        if self.is_node:
            off = self._node_offset
            if self.include_input and flat_idx == 0:
                return {"component": "input", "layer": -1}
            flat_idx -= off
            attn_count = self.num_layers * self.num_heads
            if flat_idx < attn_count:
                return {"component": "attn", "layer": flat_idx // self.num_heads,
                        "head": flat_idx % self.num_heads}
            return {"component": "mlp", "layer": flat_idx - attn_count}

        if flat_idx < self.mlp_total:
            layer = flat_idx // (self.seq_len * self.intermediate_size)
            rem = flat_idx % (self.seq_len * self.intermediate_size)
            return {"component": "mlp", "layer": layer, "pos": rem // self.intermediate_size,
                    "neuron": rem % self.intermediate_size}

        flat_idx -= self.mlp_total
        if self.mask_type == "mlp+attn_dim":
            layer = flat_idx // (self.seq_len * self.hidden_size)
            rem = flat_idx % (self.seq_len * self.hidden_size)
            return {"component": "attn", "layer": layer, "pos": rem // self.hidden_size,
                    "dim": rem % self.hidden_size}
        layer = flat_idx // (self.seq_len * self.num_heads)
        rem = flat_idx % (self.seq_len * self.num_heads)
        return {"component": "attn", "layer": layer, "pos": rem // self.num_heads,
                "head": rem % self.num_heads}
