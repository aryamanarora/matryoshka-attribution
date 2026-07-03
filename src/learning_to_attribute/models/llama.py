"""Llama-specific hook management for sigmoid top-k attribution."""

from __future__ import annotations

import torch


class LlamaAttributionHooks:
    """Manages sigmoid top-k masking hooks for Llama models.

    Mask types:
      - "mlp": per-(layer, pos, neuron) masking at MLP down_proj input
      - "attn_output": per-(layer, pos) scalar masking of full attention output
      - "attn_head": per-(layer, pos, head) masking of attention output
      - "mlp+attn_head": combined MLP neuron + attention head masking
      - "resid": per-(layer, pos) scalar masking of full layer output (residual stream)

    Score layout (flat vector):
      - mlp:           [num_layers * seq_len * intermediate_size]
      - attn_output:   [num_layers * seq_len]
      - attn_head:     [num_layers * seq_len * num_heads]
      - mlp+attn_head: [mlp_scores | attn_head_scores]
      - resid:         [num_layers * seq_len]
    """

    MASK_TYPES = {"mlp", "mlp_tied", "mlp_span", "mlp+attn_span", "mlp+attn_head_span", "mlp_sae_span", "resid_sae_span", "das_mlp_span", "das_resid_span", "attn_output", "attn_head", "mlp+attn_head", "mlp+attn_dim", "resid", "resid_dim", "node", "das", "sae"}

    def __init__(self, model, mask_type, seq_len, sufficient=False, include_input=False,
                 num_spans=None):
        assert mask_type in self.MASK_TYPES, f"Unknown mask type: {mask_type}"

        self.model = model
        self.mask_type = mask_type
        self.seq_len = seq_len
        self.sufficient = sufficient
        self.include_input = include_input and (mask_type == "node")
        self.num_spans = num_spans          # for mlp_span: # of causalgym content spans
        self.span_last = None               # per-batch [B, num_spans] long: base last-token pos/span
        self.span_last_src = None           # per-batch [B, num_spans] long: source last-token pos/span
        self.saes = None                    # {layer: LlamaScopeSAE} for *_sae_span
        self.d_sae = None; self.sae_width = None
        self.cf_acts_saemlp = {}            # cached src MLP-out (down_proj output) per layer
        self.R = {}                        # {layer: RotateLayer} for das_*_span
        self.das_dim = None

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size

        self.mlp_total = self.num_layers * seq_len * self.intermediate_size
        self.mlp_tied_total = self.num_layers * self.intermediate_size  # per-(layer,neuron), tied over pos
        self.mlp_span_total = (self.num_layers * num_spans * self.intermediate_size
                               if num_spans else 0)  # per-(layer,span,neuron)
        self.attn_span_total = (self.num_layers * num_spans * self.hidden_size
                                if num_spans else 0)  # per-(layer,span,dim) o_proj input
        self.attn_head_span_total = (self.num_layers * num_spans * self.num_heads
                                     if num_spans else 0)  # per-(layer,span,head)
        self.attn_output_total = self.num_layers * seq_len
        self.attn_head_total = self.num_layers * seq_len * self.num_heads
        self.attn_dim_total = self.num_layers * seq_len * self.hidden_size  # per-dim o_proj input
        self.resid_total = self.num_layers * seq_len
        self.node_total = self.num_layers * self.num_heads + self.num_layers + (1 if self.include_input else 0)

        if mask_type == "mlp":
            self.total = self.mlp_total
        elif mask_type == "mlp_tied":
            self.total = self.mlp_tied_total
        elif mask_type == "mlp_span":
            assert num_spans, "mlp_span requires num_spans"
            self.total = self.mlp_span_total
        elif mask_type == "mlp+attn_span":
            assert num_spans, "mlp+attn_span requires num_spans"
            self.total = self.mlp_span_total + self.attn_span_total
        elif mask_type == "mlp+attn_head_span":
            assert num_spans, "mlp+attn_head_span requires num_spans"
            self.total = self.mlp_span_total + self.attn_head_span_total
        elif mask_type in ("mlp_sae_span", "resid_sae_span"):
            assert num_spans, "*_sae_span requires num_spans"
            self.total = 0   # set by set_saes() once SAE width is known
        elif mask_type in ("das_mlp_span", "das_resid_span"):
            assert num_spans, "das_*_span requires num_spans"
            self.total = 0   # set by set_das() once das_dim is known
        elif mask_type == "attn_output":
            self.total = self.attn_output_total
        elif mask_type == "attn_head":
            self.total = self.attn_head_total
        elif mask_type == "mlp+attn_head":
            self.total = self.mlp_total + self.attn_head_total
        elif mask_type == "mlp+attn_dim":
            self.total = self.mlp_total + self.attn_dim_total
        elif mask_type == "resid":
            self.total = self.resid_total
        elif mask_type == "node":
            self.total = self.node_total

        self.mask = None
        self.cf_acts_mlp = {}
        self.cf_acts_attn = {}
        self.cf_acts_resid = {}
        self.cf_acts_embed = None
        self._hooks = []

    @property
    def has_mlp(self):
        return self.mask_type in ("mlp", "mlp_tied", "mlp_span", "mlp+attn_span", "mlp+attn_head_span", "mlp+attn_head", "mlp+attn_dim", "node")

    @property
    def has_attn(self):
        return self.mask_type in ("attn_output", "attn_head", "mlp+attn_head", "mlp+attn_dim", "mlp+attn_span", "mlp+attn_head_span", "node")

    @property
    def has_resid(self):
        return self.mask_type == "resid"

    @property
    def is_node(self):
        return self.mask_type == "node"

    @property
    def is_sae(self):
        return self.mask_type in ("mlp_sae_span", "resid_sae_span")

    def set_saes(self, saes):
        """Provide per-layer frozen SAEs (.encode/.decode_delta/.d_sae). Sets the node layout:
        per-(layer, span, d_sae feature) + 1 per-(layer,span) reconstruction-error node."""
        assert self.is_sae, "set_saes only for *_sae_span"
        self.saes = saes
        self.d_sae = next(iter(saes.values())).d_sae
        self.sae_width = self.d_sae + 1
        self.total = self.num_layers * self.num_spans * self.sae_width

    @property
    def is_das(self):
        return self.mask_type in ("das_mlp_span", "das_resid_span")

    def set_das(self, das_dim=None, device="cpu", dtype=torch.float32):
        """Create per-layer learned orthogonal DAS rotations (d_model -> das_dim) and set the
        node layout: per-(layer, span, das_dim) score in the rotated subspace."""
        assert self.is_das, "set_das only for das_*_span"
        from learning_to_attribute.sigmoid_das import make_rotate_layer
        self.das_dim = das_dim or self.hidden_size
        self.R = {li: make_rotate_layer(self.hidden_size, self.das_dim).to(device, dtype)
                  for li in range(self.num_layers)}
        self.total = self.num_layers * self.num_spans * self.das_dim

    def das_parameters(self):
        return [p for li in sorted(self.R) for p in self.R[li].parameters()]

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

    def describe(self):
        parts = []
        if self.is_sae:
            site = "MLP-out" if self.mask_type == "mlp_sae_span" else "resid"
            return (f"SAE({site}): {self.num_layers}L x {self.num_spans}span x "
                    f"({self.d_sae}feat + 1err) = {self.total:,} total")
        if self.is_das:
            site = "MLP-out" if self.mask_type == "das_mlp_span" else "resid"
            return (f"DAS({site}): {self.num_layers}L x {self.num_spans}span x "
                    f"{self.das_dim}rot-dim = {self.total:,} total")
        if self.is_node:
            inp = "+input" if self.include_input else ""
            parts.append(f"Node: {self.num_layers}L x ({self.num_heads}h + 1mlp){inp} = "
                         f"{self.node_total:,}")
        else:
            if self.mask_type == "mlp_tied":
                parts.append(f"MLP(tied over pos): {self.num_layers}L x "
                             f"{self.intermediate_size}n = {self.mlp_tied_total:,}")
            elif self.mask_type in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span"):
                parts.append(f"MLP(per-span): {self.num_layers}L x {self.num_spans}span x "
                             f"{self.intermediate_size}n = {self.mlp_span_total:,}")
            elif self.has_mlp:
                parts.append(f"MLP: {self.num_layers}L x {self.seq_len}pos x "
                             f"{self.intermediate_size}n = {self.mlp_total:,}")
            if self.has_attn:
                if self.mask_type == "attn_output":
                    parts.append(f"Attn: {self.num_layers}L x {self.seq_len}pos = "
                                 f"{self.attn_output_total:,}")
                elif self.mask_type == "mlp+attn_dim":
                    parts.append(f"Attn(pre-out per-dim): {self.num_layers}L x {self.seq_len}pos x "
                                 f"{self.hidden_size}d = {self.attn_dim_total:,}")
                elif self.mask_type == "mlp+attn_span":
                    parts.append(f"Attn(per-span dim): {self.num_layers}L x {self.num_spans}span x "
                                 f"{self.hidden_size}d = {self.attn_span_total:,}")
                elif self.mask_type == "mlp+attn_head_span":
                    parts.append(f"Attn(per-span head): {self.num_layers}L x {self.num_spans}span x "
                                 f"{self.num_heads}h = {self.attn_head_span_total:,}")
                else:
                    parts.append(f"Attn: {self.num_layers}L x {self.seq_len}pos x "
                                 f"{self.num_heads}h = {self.attn_head_total:,}")
            if self.has_resid:
                parts.append(f"Resid: {self.num_layers}L x {self.seq_len}pos = "
                             f"{self.resid_total:,}")
        return " + ".join(parts) + f" = {self.total:,} total"

    def cache_cf_activations(self, cf_input_ids):
        """Run CF input through model and cache activations at hook points."""
        self.mask = None  # disable masking hooks during CF forward
        hooks = []
        if self.include_input:
            def _embed_hook(mod, input, output):
                self.cf_acts_embed = output.detach()
            hooks.append(self._get_embed_module().register_forward_hook(_embed_hook))
        for li in range(self.num_layers):
            layer = self._get_layer(li)
            if self.is_sae or self.is_das:
                mlp_site = self.mask_type in ("mlp_sae_span", "das_mlp_span")
                mod = self._get_mlp_module(layer) if mlp_site else layer
                cf_dict = self.cf_acts_saemlp if mlp_site else self.cf_acts_resid
                def _post(idx, cfd):
                    def hook(mod, inp, output):
                        cfd[idx] = (output[0] if isinstance(output, tuple) else output).detach()
                    return hook
                hooks.append(mod.register_forward_hook(_post(li, cf_dict)))
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
            if self.has_resid:
                def _resid(idx):
                    def hook(mod, input, output):
                        x = output[0] if isinstance(output, tuple) else output
                        self.cf_acts_resid[idx] = x.detach()
                    return hook
                hooks.append(layer.register_forward_hook(_resid(li)))

        with torch.no_grad():
            cf_logits = self.model(cf_input_ids).logits[0, -1].float()

        for h in hooks:
            h.remove()
        return cf_logits

    def _interpolate(self, x, m, cf_act):
        m = m.to(x.dtype)
        if cf_act is not None:
            if self.sufficient:
                return (x * (1 - m) + cf_act * m,)
            else:
                return (x * m + cf_act * (1 - m),)
        return (x * m,)

    def _sae_interchange(self, out, cf, layer_idx):
        """SAE feature interchange at each span's LAST token, cross-aligned base<-src.
        Node = per (span, feature) + 1 per-span reconstruction-error node. Reconstruction
        error held at base; update = base + decode(masked feature delta) + masked err_diff.
        sufficient (noising): mask=1 (top-k) -> source feature; else (denoising): mask=1 -> base."""
        sae = self.saes.get(layer_idx)
        if sae is None or cf is None or self.mask is None or self.span_last is None:
            return out
        S, dm, dsae, W = self.num_spans, out.shape[-1], self.d_sae, self.sae_width
        B = self.span_last.shape[0]
        off = layer_idx * S * W
        per_span = self.mask[off:off + S * W].view(S, W)
        wdt = sae.W_enc.dtype
        mf = per_span[:, :dsae].to(wdt)[None]      # [1,S,dsae]
        me = per_span[:, dsae].to(wdt)[None, :, None]   # [1,S,1]
        bidx = self.span_last[:, :, None].expand(B, S, dm)
        sidx = self.span_last_src[:, :, None].expand(B, S, dm)
        b = out.gather(1, bidx).to(wdt)            # clean act at base span-last  [B,S,dm]
        c = cf.gather(1, sidx).to(wdt)             # source act at src span-last
        fb, fc = sae.encode(b), sae.encode(c)
        fd = fc - fb
        err_diff = (c - b) - sae.decode_delta(fd)
        if self.sufficient:
            new = b + sae.decode_delta(mf * fd) + me * err_diff
        else:
            new = b + sae.decode_delta((1.0 - mf) * fd) + (1.0 - me) * err_diff
        # numerical guard: cap per-(B,S) norm at 8x source norm; fall back to source on NaN
        nn = new.norm(dim=-1, keepdim=True); cap = 8.0 * c.norm(dim=-1, keepdim=True) + 1e-6
        new = torch.where(torch.isfinite(nn) & (nn > cap), new * cap / nn, new)
        new = torch.where(torch.isfinite(new), new, c)
        return out.scatter(1, bidx, new.to(out.dtype))

    def _das_interchange(self, out, cf, layer_idx):
        """DAS interchange in a learned orthogonal subspace at each span's LAST token,
        cross-aligned base<-src. Node = per (span, das_dim) score in the rotated space."""
        rot = self.R.get(layer_idx)
        if rot is None or cf is None or self.mask is None or self.span_last is None:
            return out
        S, dm, dd = self.num_spans, out.shape[-1], self.das_dim
        B = self.span_last.shape[0]
        off = layer_idx * S * dd
        per_span = self.mask[off:off + S * dd].view(S, dd)
        bidx = self.span_last[:, :, None].expand(B, S, dm)
        sidx = self.span_last_src[:, :, None].expand(B, S, dm)
        b = out.gather(1, bidx)
        c = cf.gather(1, sidx)
        m = per_span.to(b.dtype)[None]                 # [1, S, das_dim]
        new = rot.intervene(b, c, m, sufficient=self.sufficient)   # [B, S, d_model]
        return out.scatter(1, bidx, new.to(out.dtype))

    def register_hooks(self):
        self.remove_hooks()

        if self.is_sae or self.is_das:
            interchange = self._sae_interchange if self.is_sae else self._das_interchange
            mlp_site = self.mask_type in ("mlp_sae_span", "das_mlp_span")
            for layer_idx in range(self.num_layers):
                layer = self._get_layer(layer_idx)
                mod = self._get_mlp_module(layer) if mlp_site else layer
                cf_dict = self.cf_acts_saemlp if mlp_site else self.cf_acts_resid

                def make_post_hook(li, cfd):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        tup = isinstance(output, tuple)
                        out = output[0] if tup else output
                        new = interchange(out, cfd.get(li), li)
                        return ((new,) + tuple(output[1:])) if tup else new
                    return hook
                self._hooks.append(mod.register_forward_hook(make_post_hook(layer_idx, cf_dict)))
            return

        # Input embedding hook (node mask with include_input)
        if self.include_input:
            def make_embed_hook():
                def hook(mod, input, output):
                    if self.mask is None:
                        return
                    m = self.mask[0].view(1, 1, 1)
                    result = self._interpolate(output, m, self.cf_acts_embed)
                    return result[0]
                return hook
            self._hooks.append(
                self._get_embed_module().register_forward_hook(make_embed_hook()))

        for layer_idx in range(self.num_layers):
            layer = self._get_layer(layer_idx)

            if self.has_mlp:
                def make_mlp_hook(li):
                    def hook(mod, hook_args):
                        if self.mask is None:
                            return
                        x = hook_args[0]  # [1, seq_len, intermediate_size]
                        if self.is_node:
                            # Node: one scalar per MLP per layer, broadcast
                            off = self._node_offset
                            attn_count = self.num_layers * self.num_heads
                            m = self.mask[off + attn_count + li].view(1, 1, 1)
                        elif self.mask_type == "mlp_tied":
                            # per-(layer, neuron), tied/broadcast across all token positions
                            start = li * self.intermediate_size
                            end = start + self.intermediate_size
                            m = self.mask[start:end].view(1, 1, self.intermediate_size)
                        elif self.mask_type in ("mlp_span", "mlp+attn_span", "mlp+attn_head_span"):
                            # per-(layer, span, neuron): node lives at each span's LAST token.
                            # The cf (patch) is cross-aligned span-for-span: the SOURCE span's
                            # last-token activation is written into the BASE span's last token.
                            # Non-span positions stay clean (identity), which under _interpolate
                            # means default m=0 if sufficient else 1.
                            S, N = self.num_spans, self.intermediate_size
                            per_span = self.mask[li * S * N:(li + 1) * S * N].view(S, N).to(x.dtype)
                            B, seq = self.span_last.shape[0], x.shape[1]
                            bidx = self.span_last[:, :, None].expand(B, S, N)        # base last pos
                            default_m = 0.0 if self.sufficient else 1.0
                            m = x.new_full((B, seq, N), default_m)
                            m = m.scatter(1, bidx, per_span[None].expand(B, S, N))
                            cf_full = self.cf_acts_mlp.get(li)                       # [B, src_seq, N]
                            cf_aligned = x.detach().clone() if cf_full is not None else None
                            if cf_full is not None:
                                sidx = self.span_last_src[:, :, None].expand(B, S, N)
                                cf_aligned = cf_aligned.scatter(1, bidx, cf_full.gather(1, sidx))
                            return self._interpolate(x, m, cf_aligned)
                        else:
                            start = li * self.seq_len * self.intermediate_size
                            end = start + self.seq_len * self.intermediate_size
                            m = self.mask[start:end].view(
                                1, self.seq_len, self.intermediate_size)
                        return self._interpolate(x, m, self.cf_acts_mlp.get(li))
                    return hook
                self._hooks.append(
                    self._get_mlp_module(layer).register_forward_pre_hook(
                        make_mlp_hook(layer_idx)))

            if self.has_attn:
                def make_attn_hook(li):
                    def hook(mod, hook_args):
                        if self.mask is None:
                            return
                        x = hook_args[0]  # [1, seq_len, hidden_size]
                        cf = self.cf_acts_attn.get(li)
                        seq = x.shape[1]

                        if self.is_node:
                            # Node: [n_heads] per layer, broadcast over batch & positions
                            B = x.shape[0]
                            off = self._node_offset + li * self.num_heads
                            m = self.mask[off:off + self.num_heads].view(
                                1, 1, self.num_heads, 1)
                            x4d = x.view(B, seq, self.num_heads, self.head_dim)
                            cf4d = (cf.view(cf.shape[0], cf.shape[1], self.num_heads,
                                            self.head_dim) if cf is not None
                                    else None)
                            out = self._interpolate(x4d, m, cf4d)
                            return (out[0].reshape(B, seq, -1),)

                        if self.mask_type == "attn_output":
                            off = li * self.seq_len
                            m = self.mask[off:off + self.seq_len].view(
                                1, self.seq_len, 1)
                            return self._interpolate(x, m, cf)

                        if self.mask_type == "mlp+attn_dim":
                            # per-(pos, dim) over the o_proj input (pre-out, hidden_size)
                            off = self.mlp_total + li * self.seq_len * self.hidden_size
                            end = off + self.seq_len * self.hidden_size
                            m = self.mask[off:end].view(1, self.seq_len, self.hidden_size)
                            return self._interpolate(x, m, cf)

                        if self.mask_type == "mlp+attn_span":
                            # per-(layer, span, dim) over o_proj input, node at each span's LAST
                            # token, cross-aligned base<-src (mirrors the MLP-span branch).
                            H, S = self.hidden_size, self.num_spans
                            base = self.mlp_span_total + li * S * H
                            per_span = self.mask[base:base + S * H].view(S, H).to(x.dtype)
                            B = self.span_last.shape[0]
                            bidx = self.span_last[:, :, None].expand(B, S, H)
                            default_m = 0.0 if self.sufficient else 1.0
                            m = x.new_full((B, seq, H), default_m).scatter(1, bidx, per_span[None].expand(B, S, H))
                            cf_aligned = x.detach().clone() if cf is not None else None
                            if cf is not None:
                                sidx = self.span_last_src[:, :, None].expand(B, S, H)
                                cf_aligned = cf_aligned.scatter(1, bidx, cf.gather(1, sidx))
                            return self._interpolate(x, m, cf_aligned)

                        if self.mask_type == "mlp+attn_head_span":
                            # per-(layer, span, head): one mask value per head (broadcast over
                            # head_dim), node at each span's LAST token, cross-aligned base<-src.
                            nh, Hd, S = self.num_heads, self.head_dim, self.num_spans
                            base = self.mlp_span_total + li * S * nh
                            per_span = self.mask[base:base + S * nh].view(S, nh).to(x.dtype)
                            B = self.span_last.shape[0]
                            x4 = x.view(B, seq, nh, Hd)
                            bidx = self.span_last[:, :, None].expand(B, S, nh)
                            default_m = 0.0 if self.sufficient else 1.0
                            m = x.new_full((B, seq, nh), default_m).scatter(
                                1, bidx, per_span[None].expand(B, S, nh))[:, :, :, None]  # [B,seq,nh,1]
                            cf4 = None
                            if cf is not None:
                                cf4 = x4.detach().clone()
                                bidx4 = self.span_last[:, :, None, None].expand(B, S, nh, Hd)
                                sidx4 = self.span_last_src[:, :, None, None].expand(B, S, nh, Hd)
                                cf4 = cf4.scatter(1, bidx4, cf.view(B, -1, nh, Hd).gather(1, sidx4))
                            out = self._interpolate(x4, m, cf4)
                            return (out[0].reshape(B, seq, -1),)

                        # attn_head or mlp+attn_head
                        if self.mask_type == "attn_head":
                            off = li * self.seq_len * self.num_heads
                        else:  # mlp+attn_head
                            off = (self.mlp_total
                                   + li * self.seq_len * self.num_heads)
                        end = off + self.seq_len * self.num_heads
                        B = x.shape[0]
                        m = self.mask[off:end].view(
                            1, self.seq_len, self.num_heads, 1)   # broadcast over batch
                        x4d = x.view(B, self.seq_len, self.num_heads,
                                     self.head_dim)
                        cf4d = (cf.view(cf.shape[0], self.seq_len, self.num_heads,
                                        self.head_dim) if cf is not None
                                else None)
                        out = self._interpolate(x4d, m, cf4d)
                        return (out[0].reshape(B, self.seq_len, -1),)
                    return hook
                self._hooks.append(
                    self._get_attn_module(layer).register_forward_pre_hook(
                        make_attn_hook(layer_idx)))

            if self.has_resid:
                def make_resid_hook(li):
                    def hook(mod, input, output):
                        if self.mask is None:
                            return
                        if isinstance(output, tuple):
                            x = output[0]
                        else:
                            x = output
                        off = li * self.seq_len
                        m = self.mask[off:off + self.seq_len].view(
                            1, self.seq_len, 1)
                        cf = self.cf_acts_resid.get(li)
                        new_x = self._interpolate(x, m, cf)[0]
                        # Return same type as input
                        if isinstance(output, tuple):
                            return (new_x,) + output[1:]
                        return new_x
                    return hook
                self._hooks.append(
                    layer.register_forward_hook(make_resid_hook(layer_idx)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def decode_index(self, flat_idx):
        """Convert flat score index -> dict with layer, pos, and component info."""
        flat_idx = int(flat_idx)

        if self.is_node:
            off = self._node_offset
            if self.include_input and flat_idx == 0:
                return {"component": "input", "layer": -1}
            flat_idx -= off
            attn_count = self.num_layers * self.num_heads
            if flat_idx < attn_count:
                layer = flat_idx // self.num_heads
                head = flat_idx % self.num_heads
                return {"component": "attn", "layer": layer, "head": head}
            else:
                layer = flat_idx - attn_count
                return {"component": "mlp", "layer": layer}

        if self.mask_type == "mlp" or (
                self.mask_type == "mlp+attn_head"
                and flat_idx < self.mlp_total):
            layer = flat_idx // (self.seq_len * self.intermediate_size)
            rem = flat_idx % (self.seq_len * self.intermediate_size)
            pos = rem // self.intermediate_size
            neuron = rem % self.intermediate_size
            return {"component": "mlp", "layer": layer, "pos": pos,
                    "neuron": neuron}

        if self.mask_type in ("attn_output", "resid"):
            layer = flat_idx // self.seq_len
            pos = flat_idx % self.seq_len
            comp = "attn" if self.mask_type == "attn_output" else "resid"
            return {"component": comp, "layer": layer, "pos": pos}

        # attn_head or attn part of mlp+attn_head
        if self.mask_type == "mlp+attn_head":
            flat_idx -= self.mlp_total
        layer = flat_idx // (self.seq_len * self.num_heads)
        rem = flat_idx % (self.seq_len * self.num_heads)
        pos = rem // self.num_heads
        head = rem % self.num_heads
        return {"component": "attn", "layer": layer, "pos": pos, "head": head}

    def scores_to_heatmap(self, scores_flat):
        """Reduce scores to [num_layers, N] heatmap. N=seq_len or n_heads+1 for node."""
        device = scores_flat.device

        if self.is_node:
            # Node: heatmap is [num_layers, n_heads+1] (heads then MLP)
            attn_count = self.num_layers * self.num_heads
            attn = scores_flat[:attn_count].view(self.num_layers, self.num_heads)
            mlp = scores_flat[attn_count:].view(self.num_layers, 1)
            return torch.cat([attn, mlp], dim=1)

        heatmap = torch.full((self.num_layers, self.seq_len), float("-inf"),
                             device=device)

        if self.has_mlp:
            mlp_scores = scores_flat[:self.mlp_total].view(
                self.num_layers, self.seq_len, self.intermediate_size)
            heatmap = torch.maximum(heatmap, mlp_scores.max(dim=-1).values)

        if self.has_attn:
            if self.mask_type == "attn_output":
                attn_scores = scores_flat[-self.attn_output_total:].view(
                    self.num_layers, self.seq_len)
            else:
                offset = self.mlp_total if self.mask_type == "mlp+attn_head" else 0
                attn_scores = scores_flat[offset:offset + self.attn_head_total].view(
                    self.num_layers, self.seq_len, self.num_heads).max(dim=-1).values
            heatmap = torch.maximum(heatmap, attn_scores)

        if self.has_resid:
            resid_scores = scores_flat.view(self.num_layers, self.seq_len)
            heatmap = torch.maximum(heatmap, resid_scores)

        return heatmap


class LlamaSpanAttributionHooks:
    """Span-aware sigmoid top-k masking hooks for Llama models.

    Scores are indexed by (layer, span) instead of (layer, token_position).
    At runtime, span scores are expanded to token positions via alignment.
    Supports variable-length inputs across training steps.

    Score layout (flat vector, S = num_spans):
      - mlp:           [num_layers * S * intermediate_size]
      - attn_head:     [num_layers * S * num_heads]
      - attn_output:   [num_layers * S]
      - mlp+attn_head: [mlp_scores | attn_head_scores]
      - resid:         [num_layers * S]
    """

    MASK_TYPES = LlamaAttributionHooks.MASK_TYPES

    def __init__(self, model, mask_type: str, num_spans: int,
                 pos_strategy: str = "last", sufficient: bool = False):
        assert mask_type in self.MASK_TYPES
        assert pos_strategy in ("first", "last", "all")

        self.model = model
        self.mask_type = mask_type
        self.num_spans = num_spans
        self.pos_strategy = pos_strategy
        self.sufficient = sufficient

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size

        S = num_spans
        self.mlp_total = self.num_layers * S * self.intermediate_size
        self.attn_head_total = self.num_layers * S * self.num_heads
        self.scalar_total = self.num_layers * S  # for attn_output, resid

        if mask_type == "mlp":
            self.total = self.mlp_total
        elif mask_type == "attn_output":
            self.total = self.scalar_total
        elif mask_type == "attn_head":
            self.total = self.attn_head_total
        elif mask_type == "mlp+attn_head":
            self.total = self.mlp_total + self.attn_head_total
        elif mask_type == "resid":
            self.total = self.scalar_total
        elif mask_type == "resid_dim":
            self.total = self.num_layers * S * self.hidden_size
        elif mask_type == "das":
            # DAS: per (layer, span, das_dim) scores in rotated subspace
            # das_dim set externally via set_das_dim(); defaults to hidden_size
            self.das_dim = self.hidden_size
            self.total = self.num_layers * S * self.das_dim
        elif mask_type == "sae":
            # per (layer, span, sae feature); saes + total set externally via set_saes()
            self.saes = {}
            self.d_sae = None
            self.total = 0

        self.mask = None
        # DAS rotation layers (set externally via set_das_rotations)
        self.R: dict[int, torch.nn.Module] = {}  # layer_idx -> RotateLayer
        self.cf_acts_mlp: dict[int, torch.Tensor] = {}
        self.cf_acts_attn: dict[int, torch.Tensor] = {}
        self.cf_acts_resid: dict[int, torch.Tensor] = {}
        self._hooks: list = []

        # Per-example alignment (set before each forward)
        self.base_span_to_pos: list[list[int]] = []
        self.src_span_to_pos: list[list[int]] = []
        self.base_seq_len = 0
        self.src_seq_len = 0

    @property
    def has_mlp(self):
        return self.mask_type in ("mlp", "mlp+attn_head")

    @property
    def has_attn(self):
        return self.mask_type in ("attn_output", "attn_head", "mlp+attn_head")

    @property
    def has_resid(self):
        return self.mask_type in ("resid", "resid_dim", "das", "sae")

    @property
    def has_das(self):
        return self.mask_type == "das"

    # Override these in subclasses for different model architectures
    def _get_layer(self, li):
        return self.model.model.layers[li]

    def _get_mlp_module(self, layer):
        return layer.mlp.down_proj

    def _get_attn_module(self, layer):
        return layer.self_attn.o_proj

    def describe(self):
        S = self.num_spans
        parts = []
        if self.has_mlp:
            parts.append(f"MLP: {self.num_layers}L x {S}spans x "
                         f"{self.intermediate_size}n = {self.mlp_total:,}")
        if self.has_attn:
            if self.mask_type == "attn_output":
                parts.append(f"Attn: {self.num_layers}L x {S}spans = "
                             f"{self.scalar_total:,}")
            else:
                parts.append(f"Attn: {self.num_layers}L x {S}spans x "
                             f"{self.num_heads}h = {self.attn_head_total:,}")
        if self.has_das:
            parts.append(f"DAS: {self.num_layers}L x {S}spans x "
                         f"{self.das_dim}d = {self.total:,}")
        elif self.mask_type == "resid_dim":
            parts.append(f"Resid-dim: {self.num_layers}L x {S}spans x "
                         f"{self.hidden_size}d = {self.total:,}")
        elif self.has_resid:
            parts.append(f"Resid: {self.num_layers}L x {S}spans = "
                         f"{self.scalar_total:,}")
        return " + ".join(parts) + f" = {self.total:,} total"

    def set_alignment(self, base_alignment: list[list[int]],
                      src_alignment: list[list[int]],
                      base_seq_len: int, src_seq_len: int):
        """Set span-to-token mapping for the current example."""
        self.base_seq_len = base_seq_len
        self.src_seq_len = src_seq_len
        self.base_span_to_pos = []
        self.src_span_to_pos = []
        for span_i in range(self.num_spans):
            bt = base_alignment[span_i]
            st = src_alignment[span_i]
            if self.pos_strategy == "last":
                self.base_span_to_pos.append([bt[-1]] if bt else [])
                self.src_span_to_pos.append([st[-1]] if st else [])
            elif self.pos_strategy == "first":
                self.base_span_to_pos.append([bt[0]] if bt else [])
                self.src_span_to_pos.append([st[0]] if st else [])
            else:  # all
                self.base_span_to_pos.append(bt)
                self.src_span_to_pos.append(st)

    def set_das_dim(self, das_dim: int):
        """Set DAS subspace dimension and recompute total score count."""
        assert self.mask_type == "das"
        self.das_dim = das_dim
        S = self.num_spans
        self.total = self.num_layers * S * das_dim

    def set_saes(self, saes: dict):
        """Provide per-layer frozen SAEs (objects with .encode/.decode/.W_dec, d_sae)."""
        assert self.mask_type == "sae"
        self.saes = saes
        self.d_sae = next(iter(saes.values())).d_sae
        self.sae_width = self.d_sae + 1   # +1 per-span scored error-term node (always learned)
        self.total = self.num_layers * self.num_spans * self.sae_width

    def cache_cf_activations(self, src_input_ids: torch.Tensor):
        """Cache src activations at hook points."""
        self.mask = None  # disable masking hooks during CF forward
        hooks = []
        for li in range(self.num_layers):
            layer = self._get_layer(li)
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
            if self.has_resid:
                def _resid(idx):
                    def hook(mod, inp, output):
                        x = output[0] if isinstance(output, tuple) else output
                        self.cf_acts_resid[idx] = x.detach()
                    return hook
                hooks.append(layer.register_forward_hook(_resid(li)))

        with torch.no_grad():
            cf_logits = self.model(src_input_ids).logits[0, -1].float()

        for h in hooks:
            h.remove()
        return cf_logits

    def _span_intervene(self, base_act: torch.Tensor, cf_act: torch.Tensor | None,
                        span_mask: torch.Tensor, layer_idx: int,
                        component_dim: int | None = None) -> torch.Tensor:
        """Apply span-level mask to base activations, interpolating with CF.

        base_act: [1, base_seq_len, dim]
        cf_act:   [1, src_seq_len, dim] or None
        span_mask: [num_spans] or [num_spans, component_dim]

        Returns modified base_act (same shape).
        """
        out = base_act.clone()
        span_mask = span_mask.to(base_act.dtype)

        for span_i in range(self.num_spans):
            base_positions = self.base_span_to_pos[span_i]
            src_positions = self.src_span_to_pos[span_i]
            if not base_positions:
                continue

            # Get mask for this span
            m = span_mask[span_i]  # scalar or [component_dim]

            for j, bp in enumerate(base_positions):
                # Corresponding src position (pad with last if fewer src positions)
                sp = src_positions[min(j, len(src_positions) - 1)] if src_positions else bp

                if component_dim is not None:
                    m_expanded = m.view(1, -1)  # [1, dim]
                else:
                    m_expanded = m.view(1, 1)  # [1, 1] for broadcast

                if cf_act is not None:
                    if self.sufficient:
                        out[0, bp] = base_act[0, bp] * (1 - m_expanded) + cf_act[0, sp] * m_expanded
                    else:
                        out[0, bp] = base_act[0, bp] * m_expanded + cf_act[0, sp] * (1 - m_expanded)
                else:
                    out[0, bp] = base_act[0, bp] * m_expanded

        return out

    def _das_intervene(self, base_act: torch.Tensor, cf_act: torch.Tensor | None,
                       span_dim_mask: torch.Tensor,
                       layer_idx: int) -> torch.Tensor:
        """DAS intervention: project to learned subspace, mask, project back.

        base_act: [1, base_seq_len, d_model]
        cf_act:   [1, src_seq_len, d_model] or None
        span_dim_mask: [num_spans, das_dim] per-(span, dim) mask in rotated space
        """
        rotate_layer = self.R.get(layer_idx)
        if rotate_layer is None or cf_act is None:
            return base_act

        out = base_act.clone()
        span_dim_mask = span_dim_mask.to(base_act.dtype)

        for span_i in range(self.num_spans):
            base_positions = self.base_span_to_pos[span_i]
            src_positions = self.src_span_to_pos[span_i]
            if not base_positions:
                continue

            m = span_dim_mask[span_i]  # [das_dim]

            for j, bp in enumerate(base_positions):
                sp = src_positions[min(j, len(src_positions) - 1)] if src_positions else bp
                out[0, bp] = rotate_layer.intervene(
                    base_act[0, bp], cf_act[0, sp], m, sufficient=self.sufficient)

        return out

    def _sae_intervene(self, base_act, cf_act, span_feat_mask, layer_idx):
        """SAE feature interchange. Update = base + decode(masked feature delta), so the SAE
        reconstruction ERROR is held at BASE: mask=0 -> exactly base (lossless); mask=1 ->
        decode(f_cf)+err_base, NOT a clean cf activation. A full feature swap is thus lossy
        (cf features carry base's error, compounded across layers) -- the 100% edge of a
        sufficiency/necessity curve is not a clean cf reference.
        span_feat_mask: [num_spans, d_sae]."""
        sae = self.saes.get(layer_idx)
        if sae is None or cf_act is None:
            return base_act
        out = base_act.clone()
        for span_i in range(self.num_spans):
            bps = self.base_span_to_pos[span_i]
            sps = self.src_span_to_pos[span_i]
            if not bps:
                continue
            sm = span_feat_mask[span_i].float()               # [d_sae + 1]
            m = sm[:sae.d_sae]                                 # feature mask [d_sae]
            m_err = sm[sae.d_sae]                              # scalar: error-term node score
            for j, bp in enumerate(bps):
                sp = sps[min(j, len(sps) - 1)] if sps else bp
                wdt = sae.W_enc.dtype                      # SAE compute dtype (float32)
                b = base_act[0, bp].to(wdt); c = cf_act[0, sp].to(wdt)
                fb, fc = sae.encode(b), sae.encode(c)
                # The SAE reconstruction ERROR is a SCORED node (m_err), like every feature:
                # err_diff = err_cf - err_base = (c-b) - decode(f_cf - f_base). With both the
                # features and the error node selected, a full swap reaches the *clean* cf.
                err_diff = (c - b) - sae.decode_delta(fc - fb)
                if self.sufficient:    # noising: mask=1 (top-k) -> cf, rest base
                    new = b + sae.decode_delta(m * (fc - fb)) + m_err * err_diff
                else:                  # denoising/sufficient: mask=1 -> base, rest cf
                    new = b + sae.decode_delta((1.0 - m) * (fc - fb)) + (1.0 - m_err) * err_diff
                # Numerical guard: a near-full feature swap injects SAE reconstruction
                # error that can compound/overflow across all 32 layers. The intended
                # result is bounded by a real activation (base/cf), so cap the norm at a
                # generous 8x max(|base|,|cf|) -- only triggers on true runaway.
                cap = 8.0 * c.norm()        # c = clean cached cf act, always a sane anchor
                nn_ = new.norm()
                if torch.isfinite(nn_) and nn_ > cap:
                    new = new * (cap / nn_)
                elif not torch.isfinite(nn_):
                    new = c  # degenerate: fall back to the clean cf act rather than NaN
                out[0, bp] = new.to(base_act.dtype)
        return out

    def register_hooks(self):
        self.remove_hooks()
        S = self.num_spans

        for layer_idx in range(self.num_layers):
            layer = self._get_layer(layer_idx)

            if self.has_mlp:
                def make_mlp_hook(li):
                    def hook(mod, hook_args):
                        if self.mask is None:
                            return
                        x = hook_args[0]
                        start = li * S * self.intermediate_size
                        end = start + S * self.intermediate_size
                        span_mask = self.mask[start:end].view(S, self.intermediate_size)
                        out = self._span_intervene(
                            x, self.cf_acts_mlp.get(li), span_mask, li,
                            component_dim=self.intermediate_size)
                        return (out,)
                    return hook
                self._hooks.append(
                    self._get_mlp_module(layer).register_forward_pre_hook(
                        make_mlp_hook(layer_idx)))

            if self.has_attn:
                def make_attn_hook(li):
                    def hook(mod, hook_args):
                        if self.mask is None:
                            return
                        x = hook_args[0]
                        cf = self.cf_acts_attn.get(li)

                        if self.mask_type == "attn_output":
                            off = li * S
                            span_mask = self.mask[off:off + S]
                            out = self._span_intervene(
                                x, cf, span_mask, li, component_dim=None)
                            return (out,)

                        # attn_head or mlp+attn_head
                        if self.mask_type == "attn_head":
                            off = li * S * self.num_heads
                        else:
                            off = self.mlp_total + li * S * self.num_heads
                        span_mask = self.mask[off:off + S * self.num_heads].view(
                            S, self.num_heads)

                        x4d = x.view(1, x.shape[1], self.num_heads, self.head_dim)
                        cf4d = cf.view(1, cf.shape[1], self.num_heads, self.head_dim) if cf is not None else None
                        out4d = self._span_intervene(
                            x4d, cf4d, span_mask, li, component_dim=self.num_heads)
                        return (out4d.reshape(1, x.shape[1], self.hidden_size),)
                    return hook
                self._hooks.append(
                    self._get_attn_module(layer).register_forward_pre_hook(
                        make_attn_hook(layer_idx)))

            if self.mask_type == "resid":
                def make_resid_hook(li):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        if isinstance(output, tuple):
                            x = output[0]
                        else:
                            x = output
                        off = li * S
                        span_mask = self.mask[off:off + S]
                        new_x = self._span_intervene(
                            x, self.cf_acts_resid.get(li), span_mask, li,
                            component_dim=None)
                        if isinstance(output, tuple):
                            return (new_x,) + output[1:]
                        return new_x
                    return hook
                self._hooks.append(
                    layer.register_forward_hook(make_resid_hook(layer_idx)))

            if self.mask_type == "resid_dim":
                def make_resid_dim_hook(li):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        if isinstance(output, tuple):
                            x = output[0]
                        else:
                            x = output
                        off = li * S * self.hidden_size
                        span_mask = self.mask[off:off + S * self.hidden_size].view(
                            S, self.hidden_size)
                        new_x = self._span_intervene(
                            x, self.cf_acts_resid.get(li), span_mask, li,
                            component_dim=self.hidden_size)
                        if isinstance(output, tuple):
                            return (new_x,) + output[1:]
                        return new_x
                    return hook
                self._hooks.append(
                    layer.register_forward_hook(make_resid_dim_hook(layer_idx)))

            if self.has_das:
                def make_das_hook(li):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        if isinstance(output, tuple):
                            x = output[0]
                        else:
                            x = output
                        # Mask layout: [num_layers * num_spans * das_dim]
                        off = li * S * self.das_dim
                        span_dim_mask = self.mask[off:off + S * self.das_dim].view(
                            S, self.das_dim)
                        new_x = self._das_intervene(
                            x, self.cf_acts_resid.get(li),
                            span_dim_mask, li)
                        if isinstance(output, tuple):
                            return (new_x,) + output[1:]
                        return new_x
                    return hook
                self._hooks.append(
                    layer.register_forward_hook(make_das_hook(layer_idx)))

            if self.mask_type == "sae":
                def make_sae_hook(li):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        x = output[0] if isinstance(output, tuple) else output
                        w = self.d_sae + 1   # features + scored error node
                        off = li * S * w
                        span_feat_mask = self.mask[off:off + S * w].view(S, w)
                        new_x = self._sae_intervene(x, self.cf_acts_resid.get(li),
                                                    span_feat_mask, li)
                        if isinstance(output, tuple):
                            return (new_x,) + output[1:]
                        return new_x
                    return hook
                self._hooks.append(layer.register_forward_hook(make_sae_hook(layer_idx)))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def decode_index(self, flat_idx: int, span_names: list[str] | None = None):
        """Convert flat score index -> dict with layer, span, and component info."""
        flat_idx = int(flat_idx)
        S = self.num_spans

        if self.mask_type == "mlp" or (
                self.mask_type == "mlp+attn_head" and flat_idx < self.mlp_total):
            layer = flat_idx // (S * self.intermediate_size)
            rem = flat_idx % (S * self.intermediate_size)
            span = rem // self.intermediate_size
            neuron = rem % self.intermediate_size
            info = {"component": "mlp", "layer": layer, "span": span,
                    "neuron": neuron}
            if span_names:
                info["span_name"] = span_names[span]
            return info

        if self.mask_type == "das":
            # Layout: [num_layers * num_spans * hidden_size]
            layer = flat_idx // (S * self.hidden_size)
            rem = flat_idx % (S * self.hidden_size)
            span = rem // self.hidden_size
            dim = rem % self.hidden_size
            info = {"component": "das", "layer": layer, "span": span, "dim": dim}
            if span_names:
                info["span_name"] = span_names[span]
            return info

        if self.mask_type in ("attn_output", "resid"):
            layer = flat_idx // S
            span = flat_idx % S
            comp = "attn" if self.mask_type == "attn_output" else "resid"
            info = {"component": comp, "layer": layer, "span": span}
            if span_names:
                info["span_name"] = span_names[span]
            return info

        # attn_head or attn part of mlp+attn_head
        if self.mask_type == "mlp+attn_head":
            flat_idx -= self.mlp_total
        layer = flat_idx // (S * self.num_heads)
        rem = flat_idx % (S * self.num_heads)
        span = rem // self.num_heads
        head = rem % self.num_heads
        info = {"component": "attn", "layer": layer, "span": span, "head": head}
        if span_names:
            info["span_name"] = span_names[span]
        return info

    def scores_to_heatmap(self, scores_flat: torch.Tensor):
        """Reduce scores to [num_layers, num_spans] heatmap."""
        S = self.num_spans
        device = scores_flat.device
        heatmap = torch.full((self.num_layers, S), float("-inf"), device=device)

        if self.has_mlp:
            mlp = scores_flat[:self.mlp_total].view(
                self.num_layers, S, self.intermediate_size)
            heatmap = torch.maximum(heatmap, mlp.max(dim=-1).values)

        if self.has_attn:
            if self.mask_type == "attn_output":
                attn = scores_flat[-self.scalar_total:].view(self.num_layers, S)
            else:
                off = self.mlp_total if self.mask_type == "mlp+attn_head" else 0
                attn = scores_flat[off:off + self.attn_head_total].view(
                    self.num_layers, S, self.num_heads).max(dim=-1).values
            heatmap = torch.maximum(heatmap, attn)

        if self.has_das:
            heatmap = torch.maximum(
                heatmap, scores_flat.view(self.num_layers, S, self.das_dim).max(dim=-1).values)
        elif self.mask_type == "resid_dim":
            heatmap = torch.maximum(
                heatmap, scores_flat.view(self.num_layers, S, self.hidden_size).max(dim=-1).values)
        elif self.has_resid:
            heatmap = torch.maximum(
                heatmap, scores_flat.view(self.num_layers, S))

        return heatmap
