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

    MASK_TYPES = {"mlp", "attn_output", "attn_head", "mlp+attn_head", "resid", "node", "das"}

    def __init__(self, model, mask_type, seq_len, sufficient=False, include_input=False):
        assert mask_type in self.MASK_TYPES, f"Unknown mask type: {mask_type}"

        self.model = model
        self.mask_type = mask_type
        self.seq_len = seq_len
        self.sufficient = sufficient
        self.include_input = include_input and (mask_type == "node")

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.hidden_size = config.hidden_size

        self.mlp_total = self.num_layers * seq_len * self.intermediate_size
        self.attn_output_total = self.num_layers * seq_len
        self.attn_head_total = self.num_layers * seq_len * self.num_heads
        self.resid_total = self.num_layers * seq_len
        self.node_total = self.num_layers * self.num_heads + self.num_layers + (1 if self.include_input else 0)

        if mask_type == "mlp":
            self.total = self.mlp_total
        elif mask_type == "attn_output":
            self.total = self.attn_output_total
        elif mask_type == "attn_head":
            self.total = self.attn_head_total
        elif mask_type == "mlp+attn_head":
            self.total = self.mlp_total + self.attn_head_total
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
        return self.mask_type in ("mlp", "mlp+attn_head", "node")

    @property
    def has_attn(self):
        return self.mask_type in ("attn_output", "attn_head", "mlp+attn_head", "node")

    @property
    def has_resid(self):
        return self.mask_type == "resid"

    @property
    def is_node(self):
        return self.mask_type == "node"

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
        if self.is_node:
            inp = "+input" if self.include_input else ""
            parts.append(f"Node: {self.num_layers}L x ({self.num_heads}h + 1mlp){inp} = "
                         f"{self.node_total:,}")
        else:
            if self.has_mlp:
                parts.append(f"MLP: {self.num_layers}L x {self.seq_len}pos x "
                             f"{self.intermediate_size}n = {self.mlp_total:,}")
            if self.has_attn:
                if self.mask_type == "attn_output":
                    parts.append(f"Attn: {self.num_layers}L x {self.seq_len}pos = "
                                 f"{self.attn_output_total:,}")
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

    def register_hooks(self):
        self.remove_hooks()

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
                            # Node: [n_heads] per layer, broadcast over positions
                            off = self._node_offset + li * self.num_heads
                            m = self.mask[off:off + self.num_heads].view(
                                1, 1, self.num_heads, 1)
                            x4d = x.view(1, seq, self.num_heads, self.head_dim)
                            cf4d = (cf.view(1, cf.shape[1], self.num_heads,
                                            self.head_dim) if cf is not None
                                    else None)
                            out = self._interpolate(x4d, m, cf4d)
                            return (out[0].reshape(1, seq, -1),)

                        if self.mask_type == "attn_output":
                            off = li * self.seq_len
                            m = self.mask[off:off + self.seq_len].view(
                                1, self.seq_len, 1)
                            return self._interpolate(x, m, cf)

                        # attn_head or mlp+attn_head
                        if self.mask_type == "attn_head":
                            off = li * self.seq_len * self.num_heads
                        else:  # mlp+attn_head
                            off = (self.mlp_total
                                   + li * self.seq_len * self.num_heads)
                        end = off + self.seq_len * self.num_heads
                        m = self.mask[off:end].view(
                            1, self.seq_len, self.num_heads, 1)

                        x4d = x.view(1, self.seq_len, self.num_heads,
                                     self.head_dim)
                        cf4d = (cf.view(1, self.seq_len, self.num_heads,
                                        self.head_dim) if cf is not None
                                else None)
                        out = self._interpolate(x4d, m, cf4d)
                        return (out[0].reshape(1, self.seq_len, -1),)
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
        elif mask_type == "das":
            # DAS: per (layer, span, dim) scores — single flat vector
            # Layout: [num_layers * num_spans * hidden_size]
            self.total = self.num_layers * S * self.hidden_size

        self.mask = None
        # DAS rotation matrices (set externally per forward step)
        self.R: dict[int, torch.Tensor] = {}  # layer_idx -> [d_model, d_model]
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
        return self.mask_type in ("resid", "das")

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
        """DAS intervention: rotate, mask subspace dims per span, un-rotate.

        base_act: [1, base_seq_len, d_model]
        cf_act:   [1, src_seq_len, d_model] or None
        span_dim_mask: [num_spans, d_model] per-(span, dim) mask in rotated space
        """
        R = self.R.get(layer_idx)
        if R is None or cf_act is None:
            return base_act

        R = R.to(base_act.dtype)
        out = base_act.clone()
        span_dim_mask = span_dim_mask.to(base_act.dtype)

        for span_i in range(self.num_spans):
            base_positions = self.base_span_to_pos[span_i]
            src_positions = self.src_span_to_pos[span_i]
            if not base_positions:
                continue

            m = span_dim_mask[span_i]  # [d_model]

            for j, bp in enumerate(base_positions):
                sp = src_positions[min(j, len(src_positions) - 1)] if src_positions else bp

                # Rotate into learned basis
                rotated_base = R @ base_act[0, bp]  # [d_model]
                rotated_cf = R @ cf_act[0, sp]       # [d_model]

                # Interpolate in rotated space
                if self.sufficient:
                    interpolated = rotated_base * (1 - m) + rotated_cf * m
                else:
                    interpolated = rotated_base * m + rotated_cf * (1 - m)

                # Un-rotate back
                out[0, bp] = R.T @ interpolated

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

            if self.has_resid and not self.has_das:
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

            if self.has_das:
                def make_das_hook(li):
                    def hook(mod, inp, output):
                        if self.mask is None:
                            return
                        if isinstance(output, tuple):
                            x = output[0]
                        else:
                            x = output
                        # Mask layout: [num_layers * num_spans * hidden_size]
                        # This layer's slice: [num_spans * hidden_size]
                        off = li * S * self.hidden_size
                        span_dim_mask = self.mask[off:off + S * self.hidden_size].view(
                            S, self.hidden_size)
                        new_x = self._das_intervene(
                            x, self.cf_acts_resid.get(li),
                            span_dim_mask, li)
                        if isinstance(output, tuple):
                            return (new_x,) + output[1:]
                        return new_x
                    return hook
                self._hooks.append(
                    layer.register_forward_hook(make_das_hook(layer_idx)))

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

        if self.has_resid:
            heatmap = torch.maximum(
                heatmap, scores_flat.view(self.num_layers, S))

        return heatmap
