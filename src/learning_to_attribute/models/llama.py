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

    MASK_TYPES = {"mlp", "attn_output", "attn_head", "mlp+attn_head", "resid"}

    def __init__(self, model, mask_type, seq_len, flip=False):
        assert mask_type in self.MASK_TYPES, f"Unknown mask type: {mask_type}"

        self.model = model
        self.mask_type = mask_type
        self.seq_len = seq_len
        self.flip = flip

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size

        self.mlp_total = self.num_layers * seq_len * self.intermediate_size
        self.attn_output_total = self.num_layers * seq_len
        self.attn_head_total = self.num_layers * seq_len * self.num_heads
        self.resid_total = self.num_layers * seq_len

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

        self.mask = None
        self.cf_acts_mlp = {}
        self.cf_acts_attn = {}
        self.cf_acts_resid = {}
        self._hooks = []

    @property
    def has_mlp(self):
        return self.mask_type in ("mlp", "mlp+attn_head")

    @property
    def has_attn(self):
        return self.mask_type in ("attn_output", "attn_head", "mlp+attn_head")

    @property
    def has_resid(self):
        return self.mask_type == "resid"

    def describe(self):
        parts = []
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
        hooks = []
        for li in range(self.num_layers):
            layer = self.model.model.layers[li]
            if self.has_mlp:
                def _mlp(idx):
                    def hook(mod, args):
                        self.cf_acts_mlp[idx] = args[0].detach()
                    return hook
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mlp(li)))
            if self.has_attn:
                def _attn(idx):
                    def hook(mod, args):
                        self.cf_acts_attn[idx] = args[0].detach()
                    return hook
                hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn(li)))
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
            if self.flip:
                return (x * (1 - m) + cf_act * m,)
            else:
                return (x * m + cf_act * (1 - m),)
        return (x * m,)

    def register_hooks(self):
        self.remove_hooks()

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            if self.has_mlp:
                def make_mlp_hook(li):
                    def hook(mod, hook_args):
                        x = hook_args[0]  # [1, seq_len, intermediate_size]
                        start = li * self.seq_len * self.intermediate_size
                        end = start + self.seq_len * self.intermediate_size
                        m = self.mask[start:end].view(
                            1, self.seq_len, self.intermediate_size)
                        return self._interpolate(x, m, self.cf_acts_mlp.get(li))
                    return hook
                self._hooks.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        make_mlp_hook(layer_idx)))

            if self.has_attn:
                def make_attn_hook(li):
                    def hook(mod, hook_args):
                        x = hook_args[0]  # [1, seq_len, hidden_size]
                        cf = self.cf_acts_attn.get(li)

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
                        return (out[0].reshape(1, self.seq_len,
                                               self.hidden_size),)
                    return hook
                self._hooks.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        make_attn_hook(layer_idx)))

            if self.has_resid:
                def make_resid_hook(li):
                    def hook(mod, input, output):
                        # Extract hidden_states from output (may be tensor or tuple)
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
        """Reduce scores to [num_layers, seq_len] heatmap (max over neurons/heads)."""
        device = scores_flat.device
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
                 pos_strategy: str = "last", flip: bool = False):
        assert mask_type in self.MASK_TYPES
        assert pos_strategy in ("first", "last", "all")

        self.model = model
        self.mask_type = mask_type
        self.num_spans = num_spans
        self.pos_strategy = pos_strategy
        self.flip = flip

        config = model.config
        self.num_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
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

        self.mask = None
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
        return self.mask_type == "resid"

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
        if self.has_resid:
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
        hooks = []
        for li in range(self.num_layers):
            layer = self.model.model.layers[li]
            if self.has_mlp:
                def _mlp(idx):
                    def hook(mod, args):
                        self.cf_acts_mlp[idx] = args[0].detach()
                    return hook
                hooks.append(layer.mlp.down_proj.register_forward_pre_hook(_mlp(li)))
            if self.has_attn:
                def _attn(idx):
                    def hook(mod, args):
                        self.cf_acts_attn[idx] = args[0].detach()
                    return hook
                hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(_attn(li)))
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
                    if self.flip:
                        out[0, bp] = base_act[0, bp] * (1 - m_expanded) + cf_act[0, sp] * m_expanded
                    else:
                        out[0, bp] = base_act[0, bp] * m_expanded + cf_act[0, sp] * (1 - m_expanded)
                else:
                    out[0, bp] = base_act[0, bp] * m_expanded

        return out

    def register_hooks(self):
        self.remove_hooks()
        S = self.num_spans

        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]

            if self.has_mlp:
                def make_mlp_hook(li):
                    def hook(mod, hook_args):
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
                    layer.mlp.down_proj.register_forward_pre_hook(
                        make_mlp_hook(layer_idx)))

            if self.has_attn:
                def make_attn_hook(li):
                    def hook(mod, hook_args):
                        x = hook_args[0]
                        cf = self.cf_acts_attn.get(li)

                        if self.mask_type == "attn_output":
                            off = li * S
                            span_mask = self.mask[off:off + S]  # [S]
                            # Expand scalar per-span to full hidden dim
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

                        # Reshape to per-head
                        x4d = x.view(1, x.shape[1], self.num_heads, self.head_dim)
                        cf4d = cf.view(1, cf.shape[1], self.num_heads, self.head_dim) if cf is not None else None
                        out4d = self._span_intervene(
                            x4d, cf4d, span_mask, li, component_dim=self.num_heads)
                        return (out4d.reshape(1, x.shape[1], self.hidden_size),)
                    return hook
                self._hooks.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(
                        make_attn_hook(layer_idx)))

            if self.has_resid:
                def make_resid_hook(li):
                    def hook(mod, inp, output):
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
