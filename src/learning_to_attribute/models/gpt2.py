"""GPT-2 hook management for sigmoid top-k attribution."""

from .llama import LlamaAttributionHooks, LlamaSpanAttributionHooks


class GPT2AttributionHooks(LlamaAttributionHooks):
    """Attribution hooks for GPT-2 models.

    Module paths:
      - Layers: model.transformer.h[i]
      - MLP down: layer.mlp.c_proj
      - Attn out: layer.attn.c_proj

    Config uses n_layer/n_head/n_embd instead of num_hidden_layers etc.
    """

    def __init__(self, model, mask_type, seq_len, sufficient=False, **kwargs):
        # GPT-2 config uses different names
        config = model.config
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head
        config.hidden_size = config.n_embd
        if not hasattr(config, "intermediate_size"):
            config.intermediate_size = config.n_embd * 4
        super().__init__(model, mask_type, seq_len, sufficient=sufficient, **kwargs)

    def _get_layer(self, li):
        return self.model.transformer.h[li]

    def _get_embed_module(self):
        return self.model.transformer.wte

    def _get_mlp_module(self, layer):
        return layer.mlp.c_proj

    def _get_attn_module(self, layer):
        return layer.attn.c_proj


class GPT2SpanAttributionHooks(LlamaSpanAttributionHooks):
    """Span-aware attribution hooks for GPT-2 models."""

    def __init__(self, model, mask_type, num_spans, pos_strategy="last", sufficient=False):
        config = model.config
        config.num_hidden_layers = config.n_layer
        config.num_attention_heads = config.n_head
        config.hidden_size = config.n_embd
        if not hasattr(config, "intermediate_size"):
            config.intermediate_size = config.n_embd * 4
        super().__init__(model, mask_type, num_spans, pos_strategy=pos_strategy, sufficient=sufficient)

    def _get_layer(self, li):
        return self.model.transformer.h[li]

    def _get_mlp_module(self, layer):
        return layer.mlp.c_proj

    def _get_attn_module(self, layer):
        return layer.attn.c_proj
