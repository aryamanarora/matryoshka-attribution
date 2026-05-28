"""GPT-NeoX/Pythia hook management for sigmoid top-k attribution."""

from .llama import LlamaAttributionHooks, LlamaSpanAttributionHooks


class GPTNeoXAttributionHooks(LlamaAttributionHooks):
    """Attribution hooks for GPT-NeoX/Pythia models.

    Module paths:
      - Layers: model.gpt_neox.layers[i]
      - MLP down: layer.mlp.dense_4h_to_h
      - Attn out: layer.attention.dense
    """

    def _get_layer(self, li):
        return self.model.gpt_neox.layers[li]

    def _get_mlp_module(self, layer):
        return layer.mlp.dense_4h_to_h

    def _get_attn_module(self, layer):
        return layer.attention.dense


class GPTNeoXSpanAttributionHooks(LlamaSpanAttributionHooks):
    """Span-aware attribution hooks for GPT-NeoX/Pythia models."""

    def _get_layer(self, li):
        return self.model.gpt_neox.layers[li]

    def _get_mlp_module(self, layer):
        return layer.mlp.dense_4h_to_h

    def _get_attn_module(self, layer):
        return layer.attention.dense
