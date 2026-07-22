from .sigmoid_topk import (
    SigmoidTopK, sigmoid_topk, sigmoid_topk_hard, sigmoid_topk_detached_tau,
    test_gradcheck,
)
from .sigmoid_das import RotateLayer, make_rotate_layer, householder_product, cayley
# Core MAttr learning algorithm — depends only on torch, so always importable. Kept above
# the heavy model/data imports so other repos (e.g. circuits) can `from learning_to_attribute
# import build_mask` without pulling transformers/datasets.
from .masks import MaskResult, build_mask, build_bias_mask, VARIANTS
from .schedules import sample_k, sample_k_sum_pow2, natural_k
from .trainer import learn_scores, TrainResult
from .edge_pruning import learn_scores_edge_pruning
from .evaluate import sparsity_sweep
from .modes import normalize_mode, preferred_mode, MODE_CHOICES, ISO, CAUSE

# Model/data adapters need transformers/datasets; degrade gracefully if those aren't installed
# (a consumer may only want the core learning algorithm).
try:
    from .models import LlamaAttributionHooks, LlamaSpanAttributionHooks
    from .data import CounterfactualDataset, CausalGymDataset, SVADataset, SVA_TASKS
except ImportError as _e:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"learning_to_attribute: model/data adapters unavailable ({_e}); "
                   "core (build_mask/learn_scores/...) still usable.")
    LlamaAttributionHooks = LlamaSpanAttributionHooks = None
    CounterfactualDataset = CausalGymDataset = None
    SVADataset = None
    SVA_TASKS = ()

__all__ = [
    "SigmoidTopK", "sigmoid_topk", "sigmoid_topk_hard", "sigmoid_topk_detached_tau",
    "test_gradcheck",
    "RotateLayer", "make_rotate_layer", "householder_product", "cayley",
    "LlamaAttributionHooks", "LlamaSpanAttributionHooks",
    "CounterfactualDataset", "CausalGymDataset", "SVADataset", "SVA_TASKS",
    # MAttr learning algorithm (shared core)
    "MaskResult", "build_mask", "build_bias_mask", "VARIANTS",
    "sample_k", "sample_k_sum_pow2", "natural_k",
    "learn_scores", "TrainResult", "sparsity_sweep",
    "learn_scores_edge_pruning",
    "normalize_mode", "preferred_mode", "MODE_CHOICES", "ISO", "CAUSE",
]
