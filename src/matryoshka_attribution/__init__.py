from .sigmoid_topk import SigmoidTopK, sigmoid_topk, sigmoid_topk_detached_tau
# Core MAttr learning algorithm — depends only on torch, so always importable. Kept above
# the heavy model/data imports so other repos (e.g. circuits) can `from matryoshka_attribution
# import build_mask` without pulling transformers/datasets.
from .masks import MaskResult, build_mask, VARIANTS
from .schedules import sample_k
from .trainer import learn_scores, expected_gradients, TrainResult
from .cf_cache import CFActivationCache
from .edge_pruning import learn_scores_edge_pruning
from .evaluate import sparsity_sweep
from .modes import normalize_mode, MODE_CHOICES, ISO, CAUSE

# Model/data adapters need transformers/datasets; degrade gracefully if those aren't installed
# (a consumer may only want the core learning algorithm).
try:
    from .models import LlamaAttributionHooks
    from .data import SVADataset, SVA_TASKS
except ImportError as _e:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(f"matryoshka_attribution: model/data adapters unavailable ({_e}); "
                   "core (build_mask/learn_scores/...) still usable.")
    LlamaAttributionHooks = None
    SVADataset = None
    SVA_TASKS = ()

__all__ = [
    "SigmoidTopK", "sigmoid_topk", "sigmoid_topk_detached_tau",
    "LlamaAttributionHooks",
    "SVADataset", "SVA_TASKS",
    # MAttr learning algorithm (shared core)
    "MaskResult", "build_mask", "VARIANTS",
    "sample_k",
    "learn_scores", "expected_gradients", "TrainResult", "sparsity_sweep",
    "CFActivationCache",
    "learn_scores_edge_pruning",
    "normalize_mode", "MODE_CHOICES", "ISO", "CAUSE",
]
