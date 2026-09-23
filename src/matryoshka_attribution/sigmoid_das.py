"""DAS (Distributed Alignment Search) rotation parameterizations."""

import torch
import torch.nn as nn


class RotateLayer(nn.Module):
    """Low-rank orthogonal projection for DAS.

    Projects from d_model to das_dim via a semi-orthogonal matrix W: [d_model, das_dim].
    Orthogonality (W^T W = I) is enforced by torch parametrizations.
    """

    def __init__(self, d_model, das_dim=None):
        super().__init__()
        if das_dim is None:
            das_dim = d_model
        weight = torch.empty(d_model, das_dim)
        nn.init.orthogonal_(weight)
        self.weight = nn.Parameter(weight)

    def forward(self, x):
        return x.to(self.weight.dtype) @ self.weight

    def intervene(self, base, cf, mask, sufficient=True):
        """Apply masked intervention in the rotated subspace.

        base: [..., d_model]
        cf:   [..., d_model]
        mask: [..., das_dim] values in [0, 1]
        sufficient: if True, null space keeps base, mask=1 applies CF.
                    if False (necessary), null space uses CF, mask=1 keeps base.
        """
        W = self.weight.to(base.dtype)
        rotated_base = base @ W
        rotated_cf = cf @ W
        if sufficient:
            return base + ((rotated_cf - rotated_base) * mask) @ W.T
        else:
            return cf + ((rotated_base - rotated_cf) * mask) @ W.T


def make_rotate_layer(d_model, das_dim=None):
    layer = RotateLayer(d_model, das_dim)
    # householder: semi-orthogonal weight stored low-rank as das_dim reflection vectors
    # ([d_model, das_dim]). NOTE PyTorch already auto-selects householder for tall matrices,
    # so this is explicit-not-a-speedup. The cost is APPLYING the product of das_dim
    # reflections, recomputed (+ backprop) per layer per step; across many layers this is the
    # DAS-training bottleneck, not the d_model x das_dim storage.
    return nn.utils.parametrizations.orthogonal(layer, orthogonal_map="householder")
