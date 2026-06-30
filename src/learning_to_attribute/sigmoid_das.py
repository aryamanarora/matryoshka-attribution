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


def householder_product(V):
    """Build orthogonal matrix from Householder reflections.

    V: [k, d] — k Householder vectors of dimension d.
    Returns: [d, d] orthogonal matrix R = H_1 H_2 ... H_k
    where H_i = I - 2 v_i v_i^T / ||v_i||^2.

    Using k < d restricts the rotation to a k-dimensional subspace.
    """
    k, d = V.shape
    R = torch.eye(d, device=V.device, dtype=V.dtype)
    for i in range(k):
        v = V[i]
        v_norm_sq = v @ v
        if v_norm_sq < 1e-12:
            continue
        R = R - (2.0 / v_norm_sq) * torch.outer(R @ v, v)
    return R


def cayley(W):
    """Cayley transform: skew-symmetric params -> orthogonal matrix.

    W: [d, d] — arbitrary matrix, upper triangle used to build skew-symmetric A.
    Returns: [d, d] orthogonal matrix R = (I - A)(I + A)^{-1}.
    """
    A = W.triu(1) - W.triu(1).T
    I = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
    return torch.linalg.solve(I + A, I - A)
