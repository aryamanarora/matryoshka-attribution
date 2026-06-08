"""DAS (Distributed Alignment Search) rotation parameterizations."""

import torch


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
