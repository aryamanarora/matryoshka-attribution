import torch
from torch.autograd import Function

EPS = 1e-8


class SigmoidTopK(Function):
    """Differentiable top-k mask via sigmoid + bisection with implicit-diff backward."""

    @staticmethod
    def forward(ctx, scores, k, T, n_iters):
        # scores: [..., d], k: float, T: float, n_iters: int
        # Bracket for bisection: tau in [lo, hi] per batch element
        lo = scores.min(dim=-1, keepdim=True).values - 10 * T
        hi = scores.max(dim=-1, keepdim=True).values + 10 * T

        # Bisection to find tau such that sum_i sigma((s_i - tau) / T) = k
        with torch.no_grad():
            for _ in range(n_iters):
                mid = (lo + hi) / 2
                f_mid = torch.special.expit((scores - mid) / T).sum(dim=-1, keepdim=True)
                # f is strictly decreasing in tau:
                #   f_mid > k  =>  tau too low, raise lo
                #   f_mid <= k =>  tau too high, lower hi
                lo = torch.where(f_mid > k, mid, lo)
                hi = torch.where(f_mid > k, hi, mid)

        tau = (lo + hi) / 2
        mask = torch.special.expit((scores - tau) / T)

        ctx.save_for_backward(mask)
        ctx.T = T
        return mask

    @staticmethod
    def backward(ctx, grad_output):
        (mask,) = ctx.saved_tensors
        T = ctx.T

        # Implicit differentiation of the constraint sum_i sigma((s_i - tau)/T) = k.
        #
        # Let z_i = (s_i - tau) / T, so m_i = sigma(z_i).
        # sigma'(z_i) = m_i * (1 - m_i)  =: sp_i
        #
        # Differentiating the constraint w.r.t. s_j:
        #   sum_i sp_i * (delta_{ij} - dtau/ds_j) / T = 0
        #   => dtau/ds_j = sp_j / sum_i sp_i
        #
        # Jacobian of mask w.r.t. scores:
        #   dm_i/ds_j = (sp_i / T) * (delta_{ij} - dtau/ds_j)
        #             = (sp_i / T) * (delta_{ij} - sp_j / sum_l sp_l)
        #
        # Vector-Jacobian product with g = grad_output:
        #   dL/ds_j = sum_i g_i * dm_i/ds_j
        #           = (sp_j / T) * (g_j - sum_i g_i sp_i / sum_i sp_i)

        sp = mask * (1.0 - mask)  # [..., d]
        sp_sum = sp.sum(dim=-1, keepdim=True).clamp(min=EPS)  # [..., 1]
        gsp = (grad_output * sp).sum(dim=-1, keepdim=True)  # [..., 1]

        grad_scores = (sp / T) * (grad_output - gsp / sp_sum)
        return grad_scores, None, None, None


def sigmoid_topk(scores, k, T=1.0, n_iters=50):
    return SigmoidTopK.apply(scores, k, T, n_iters)


def sigmoid_topk_detached_tau(scores, k, T=1.0, n_iters=50):
    """Same forward as sigmoid_topk, but tau is detached in backward.

    Removes the cross-score coupling term from the gradient.
    Each score gets independent sigmoid'(s_i - tau) gradient,
    equivalent to independent sigmoid gates with a shared threshold.
    """
    # Bisection to find tau (same as SigmoidTopK.forward)
    lo = scores.min(dim=-1, keepdim=True).values - 10 * T
    hi = scores.max(dim=-1, keepdim=True).values + 10 * T
    with torch.no_grad():
        for _ in range(n_iters):
            mid = (lo + hi) / 2
            f_mid = torch.special.expit((scores - mid) / T).sum(dim=-1, keepdim=True)
            lo = torch.where(f_mid > k, mid, lo)
            hi = torch.where(f_mid > k, hi, mid)
    tau = ((lo + hi) / 2).detach()  # DETACHED
    return torch.special.expit((scores - tau) / T)
