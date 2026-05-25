import torch
import torch.nn as nn
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


def test_gradcheck():
    scores = torch.randn(3, 10, dtype=torch.float64, requires_grad=True)
    k = 4.0
    T = 1.0
    n_iters = 100
    assert torch.autograd.gradcheck(
        lambda s: SigmoidTopK.apply(s, k, T, n_iters),
        (scores,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print("Gradient check passed!")


def train_one(elements, T, num_steps=2000, lr=0.01, n_iters=50, eval_every=10, seed=0):
    from scipy.stats import spearmanr

    torch.manual_seed(seed)
    num_elements = len(elements)

    scores = nn.Parameter(torch.randn(num_elements))
    optimizer = torch.optim.Adam([scores], lr=lr)

    true_order = elements.argsort(descending=True)
    true_top10 = set(true_order[:10].tolist())
    true_ranks = torch.zeros(num_elements)
    true_ranks[true_order] = torch.arange(num_elements, dtype=torch.float)
    n_pairs = num_elements * (num_elements - 1) // 2
    mask_ut = torch.triu(torch.ones(num_elements, num_elements, dtype=torch.bool), diagonal=1)
    true_diff = true_ranks.unsqueeze(1) - true_ranks.unsqueeze(0)

    steps, spearman, top10, pairwise = [], [], [], []

    for step in range(num_steps):
        k = 1.0 + (num_elements - 1.0) * torch.rand(1).item()
        m = sigmoid_topk(scores, k=k, T=T, n_iters=n_iters)
        loss = -(elements * m).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % eval_every == 0:
            with torch.no_grad():
                lo = scores.data.argsort(descending=True)
                corr, _ = spearmanr(scores.data.numpy(), elements.numpy())
                lr_ = torch.zeros(num_elements)
                lr_[lo] = torch.arange(num_elements, dtype=torch.float)
                ld = lr_.unsqueeze(1) - lr_.unsqueeze(0)
                pa = ((true_diff[mask_ut] * ld[mask_ut]) > 0).sum().item() / n_pairs
            steps.append(step + 1)
            spearman.append(corr)
            top10.append(len(set(lo[:10].tolist()) & true_top10))
            pairwise.append(pa)

    return steps, spearman, top10, pairwise


def main():
    import matplotlib.pyplot as plt

    torch.manual_seed(42)
    num_elements = 100
    elements = torch.randn(num_elements)

    temps = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    results = {}
    for T in temps:
        print(f"T={T} ...", end=" ", flush=True)
        steps, spearman, top10, pairwise = train_one(
            elements, T=T, num_steps=2000, eval_every=10, lr=0.01,
        )
        results[T] = (steps, spearman, top10, pairwise)
        print(f"final spearman={spearman[-1]:.4f}  pairwise={pairwise[-1]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    for T, (steps, spearman, top10, pairwise) in results.items():
        axes[0, 0].plot(steps, spearman, label=f"T={T}", linewidth=0.8, alpha=0.85)
        axes[0, 1].plot(steps, top10, label=f"T={T}", linewidth=0.8, alpha=0.85)
        axes[1, 0].plot(steps, pairwise, label=f"T={T}", linewidth=0.8, alpha=0.85)

    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Spearman ρ")
    axes[0, 0].set_title("Spearman Rank Correlation")
    axes[0, 0].set_ylim(-0.1, 1.05)
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Overlap")
    axes[0, 1].set_title("Top-10 Set Overlap with Ground Truth")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Pairwise Accuracy")
    axes[1, 0].set_title("Pairwise Ordering Accuracy")
    axes[1, 0].set_ylim(0.4, 1.02)
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].axis("off")

    fig.suptitle("Temperature Sweep (d=100) — Adaptive Sigmoid Top-K", fontsize=13)
    fig.tight_layout()
    fig.savefig("temp_sweep.png", dpi=150)
    print("Saved temp_sweep.png")


if __name__ == "__main__":
    test_gradcheck()
    main()
