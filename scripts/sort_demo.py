"""Sorting demo: learn a total ordering of random elements via adaptive sigmoid top-k."""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from learning_to_attribute import sigmoid_topk


def train_one(elements, T, num_steps=2000, lr=0.01, n_iters=50, eval_every=10, seed=0):
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
    fig.savefig("plots/temp_sweep.png", dpi=150)
    print("Saved plots/temp_sweep.png")


if __name__ == "__main__":
    from learning_to_attribute import test_gradcheck
    test_gradcheck()
    main()
