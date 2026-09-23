"""Gradient check for the sigmoid top-k custom autograd function (pure torch, CPU)."""

import torch

from matryoshka_attribution.sigmoid_topk import SigmoidTopK


def test_gradcheck():
    scores = torch.randn(3, 10, dtype=torch.float64, requires_grad=True)
    k, T, n_iters = 4.0, 1.0, 100
    assert torch.autograd.gradcheck(
        lambda s: SigmoidTopK.apply(s, k, T, n_iters), (scores,),
        eps=1e-6, atol=1e-4, rtol=1e-3)


if __name__ == "__main__":
    test_gradcheck()
    print("Gradient check passed!")
