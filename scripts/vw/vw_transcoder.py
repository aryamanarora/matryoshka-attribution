"""The note's single-layer transcoder, for the Features->Logits weight family.

    sbatch -J vw_tc --time=01:00:00 scripts/vw/launch/vw.sbatch scripts/vw/vw_transcoder.py --run results/vw/base

Spec quoted from Turner, Wu & Batson (2026), Appendix > Training details > Transcoder, and
followed exactly except where marked:

  "A single-layer transcoder (SLT) was trained on the transformer's activations: it reads the
   pre-MLP residual-stream activation and is trained to predict the MLP output. It has 4,096
   features over the 256-dimensional input, with a JumpReLU activation (bandwidth 2,
   threshold initialized at 0.1); encoder biases were calibrated from 10^4 training
   activations so that each feature's pre-activation is positive on roughly half of the
   inputs at initialization.
   The loss is a reconstruction term (squared error summed over output dimensions, averaged
   over the batch) plus a sparsity penalty (the L1 norm of feature activations weighted by
   the corresponding decoder-column norms). The sparsity coefficient ramped linearly from 0
   to 2.0 over the course of training. Training used Adam with beta = (0.9, 0.999) and no
   weight decay. The learning rate was 2x10^-5 at the start of training and then linearly
   decayed to zero over the final 20%. The batch size was 16,384 activation vectors and
   training proceeded for 100,000 steps (~1.64x10^9 activation samples in total)...
   Gradients were clipped to a global L2 norm of 1.0.
   Activations were centered and normalized for training per input dimension using statistics
   estimated from 10^8 samples; after training these constants were folded into the
   transcoder weights, so the released transcoder consumes raw activations."

WHY THIS IS AFFORDABLE. 100,000 steps x 16,384 activations is 1.64e9 activation samples, but
the transcoder is two 256x4096 matrices, so that is ~2e16 FLOPs -- about a minute of an H100's
peak. The real cost is *generating* the activations: 1.64e9 token positions is ~17 epochs of
the 97.7M-token training corpus through a 2.9M-parameter transformer. Activations are produced
on the fly (caching them would be 840 GB) and the whole run is well under an hour.

THE ONE THING NOT STATED is the JumpReLU's gradient estimator. "bandwidth 2" is the width of
the rectangular kernel in the straight-through estimators of Rajamanoharan et al., which is
where the JumpReLU-SAE formulation and that hyperparameter name both come from, so that is
what is implemented: value `z*H(z-theta)`, `d/dz -> H(z-theta)`, and
`d/dtheta -> -(theta/bandwidth) * rect((z-theta)/bandwidth)`.

WHAT THIS BUYS. The Features->Logits family, `W_dec^T @ W_U`, is 4,096 x 4,096 = 16,777,216
virtual weights, and it is the note's second logit-targeting family -- so its helpfulness also
has an exact closed form. Crucially its source activation is CONTINUOUS (a feature
activation) where Tokens->Logits is a one-hot indicator, which is a genuinely different
regime for both metrics: Fisher's `E[s^2 p(1-p)]` and helpfulness's `e^{-sw}` both stop
factoring out of the data. Whether the two-regime result of the main experiment survives that
is the question this family answers.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_scores import load_run, load_tokens  # noqa: E402


class JumpReLU(torch.autograd.Function):
    """z * H(z - theta), with the rectangle-kernel STEs; `bandwidth` is the kernel width."""

    @staticmethod
    def forward(ctx, z, log_theta, bandwidth):
        theta = log_theta.exp()
        ctx.save_for_backward(z, theta)
        ctx.bandwidth = bandwidth
        return z * (z > theta).to(z.dtype)

    @staticmethod
    def backward(ctx, g):
        z, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        gate = (z > theta).to(z.dtype)
        rect = ((z - theta).abs() < bw / 2).to(z.dtype) / bw
        # d/dtheta of the *value*; the chain rule to log_theta multiplies by theta again
        return g * gate, (g * (-theta) * rect * theta).sum(0), None


class Transcoder(nn.Module):
    def __init__(self, d_in=256, d_feat=4096, thresh=0.1, bandwidth=2.0, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d_feat, d_in, generator=g) / math.sqrt(d_in)
        self.W_enc = nn.Parameter(w.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_feat))
        self.W_dec = nn.Parameter(w.t().contiguous().clone())     # (d_in, d_feat)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_theta = nn.Parameter(torch.full((d_feat,), math.log(thresh)))
        self.bandwidth = bandwidth

    def features(self, x):
        return JumpReLU.apply(x @ self.W_enc.t() + self.b_enc, self.log_theta, self.bandwidth)

    def forward(self, x):
        f = self.features(x)
        return f @ self.W_dec.t() + self.b_dec, f


@torch.no_grad()
def activations(model, toks, idx, device):
    """(pre-MLP residual stream, MLP output) for a batch of sequences, flattened."""
    b = toks[idx].to(device)
    x = model.W_E[b] + model.P[:b.shape[1]]
    q = torch.einsum("btm,hmd->bhtd", x, model.W_Q)
    k = torch.einsum("btm,hmd->bhtd", x, model.W_K)
    v = torch.einsum("btm,hmd->bhtd", x, model.W_V)
    z = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                       scale=1.0 / math.sqrt(model.d_head))
    pre = x + torch.einsum("bhtd,hdm->btm", z, model.W_O)
    out = F.relu(pre @ model.W_in) @ model.W_out
    return pre.reshape(-1, pre.shape[-1]), out.reshape(-1, out.shape[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=16_384, help="activation vectors per step")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--l1", type=float, default=2.0, help="final sparsity coefficient")
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--stat-samples", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="transcoder.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_run(args.run, dev)
    toks = load_tokens("train")
    seqs = max(1, args.batch // toks.shape[1])          # sequences per step
    g = torch.Generator().manual_seed(args.seed)

    # --- per-dimension centering / scaling statistics -------------------------------------
    n, mean, m2 = 0, torch.zeros(256, device=dev), torch.zeros(256, device=dev)
    while n < args.stat_samples:
        pre, _ = activations(model, toks, torch.randint(0, len(toks), (64,), generator=g), dev)
        c = pre.shape[0]
        d = pre.mean(0) - mean
        mean += d * c / (n + c)
        m2 += pre.var(0, unbiased=False) * c + d.pow(2) * n * c / (n + c)
        n += c
    std = (m2 / n).sqrt().clamp_min(1e-6)
    print(f"activation stats over {n:,} samples | mean |{mean.abs().mean():.4f}| "
          f"std {std.mean():.4f}", flush=True)

    tc = Transcoder(seed=args.seed).to(dev)
    # --- encoder bias calibrated so each feature is on for ~half the inputs at init --------
    pre, _ = activations(model, toks, torch.randint(0, len(toks), (16,), generator=g), dev)
    z = ((pre[:10_000] - mean) / std) @ tc.W_enc.t()
    with torch.no_grad():
        tc.b_enc.copy_(tc.log_theta.exp() - z.median(0).values)
    print(f"b_enc calibrated: {(z + tc.b_enc > tc.log_theta.exp()).float().mean():.3f} of "
          f"features active at init (target ~0.5)", flush=True)

    opt = torch.optim.Adam(tc.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    t0, log = time.time(), []
    for step in range(args.steps):
        f_ = min(1.0, (args.steps - step) / max(1e-9, args.decay_frac * args.steps))
        for gp in opt.param_groups:
            gp["lr"] = args.lr * f_
        lam = args.l1 * (step + 1) / args.steps            # linear 0 -> 2.0
        pre, out = activations(model, toks, torch.randint(0, len(toks), (seqs,), generator=g), dev)
        xin = (pre - mean) / std
        yhat, feats = tc(xin)
        recon = (yhat - out).pow(2).sum(-1).mean()
        l1 = (feats.abs() * tc.W_dec.norm(dim=0)).sum(-1).mean()
        loss = recon + lam * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tc.parameters(), 1.0)
        opt.step()
        if step % 2000 == 0 or step == args.steps - 1:
            l0 = float((feats > 0).float().sum(-1).mean())
            fvu = float((recon / out.pow(2).sum(-1).mean()).detach())
            log.append({"step": step, "recon": float(recon), "l0": l0, "fvu": fvu, "lam": lam})
            print(f"  {step:6d} recon {recon:9.4f} FVU {fvu:.4f} L0 {l0:7.1f} "
                  f"lam {lam:.3f} | {time.time()-t0:.0f}s", flush=True)

    # fold the normalization into the weights: the saved transcoder consumes RAW activations
    with torch.no_grad():
        W_enc_raw = tc.W_enc / std
        b_enc_raw = tc.b_enc - (tc.W_enc * (mean / std)).sum(-1)
    torch.save({"W_enc": W_enc_raw.cpu(), "b_enc": b_enc_raw.cpu(),
                "W_dec": tc.W_dec.detach().cpu(), "b_dec": tc.b_dec.detach().cpu(),
                "log_theta": tc.log_theta.detach().cpu(), "bandwidth": tc.bandwidth,
                "mean": mean.cpu(), "std": std.cpu(), "args": vars(args), "log": log},
               args.run / args.out)
    print(json.dumps(log[-1], indent=2))
    print(f"-> {args.run/args.out}")


if __name__ == "__main__":
    main()
