"""MAttr (and Expected Gradients) over the Tokens->Logits virtual weights of the note's 1L model.

    sbatch -J vw_ma scripts/vw/launch/vw.sbatch scripts/vw/vw_mattr.py --run results/vw/base \
        --method mattr --optimizer adam --adam-eps 1e-2 --steps 3000

THE OBJECT BEING SCORED is the same one vw_scores.py measures: the 5120 x 4096 matrix
W_TL = [W_E ; P] @ W_U, whose (i, j) entry is the note's Tokens->Logits virtual weight.
Ablation means setting an entry to zero, matching the note's helpfulness definition
exactly, so every method here answers the same question the oracle does.

    loss_fn(mask) = CE( (W_TL * mask)[tok] + (W_TL * mask)[4096 + pos] + rest , next token )

`rest` is the attention + MLP contribution to the logits. It does NOT depend on the mask --
masking a *product* [W_E;P] @ W_U changes only the direct path, never the residual stream
that feeds attention and the MLP -- so it is recomputed under no_grad each step and added
as a constant. That is what makes 21M mask units affordable: the graph the optimizer sees
is one elementwise multiply, two index_selects and a cross entropy.

WHY THIS IS THE INTERESTING COMPARISON. Fisher effectiveness and helpfulness are both
MARGINAL scores: they ablate one weight against the full model. Pruning is a JOINT
selection. The note explicitly leaves that gap open -- "Up to nonlinear effects in removing
multiple weights at once" -- and then concludes from helpfulness that "no saliency scheme
will perform much better" than Fisher. MAttr is not a saliency scheme; it optimises the
selection. So the note's conclusion is a testable prediction here, not an assumption.

ADAM EPS IS A REAL KNOB AT THIS WIDTH, not a numerical guard: with 21M mask logits nearly
every per-step gradient sits far below the default 1e-8, so Adam's update degenerates to
sign(g) and the learned score becomes a signed COUNT of steps with all effect magnitude
divided out. `--adam-eps` is swept, not fixed. (Same failure, at 2.3M logits, is recorded
in the parent project's neuron-scale runs.)

GRANULARITY. This is mask learning over WEIGHTS, not over representations, so it inherits
the unit-granularity axis of the sibling repo's `masks/layout.py` rather than MIB's
node/edge vocabulary. Three settings, all meaningful in this basis:

  weight  one score per virtual weight (5120 x 4096). The note's own unit -- it prunes
          individual virtual weights -- and the headline here.
  row     one score per SOURCE, i.e. per [vocabulary, position] entry (5120 units):
          "does this token's (or position's) direct path to the logits survive at all".
  col     one score per TARGET logit (4096 units): "does this logit receive a direct path".

row/col are ~4000x cheaper and are the granularities at which a human would actually read
the result, so they are worth having even though the faithful comparison is `weight`.
A row/col score is broadcast to the full (5120, 4096) shape on save, so every downstream
ranking script treats all three identically (ties inside a row are then broken arbitrarily,
which is the intended semantics: the mask keeps or drops whole rows).

PER-ITEM k (`--k-per-item`). By default `sample_k` draws ONE k per step and the whole batch
shares it, so a step supervises a single point on the density curve and the gradient carries
that draw's noise wholesale. With a k per BATCH ITEM, a batch of 8 supervises 8 densities per
step for exactly the same number of forwards -- the same variance reduction `k_avg` buys, at
no extra compute, because the k-averaging happens inside the batch rather than across repeat
steps.

This is the obvious suspect for the flat pruning curves: MAttr's ordering is good at the head
and uninformative below it, which is what optimising ~one k-draw per step against a
log-uniform schedule would produce.

It needs NO change to the frozen primitives. `sigmoid_topk` bisects on `dim=-1` and compares
`f_mid > k` elementwise, so scores of shape (B, N) with k of shape (B, 1) broadcast and give
per-item masks; verified against the scalar-k path (max |difference| 3e-8, and each item's
mask sums to its own k). What is new is only the outer loop, which is why this lives here
rather than as an edit to `learn_scores` -- per that repo's rule, a numerics change to an
existing variant is added as a new variant, not made in place. If it wins it should be
upstreamed as one.

SIGNS. MAttr needs no flip: scores rise for units whose inclusion lowers the loss, which is
the direction the oracle's `helpfulness > 0` also means. Expected Gradients carries the leading
minus for the same reason it does in the toy-model replication -- to first order
dL(w_ij) ~= -w_ij * dL/dw_ij.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import CTX, VOCAB, TinyLM  # noqa: E402
from vw_scores import D_VP, load_run, load_tokens  # noqa: E402

from matryoshka_attribution import learn_scores  # noqa: E402


class Batches:
    """Endless shuffled stream of training sequences, with `rest` precomputed per batch."""

    def __init__(self, model, toks, batch, device, seed=0):
        self.model, self.toks, self.batch, self.device = model, toks, batch, device
        self.g = torch.Generator().manual_seed(seed)
        self.order, self.i = torch.randperm(len(toks), generator=self.g), 0

    def next(self):
        if self.i + self.batch > len(self.order):
            self.order, self.i = torch.randperm(len(self.toks), generator=self.g), 0
        idx = self.order[self.i:self.i + self.batch]
        self.i += self.batch
        b = self.toks[idx].to(self.device)
        with torch.no_grad():
            _, a, m = self.model.paths(b)
        return b, a + m


SHAPE = {"weight": (D_VP, VOCAB), "row": (D_VP, 1), "col": (1, VOCAB)}


def masked_loss(W, mask, b, rest, pos_rows, gran="weight"):
    Wm = W * mask.view(*SHAPE[gran])
    logits = Wm[b] + Wm[pos_rows] + rest
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), b[:, 1:].reshape(-1))


def run_mattr_per_item(model, stream, args, device):
    """MAttr with an independent k per batch item; see the module docstring.

    A local training loop rather than `learn_scores`, because k there is a per-step scalar.
    The mask primitive, the k-schedule and the optimizer are all the shipped ones.
    """
    from matryoshka_attribution.masks import build_mask
    from matryoshka_attribution.schedules import sample_k

    W = model.W_TL().float()
    pos_rows = VOCAB + torch.arange(CTX, device=device)
    total = D_VP * VOCAB
    scores = torch.zeros(total, device=device, requires_grad=True)
    opt = (torch.optim.Adam([scores], lr=args.lr, eps=args.adam_eps)
           if args.optimizer == "adam" else torch.optim.SGD([scores], lr=args.lr))
    hist = []
    for step in range(args.steps):
        b, rest = stream.next()
        B = b.shape[0]
        ks = torch.tensor([[sample_k(total, args.k_schedule)] for _ in range(B)],
                          device=device, dtype=torch.float32)
        mask = build_mask(scores.expand(B, total), ks, args.variant,
                          T=args.T, n_iters=50).mask.view(B, D_VP, VOCAB)
        # gather the masked rows rather than materialising (B, 5120, 4096) * W
        bi = torch.arange(B, device=device)[:, None]
        direct = W[b] * mask[bi, b] + W[pos_rows] * mask[:, VOCAB:VOCAB + b.shape[1]]
        logits = direct + rest
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), b[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if args.log_every and step % args.log_every == 0:
            print(f"  {step:5d} loss {loss.item():.4f} k[{ks.min():.0f},{ks.max():.0f}]", flush=True)
    return scores.detach().view(D_VP, VOCAB), hist


def run_mattr(model, stream, args, device):
    W = model.W_TL().float()
    pos_rows = VOCAB + torch.arange(CTX, device=device)
    hist = []
    total = int(np.prod(SHAPE[args.granularity]))

    def loss_fn(mask):
        b, rest = stream.next()
        loss = masked_loss(W, mask, b, rest, pos_rows, args.granularity)
        hist.append(loss.item())
        return loss

    res = learn_scores(
        total, loss_fn, steps=args.steps, variant=args.variant,
        k_schedule=args.k_schedule, T=args.T, lr=args.lr, optimizer=args.optimizer,
        adam_eps=args.adam_eps, device=device, log_every=args.log_every)
    # broadcast row/col scores to the full family shape so every ranking script is identical
    return res.scores.detach().view(*SHAPE[args.granularity]).expand(D_VP, VOCAB).contiguous(), hist


def run_ig(model, stream, args, device):
    """Expected Gradients along the path alpha * W_TL, alpha ~ U(0,1) drawn PER EXAMPLE.

    Per-example rather than per-batch: the estimator's variance falls with the number of
    independent alphas, and one batch of B sequences carries B of them."""
    W = model.W_TL().float()
    pos_rows = VOCAB + torch.arange(CTX, device=device)
    acc = torch.zeros_like(W)
    g = torch.Generator(device=device).manual_seed(args.seed + 30_000)
    hist = []
    for step in range(args.steps):
        b, rest = stream.next()
        Wv = W.clone().requires_grad_(True)
        alpha = (torch.rand(b.shape[0], 1, 1, device=device, generator=g)
                 if args.method == "ig" else torch.ones(b.shape[0], 1, 1, device=device))
        logits = alpha * (Wv[b] + Wv[pos_rows]) + rest
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), b[:, 1:].reshape(-1))
        loss.backward()
        acc += Wv.grad
        hist.append(loss.item())
        if args.log_every and step % args.log_every == 0:
            print(f"  {step:5d} loss {loss.item():.4f}", flush=True)
    return -(acc / args.steps) * W, hist       # see the sign note in the module docstring


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--method", default="mattr", choices=["mattr", "ig", "ixg"])
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--granularity", default="weight", choices=["weight", "row", "col"])
    ap.add_argument("--k-per-item", action="store_true",
                    help="draw an independent k for every batch item (see module docstring)")
    ap.add_argument("--variant", default="topk")
    ap.add_argument("--k-schedule", default="log")
    ap.add_argument("--T", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model = load_run(args.run, device)
    stream = Batches(model, load_tokens("train"), args.batch, device, args.seed)
    g = "" if args.granularity == "weight" else f"_{args.granularity}"
    # k-schedule enters the tag only when it is not the default, so every tag written before
    # this arm existed keeps its name and nothing on disk is orphaned.
    g += "" if args.k_schedule == "log" else f"_k{args.k_schedule}"
    g += "_kitem" if args.k_per_item else ""
    tag = args.tag or (f"{args.method}_{args.optimizer}_lr{args.lr}_eps{args.adam_eps:g}{g}"
                       if args.method == "mattr" else f"{args.method}_s{args.steps}")
    print(f"[{tag}] {int(np.prod(SHAPE[args.granularity])):,} mask units | batch {args.batch} | {args.steps} steps", flush=True)

    t0 = time.time()
    fn = (run_mattr_per_item if (args.method == "mattr" and args.k_per_item)
          else run_mattr if args.method == "mattr" else run_ig)
    scores, hist = fn(model, stream, args, device)
    dt = time.time() - t0

    d = args.run / "attrib"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"scores": scores.cpu(), "args": vars(args), "loss_hist": hist,
                "seconds": dt}, d / f"{tag}.pt")
    s = scores.flatten().abs()
    # torch.quantile refuses tensors above ~16M elements and this family has 21M, so the
    # spread diagnostic is read off a fixed random subsample instead of the full tensor.
    sub = s[torch.randperm(s.numel(), generator=torch.Generator(device=s.device).manual_seed(0),
                           device=s.device)[:1_000_000]]
    print(json.dumps({"tag": tag, "seconds": round(dt, 1),
                      "loss_first50": float(np.mean(hist[:50])),
                      "loss_last50": float(np.mean(hist[-50:])),
                      "score_absmax": float(s.max()),
                      "score_p99_over_p50": float(sub.quantile(0.99) / sub.median()),
                      "frac_nonzero": float((s != 0).float().mean())}, indent=2))
    print(f"-> {d/(tag + '.pt')}")


if __name__ == "__main__":
    main()
