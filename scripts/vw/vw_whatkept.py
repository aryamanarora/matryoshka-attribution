"""What each ranking's kept set actually IS, at a fixed density -- and the bias control.

    uv run python scripts/vw/vw_whatkept.py --run results/vw/base --density 0.02

Motivated by the finding in the sibling toy-model replication (mask-learning-finetuning,
docs/interference_toy.md): there, MAttr+Adam beat both the trained model and the true
circuit by keeping ~1250 individually-worthless off-circuit weights, and the mechanism was
that their SUM supplied a per-row constant the frozen bias could no longer provide. The
weights were not computing anything; they were standing in for a scalar.

The Tokens->Logits family has an exact analogue, and it is a more interesting one because
the constant has a name. Every position's direct-path logit vector is a sum of two rows of
W_TL, so across the corpus the family's average contribution to logit j is

    b_j = (1/N) sum_i count(i) * W_ij ,     count(i) = #positions where source i is active

which is a fixed vector over the vocabulary -- a UNIGRAM PRIOR the direct path installs on
every token. A ranking that keeps weights mainly to preserve that prior is not localising
computation, and a sparsity curve that looks good for that reason is measuring model
capacity, not attribution.

So for each ranking at a given density we report:

  loss            the masked model's held-out loss
  loss_meanonly   the same kept set REPLACED by its mean effect: no direct path at all,
                  plus the constant vector b_j above restricted to the kept weights. If this
                  matches `loss`, the kept set is a bias and nothing else.
  composition     median |W| kept vs the population (the toy's "many small" signature),
                  fraction positive, fraction of keeps that are POSITION rows (which are
                  active once per sequence and so are the purest bias carriers)
  rho_unigram     Spearman of b_j against log unigram frequency of token j
  loss_refit      the kept set with a FREE per-target bias fitted on top (4,096 parameters,
                  everything else frozen). This is the decisive diagnostic from the toy-model
                  replication, transplanted: there, a free bias was worth 0.478 nats to the
                  circuit-only model and 0.003 to MAttr+Adam's kept set, which showed that
                  what Adam's extra weights were FOR was exactly what a bias would supply.
                  A ranking whose `loss_refit` is far below its `loss` was spending its
                  budget on something a bias could have done; one where the two coincide has
                  already bought that, in weights.

with three reference rows: the full model, no direct path at all, and no direct path plus
the FULL family's mean effect (the best any pure-bias story can do).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import CTX, VOCAB  # noqa: E402
from vw_prune import rankings  # noqa: E402
from vw_scores import D_VP, load_run, load_tokens  # noqa: E402


@torch.no_grad()
def eval_direct(model, toks, W, bias, batch, device):
    """Held-out loss with direct path `W` (may be all-zero) plus a constant logit `bias`."""
    pos = VOCAB + torch.arange(CTX, device=device)
    tot, n = 0.0, 0
    for i in range(0, len(toks), batch):
        b = toks[i:i + batch].to(device)
        _, a, m = model.paths(b)
        logits = a + m + bias
        if W is not None:
            logits = logits + W[b] + W[pos]
        tot += F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(),
                               b[:, 1:].reshape(-1)).item() * b.shape[0]
        n += b.shape[0]
    return tot / n


def refit_bias(model, tr, va, Wk, steps, batch, device, lr=0.3, seed=0):
    """Fit a free per-target logit bias on top of a fixed kept set; return held-out loss.

    4,096 parameters, everything else frozen, Adam on the TRAINING split and reported on the
    held-out one -- so this is what a bias is worth, not what overfitting one is worth."""
    pos = VOCAB + torch.arange(CTX, device=device)
    b = torch.zeros(VOCAB, device=device, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(0, len(tr), (batch,), generator=g)
        x = tr[idx].to(device)
        with torch.no_grad():
            _, a, m = model.paths(x)
            base = a + m + (Wk[x] + Wk[pos] if Wk is not None else 0)
        loss = F.cross_entropy((base + b)[:, :-1].reshape(-1, VOCAB).float(),
                               x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return eval_direct(model, va, Wk, b.detach(), batch, device)


def spearman(a, b):
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    ra, rb = (ra - ra.mean()) / ra.std(), (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--density", type=float, nargs="*", default=[0.02])
    ap.add_argument("--seqs", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="whatkept.json")
    ap.add_argument("--refit-steps", type=int, default=300,
                    help="Adam steps for the free-bias refit; 0 disables it")
    ap.add_argument("--refit-density", type=float, nargs="*", default=[0.02],
                    help="densities at which to pay for the refit (it is the expensive part)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_run(args.run, dev)
    toks = load_tokens("val")[:args.seqs]
    sc = torch.load(args.run / "scores" / "tl.pt", map_location=dev, weights_only=False)
    W = sc["W"].to(dev).float()
    cnt = sc["src_count"].to(dev).float()
    n_pos = float(sc["n_positions"])
    ranks = rankings(args.run, dev)
    N = D_VP * VOCAB
    zero = torch.zeros(VOCAB, device=dev)

    # unigram log-frequency of each target token over the corpus the scores were taken on
    uni = torch.zeros(VOCAB, device=dev)
    tr = load_tokens("train")
    uni.index_add_(0, tr[:, 1:].reshape(-1).to(dev),
                   torch.ones(tr[:, 1:].numel(), device=dev))
    log_uni = torch.log(uni + 1)

    full = eval_direct(model, toks, W, zero, args.batch, dev)
    none = eval_direct(model, toks, None, zero, args.batch, dev)
    b_all = (cnt[:, None] * W).sum(0) / n_pos
    mean_all = eval_direct(model, toks, None, b_all, args.batch, dev)
    tr = load_tokens("train")
    print(f"full {full:.4f} | no direct path {none:.4f} | "
          f"no direct path + its MEAN effect {mean_all:.4f}\n"
          f"  (the mean effect alone recovers {100*(none-mean_all)/(none-full):.1f}% of the "
          f"direct path's {none-full:.4f} nats)", flush=True)

    res = {"full": full, "none": none, "mean_all": mean_all,
           "rho_unigram_all": spearman(b_all, log_uni), "rows": {}}
    if args.refit_steps:
        # the two reference rows for the bias diagnostic
        res["none_refit"] = refit_bias(model, tr, toks, None, args.refit_steps, args.batch, dev)
        res["full_refit"] = refit_bias(model, tr, toks, W, args.refit_steps, args.batch, dev)
        print(f"  a free per-target bias is worth: {none - res['none_refit']:+.4f} nats to the "
              f"no-direct-path model, {full - res['full_refit']:+.4f} to the full model",
              flush=True)
    med_pop = float(W.abs().median())
    for dens in args.density:
        k = max(1, int(round(dens * N)))
        for name, s in ranks.items():
            keep = torch.zeros(N, dtype=torch.bool, device=dev)
            keep[s.argsort(descending=True)[:k]] = True
            keep = keep.view(D_VP, VOCAB)
            Wk = W * keep
            bk = (cnt[:, None] * Wk).sum(0) / n_pos
            r = {
                "density": dens,
                "loss": eval_direct(model, toks, Wk, zero, args.batch, dev),
                "loss_meanonly": eval_direct(model, toks, None, bk, args.batch, dev),
                "median_absW_kept_over_pop": float(W.abs()[keep].median()) / med_pop,
                "frac_positive": float((W[keep] > 0).float().mean()),
                "frac_position_rows": float(keep[VOCAB:].sum() / keep.sum()),
                "frac_position_rows_base": CTX / D_VP,
                "rho_unigram": spearman(bk, log_uni),
            }
            if args.refit_steps and dens in args.refit_density:
                r["loss_refit"] = refit_bias(model, tr, toks, Wk, args.refit_steps,
                                             args.batch, dev)
                r["bias_worth"] = r["loss"] - r["loss_refit"]
            res["rows"][f"{name}@{dens}"] = r
            print(f"  d={dens:<7} {name:<32} loss {r['loss']:.4f}  mean-only "
                  f"{r['loss_meanonly']:.4f}  |W|med/pop {r['median_absW_kept_over_pop']:.2f}  "
                  f"pos-rows {r['frac_position_rows']:.3f}  rho(uni) {r['rho_unigram']:+.2f}"
                  + (f"  bias worth {r['bias_worth']:+.4f}" if "bias_worth" in r else ""),
                  flush=True)
            (args.run / args.out).write_text(json.dumps(res, indent=2))
    print(f"-> {args.run/args.out}")


if __name__ == "__main__":
    main()
