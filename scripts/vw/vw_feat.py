"""MAttr, gradient baselines and the pruning sweep for the Features->Logits family.

    uv run python scripts/vw/vw_feat.py attrib --run results/vw/base --method mattr --optimizer adam
    uv run python scripts/vw/vw_feat.py prune  --run results/vw/base

A separate entry point from vw_mattr.py / vw_prune.py rather than a `--family` flag on
those, deliberately: those two are in the middle of a multi-stage cluster run, and adding a
branch to a script that queued jobs will launch is how you break an experiment you cannot
watch. The masking contract is identical; only the forward differs.

THE FORWARD. In the VW model the MLP is replaced by the transcoder, and because the split is
exact the mask multiplies a term that is otherwise the model's own:

    mlp_out @ W_U = f @ (W_dec^T W_U) + b_dec @ W_U + e @ W_U ,   e = MLP_out - SLT_pred

so masking `W_FL = W_dec^T W_U` gives

    logits(mask) = direct + attention + f @ (W_FL * mask) + b_dec @ W_U + e @ W_U
                   `------------------ mask-independent -----------------'

with `mask = 1` reproducing the original transformer's logits exactly. The transcoder ERROR
TERM `e @ W_U` is carried through unscored, exactly as the note does: "This term is not
attributable to any single virtual weight, so we hold it separate from the scoring."
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
from vw_model import VOCAB  # noqa: E402
from vw_prune import DENSITIES  # noqa: E402
from vw_scores import load_run, load_tokens  # noqa: E402
from vw_scores_feat import load_transcoder  # noqa: E402
from vw_transcoder import activations  # noqa: E402

from learning_to_attribute import learn_scores  # noqa: E402


@torch.no_grad()
def batch_parts(model, tc, idx, toks, device):
    pre, mlp_out = activations(model, toks, idx, device)
    b = toks[idx].to(device)
    B, T = b.shape
    f = tc.features(pre)                                   # (B*T, d_feat)
    pred = f @ tc.W_dec.t() + tc.b_dec
    err = mlp_out - pred                                   # unscored, carried through
    direct, attn, _ = model.paths(b)
    rest = direct + attn + ((tc.b_dec + err) @ model.W_U).view(B, T, -1)
    return b, f.view(B, T, -1), rest


def masked_loss(W, mask, b, f, rest):
    # f/rest may be cached in bf16 (see the prune path); upcast per batch rather than
    # holding a second fp32 copy of a (seqs, 1024, 4096) tensor.
    logits = f.float() @ (W * mask.view(*W.shape)) + rest.float()
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), b[:, 1:].reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["attrib", "prune"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--transcoder", default="transcoder.pt")
    ap.add_argument("--method", default="mattr", choices=["mattr", "ig", "ixg"])
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--adam-eps", type=float, default=1e-2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--k-schedule", default="log")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seqs", type=int, default=256, help="held-out sequences for prune")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ablation", default="zero", choices=["zero", "mean"],
                    help="zero = the note's definition; mean = the bias-preserving control")
    ap.add_argument("--out", default="prune_fl.json")
    ap.add_argument("--sign", default="all", choices=["all", "pos", "neg"],
                    help="prune only weights of this sign (the note's per-sign appendix split)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict prune to these rankings")
    ap.add_argument("--densities", type=float, nargs="*", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    model = load_run(args.run, dev)
    tc, _ = load_transcoder(args.run, args.transcoder, dev)
    # DETACH: W_dec and W_U are nn.Parameters, so this product is a non-leaf tensor.
    # Without the detach, `W.clone().requires_grad_(True)` in the IG path is still
    # non-leaf and `.grad` comes back None, and the MAttr path would push gradient
    # into the transcoder and the transformer for nothing.
    W = (tc.W_dec.t() @ model.W_U).float().detach()
    N = W.numel()

    if args.cmd == "attrib":
        toks = load_tokens("train")
        g = torch.Generator().manual_seed(args.seed)
        order, ptr = torch.randperm(len(toks), generator=g), [0]

        def nxt():
            if ptr[0] + args.batch > len(order):
                ptr[0] = 0
            i = order[ptr[0]:ptr[0] + args.batch]
            ptr[0] += args.batch
            return batch_parts(model, tc, i, toks, dev)

        hist, t0 = [], time.time()
        if args.method == "mattr":
            def loss_fn(mask):
                b, f, rest = nxt()
                l = masked_loss(W, mask, b, f, rest)
                hist.append(l.item())
                return l
            res = learn_scores(N, loss_fn, steps=args.steps, variant="topk",
                               k_schedule=args.k_schedule, T=0.5, lr=args.lr,
                               optimizer=args.optimizer, adam_eps=args.adam_eps, device=dev)
            scores = res.scores.detach().view(*W.shape)
            tag = args.tag or f"mattr_{args.optimizer}_lr{args.lr}_eps{args.adam_eps:g}"
        else:
            acc = torch.zeros_like(W)
            gg = torch.Generator(device=dev).manual_seed(args.seed + 30_000)
            for _ in range(args.steps):
                b, f, rest = nxt()
                Wv = W.clone().requires_grad_(True)
                al = (torch.rand(b.shape[0], 1, 1, device=dev, generator=gg)
                      if args.method == "ig" else 1.0)
                logits = al * (f @ Wv) + rest
                l = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(),
                                    b[:, 1:].reshape(-1))
                l.backward()
                acc += Wv.grad
                hist.append(l.item())
            scores = -(acc / args.steps) * W          # see vw_mattr.py's sign note
            tag = args.tag or f"{args.method}_s{args.steps}"
        d = args.run / "attrib_fl"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"scores": scores.cpu(), "args": vars(args), "loss_hist": hist,
                    "seconds": time.time() - t0}, d / f"{tag}.pt")
        print(json.dumps({"tag": tag, "seconds": round(time.time() - t0, 1),
                          "loss_first50": float(np.mean(hist[:50])),
                          "loss_last50": float(np.mean(hist[-50:]))}, indent=2))
        print(f"-> {d/(tag + '.pt')}")
        return

    # ---- prune ---------------------------------------------------------------------------
    toks = load_tokens("val")[:args.seqs]
    # Cache the mask-independent parts once, in bf16: `f` and `rest` are each
    # (seqs, 1024, 4096), which is 4.3 GB at seqs=256 in bf16 and 17 GB in fp32.
    parts = []
    for i in range(0, len(toks), 8):
        b, f, rest = batch_parts(model, tc, torch.arange(i, min(i + 8, len(toks))), toks, dev)
        parts.append((b, f.bfloat16(), rest.bfloat16()))

    sc = torch.load(args.run / "scores" / "fl.pt", map_location=dev, weights_only=False)
    # MEAN ABLATION for a CONTINUOUS source: a dropped weight's expected contribution is
    # W_ij * E[s_i], and E[s_i] is recoverable from the stored (source, target) activation
    # sums -- SW[i, t] sums s over positions where feature i fired with target t, so
    # SW[i].sum() / n_positions is the feature's mean activation over the whole corpus.
    mean_s = (sc["pair"].to(dev).float().sum(1) / float(sc["n_positions"]))[:, None]
    W_all = sc["W"].to(dev).float()
    mean_of = ((lambda keep: ((1 - keep.float()).view(*W.shape) * W_all * mean_s).sum(0))
               if args.ablation == "mean" else (lambda keep: None))

    @torch.no_grad()
    def ev(mask):
        bias = mean_of(mask)
        tot = 0.0
        for b, f, rest in parts:
            r = rest if bias is None else rest.float() + bias
            tot += masked_loss(W, mask, b, f, r).item() * b.shape[0]
        return tot / len(toks)

    ranks = {"weight_abs": sc["W"].abs(), "weight": sc["W"], "era": sc["era"],
             "fisher": sc["fisher"], "helpfulness": sc["helpfulness"]}
    for f_ in sorted((args.run / "attrib_fl").glob("*.pt")):
        ranks[f_.stem] = torch.load(f_, map_location=dev, weights_only=False)["scores"]
    ranks["random_s0"] = torch.rand(*W.shape, generator=torch.Generator().manual_seed(1234))
    ranks = {k: v.to(dev).float().flatten() for k, v in ranks.items()}
    if args.only:
        ranks = {k: v for k, v in ranks.items() if k in args.only}
    dens_grid = list(args.densities) if args.densities else DENSITIES
    # prunable set: all weights, or only those of one sign (the rest are held on)
    Wf = W_all.flatten()
    sel = torch.ones(N, dtype=torch.bool, device=dev)
    if args.sign == "pos":
        sel &= Wf > 0
    elif args.sign == "neg":
        sel &= Wf < 0
    held = ~sel
    Nsel = int(sel.sum())

    full = ev(torch.ones(N, device=dev))
    empty = ev(held.float())
    print(f"[Features->Logits, {args.ablation} ablation, sign={args.sign}] full {full:.4f} | "
          f"{Nsel:,} prunable removed {empty:.4f} (dL {empty-full:+.4f}) | {len(toks)} held-out sequences",
          flush=True)
    res = {"full_loss": full, "empty_loss": empty, "n_weights": Nsel, "ablation": args.ablation,
           "rows": "all", "sign": args.sign, "seqs": len(toks), "split": "val",
           "ablation_mode": args.ablation,
           "densities": dens_grid, "curves": {}}
    for name, s in ranks.items():
        order = torch.argsort(s.masked_fill(held, -float("inf")), descending=True)
        row = []
        for dens in dens_grid:
            keep = held.clone()
            keep[order[:max(1, int(round(dens * Nsel)))]] = True
            row.append(ev(keep))
        res["curves"][name] = row
        pick = {d: row[dens_grid.index(d)] - full for d in (0.55, 0.3, 0.15, 0.05, 0.01) if d in dens_grid}
        print(f"  {name:<34} " + " ".join(f"d{d}={v:+.4f}" for d, v in pick.items()), flush=True)
        (args.run / args.out).write_text(json.dumps(res, indent=2))
    print(f"-> {args.run/args.out}")


if __name__ == "__main__":
    main()
