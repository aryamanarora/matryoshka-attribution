"""Fisher effectiveness and exact helpfulness for the Features->Logits virtual weights.

    sbatch -J vw_fs scripts/vw/launch/vw.sbatch scripts/vw/vw_scores_feat.py --run results/vw/base

The note's second logit-targeting family: W_FL = W_dec^T @ W_U, 4,096 features x 4,096
logits = 16,777,216 virtual weights. Same two closed forms as vw_scores.py, and both are
still exact --

    fisher(w)      = 1/2 w^2 E[ s^2 p_j (1 - p_j) ]
    helpfulness(w) = E[ log(1 - p_j (1 - e^{-sw})) + s w 1_{j=t} ]

-- but `s` is now a **continuous** feature activation rather than a one-hot indicator, and
that changes what can be precomputed. Three consequences, and they are the whole reason this
family is a different experiment rather than more of the same:

1. `e^{-sw}` no longer factors. In Tokens->Logits, `s in {0,1}` let us precompute `exp(-W)`
   once and gather rows; here the exponent depends on the position's activation, so the
   (n_active, 4096) block has to be built per chunk.
2. `s w 1_{j=t}` is no longer `w` times a co-occurrence COUNT. It is `w` times the summed
   ACTIVATION `SW[i,t] = sum_q s_q` over positions where feature i is on and the target is t.
3. Cost is set by activation sparsity, not by the weight count -- we only touch (position,
   feature) pairs where the feature fires. That is the same structure the note describes for
   its counterfactual scores: "their cost scales with activation sparsity rather than with
   the full weight-matrix size".

`p` is the REAL model's output distribution (true MLP, not the transcoder's prediction), per
the note: `p` is "the model's output probabilities". The transcoder enters only through which
features exist and what they write to the logits.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import VOCAB  # noqa: E402
from vw_scores import load_run, load_tokens  # noqa: E402
from vw_transcoder import Transcoder, activations  # noqa: E402


def load_transcoder(run, name, device):
    d = torch.load(run / name, map_location=device, weights_only=False)
    tc = Transcoder(bandwidth=d["bandwidth"]).to(device)
    with torch.no_grad():                      # the saved weights consume RAW activations
        tc.W_enc.copy_(d["W_enc"]); tc.b_enc.copy_(d["b_enc"])
        tc.W_dec.copy_(d["W_dec"]); tc.b_dec.copy_(d["b_dec"])
        tc.log_theta.copy_(d["log_theta"])
    return tc.eval(), d


@torch.no_grad()
def accumulate(model, tc, toks, batch, device, positions=None, chunk=8192,
               time_budget=0.0, log_every=20):
    V = VOCAB
    W = (tc.W_dec.t() @ model.W_U).float()     # (d_feat, vocab) virtual weights
    F_ = W.shape[0]
    z = lambda: torch.zeros(F_, V, device=device, dtype=torch.float32)
    C, D, D2, Dts, SW, S2 = (z() for _ in range(6))
    fire = torch.zeros(F_, device=device, dtype=torch.float32)
    n, t0 = 0, time.time()

    for bi in range(0, len(toks), batch):
        idx = torch.arange(bi, min(bi + batch, len(toks)))
        pre, _ = activations(model, toks, idx, device)          # (B*T, d_m)
        b = toks[idx].to(device)
        B, T = b.shape
        p = model(b)[:, :-1].float().softmax(-1).reshape(-1, V)  # (B*(T-1), V)
        tgt = b[:, 1:].reshape(-1)
        feats = tc.features(pre).view(B, T, -1)[:, :-1].reshape(-1, F_)   # drop last position
        qv = p * (1 - p)

        pos_i, feat_i = feats.nonzero(as_tuple=True)
        s_all = feats[pos_i, feat_i]
        for c in range(0, pos_i.numel(), chunk):
            pi, fi = pos_i[c:c + chunk], feat_i[c:c + chunk]
            s = s_all[c:c + chunk][:, None]
            pc, tc_ = p[pi], tgt[pi]
            C.index_add_(0, fi, s.pow(2) * qv[pi])
            d = torch.log1p(-pc * (1 - torch.exp(-s * W[fi])))
            D.index_add_(0, fi, d)
            D2.index_add_(0, fi, d * d)
            flat = fi * V + tc_
            dt = d.gather(1, tc_[:, None]).squeeze(1)
            Dts.view(-1).index_add_(0, flat, dt * s.squeeze(1))
            SW.view(-1).index_add_(0, flat, s.squeeze(1))
            S2.view(-1).index_add_(0, flat, s.squeeze(1).pow(2))
            del d
        fire.index_add_(0, feat_i, torch.ones_like(feat_i, dtype=torch.float32))
        n += B * (T - 1)
        if (bi // batch) % log_every == 0:
            print(f"  {n/1e6:7.2f}M positions | L0 {pos_i.numel()/(B*(T-1)):6.1f} "
                  f"| {time.time()-t0:6.0f}s", flush=True)
        if positions and n >= positions:
            break
        if time_budget and time.time() - t0 > time_budget:
            print(f"  stopping at the {time_budget:.0f}s budget after {n/1e6:.1f}M positions",
                  flush=True)
            break

    total = D + W * SW                                  # sum_q dl_q, exactly
    sq = D2 + 2 * W * Dts + W.pow(2) * S2               # sum_q dl_q^2, exactly
    mean = total / n
    var = torch.clamp(sq / n - mean.pow(2), min=0)
    return dict(W=W, C=C / n, H=mean, H_se=(var / n).sqrt(), fire=fire, SW=SW, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--transcoder", default="transcoder.pt")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--positions", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=2400.0)
    ap.add_argument("--tag", default="fl")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_run(args.run, dev)
    tc, meta = load_transcoder(args.run, args.transcoder, dev)
    toks = load_tokens(args.split)
    print(f"scoring {tc.W_dec.shape[1]}x{VOCAB} = {tc.W_dec.shape[1]*VOCAB:,} "
          f"Features->Logits weights | transcoder FVU {meta['log'][-1]['fvu']:.4f} "
          f"L0 {meta['log'][-1]['l0']:.1f}", flush=True)

    a = accumulate(model, tc, toks, args.batch, dev, args.positions or None,
                   args.chunk, args.time_budget)
    W, h = a["W"], a["H"]
    out = {"W": W.cpu(), "fisher": (0.5 * W.pow(2) * a["C"]).cpu(),
           "helpfulness": h.cpu(), "helpfulness_se": a["H_se"].cpu(),
           "era": (W.abs() * (a["fire"] / a["n"])[:, None]).cpu(),
           "src_count": a["fire"].cpu(), "pair": a["SW"].cpu(), "n_positions": a["n"]}
    d = args.run / "scores"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / f"{args.tag}.pt")
    print(json.dumps({"n_positions": a["n"], "weights": int(W.numel()),
                      "helpful_frac": float((h > 0).float().mean()),
                      "dead_frac": float((a["fire"] == 0).float().mean()),
                      "sum_pos_helpfulness": float(h[h > 0].sum()),
                      "sum_neg_helpfulness": float(-h[h < 0].sum())}, indent=2))
    print(f"-> {d/(args.tag + '.pt')}")


if __name__ == "__main__":
    main()
