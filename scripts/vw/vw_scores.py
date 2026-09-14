"""Fisher effectiveness and EXACT helpfulness for every Tokens->Logits virtual weight.

    sbatch -J vw_sc scripts/vw/launch/vw.sbatch scripts/vw/vw_scores.py --run results/vw/base

Both quantities are the note's (Turner, Wu & Batson 2026), specialised to the one weight
family whose source activation `s` is a one-hot indicator, so both closed forms are exact
and both are computable for ALL 5120 x 4096 = 20,971,520 weights in a single streaming
pass. The note only samples helpfulness (7,765 weights over 1B tokens) because it must
cover six families; restricted to this one it says so itself -- "Helpfulness can be
computed cheaply for weight families that target the logits".

FISHER EFFECTIVENESS (note, Appendix > Paths to logits):

    fisher(w) = 1/2 E[a^T F a] = 1/2 w^2 E[ s^2 p_j (1 - p_j) ]

  With s in {0,1} this is  1/2 W_ij^2 * C_ij,  C_ij = (1/N) sum_{q : source i active} p_j(1-p_j)(q).

HELPFULNESS (note, Appendix > Helpfulness computations for weight families targeting the
logits) -- the average change in loss when the weight is ablated:

    helpfulness(w) = E[ log(1 - p_j (1 - e^{-sw})) + s w 1_{j = t} ]

  Positive = ablation raises the loss = the weight was helping. Positions where the source
  is inactive contribute EXACTLY zero (s = 0 kills both terms), which is why the sum runs
  over active positions only and why the estimator has no variance contribution from them.

WHAT WE GET THAT THE NOTE DOES NOT: because our corpus is 97.7M tokens and we sweep all of
it, these are not Monte-Carlo estimates of the training-distribution expectation -- they
are that expectation, exactly, over the empirical training set the model saw. The standard
errors we also accumulate are therefore about generalisation to new text, not about
sampling error in the score. `--positions` caps the pass for a quick run.

Also written, as cheap baselines that need no extra pass:
  `weight`   the raw virtual weight W_ij (the note's naive baseline; sorted SIGNED, and
             `abs` for the magnitude version the pruning figure uses)
  `era`      |W_ij| * P(source i active) -- the coactivation-weighted magnitude that plays
             the role of ERA here. With a one-hot source and a logit target there is no
             target-gating term, so ERA collapses to this and TWERA has no analogue.

Outputs results/<run>/scores/tl.pt with fp32 (5120, 4096) tensors:
  W, fisher, helpfulness, helpfulness_se, count (positions per source), pair_count.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import CTX, VOCAB, TinyLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "vw"
D_VP = VOCAB + CTX          # 5120 = the note's d_{v'}


def load_run(run, device):
    ck = torch.load(run / "model.pt", map_location=device, weights_only=False)
    m = TinyLM(seed=ck["args"].get("seed", 0)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def load_tokens(tag):
    a = np.fromfile(DATA / f"{tag}.bin", dtype=np.uint16)
    return torch.from_numpy(a.reshape(-1, CTX).astype(np.int64))


@torch.no_grad()
def accumulate(model, toks, batch, device, positions=None, log_every=50, se=True,
               time_budget=0.0):
    """One streaming pass over the split; every accumulator is a (5120, 4096) fp32 sum.

    Per position q with active source row i, target token t and probabilities p, the
    per-position ablation delta for weight (i, j) is

        dl_q(i, j) = log(1 - p_j (1 - e^{-W_ij}))  +  W_ij 1_{j = t}
                     `-------- d_q(j) ---------'      `--- only at j = t ---'

    so the mean needs sum_q d_q and the co-occurrence count of (i, t), and the second
    moment needs sum_q d_q^2, sum_q d_q(t) and that same count -- five accumulators, all
    obtained without ever materialising a per-weight, per-position tensor."""
    V, T = model.vocab, toks.shape[1]
    d_vp = model.vocab + model.ctx
    W = model.W_TL().float()                       # (d_v', d_v) virtual weights
    Em = torch.exp(-W)                             # e^{-w}, the ablation factor
    z = lambda: torch.zeros(d_vp, V, device=device, dtype=torch.float32)
    C, D, D2 = z(), z(), z()                       # Fisher stat, sum d, sum d^2
    Dt, pair = z(), z()                            # sum d at j=t, and #(source, target)
    src_count = torch.zeros(d_vp, device=device, dtype=torch.float32)
    pos_rows = model.vocab + torch.arange(T - 1, device=device)
    n, t0 = 0, time.time()

    for bi in range(0, len(toks), batch):
        b = toks[bi:bi + batch].to(device)
        B = b.shape[0]
        p = model(b)[:, :-1].float().softmax(-1)   # (B, T-1, V): position q predicts q+1
        tgt = b[:, 1:]                             # (B, T-1)
        src = b[:, :-1].reshape(-1)                # token source row at each position
        flat_t = tgt.reshape(-1)
        pf = p.reshape(-1, V)

        C.index_add_(0, src, (p * (1 - p)).reshape(-1, V))    # token source
        C[pos_rows] += (p * (1 - p)).sum(0)                   # position source

        # ---- token source: real scatters over B*(T-1) rows ---------------------------
        d = torch.log1p(-pf * (1 - Em[src]))
        D.index_add_(0, src, d)
        if se:      # the second moment is only needed for the 95% CI; ~40% of the pass
            D2.index_add_(0, src, d * d)
            Dt.view(-1).index_add_(0, src * V + flat_t, d.gather(1, flat_t[:, None]).squeeze(1))
        del d
        # ---- position source: the row is determined by q, so this is a reduction ------
        dp = torch.log1p(-p * (1 - Em[pos_rows]))
        D[pos_rows] += dp.sum(0)
        flat_pos = (pos_rows[None, :] * V + tgt).reshape(-1)
        if se:
            D2[pos_rows] += (dp * dp).sum(0)
            Dt.view(-1).index_add_(0, flat_pos, dp.gather(2, tgt[..., None]).reshape(-1))
        del dp

        ones = torch.ones_like(flat_t, dtype=torch.float32)
        pair.view(-1).index_add_(0, src * V + flat_t, ones)
        pair.view(-1).index_add_(0, flat_pos, ones)
        src_count.index_add_(0, src, ones)
        src_count[pos_rows] += B
        n += B * (T - 1)
        if (bi // batch) % log_every == 0:
            print(f"  {n/1e6:7.2f}M positions | {time.time()-t0:6.0f}s", flush=True)
        if positions and n >= positions:
            break
        if time_budget and time.time() - t0 > time_budget:
            print(f"  stopping at the {time_budget:.0f}s budget after {n/1e6:.1f}M positions "
                  f"of {toks.numel()/1e6:.1f}M", flush=True)
            break

    total = D + W * pair                                   # sum_q dl_q, exactly
    sq = D2 + 2 * W * Dt + W.pow(2) * pair                 # sum_q dl_q^2, exactly
    mean = total / n
    var = torch.clamp(sq / n - mean.pow(2), min=0)         # over all n positions
    return dict(W=W, C=C / n, H=mean, H_se=(var / n).sqrt() if se else torch.zeros_like(mean),
                pair=pair, src_count=src_count, n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--positions", type=int, default=0, help="0 = the whole split")
    # A wall-clock cap, so the pass degrades to a (still unbiased) subsample of the corpus
    # rather than dying at the slurm limit. Which it did is recorded in `n_positions`.
    ap.add_argument("--time-budget", type=float, default=0.0, help="seconds; 0 = no cap")
    ap.add_argument("--tag", default="tl")
    ap.add_argument("--no-se", action="store_true",
                    help="skip the second-moment accumulators (no 95%% CIs, ~40%% faster)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_run(args.run, device)
    toks = load_tokens(args.split)
    print(f"scoring {D_VP}x{VOCAB} = {D_VP*VOCAB:,} Tokens->Logits weights over "
          f"{args.split} ({toks.numel():,} tokens)", flush=True)

    a = accumulate(model, toks, args.batch, device, args.positions or None,
                   se=not args.no_se, time_budget=args.time_budget)
    W, C = a["W"], a["C"]
    out = {
        "W": W.cpu(),
        "fisher": (0.5 * W.pow(2) * C).cpu(),
        "helpfulness": a["H"].cpu(),
        "helpfulness_se": a["H_se"].cpu(),
        "era": (W.abs() * (a["src_count"] / a["n"])[:, None]).cpu(),
        "src_count": a["src_count"].cpu(),
        "pair": a["pair"].cpu(),
        "n_positions": a["n"],
    }
    d = args.run / "scores"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(out, d / f"{args.tag}.pt")
    h = out["helpfulness"]
    print(json.dumps({
        "n_positions": a["n"],
        "weights": int(W.numel()),
        "helpful_frac": float((h > 0).float().mean()),
        "dead_frac": float((out["src_count"] == 0).float().mean()),
        "sum_pos_helpfulness": float(h[h > 0].sum()),
        "sum_neg_helpfulness": float(-h[h < 0].sum()),
        "fisher_max": float(out["fisher"].max()),
        "fisher_median": float(out["fisher"].median()),
    }, indent=2))
    print(f"-> {d/(args.tag + '.pt')}")


if __name__ == "__main__":
    main()
