"""Loss-vs-density for every ranking of the Tokens->Logits virtual weights.

    sbatch -J vw_pr scripts/vw/launch/vw.sbatch scripts/vw/vw_prune.py --run results/vw/base

This is the note's headline pruning figure ("Fisher effectiveness can cheaply filter tens
of percent"), restricted to the Tokens->Logits family and with our methods added:

  "we remove weights from the VW model in order from least to most effective (moving from
   right to left above) and measure the pruned model's loss on a held-out test set."

x is DENSITY (fraction of the family's 20,971,520 weights kept), y is
dL = L(pruned) - L(full), on the held-out shard (common_corpus_10), which the model never
saw and which none of the scores were estimated on.

RANKINGS. All are "keep the top-scoring fraction", so a ranking only has to order weights;
the sign convention is fixed by what each score means and is NOT free:

  weight_abs   |W_ij|              the note's own baseline on this figure
  weight       W_ij (signed)       the toy-model note's ranking; kept as a control
  era          |W_ij| P(source i)  coactivation-weighted magnitude
  fisher       the note's metric    (non-negative by construction)
  helpfulness  the ORACLE, signed   -- available for every weight in this family, which is
                                    what makes this a strong benchmark rather than a proxy
  mattr/ig     whatever vw_mattr.py wrote into <run>/attrib/
  random       the control, one draw per seed

WHICH ROWS ARE PRUNABLE (`--rows`). The family's 1,024 POSITION source rows carry ~98% of
its positive helpfulness mass, because position enters through a fixed sinusoidal table
whose rows have norm ~11.3 against token embeddings trained up from 0.0375. An aggregate
over all 5,120 rows is therefore mostly a statement about the position embedding.
`--rows token` holds every position row permanently, prunes only the 16,777,216
token->logit weights, and measures density over those -- the linguistically meaningful part
of the family, and the part where a ranking has to know something about language rather
than about a sinusoid. `--rows all` (default) is the note's own object.

TWO ABLATIONS, and the second is the control the note does not have.

  `--ablation zero` (default, the note's) sets a pruned weight to 0. This is what its
  helpfulness definition ablates to, so it is the faithful curve.

  `--ablation mean` replaces a pruned weight's contribution by its EXPECTATION over the
  corpus instead of by zero. For this family that is exact and free: the contribution of
  weight (i, j) at a position is W_ij * s_i with s_i in {0,1}, so its mean is W_ij * P(i),
  and mean-ablating a set is the same forward pass plus the constant logit vector
      c_j = sum_{(i,j) dropped} W_ij P(i).

  Why it matters: the direct path's average contribution to logit j is a FIXED vector over
  the vocabulary -- effectively a unigram prior. Under zero ablation, pruning destroys it,
  and a ranking can score well simply by keeping weights that rebuild it (which is exactly
  the mechanism the sibling toy-model replication found MAttr+Adam exploiting). Under mean
  ablation that constant is preserved for free, so the curve measures only which weights
  carry INPUT-DEPENDENT signal. Reporting both separates the two effects instead of letting
  one masquerade as the other.

WHY THE ORACLE IS NOT AUTOMATICALLY THE BEST CURVE, and why that is the point: helpfulness
is a MARGINAL quantity, the loss change from ablating one weight with all others present.
The curve asks a JOINT question. The note flags the gap in a footnote ("Up to nonlinear
effects in removing multiple weights at once") and then reasons past it. Any ranking that
beats the oracle here is measuring that gap.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_model import CTX, VOCAB  # noqa: E402
from vw_scores import D_VP, load_run, load_tokens  # noqa: E402

DENSITIES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1,
             0.07, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]


@torch.no_grad()
def eval_masked(model, toks, keep, batch, device, bias=None):
    """Held-out loss with the direct path restricted to `keep` (a bool (5120,4096))."""
    W = model.W_TL().float() * keep
    pos_rows = VOCAB + torch.arange(CTX, device=device)
    tot, n = 0.0, 0
    for i in range(0, len(toks), batch):
        b = toks[i:i + batch].to(device)
        with torch.no_grad():
            _, a, m = model.paths(b)
        logits = W[b] + W[pos_rows] + a + m
        if bias is not None:
            logits = logits + bias
        l = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB).float(), b[:, 1:].reshape(-1))
        tot += l.item() * b.shape[0]
        n += b.shape[0]
    return tot / n


def rankings(run, device, seeds=(0,)):
    """Every score tensor on disk for this run, flattened, higher = keep first."""
    sc = torch.load(run / "scores" / "tl.pt", map_location=device, weights_only=False)
    out = {
        "weight_abs": sc["W"].abs(),
        "weight": sc["W"],
        "era": sc["era"],
        "fisher": sc["fisher"],
        "helpfulness": sc["helpfulness"],
    }
    d = run / "attrib"
    if d.exists():
        for f in sorted(d.glob("*.pt")):
            out[f.stem] = torch.load(f, map_location=device, weights_only=False)["scores"]
    for s in seeds:
        g = torch.Generator(device="cpu").manual_seed(1234 + s)
        out[f"random_s{s}"] = torch.rand(D_VP, VOCAB, generator=g).to(device)
    return {k: v.to(device).float().flatten() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--seqs", type=int, default=1024, help="held-out sequences to score on")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these rankings")
    ap.add_argument("--rows", default="all", choices=["all", "token", "position"],
                    help="which source rows are prunable; the rest are held permanently")
    ap.add_argument("--ablation", default="zero", choices=["zero", "mean"],
                    help="zero = the note's definition; mean = the bias-preserving control")
    # the note's appendix splits the pruning curve by the SIGN of the virtual weight ("except
    # for negative Features->Logits weights"); `pos`/`neg` prune only weights of that sign and
    # hold the others on, like --rows does for source rows
    ap.add_argument("--sign", default="all", choices=["all", "pos", "neg"])
    ap.add_argument("--densities", type=float, nargs="*", default=None,
                    help="override the density grid (default: the 21-point grid)")
    ap.add_argument("--out", default="prune.json")
    args = ap.parse_args()
    global DENSITIES
    if args.densities:
        DENSITIES = list(args.densities)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_run(args.run, device)
    toks = load_tokens(args.split)[:args.seqs]
    ranks = rankings(args.run, device)
    if args.only:
        ranks = {k: v for k, v in ranks.items() if k in args.only}
    # `sel` = prunable positions; everything outside it is held on at every density.
    sel = torch.ones(D_VP, VOCAB, dtype=torch.bool, device=device)
    if args.rows == "token":
        sel[VOCAB:] = False
    elif args.rows == "position":
        sel[:VOCAB] = False
    sc = torch.load(args.run / "scores" / "tl.pt", map_location=device, weights_only=False)
    W_all = sc["W"].to(device).float()
    if args.sign == "pos":
        sel &= W_all > 0
    elif args.sign == "neg":
        sel &= W_all < 0
    sel = sel.flatten()
    held = ~sel
    N = int(sel.sum())
    # P(source i active), so a dropped weight's mean contribution is W_ij * P(i)
    p_src = (sc["src_count"].to(device).float() / float(sc["n_positions"]))[:, None]
    mean_of = (lambda keep: ((1 - keep.float()) * W_all * p_src).sum(0)) \
        if args.ablation == "mean" else (lambda keep: None)

    ones = torch.ones(D_VP, VOCAB, device=device)
    zeros = held.view(D_VP, VOCAB).float()   # "nothing kept" still holds the frozen rows
    full = eval_masked(model, toks, ones, args.batch, device, mean_of(ones))
    # the floor: no direct path at all (every Tokens->Logits weight removed)
    empty = eval_masked(model, toks, zeros, args.batch, device, mean_of(zeros))
    print(f"[{args.ablation} ablation, rows={args.rows}, sign={args.sign}] full {full:.4f} | all {N:,} prunable "
          f"weights removed {empty:.4f} (dL {empty-full:+.4f}) | {len(toks)} held-out sequences",
          flush=True)

    res = {"full_loss": full, "empty_loss": empty, "n_weights": N, "ablation": args.ablation,
           "rows": args.rows, "sign": args.sign,
           "seqs": len(toks), "split": args.split, "densities": DENSITIES, "curves": {}}
    for name, s in ranks.items():
        # rank only within the prunable set: send held-out entries to the bottom
        order = torch.argsort(s.masked_fill(held, -float("inf")), descending=True)
        row, t0 = [], time.time()
        for dsty in DENSITIES:
            k = max(1, int(round(dsty * N)))
            keep = held.clone()
            keep[order[:k]] = True
            kv = keep.view(D_VP, VOCAB)
            row.append(eval_masked(model, toks, kv, args.batch, device, mean_of(kv)))
        res["curves"][name] = row
        # the note quotes dL at density 0.55 / 0.30 / 0.15
        pick = {d: row[DENSITIES.index(d)] - full for d in (0.55, 0.3, 0.15, 0.05, 0.01)}
        print(f"  {name:<34} " + " ".join(f"d{d}={v:+.4f}" for d, v in pick.items())
              + f" | {time.time()-t0:.0f}s", flush=True)
        (args.run / args.out).write_text(json.dumps(res, indent=2))
    print(f"-> {args.run/args.out}")


if __name__ == "__main__":
    main()
