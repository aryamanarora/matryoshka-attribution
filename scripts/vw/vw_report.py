"""Every quantitative claim the note makes about its weight population, recomputed here.

    uv run python scripts/vw/vw_report.py --run results/vw/base

The note (Turner, Wu & Batson 2026) states its distributional findings as numbers over all
six virtual weight families; we can only check them on Tokens->Logits, so each row below
carries the note's figure alongside ours and the comparison is family-restricted by
construction. That is a caveat on the comparison, not a defect in the measurement: this
family is 21M of the note's 331M weights and is the one it analyses most closely.

Claims checked (quoted):
  "The median magnitude of virtual weight is approximately one third of the magnitude of
   the largest weight, while the median effectiveness of a virtual weight is 10,000x less
   than the maximum effectiveness."
  "virtual weights appear approximately normally distributed, while Fisher effectiveness is
   roughly lognormally distributed, varying by 10 orders of magnitude"
  "roughly half (47.6%) of all weights have positive mean helpfulness (with 12.7% dead)"
  "The most effective helpful weight out-measures any harmful weight by an order of
   magnitude or more within each weight family."
  "we estimate that we'd only need 2.43% density [for 90% of positive helpfulness mass],
   but this fraction quickly climbs to a similar regime (13.6%) when accounting for 99%".

The second block is what the note could not do: agreement of every ranking with the
helpfulness ORACLE, over all 21M weights rather than a 7,765-weight sample.
"""

import argparse
import json
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vw_prune import rankings  # noqa: E402
from vw_scores import D_VP  # noqa: E402
from vw_model import VOCAB  # noqa: E402

NOTE = {"median_absW_over_max": 1 / 3, "median_fisher_over_max": 1e-4,
        "frac_helpful": 0.476, "frac_dead": 0.127,
        "density_for_90pct_mass": 0.0243, "density_for_99pct_mass": 0.136}


def spearman(a, b, n=2_000_000, seed=0):
    """Spearman on a uniform subsample, with TIES AVERAGED.

    Ties are not a technicality here: every weight whose source token never appears in the
    corpus has helpfulness identically zero, and Fisher effectiveness is exactly zero on the
    same set, so a large block of the 21M is tied in both variables. `argsort().argsort()`
    would break those ties by array index -- an arbitrary order that correlates with nothing
    and silently dilutes every coefficient. scipy's rankdata averages them instead."""
    from scipy.stats import rankdata
    g = torch.Generator(device=a.device).manual_seed(seed)
    idx = torch.randperm(a.numel(), generator=g, device=a.device)[:n]
    ra = rankdata(a[idx].cpu().numpy())
    rb = rankdata(b[idx].cpu().numpy())
    ra = (ra - ra.mean()) / ra.std()
    rb = (rb - rb.mean()) / rb.std()
    return float((ra * rb).mean())


def mass_density(h, frac):
    """Smallest density that captures `frac` of the total POSITIVE helpfulness mass."""
    pos = h[h > 0].sort(descending=True).values
    c = pos.cumsum(0)
    k = int(torch.searchsorted(c, frac * c[-1]).item()) + 1
    return k / h.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--sig", type=float, default=1.96, help="z for the 95% CI on helpfulness")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sc = torch.load(args.run / "scores" / "tl.pt", map_location=device, weights_only=False)
    W, F, H, SE = (sc[k].to(device).float() for k in ("W", "fisher", "helpfulness", "helpfulness_se"))
    # "dead" in the note's sense: the weight is never exercised, i.e. its source row never
    # activates on the corpus, so its helpfulness is exactly (not approximately) zero.
    dead = (sc["src_count"].to(device) == 0)[:, None].expand_as(W)
    Wf, Ff, Hf, SEf = (x.flatten() for x in (W, F, H, SE))
    nz = Ff > 0
    sig_pos, sig_neg = (Hf - args.sig * SEf) > 0, (Hf + args.sig * SEf) < 0

    dist = {
        "n_weights": int(Wf.numel()),
        "median_absW_over_max": float(W.abs().median() / W.abs().max()),
        "median_fisher_over_max": float(Ff[nz].median() / Ff.max()),
        "fisher_log10_range": float(torch.log10(Ff[nz].max() / Ff[nz].min())),
        "W_skew": float((((Wf - Wf.mean()) / Wf.std()) ** 3).mean()),
        "log10_fisher_skew": float((lambda z: ((z - z.mean()) / z.std()).pow(3).mean())(torch.log10(Ff[nz]))),
        "frac_helpful": float((Hf > 0).float().mean()),
        "frac_dead": float(dead.float().mean()),
        "frac_sig_helpful": float(sig_pos.float().mean()),
        "frac_sig_harmful": float(sig_neg.float().mean()),
        "sum_pos_helpfulness": float(Hf[Hf > 0].sum()),
        "sum_neg_helpfulness": float(-Hf[Hf < 0].sum()),
        "max_fisher_of_helpful": float(Ff[sig_pos].max()) if sig_pos.any() else None,
        "max_fisher_of_harmful": float(Ff[sig_neg].max()) if sig_neg.any() else None,
        "density_for_90pct_mass": mass_density(Hf, 0.90),
        "density_for_99pct_mass": mass_density(Hf, 0.99),
    }
    if dist["max_fisher_of_harmful"]:
        dist["helpful_over_harmful_fisher"] = dist["max_fisher_of_helpful"] / dist["max_fisher_of_harmful"]

    print("=== distribution (note's number in brackets; note covers all six families) ===")
    for k, v in dist.items():
        n = f"   [note {NOTE[k]:g}]" if k in NOTE else ""
        print(f"  {k:<28} {v:.6g}{n}" if isinstance(v, float) else f"  {k:<28} {v}{n}")

    # TOKEN vs POSITION sources are structurally different populations in this family and
    # are reported apart. Position information enters through a FIXED sinusoidal table whose
    # rows have norm ~sqrt(d_m/2) = 11.3, against token embeddings initialised at 0.0375 and
    # trained from there -- so a position row's virtual weights are systematically larger for
    # a reason that has nothing to do with what they do. The note's model is built the same
    # way and shows the same thing ("The largest positive Tokens->Features virtual weights to
    # this feature originate from positions"), so this is replicated, not an artifact here.
    print("\n=== token-source vs position-source rows ===")
    for lab, sl in (("token   ", slice(0, VOCAB)), ("position", slice(VOCAB, D_VP))):
        w, f, h = W[sl].flatten(), F[sl].flatten(), H[sl].flatten()
        sp = (h - args.sig * SE[sl].flatten()) > 0
        dist[f"{lab.strip()}_rows"] = {
            "n": int(w.numel()), "median_absW": float(w.abs().median()),
            "median_fisher": float(f[f > 0].median()) if (f > 0).any() else 0.0,
            "frac_sig_helpful": float(sp.float().mean()),
            "sum_pos_helpfulness": float(h[h > 0].sum()),
        }
        d = dist[f"{lab.strip()}_rows"]
        print(f"  {lab} {d['n']:>10,} weights  median|W| {d['median_absW']:.4f}  "
              f"median fisher {d['median_fisher']:.3e}  sig-helpful {d['frac_sig_helpful']:.3f}  "
              f"pos-help mass {d['sum_pos_helpfulness']:.3e}")

    print("\n=== agreement with the helpfulness oracle, over all 21M weights ===")
    agree = {}
    ranks = rankings(args.run, device)
    ntop = int(sig_pos.sum())
    for name, s in ranks.items():
        if name == "helpfulness":
            continue
        top = s.argsort(descending=True)[:ntop]
        agree[name] = {
            "spearman_vs_helpfulness": spearman(s, Hf),
            "spearman_vs_fisher": spearman(s, Ff),
            f"precision@{ntop}_sig_helpful": float(sig_pos[top].float().mean()),
            f"harmful_rate@{ntop}": float(sig_neg[top].float().mean()),
        }
        a = agree[name]
        print(f"  {name:<34} rho(help) {a['spearman_vs_helpfulness']:+.3f}  "
              f"rho(fisher) {a['spearman_vs_fisher']:+.3f}  "
              f"P@sig {a[f'precision@{ntop}_sig_helpful']:.3f}  "
              f"harmful {a[f'harmful_rate@{ntop}']:.4f}")
    base = float(sig_pos.float().mean())
    print(f"  {'(base rate of sig-helpful)':<34} {base:.4f}")

    out = {"dist": dist, "note": NOTE, "agreement": agree, "base_rate_sig_helpful": base,
           "n_positions": sc["n_positions"]}
    (args.run / "report.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.run/'report.json'}")


if __name__ == "__main__":
    main()
