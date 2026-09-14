"""Figure-by-figure audit of Turner, Wu & Batson (2026) under a learned attribution method.

    uv run python scripts/vw/vw_audit.py --run results/vw/base

The note's claims about effectiveness and helpfulness are claims about ONE ranking -- Fisher
effectiveness -- read off figures drawn for that ranking. Restricted to the two families whose
helpfulness we have exactly for every weight (Tokens->Logits, Features->Logits), every one of
those figures can be redrawn for any ranking. This script does that for the best MAttr arm in
each setting and the baselines, and writes the statistics each figure's caption rests on:

  A. "Effectiveness surfaces helpful weights but does not isolate them" (the IN and Feature
     3013 per-node panels). Per source node, the top-n weights by each score: what fraction
     are significantly helpful, harmful, or neither; and how many nodes have ANY significantly
     harmful weight in their top-n.
  B. "The most effective weights are helpful" (the 1,111-weight three-regime figure). Along
     each ranking: helpfulness in log-spaced rank bins, the rank of the first significantly
     harmful weight, and the fraction of harmful weights in the top 0.01% / 0.1% / 1% / 10%.
  C. "Weight filters in this basis won't yield much sparser models" (the helpfulness table and
     mass analysis). For each ranking and density: positive-helpfulness mass captured, fraction
     of significantly-helpful weights kept, fraction of significantly-harmful weights kept,
     next to the measured d-loss of that kept set from the pruning sweeps. Also the density
     each ranking needs for 90% / 99% of positive mass.
  D. The IN -> logits worked example, re-sorted by the MAttr score.

"Significant" is the note's convention: the 95% Gaussian CI on mean helpfulness excludes 0.
Everything is exact over the 97.7M training positions, not a sample.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "plots"))
sys.path.insert(0, str(ROOT / "scripts"))
from palette import METHOD, OTHER, RC, furnish  # noqa: E402
from vw_model import VOCAB  # noqa: E402

plt.rcParams.update(RC)
HELPFUL, HARMFUL, NOTSIG = "#0072b2", "#cc3311", "#cccccc"
DENS = [0.55, 0.30, 0.15, 0.05, 0.02, 0.01, 0.002]
TOPF = [1e-4, 1e-3, 1e-2, 1e-1]

# (key, label, colour) per family; the MAttr arm is the best one measured in that setting
FAMILIES = {
    "tl_tok": dict(
        scores="scores/tl.pt", attrib="attrib", rows=slice(0, VOCAB), prune="prune_tok_mean.json",
        prune_zero="prune_tok.json", title="Tokens→Logits (token rows)",
        rank=[("fisher", "Fisher effectiveness", METHOD["IG"]),
              ("weight_abs", "Virtual weight magnitude", METHOD["I×G"]),
              ("era", "Expected attribution", METHOD["AttnLRP"]),
              ("ig_long", "Expected Gradients", METHOD["Expected Gradients"]),
              ("mattr_adam_lr0.05_eps0.01", "MAttr (default: lr .05, ε=1e-2)", METHOD["MAttr"]),
              ("E_eps1e-8_lr0.01_s12000_b32", "MAttr (lr .01, ε=1e-8, b32, 12k)", "#c77dff"),
              ("J_eps1e-8_lr0.01_s24000_b64", "MAttr (best: lr .01, ε=1e-8, b64, 24k)", METHOD["MAttr (Adam, default eps)"]),
              ("helpfulness", "Helpfulness (oracle)", "#000000")]),
    "fl": dict(
        scores="scores/fl.pt", attrib="attrib_fl", rows=slice(0, VOCAB), prune="prune_fl_mean.json",
        prune_zero="prune_fl.json", title="Features→Logits",
        rank=[("fisher", "Fisher effectiveness", METHOD["IG"]),
              ("weight_abs", "Virtual weight magnitude", METHOD["I×G"]),
              ("era", "Expected attribution", METHOD["AttnLRP"]),
              ("ig_s3000", "Expected Gradients", METHOD["Expected Gradients"]),
              ("mattr_adam_lr0.05_eps0.01", "MAttr (default: lr .05, ε=1e-2)", METHOD["MAttr"]),
              ("H_eps1e-8_lr0.01_s12000_b32", "MAttr (best: lr .01, ε=1e-8, b32)", METHOD["MAttr (Adam, default eps)"]),
              ("helpfulness", "Helpfulness (oracle)", "#000000")]),
}


def load_family(run, fam, sig):
    f = FAMILIES[fam]
    sc = torch.load(run / f["scores"], map_location="cpu", weights_only=False)
    rows = f["rows"]
    W, F, H, SE = (sc[k][rows].float() for k in ("W", "fisher", "helpfulness", "helpfulness_se"))
    scores = {"fisher": F, "weight_abs": W.abs(), "era": sc["era"][rows].float(), "helpfulness": H}
    for key, _, _ in f["rank"]:
        if key in scores:
            continue
        p = run / f["attrib"] / f"{key}.pt"
        if not p.exists():
            print(f"  [{fam}] missing {p.name}; skipped")
            continue
        s = torch.load(p, map_location="cpu", weights_only=False)["scores"]
        if s.shape[0] != W.shape[0]:
            s = s[rows]
        scores[key] = s.float()
    sig_pos, sig_neg = (H - sig * SE) > 0, (H + sig * SE) < 0
    return dict(W=W, F=F, H=H, SE=SE, pair=sc["pair"][rows].float(), src=sc["src_count"][rows].float(),
                scores=scores, sig_pos=sig_pos, sig_neg=sig_neg)


def per_node(d, key, n):
    """Top-n of each source row by `key`: mean helpful / harmful / not-sig fraction, and the
    fraction of rows with >=1 harmful weight in their top-n. Dead rows (source never active)
    are excluded -- every score is 0 there and the top-n is arbitrary."""
    s = d["scores"][key]
    alive = d["src"] > 0
    top = s[alive].topk(n, dim=1).indices
    sp = d["sig_pos"][alive].gather(1, top).float()
    sn = d["sig_neg"][alive].gather(1, top).float()
    return dict(helpful=float(sp.mean()), harmful=float(sn.mean()),
                notsig=float(1 - sp.mean() - sn.mean()),
                rows_with_harmful=float((sn.sum(1) > 0).float().mean()), n_rows=int(alive.sum()))


def along_ranking(d, key):
    s = d["scores"][key].flatten()
    order = s.argsort(descending=True)
    H = d["H"].flatten()[order]
    sp = d["sig_pos"].flatten()[order]
    sn = d["sig_neg"].flatten()[order]
    N = H.numel()
    out = {}
    # B: first harmful, harmful fraction in the top slices, and log-rank bins
    first = int(sn.nonzero()[0]) + 1 if sn.any() else None
    out["first_harmful_rank"] = first
    out["first_harmful_frac"] = first / N if first else None
    for f in TOPF:
        k = max(1, int(f * N))
        out[f"harmful_in_top_{f:g}"] = float(sn[:k].float().mean())
        out[f"helpful_in_top_{f:g}"] = float(sp[:k].float().mean())
    edges = np.unique(np.round(np.geomspace(1, N, 13)).astype(np.int64))
    bins = []
    for a, b in zip(edges[:-1], edges[1:]):
        bins.append(dict(lo=int(a), hi=int(b), helpful=float(sp[a - 1:b].float().mean()),
                         harmful=float(sn[a - 1:b].float().mean()),
                         mean_help=float(H[a - 1:b].mean())))
    out["bins"] = bins
    # the note's "out-measures by an order of magnitude": max score of a helpful weight
    # against max score of a harmful one, only meaningful for non-negative scores
    sc = s[order]
    if float(sc.min()) >= 0:
        out["max_score_helpful_over_harmful"] = float(sc[sp].max() / sc[sn].max()) if sn.any() else None
    # C: mass and counts at each density
    cum_pos = torch.cumsum(H.clamp(min=0), 0)
    cum_sp = torch.cumsum(sp.float(), 0)
    cum_sn = torch.cumsum(sn.float(), 0)
    tot_pos, tot_sp, tot_sn = float(cum_pos[-1]), float(cum_sp[-1]), float(cum_sn[-1])
    dens = {}
    for dd in DENS:
        k = max(1, int(round(dd * N))) - 1
        dens[str(dd)] = dict(pos_mass=float(cum_pos[k]) / tot_pos, helpful_kept=float(cum_sp[k]) / tot_sp,
                             harmful_kept=float(cum_sn[k]) / tot_sn)
    out["at_density"] = dens
    for f in (0.90, 0.99):
        j = int(torch.searchsorted(cum_pos, torch.tensor(f * tot_pos))) + 1
        out[f"density_for_{int(f*100)}pct_mass"] = j / N
    out["_curves"] = dict(cum_pos=(cum_pos / tot_pos).numpy(), cum_sn=cum_sn.numpy(),
                          H=H.numpy(), sp=sp.numpy(), sn=sn.numpy())
    return out


def in_example(d, run, key, top=8):
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "data" / "vw" / "tok4096.json"))
    i = tk.encode("IN", add_special_tokens=False).ids[0]
    s = d["scores"][key][i]
    order = s.argsort(descending=True)[:top]
    rows = []
    for j in order.tolist():
        rows.append(dict(target=tk.decode([j]), W=float(d["W"][i, j]), fisher=float(d["F"][i, j]),
                         helpfulness=float(d["H"][i, j]), followed=int(d["pair"][i, j]),
                         sig="helpful" if d["sig_pos"][i, j] else ("harmful" if d["sig_neg"][i, j] else "ns")))
    return rows


def dl_from_prune(run, fname, key):
    p = run / fname
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    if key not in d["curves"]:
        return {}
    x, full = d["densities"], d["full_loss"]
    return {str(dd): d["curves"][key][x.index(dd)] - full for dd in DENS if dd in x}



BUDGET = float(np.log(VOCAB)) - 3.3796          # the note's yardstick: ln|V| minus held-out loss
SIGNED = {"E_eps1e-8_lr0.01_s12000_b32", "J_eps1e-8_lr0.01_s24000_b64", "H_eps1e-8_lr0.01_s12000_b32", "mattr_adam_lr0.05_eps0.01",
          "ig_long", "ig_s3000"}
IN_TARGETS = ["E", "T", "TER", "ES", "D", "ST", "vir", "tig", "ú", "SA", "DE"]


def distribution(d, n_positions, sig):
    """The appendix's distribution figures as numbers: the mean-helpfulness histogram (count
    symmetry vs mass asymmetry), the Fisher mass split by the sign of the weight, how many
    tokens it takes to recover a weight's sign, and the ROPE classification."""
    H, SE, W, F = (d[k].flatten() for k in ("H", "SE", "W", "F"))
    live = SE > 0
    out = dict(frac_pos=float((H > 0).float().mean()), frac_neg=float((H < 0).float().mean()),
               frac_zero=float((H == 0).float().mean()),
               median_abs_pos=float(H[H > 0].median()), median_abs_neg=float((-H[H < 0]).median()),
               mass_pos=float(H.clamp(min=0).sum()), mass_neg=float((-H).clamp(min=0).sum()),
               fisher_nonzero=float((F > 0).float().mean()),
               fisher_mass_Wpos=float(F[W > 0].sum()), fisher_mass_Wneg=float(F[W < 0].sum()))
    # tokens needed for the 95% CI to exclude zero: n * (1.96 SE / |H|)^2, SE being at n positions
    z = (H[live].abs() / SE[live])
    need = n_positions * (sig / z.clamp(min=1e-12)) ** 2
    out["tokens_to_recover_sign"] = {f"{n:g}": float((need <= n).float().mean())
                                     for n in (1e6, 1e7, 1e8, 1e9, 1e10)}
    out["median_tokens_to_recover_sign"] = float(need.median())
    out["frac_sig_at_full_data"] = float((z > sig).float().mean())
    # ROPE
    lo, hi = H - sig * SE, H + sig * SE
    out["budget"] = BUDGET
    rope = {}
    for name, eps in (("budget/N", BUDGET / H.numel()), ("budget*1e-6", BUDGET * 1e-6),
                      ("budget*1e-5", BUDGET * 1e-5), ("budget*1e-4", BUDGET * 1e-4)):
        pos, neg = lo > eps, hi < -eps
        zero = (lo > -eps) & (hi < eps)
        rope[name] = dict(eps=eps, positive=float(pos.float().mean()), negative=float(neg.float().mean()),
                          practically_zero=float(zero.float().mean()),
                          uncertain=float(1 - pos.float().mean() - neg.float().mean() - zero.float().mean()))
    out["rope"] = rope
    return out


def rope_kept(d, key, sig, dens=0.15):
    """Of the weights a ranking keeps at `dens`, what fraction are practically zero at the
    per-family yardstick eps = budget / N (the ROPE appendix's own threshold)?"""
    H, SE = d["H"].flatten(), d["SE"].flatten()
    eps = BUDGET / H.numel()
    zero = ((H - sig * SE) > -eps) & ((H + sig * SE) < eps)
    s = d["scores"][key].flatten()
    k = max(1, int(round(dens * s.numel())))
    top = s.topk(k).indices
    return float(zero[top].float().mean()), float(zero.float().mean())


def sign_agreement(d, key, sig):
    """P(sign(score) == sign(helpfulness)) by how well-determined the helpfulness sign is
    (|mean| / SE), for scores that carry a sign. MAttr scores are mask logits from a zero
    init, so >0 means the mask wants the weight."""
    s = d["scores"][key].flatten()
    H, SE = d["H"].flatten(), d["SE"].flatten()
    live = (SE > 0) & (H != 0) & (s != 0)
    z = (H[live].abs() / SE[live])
    agree = (torch.sign(s[live]) == torch.sign(H[live]))
    edges = [0, 1, sig, 5, 20, float("inf")]
    out = {"all": float(agree.float().mean())}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        out[f"z[{a:g},{b:g})"] = dict(agree=float(agree[m].float().mean()) if m.any() else None,
                                      frac=float(m.float().mean()))
    return out


def rank_agreement(d, key, sig):
    """P(kept == helpful) when the ranking keeps exactly as many weights as have positive
    helpfulness, by how well-determined the helpfulness sign is (|mean| / SE). A ranking's
    analogue of sign agreement that does not depend on where its zero sits (MAttr logits
    under a log-uniform k drift negative, so their raw sign says nothing)."""
    s = d["scores"][key].flatten()
    H, SE = d["H"].flatten(), d["SE"].flatten()
    k = int((H > 0).sum())
    kept = torch.zeros_like(H, dtype=torch.bool)
    kept[s.topk(k).indices] = True
    live = (SE > 0) & (H != 0)
    z = (H[live].abs() / SE[live])
    agree = kept[live] == (H[live] > 0)
    edges = [0, 1, sig, 5, 20, float("inf")]
    out = {"all": float(agree.float().mean()), "k_frac": k / H.numel()}
    for a, b in zip(edges[:-1], edges[1:]):
        m = (z >= a) & (z < b)
        out[f"z[{a:g},{b:g})"] = dict(agree=float(agree[m].float().mean()) if m.any() else None,
                                      frac=float(m.float().mean()))
    return out


def in_targets(d, keys):
    """The motivating example as a membership question: for each named target out of IN, its
    rank within the row and the density at which each ranking first keeps it."""
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(ROOT / "data" / "vw" / "tok4096.json"))
    i = tk.encode("IN", add_special_tokens=False).ids[0]
    N = d["H"].numel()
    out = {}
    for t in IN_TARGETS:
        ids = tk.encode(t, add_special_tokens=False).ids
        if len(ids) != 1:
            continue
        j = ids[0]
        rec = dict(w=float(d["W"][i, j]), help=float(d["H"][i, j]), followed=int(d["pair"][i, j]),
                   sig="helpful" if d["sig_pos"][i, j] else ("harmful" if d["sig_neg"][i, j] else "ns"))
        for key in keys:                      # 'helpfulness' is also a ranking key, hence 'help'

            s = d["scores"][key]
            v = s[i, j]
            rec[key] = dict(row_rank=int((s[i] > v).sum()) + 1, kept_at_density=(int((s > v).sum()) + 1) / N)
        out[t] = rec
    return out


def fig_hist(ds, fams, out):
    """The appendix's mean-helpfulness histogram, per family, with the count and mass split."""
    fig, axes = plt.subplots(1, len(fams), figsize=(4.0 * len(fams), 2.4), squeeze=False)
    for c, fam in enumerate(fams):
        H = ds[fam]["H"].flatten().numpy()
        ax = axes[0, c]
        furnish(ax)
        nz = H[H != 0]
        mag = np.abs(nz)
        bins = np.geomspace(max(mag.min(), 1e-14), mag.max(), 80)
        ax.hist(mag[nz > 0], bins=bins, color=HELPFUL, alpha=0.8, label=f"helpful ({100*(H>0).mean():.1f}%, mass {H[H>0].sum():.3g})")
        ax.hist(mag[nz < 0], bins=bins, color=HARMFUL, alpha=0.6, label=f"harmful ({100*(H<0).mean():.1f}%, mass {(-H[H<0]).sum():.3g})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("|mean helpfulness| (nats/token)", fontsize=7)
        ax.set_ylabel("weights", fontsize=7)
        ax.set_title(FAMILIES[fam]["title"], fontsize=8)
        ax.legend(fontsize=6, frameon=False, loc="upper left")
        ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out}")


def sign_split(run, fam):
    """Fold the per-sign pruning sweeps (scripts/vw/launch/vw_stageK_cpu.sh) in, if they exist."""
    stem = {"tl_tok": "prune_tok", "fl": "prune_fl"}[fam]
    out = {}
    for sgn in ("pos", "neg"):
        p = run / f"{stem}_{sgn}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        full = d["full_loss"]
        out[sgn] = dict(n=d["n_weights"], all_removed=d["empty_loss"] - full, densities=d["densities"],
                        curves={k: [v - full for v in vals] for k, vals in d["curves"].items()})
    return out


def fig_regimes(res, fams, out, sample=3000):
    """The note's 'most effective weights are helpful' panel for every ranking, both families."""
    keys = [k for k in ("fisher", "weight_abs", "mattr_adam_lr0.05_eps0.01",
                        "J_eps1e-8_lr0.01_s24000_b64", "H_eps1e-8_lr0.01_s12000_b32")]
    fig, axes = plt.subplots(len(fams), 4, figsize=(8.6, 2.3 * len(fams)), squeeze=False)
    for r, fam in enumerate(fams):
        labs = {k: (lab, col) for k, lab, col in FAMILIES[fam]["rank"]}
        cols = [k for k in keys if k in res[fam]["along"]]
        lim = None
        for c, k in enumerate(cols):
            cv = res[fam]["along"][k]["_curves"]
            H, sp, sn = cv["H"], cv["sp"], cv["sn"]
            N = len(H)
            idx = np.unique(np.round(np.exp(np.linspace(0, np.log(N - 1), sample))).astype(np.int64))
            lim = lim or float(np.abs(H).max()) * 1.5
            ax = axes[r, c]
            furnish(ax)
            colr = np.where(sp[idx], HELPFUL, np.where(sn[idx], HARMFUL, NOTSIG))
            ax.scatter(idx + 1, H[idx], s=1.6, c=colr, lw=0, rasterized=True)
            ax.axhline(0, color="#999999", lw=0.5)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=1e-9)
            ax.set_ylim(-lim, lim)
            fh = res[fam]["along"][k]["first_harmful_rank"]
            if fh:
                ax.axvline(fh, color=HARMFUL, lw=0.6, ls=":")
            ax.set_title(labs[k][0], fontsize=7, color=labs[k][1])
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"{FAMILIES[fam]['title']}\nmean helpfulness", fontsize=7)
            else:
                ax.set_yticklabels([])
            if r == len(fams) - 1:
                ax.set_xlabel("rank under the score (1 = kept first)", fontsize=7)
    for lbl, c in (("helpful (p<.05)", HELPFUL), ("harmful (p<.05)", HARMFUL), ("not significant", NOTSIG)):
        axes[0, 0].scatter([], [], s=8, c=c, lw=0, label=lbl)
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out}")


def fig_curves(res, fams, out):
    """Two statistics per family, as curves against density: fraction of the kept set that is
    significantly harmful (does the ranking ISOLATE helpful weights?) and positive helpfulness
    mass captured (the note's density-from-mass argument)."""
    fig, axes = plt.subplots(2, len(fams), figsize=(4.2 * len(fams), 4.6), squeeze=False)
    for c, fam in enumerate(fams):
        for k, lab, col in FAMILIES[fam]["rank"]:
            if k not in res[fam]["along"]:
                continue
            cv = res[fam]["along"][k]["_curves"]
            N = len(cv["H"])
            dens = np.geomspace(1 / N, 1.0, 200)
            ks = np.clip((dens * N).astype(np.int64), 1, N) - 1
            ls = "--" if k.startswith(("E_", "H_", "J_")) else ("-" if not k == "helpfulness" else (0, (1, 1)))
            axes[0, c].plot(dens, cv["cum_sn"][ks] / (ks + 1), color=col, lw=1.3, linestyle=ls, label=lab)
            axes[1, c].plot(dens, cv["cum_pos"][ks], color=col, lw=1.3, linestyle=ls)
        for r in range(2):
            furnish(axes[r, c])
            axes[r, c].set_xscale("log")
            axes[r, c].tick_params(labelsize=6.5)
        axes[0, c].set_title(FAMILIES[fam]["title"], fontsize=8)
        axes[0, c].set_ylabel("fraction of kept weights\nsignificantly harmful", fontsize=7)
        axes[1, c].set_ylabel("positive helpfulness\nmass captured", fontsize=7)
        axes[1, c].set_xlabel("density (fraction of the family kept)", fontsize=7)
        axes[1, c].plot([1e-6, 1], [1e-6, 1], color="#bbbbbb", lw=0.7, ls="--")
        axes[0, c].set_yscale("log")
        axes[0, c].set_ylim(1e-4, 1)
        axes[1, c].set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=5.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=600)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/vw/base"))
    ap.add_argument("--sig", type=float, default=1.96)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(16)
    res, loaded = {}, {}
    for fam in args.families:
        print(f"\n===== {FAMILIES[fam]['title']} =====")
        d = load_family(args.run, fam, args.sig)
        N = d["H"].numel()
        base = dict(n=N, frac_sig_helpful=float(d["sig_pos"].float().mean()),
                    frac_sig_harmful=float(d["sig_neg"].float().mean()),
                    frac_pos=float((d["H"] > 0).float().mean()),
                    sum_pos=float(d["H"].clamp(min=0).sum()), sum_neg=float((-d["H"]).clamp(min=0).sum()))
        print(f"  {N:,} weights; sig-helpful {base['frac_sig_helpful']:.3f}, sig-harmful {base['frac_sig_harmful']:.3f}")
        R = dict(base=base, per_node={}, along={}, in_example={})
        for key, lab, _ in FAMILIES[fam]["rank"]:
            if key not in d["scores"]:
                continue
            R["per_node"][key] = {n: per_node(d, key, n) for n in (10, 50)}
            R["along"][key] = along_ranking(d, key)
            R["along"][key]["dl_mean"] = dl_from_prune(args.run, FAMILIES[fam]["prune"], key)
            R["along"][key]["dl_zero"] = dl_from_prune(args.run, FAMILIES[fam]["prune_zero"], key)
            pn, al = R["per_node"][key], R["along"][key]
            print(f"\n  {lab}")
            print(f"    per node top-10: helpful {pn[10]['helpful']:.3f} harmful {pn[10]['harmful']:.3f} "
                  f"ns {pn[10]['notsig']:.3f}; rows with a harmful weight in top-10 {pn[10]['rows_with_harmful']:.3f}"
                  f"  | top-50: helpful {pn[50]['helpful']:.3f} harmful {pn[50]['harmful']:.3f}")
            print(f"    first harmful at rank {al['first_harmful_rank']:,} ({al['first_harmful_frac']:.2e} of family); "
                  f"harmful in top 1e-4/1e-3/1e-2/1e-1: "
                  + " / ".join(f"{al[f'harmful_in_top_{f:g}']:.3f}" for f in TOPF)
                  + (f"; max-score helpful/harmful {al['max_score_helpful_over_harmful']:.1f}x"
                     if al.get("max_score_helpful_over_harmful") else ""))
            print(f"    density for 90% / 99% pos mass: {al['density_for_90pct_mass']:.4f} / {al['density_for_99pct_mass']:.4f}")
            print("    density : pos-mass  helpful-kept  harmful-kept   dL(mean)  dL(zero)")
            for dd in DENS:
                a = al["at_density"][str(dd)]
                dm, dz = al["dl_mean"].get(str(dd)), al["dl_zero"].get(str(dd))
                print(f"      {dd:<6}: {a['pos_mass']:.3f}     {a['helpful_kept']:.3f}         {a['harmful_kept']:.3f}      "
                      f"{'' if dm is None else f'{dm:+.3f}':>7}   {'' if dz is None else f'{dz:+.3f}':>7}")
            if fam == "tl_tok":
                R["in_example"][key] = in_example(d, args.run, key)
        n_pos = int(torch.load(args.run / FAMILIES[fam]["scores"], map_location="cpu", weights_only=False)["n_positions"])
        R["distribution"] = distribution(d, n_pos, args.sig)
        R["rope_kept_at_0.15"] = {k: rope_kept(d, k, args.sig) for k in R["along"]}
        R["rope_kept_at_0.01"] = {k: rope_kept(d, k, args.sig, dens=0.01) for k in R["along"]}
        R["rank_agreement"] = {k: rank_agreement(d, k, args.sig) for k in R["along"]}
        # raw sign only where the score has a meaningful zero: gradient scores and the weight
        R["sign_agreement"] = {k: sign_agreement(d, k, args.sig) for k in R["along"] if k.startswith("ig")}
        R["sign_agreement"]["weight_signed"] = sign_agreement({"scores": {"w": d["W"]}, "H": d["H"], "SE": d["SE"]}, "w", args.sig)
        R["sign_split"] = sign_split(args.run, fam)
        dist = R["distribution"]
        print(f"\n  distribution: pos {dist['frac_pos']:.3f} / neg {dist['frac_neg']:.3f} / zero {dist['frac_zero']:.4f}; "
              f"mass pos {dist['mass_pos']:.3f} neg {dist['mass_neg']:.3f}; median|h| pos {dist['median_abs_pos']:.2e} neg {dist['median_abs_neg']:.2e}")
        print(f"  fisher: nonzero {dist['fisher_nonzero']:.4f}; mass over W>0 {dist['fisher_mass_Wpos']:.3f}, W<0 {dist['fisher_mass_Wneg']:.3f}")
        print(f"  sign recoverable with 1e6..1e10 tokens: " + ", ".join(f"{k}: {v:.3f}" for k, v in dist['tokens_to_recover_sign'].items())
              + f"; median tokens needed {dist['median_tokens_to_recover_sign']:.3g}")
        for name, r in dist["rope"].items():
            print(f"  ROPE eps={name} ({r['eps']:.2e}): +{r['positive']:.3f} -{r['negative']:.3f} zero {r['practically_zero']:.3f} unc {r['uncertain']:.3f}")
        print("  practically-zero share of the kept set at d=0.15 (eps=budget/N): " +
              ", ".join(f"{k}: {v[0]:.3f}" for k, v in R["rope_kept_at_0.15"].items()) +
              f"  [population {list(R['rope_kept_at_0.15'].values())[0][1]:.3f}]")
        print("  practically-zero share of the kept set at d=0.01: " +
              ", ".join(f"{k}: {v[0]:.3f}" for k, v in R["rope_kept_at_0.01"].items()))
        for k, sa in R["sign_agreement"].items():
            print(f"  sign agreement {k}: all {sa['all']:.3f}; " + ", ".join(f"{b}: {v['agree']:.3f}" for b, v in sa.items() if b != "all" and v['agree'] is not None))
        for k, sa in R["rank_agreement"].items():
            print(f"  rank agreement (keep top {100*sa['k_frac']:.1f}%) {k}: all {sa['all']:.3f}; " +
                  ", ".join(f"{b}: {v['agree']:.3f}" for b, v in sa.items() if b not in ("all", "k_frac") and v['agree'] is not None))
        if fam == "tl_tok":
            R["in_targets"] = in_targets(d, list(R["along"]))
            print("\n  IN -> targets: within-row rank / density at which first kept")
            for t, rec in R["in_targets"].items():
                print(f"    {t!r:<6} W {rec['w']:+.2f} help {rec['help']:+.2e} followed {rec['followed']:>5} {rec['sig'][:2]}: " +
                      " | ".join(f"{k[:6]} {rec[k]['row_rank']:>4}/{rec[k]['kept_at_density']:.3f}" for k in R["along"]))
            print("\n  IN -> logits, top-8 under each score (target, W, helpfulness, followed-count, sig):")
            for key, lab, _ in FAMILIES[fam]["rank"]:
                if key not in R["in_example"]:
                    continue
                print(f"    {lab}: " + ", ".join(
                    f"{r['target']!r}[{r['followed']},{r['sig'][:2]}]" for r in R["in_example"][key]))
        res[fam] = R
        loaded[fam] = d

    fams = [f for f in args.families if f in res]
    figs = ROOT / "paper" / "figs"
    fig_hist(loaded, fams, figs / "vw_audit_hist.pdf")
    fig_regimes(res, fams, figs / "vw_audit_regimes.pdf")
    fig_curves(res, fams, figs / "vw_audit_curves.pdf")
    # drop the raw curves before writing json
    for fam in res:
        for k in res[fam]["along"]:
            res[fam]["along"][k].pop("_curves", None)
    out = args.out or (args.run / "audit.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
