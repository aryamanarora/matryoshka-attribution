"""Gradient geometry at MAttr's MASKED operating points: where do CE and logit-diff agree?

Reads ONLY the observational records written by ``attribute_sae.py --record-gradient-geometry``
(which are proven not to perturb training: section A checks the instrumented runs' scores,
losses and log-AUC are bitwise identical to the originals) plus the causal numbers already
present in the attribution results. Nothing here runs a model, an SAE, or a backward pass.

Section 5 of the README established that at I x G's base point the two objectives' OUTPUT
gradients sit at cosine ~0.68 uniformly on every task, which explains nothing task-specific.
This measures the geometry MAttr actually optimises through. Predeclared before the last
8 of the 12 runs finished (with only npi_ever_subj-relc and filler_gap_pp observed):

- primary statistic: median cos(dL_CE/dS, dL_LD/dS) at states with k > 128 (the regime
  approaching I x G's fully-unmasked operating point m=1 <=> k=total);
- prediction 1: it decreases with the task's I x G causal swing (Spearman over n=6);
- prediction 2: k<=8 cosines are uniformly high and output-gradient cosines are uniformly
  ~0.68 -- neither separates any task (replicating the base-point negative).

Outcome: prediction 2 held; prediction 1 held in direction (rho = -0.66) but the statistic
does NOT separate the NPI pair -- npi_ever collapses to full orthogonality (0.007) while
npi_any (0.406) sits inside the control range. See the README for the full reading.

Usage:
    python sae_pilot/analyze_masked_geometry.py --diag results/sae_mattr_diag \
        --root results/sae_variant_pilot [--figure sae_pilot/sae_masked_gradient_geometry.png]
"""
import argparse, json, math, os
import numpy as np
import torch

TASKS = ["npi_ever_subj-relc","npi_any_obj-relc","agr_sv_num_subj-relc",
         "garden_npz_v-trans","garden_npz_obj_mod","filler_gap_pp"]
NPI = {"npi_ever_subj-relc","npi_any_obj-relc"}
OBJ = [("ce","ce"), ("logit_diff","ld")]
COL = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
# coarse bins predeclared with the two-task pilot; the fine tail added before the 6-task read.
# upper bins are UNBOUNDED: k runs to the task's own total score count (98,310-147,465 across
# these six tasks), so any fixed cap would silently drop tail records for the larger tasks.
INF = float("inf")
BINS  = [("k<=8",1,8),("8<k<=32",8,32),("32<k<=128",32,128),("k>128",128,INF)]
FBINS = [("128<k<=2048",128,2048),("2048<k<=32768",2048,32768),("k>32768",32768,INF)]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = x.argsort().argsort().astype(float), y.argsort().argsort().astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def sep_detail(vals):
    """NPI-pair separation with the normalised-gap yardstick of section 5 of the README."""
    npi = [vals[i] for i, t in enumerate(TASKS) if t in NPI]
    oth = [vals[i] for i, t in enumerate(TASKS) if t not in NPI]
    rng = max(vals) - min(vals)
    if min(npi) > max(oth):   gap = min(npi) - max(oth)
    elif max(npi) < min(oth): gap = min(oth) - max(npi)
    else:                     gap = -min(abs(a - b) for a in npi for b in oth)
    return gap > 0, gap / rng if rng else 0.0


def geom_dir(diag, task, loss):
    for d in (f"{diag}/{task}/MAttr_{loss}_s0_geom10",             # cluster layout
              f"{diag}/{task}_MAttr_{loss}_s0_geom"):              # flat mirror layout
        if os.path.exists(f"{d}/gradient_geometry.json"):
            return d
    raise SystemExit(f"no geometry run for {task}/{loss} under {diag}")


def med(x): return float(np.median(np.asarray(x, dtype=float)))


def pooled(G, t, key, lo=None, hi=None, which="score_grad"):
    out = []
    for _, otag in OBJ:
        for x in G[(t, otag)][which]:
            if lo is None or lo < x["k"] <= hi or (lo == 1 and x["k"] <= hi):
                out.append(x[key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="results/sae_mattr_diag")
    ap.add_argument("--root", default="results/sae_variant_pilot")
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    G, ok_all = {}, True
    print("=== A. INVARIANCE: the diagnostics must not have perturbed training ===")
    for t in TASKS:
        for oflag, otag in OBJ:
            d = geom_dir(args.diag, t, oflag)
            G[(t, otag)] = json.load(open(f"{d}/gradient_geometry.json"))
            o = f"{args.root}/{t}/n200/MAttr_{otag}_s0"
            sa = torch.load(f"{o}/scores.pt", map_location="cpu", weights_only=True)
            sb = torch.load(f"{d}/scores.pt", map_location="cpu", weights_only=True)
            ja, jb = json.load(open(f"{o}/results.json")), json.load(open(f"{d}/results.json"))
            same = (torch.equal(sa, sb) and ja["losses"] == jb["losses"]
                    and ja["metrics"]["learned"]["log_auc"] == jb["metrics"]["learned"]["log_auc"])
            ok_all &= same
            if not same: print(f"  FAIL {t} {otag}")
    print(f"  ALL 12 RUNS BITWISE IDENTICAL (scores, losses, log-AUC): {ok_all}")
    if not ok_all: raise SystemExit("INVARIANCE FAILED -- these records cannot be used")

    swing, mattr_swing = {}, {}
    for t in TASKS:
        for meth, store in (("IxG", swing), ("MAttr", mattr_swing)):
            s = []
            for seed in (0, 1):
                a = json.load(open(f"{args.root}/{t}/n200/{meth}_ld_s{seed}/results.json"))
                b = json.load(open(f"{args.root}/{t}/n200/{meth}_ce_s{seed}/results.json"))
                s.append(a["metrics"]["learned"]["log_auc"] - b["metrics"]["learned"]["log_auc"])
            store[t] = float(np.mean(s))

    stats = {t: {
        "swing": swing[t], "mattr_swing": mattr_swing[t],
        "cos_kgt128": med(pooled(G, t, "score_cos", 128, INF)),
        "cos_all":    med(pooled(G, t, "score_cos", 1, INF)),
        "cos_kle8":   med(pooled(G, t, "score_cos", 1, 8)),
        "out_cos":    med(pooled(G, t, "out_cos", which="per_step")),
        "abs_kgt128": med(pooled(G, t, "abs_cos", 128, INF)),
        "ovl_kgt128": med(pooled(G, t, "top256_overlap", 128, INF)),
    } for t in TASKS}
    order = sorted(TASKS, key=lambda t: -swing[t])

    print("\n=== B. PER-TASK SUMMARY (both trajectories pooled; sorted by I x G swing) ===")
    print(f"  {'task':<22}{'IxG swing':>10}{'MAttr sw':>9}{'cos k>128':>10}{'cos all':>9}"
          f"{'cos k<=8':>9}{'out cos':>9}")
    for t in order:
        s = stats[t]
        print(f"  {t:<22}{s['swing']:>+10.3f}{s['mattr_swing']:>+9.3f}{s['cos_kgt128']:>10.3f}"
              f"{s['cos_all']:>9.3f}{s['cos_kle8']:>9.3f}{s['out_cos']:>9.3f}")

    print("\n=== C. CANDIDATE vs I x G CAUSAL SWING (n=6, descriptive only) ===")
    sw = [stats[t]["swing"] for t in TASKS]
    print(f"  {'candidate':<12}{'rho':>7}{'separates NPI?':>16}{'norm gap':>10}")
    for cand in ["cos_kgt128", "cos_all", "cos_kle8", "out_cos"]:
        v = [stats[t][cand] for t in TASKS]
        sep, ngap = sep_detail(v)
        print(f"  {cand:<12}{spearman(v, sw):>+7.3f}{('YES' if sep else 'no'):>16}{ngap:>10.2f}")
    print(f"  control: cos_kgt128 vs the MATTR swing rho = "
          f"{spearman([stats[t]['cos_kgt128'] for t in TASKS], [stats[t]['mattr_swing'] for t in TASKS]):+.3f}"
          f"   (MAttr swings {min(mattr_swing.values()):+.3f}..{max(mattr_swing.values()):+.3f})")

    print("\n=== D. WHERE the k>128 disagreement lives ===")
    print("  abs_cos compares magnitude profiles only; high abs_cos with low cos means the two")
    print("  objectives load the SAME latents with DISAGREEING signs, not different latents.")
    print(f"  {'task':<22}{'cos':>8}{'abs_cos':>9}{'top256 ovl':>12}")
    for t in order:
        s = stats[t]
        print(f"  {t:<22}{s['cos_kgt128']:>8.3f}{s['abs_kgt128']:>9.3f}{s['ovl_kgt128']:>12.3f}")

    print("\n=== E. SCORE-GRAD COSINE BY k (coarse + fine tail; median n_boundary in brackets) ===")
    allb = BINS + FBINS
    print(f"  {'task':<22}" + "".join(f"{n:>15}" for n, _, _ in allb))
    for t in order:
        row = ""
        for _, lo, hi in allb:
            c = med(pooled(G, t, "score_cos", lo, hi))
            nb = int(med(pooled(G, t, "n_boundary", lo, hi)))
            row += f"{c:>9.3f}[{nb:>4d}]"
        print(f"  {t:<22}{row}")
    print("  CAVEAT: at k<=8 the mask leaves only ~2-3 coordinates undecided, so near-perfect")
    print("  agreement there is partly forced by dimensionality, not evidence about the task.")

    if args.figure:
        make_figure(G, stats, order, args.figure)


def make_figure(G, stats, order, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":6.2,
                         "xtick.labelsize":7,"ytick.labelsize":7,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    allb = BINS[:3] + FBINS
    mids = [math.sqrt(max(lo,1)*(hi if hi != INF else 147465)) for _, lo, hi in allb]

    ax = axes[0]
    ax.axhspan(0.60, 0.74, color="#999999", alpha=0.18, zorder=0)
    ax.text(1.35, 0.665, "output-gradient cosine: uniform on every task\n(masked points and base point alike)",
            fontsize=5.8, color="#555555", va="center")
    for t, c in zip(TASKS, COL):
        y = [np.median(pooled(G, t, "score_cos", lo, hi)) for _, lo, hi in allb]
        ax.plot(mids, y, "-o", color=c, lw=1.2, ms=2.6,
                label=t + ("  (NPI)" if t in NPI else ""))
    ax.axhline(0, lw=0.8, color="#666666", ls=":")
    ax.axvspan(98310, 147465, color="#e41a1c", alpha=0.10, zorder=0)
    ax.annotate("I$\\times$G differentiates here\n($m{=}1$; $k{=}$total, 98k-147k)", xy=(120000, 0.30),
                xytext=(7500, 0.11), fontsize=6.2, color="#a94442", ha="center",
                linespacing=1.15, arrowprops=dict(arrowstyle="->", color="#a94442", lw=0.8))
    ax.set_xscale("log"); ax.set_ylim(-0.12, 1.05); ax.set_xlim(1.3, 2.6e5)
    ax.set_xlabel("states visited during MAttr training: top-$k$ kept clean (log-uniform $k$)")
    ax.set_ylabel(r"median cosine$\left(\partial L_{\mathrm{CE}}/\partial S,\ \partial L_{\mathrm{LD}}/\partial S\right)$")
    ax.set_title("(a) the two objectives' SCORE gradients agree at small $k$\nand decorrelate toward the clean model",
                 fontsize=7.5, loc="left")
    ax.legend(loc="lower left", frameon=False, handlelength=1.4)
    ax.grid(True, axis="y", lw=0.25, color="#dddddd"); ax.set_axisbelow(True)

    ax = axes[1]
    for t, c in zip(TASKS, COL):
        ax.scatter(stats[t]["cos_kgt128"], stats[t]["swing"], color=c, s=22, zorder=3)
        dy = 0.013 if t != "agr_sv_num_subj-relc" else -0.030
        ax.annotate(t, (stats[t]["cos_kgt128"], stats[t]["swing"]),
                    xytext=(stats[t]["cos_kgt128"], stats[t]["swing"] + dy),
                    fontsize=5.6, ha="center", color="#333333")
    v = [stats[t]["cos_kgt128"] for t in TASKS]; sw = [stats[t]["swing"] for t in TASKS]
    rho = spearman(v, sw)
    ax.set_xlabel(r"median score-gradient cosine at $k>128$")
    ax.set_ylabel(r"I$\times$G causal swing (log-AUC, LD$-$CE)")
    ax.set_title(f"(b) tracks the swing ($\\rho={rho:+.2f}$, $n=6$, descriptive)\nbut does NOT separate the NPI pair",
                 fontsize=7.5, loc="left")
    ax.set_xlim(-0.06, 0.68); ax.set_ylim(-0.02, 0.45)
    ax.grid(True, lw=0.25, color="#dddddd"); ax.set_axisbelow(True)

    fig.suptitle(r"CE-vs-logit-diff gradient geometry at MAttr's masked operating points "
                 r"(gemma-2-2b L12, 6 tasks $\times$ 2 trajectories $\times$ 401 states)",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
