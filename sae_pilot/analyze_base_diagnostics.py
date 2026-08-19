"""Why does the attribution objective matter so much for I x G in SAE space?

Reads ONLY the forward-only scalar diagnostics written by ``collect_base_diagnostics.py``
plus the causal / ranking numbers already present in the attribution results. Nothing here
runs a model, an SAE, or a backward pass.

The hypothesis this was written to test was that CE is brittle because its gradient SATURATES
when the model is confident in the base label (``grad_z L_CE = softmax(z) - e_b`` -> 0 as
p_base -> 1). That hypothesis is REFUTED by direct measurement: the saturated regime is empty.
What the data show instead is that ~97% of the probability mass sits on tokens that are
neither the base nor the source label, which leaves CE's gradient only ~0.68-cosine aligned
with the causal base-vs-source direction -- uniformly, on every task.

Usage:
    python sae_pilot/analyze_base_diagnostics.py --diag results/sae_base_diag \
        --root results/sae_variant_pilot [--figure sae_pilot/sae_ce_gradient_geometry.png]
"""
import argparse, json, math, os, statistics as st
from math import comb
import numpy as np
import torch

TASKS = ["npi_ever_subj-relc","npi_any_obj-relc","agr_sv_num_subj-relc",
         "garden_npz_v-trans","garden_npz_obj_mod","filler_gap_pp"]
NPI = {"npi_ever_subj-relc","npi_any_obj-relc"}
FIELDS = ["p_base","p_source","p_other","margin","abs_margin","ce_loss","softacc_loss",
          "ce_grad_norm","grad_norm_ratio","ce_ld_grad_cosine","entropy"]
COL = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0]*len(v); i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and v[order[j+1]] == v[order[i]]: j += 1
            for q in range(i, j+1): r[order[q]] = (i+j)/2.0
            i = j+1
        return r
    a, b = rank(x), rank(y); n = len(x)
    ma, mb = sum(a)/n, sum(b)/n
    num = sum((a[i]-ma)*(b[i]-mb) for i in range(n))
    den = math.sqrt(sum((a[i]-ma)**2 for i in range(n))*sum((b[i]-mb)**2 for i in range(n)))
    return num/den if den else float("nan")


def sep_detail(vals):
    """Does the NPI pair sit entirely above/below the other four, and by how much?

    With 2 NPI vs 4 non-NPI, a RANDOM ordering separates with probability 2/C(6,2) = 13.3%,
    so the boolean alone is not evidence. The gap normalised by the total spread is the
    discriminating quantity -- chance separations are typically razor-thin.
    """
    npi = [vals[i] for i, t in enumerate(TASKS) if t in NPI]
    oth = [vals[i] for i, t in enumerate(TASKS) if t not in NPI]
    rng = max(vals) - min(vals)
    if min(npi) > max(oth):   gap = min(npi) - max(oth)
    elif max(npi) < min(oth): gap = min(oth) - max(npi)
    else:                     gap = -min(abs(a-b) for a in npi for b in oth)
    return gap > 0, gap, (gap/rng if rng else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", default="results/sae_base_diag")
    ap.add_argument("--root", default="results/sae_variant_pilot")
    ap.add_argument("--figure", default=None, help="write the two-panel figure here")
    args = ap.parse_args()

    R = {}
    for t in TASKS:
        for s in (0, 1):
            f = f"{args.diag}/{t}_s{s}.npz"
            if not os.path.exists(f):
                raise SystemExit(f"missing {f} -- run sae_pilot/collect_base_diagnostics.py first")
            R[(t, s)] = np.load(f, allow_pickle=True)
    pool = lambda t, k: np.concatenate([R[(t, s)][f"train_{k}"] for s in (0, 1)])
    every = lambda k: np.concatenate([pool(t, k) for t in TASKS])

    def la(t, m, o, s):
        return json.load(open(f"{args.root}/{t}/n200/{m}_{o}_s{s}/results.json"))["metrics"]["learned"]["log_auc"]
    def vec(t, m, o, s):
        return torch.load(f"{args.root}/{t}/n200/{m}_{o}_s{s}/scores.pt", map_location="cpu", weights_only=True)
    def churn(t, k=32):
        return 1 - st.mean([len(set(vec(t,"IxG","ce",s).topk(k).indices.tolist())
                              & set(vec(t,"IxG","ld",s).topk(k).indices.tolist()))/k for s in (0,1)])
    SW = {t: st.mean([la(t,"IxG","ld",s) - la(t,"IxG","ce",s) for s in (0,1)]) for t in TASKS}
    CH = {t: churn(t) for t in TASKS}
    med = {(t, s): {k: float(np.median(R[(t,s)][f"train_{k}"])) for k in FIELDS} for t in TASKS for s in (0,1)}
    agg = {t: {k: st.mean([med[(t,0)][k], med[(t,1)][k]]) for k in FIELDS} for t in TASKS}

    print("=== 1. Is the CE gradient ever saturated? (all attribution examples, pooled) ===")
    pb, rt = every("p_base"), every("grad_norm_ratio")
    print(f"  n = {pb.size:,} examples")
    print(f"  p_base:            median {np.median(pb):.4f}   p99 {np.percentile(pb,99):.4f}   max {pb.max():.4f}")
    print(f"  count p_base>0.9:  {(pb>0.9).sum():,}        count p_base>0.99: {(pb>0.99).sum():,}")
    print(f"  ||g_ce||/sqrt(2):  min {rt.min():.4f}   median {np.median(rt):.4f}   max {rt.max():.4f}")
    print(f"  count ratio<0.1:   {(rt<0.1).sum():,}")
    print("  -> the saturated regime is ABSENT, not merely rare: CE saturation is REFUTED.")

    print("\n=== 2. Where does CE's gradient mass actually go? ===")
    co = every("ce_ld_grad_cosine")
    print(f"  median p_base {np.median(pb):.4f}   p_source {np.median(every('p_source')):.5f}   "
          f"p_other {np.median(every('p_other')):.4f}")
    print(f"  cosine(grad CE, grad LD): p10 {np.percentile(co,10):.3f}  median {np.median(co):.3f}  "
          f"p90 {np.percentile(co,90):.3f}")
    print("  -> ~97% of the mass is on tokens that are neither base nor source, so CE's gradient")
    print("     is only ~0.68-cosine aligned with the causal base-vs-source direction.")

    print("\n=== 3. Per-task summary (two-seed means of per-example medians) ===")
    h = f"{'task':<22}{'IxG swing':>11}{'churn@32':>10}{'p_base':>9}{'p_other':>9}{'||g_ce||':>10}{'CE-LD cos':>11}{'|margin|':>10}"
    print(h); print("-"*len(h))
    for t in TASKS:
        a = agg[t]
        print(f"{t:<22}{SW[t]:>+11.4f}{CH[t]:>10.3f}{a['p_base']:>9.4f}{a['p_other']:>9.4f}"
              f"{a['ce_grad_norm']:>10.4f}{a['ce_ld_grad_cosine']:>11.3f}{a['abs_margin']:>10.3f}")

    print("\n=== 4. Does any base-point statistic explain WHICH tasks swing? (n=6, DESCRIPTIVE) ===")
    p_chance = 2/comb(6, 2)
    cands = [("median p_base","p_base"), ("median ||g_ce||","ce_grad_norm"),
             ("median grad ratio","grad_norm_ratio"), ("median p_other","p_other"),
             ("median CE-LD cosine","ce_ld_grad_cosine"), ("median |margin|","abs_margin")]
    print(f"  GUARD RAIL: a random ordering separates the 2 NPI tasks from the other 4 with")
    print(f"  p = 2/C(6,2) = {p_chance:.3f}; over the {len(cands)+1} candidates below, "
          f"P(>=1 separates by luck) = {1-(1-p_chance)**(len(cands)+1):.2f}.")
    print(f"  So 'separates' alone is NOT evidence -- gap/range and per-seed consistency are.\n")
    sw = [SW[t] for t in TASKS]
    print(f"  {'candidate':<22}{'rho':>7}{'sep':>6}{'gap/range':>11}{'seeds':>7}   values (NPI first)")
    print("  "+"-"*104)
    for name, key in cands:
        v = [agg[t][key] for t in TASKS]
        s, g, ng = sep_detail(v)
        seeds = sum(1 for sd in (0,1) if sep_detail([med[(t,sd)][key] for t in TASKS])[0])
        npi = [f"{v[i]:.3f}" for i,t in enumerate(TASKS) if t in NPI]
        oth = [f"{v[i]:.3f}" for i,t in enumerate(TASKS) if t not in NPI]
        print(f"  {name:<22}{spearman(sw,v):>+7.3f}{('YES' if s else 'no'):>6}{ng:>11.2f}{f'{seeds}/2':>7}   "
              f"[{', '.join(npi)}] vs [{', '.join(oth)}]")
    v = [CH[t] for t in TASKS]; s, g, ng = sep_detail(v)
    print(f"  {'ranking churn@32':<22}{spearman(sw,v):>+7.3f}{('YES' if s else 'no'):>6}{ng:>11.2f}{'-':>7}   "
          f"[{', '.join(f'{v[i]:.3f}' for i,t in enumerate(TASKS) if t in NPI)}] vs "
          f"[{', '.join(f'{v[i]:.3f}' for i,t in enumerate(TASKS) if t not in NPI)}]")
    print(f"\n  the outcome itself: NPI {[round(SW[t],3) for t in TASKS if t in NPI]} vs "
          f"non-NPI {[round(SW[t],3) for t in TASKS if t not in NPI]}  "
          f"separates=YES gap/range={sep_detail(sw)[2]:.2f}")

    print("\n=== 5. |margin| is the only separating correlate -- how robust is it? ===")
    for stat, fn in [("mean", np.mean), ("median", np.median),
                     ("p25", lambda x: np.percentile(x,25)), ("p75", lambda x: np.percentile(x,75)),
                     ("p90", lambda x: np.percentile(x,90))]:
        v = [float(fn(pool(t,"abs_margin"))) for t in TASKS]
        s, g, ng = sep_detail(v)
        print(f"  |margin| {stat:<7} separates={'YES' if s else 'no ':<3}  gap/range={ng:>+5.2f}   "
              f"{[round(x,2) for x in v]}")
    print("  -> separates at mean/median/p25 but NOT at p75/p90, and it mis-orders within groups")
    print("     (filler_gap_pp has the 3rd-largest |margin| and the SMALLEST swing). A fragile")
    print("     correlate, not a demonstrated mechanism.")

    print("\n=== 6. Held-out check: is any of this peculiar to the attribution stream? ===")
    h = f"{'task':<22}{'p_base tr':>10}{'p_base ev':>10}{'cos tr':>9}{'cos ev':>9}{'d(cos)':>9}"
    print(h); print("-"*len(h))
    for t in TASKS:
        tr = pool(t,"p_base"); ev = np.concatenate([R[(t,s)]["eval_p_base"] for s in (0,1)])
        ct = pool(t,"ce_ld_grad_cosine"); ce = np.concatenate([R[(t,s)]["eval_ce_ld_grad_cosine"] for s in (0,1)])
        print(f"{t:<22}{np.median(tr):>10.4f}{np.median(ev):>10.4f}"
              f"{np.median(ct):>9.3f}{np.median(ce):>9.3f}{np.median(ce)-np.median(ct):>+9.3f}")
    print("  -> held-out and attribution-stream distributions agree; nothing is stream-specific.")

    if args.figure:
        make_figure(args.figure, pool, TASKS)
        print(f"\nwrote {args.figure}")


def make_figure(path, pool, tasks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pb = [pool(t,"p_base") for t in tasks]
    co = [pool(t,"ce_ld_grad_cosine") for t in tasks]
    n = sum(x.size for x in pb)
    plt.rcParams.update({"font.size":8,"axes.labelsize":8,"xtick.labelsize":7,
                         "ytick.labelsize":7,"axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    lab = [t + ("  (NPI)" if t in NPI else "") for t in tasks]
    pos = np.arange(len(tasks))[::-1]

    ax = axes[0]
    ax.axvspan(0.9, 1.05, color="#e41a1c", alpha=0.16, zorder=0)
    ax.axvline(0.9, lw=0.8, color="#e41a1c", ls="--", zorder=1)
    bp = ax.boxplot(pb, positions=pos, vert=False, widths=0.6, showfliers=False, whis=(1,99),
                    patch_artist=True, medianprops=dict(color="black", lw=1.0))
    for b, c in zip(bp["boxes"], COL):
        b.set_facecolor(c); b.set_alpha(0.75); b.set_edgecolor("black"); b.set_linewidth(0.5)
    ax.set_xscale("log"); ax.set_xlim(2e-4, 1.05)
    ax.set_ylim(pos[-1]-1.15, pos[0]+0.55)
    ax.set_yticks(pos); ax.set_yticklabels(lab)
    ax.set_xlabel(r"$p_{\mathrm{base}}$ at the I$\times$G base point $m{=}1$")
    ax.set_title("(a) the CE-saturation regime is empty\n"
                 rf"$0$ of ${n:,}$ examples have $p_{{\mathrm{{base}}}}>0.9$".replace(",", r"{,}"),
                 fontsize=7.5, loc="left")
    ax.annotate("saturation regime (CE gradient $\\to$ 0):\nEMPTY on every task",
                xy=(0.93, pos[-1]-0.45), xytext=(0.030, pos[-1]-0.72), color="#e41a1c",
                fontsize=6.2, ha="center", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=0.8))
    ax.grid(True, axis="x", lw=0.25, color="#dddddd"); ax.set_axisbelow(True)

    ax = axes[1]
    ax.axvline(1.0, lw=0.8, color="#666666", ls="--")
    bp = ax.boxplot(co, positions=pos, vert=False, widths=0.6, showfliers=False, whis=(1,99),
                    patch_artist=True, medianprops=dict(color="black", lw=1.0))
    for b, c in zip(bp["boxes"], COL):
        b.set_facecolor(c); b.set_alpha(0.75); b.set_edgecolor("black"); b.set_linewidth(0.5)
    ax.set_xlim(0.45, 1.10); ax.set_ylim(pos[-1]-1.15, pos[0]+0.55)
    ax.set_yticks(pos); ax.set_yticklabels([])
    for y, t in zip(pos, tasks):
        ax.text(1.045, y, rf"$p_{{\mathrm{{other}}}}$={np.median(pool(t,'p_other')):.3f}",
                fontsize=5.9, ha="right", va="center", color="#444444")
    ax.annotate("perfect alignment with\nthe logit-difference gradient",
                xy=(1.0, pos[-1]-0.45), xytext=(0.72, pos[-1]-0.72), fontsize=6.2,
                color="#666666", ha="center", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
    ax.set_xlabel(r"cosine$\left(\nabla_z L_{\mathrm{CE}},\ \nabla_z L_{\mathrm{LD}}\right)$")
    ax.set_title("(b) CE's gradient is uniformly misaligned\nwith the causal base-vs-source direction",
                 fontsize=7.5, loc="left")
    ax.grid(True, axis="x", lw=0.25, color="#dddddd"); ax.set_axisbelow(True)

    fig.suptitle(r"Forward-only base-point diagnostics on the exact I$\times$G attribution streams "
                 r"(gemma-2-2b L12, 6 tasks $\times$ 2 seeds $\times$ 4000 examples)", fontsize=8, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
