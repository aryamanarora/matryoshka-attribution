"""Figures for the "why does Adam lose to SGD at MLP-neuron scale" investigation.

ONE cell throughout: addition / llama3 / --nodes mlp, 2,293,760 neurons, sufficient
(denoising), bs=1, log-k, 2000 steps unless a panel says otherwise. Sources:

  results/sva_sweep/addition_llama3_mlp_*        the pre-existing 51-run loss x gate x optimizer
                                                 grid + the IG / IxG / AttnLRP baselines
  results/sva_mlp_lr/{topk_adam,topk_sgd}/lr_*   the two lr brackets
  results/sva_mlp_steps20k/*/lr_*                the 20k step-matched pair
  results/adamsgd_mlp/{A_eps,B_loss,C_fixedk,    this investigation's arms
                       D_perexample,E_steplessig}

Every figure reads only from disk; safe to re-run. `uv run python plots/plot_adamsgd_mlp_diag.py`
writes PDFs (for LaTeX) and PNGs (for the HTML report) into plots/.

Raw matplotlib rather than plotnine throughout: fig 1/4/6 need twin reference lines and
hand-placed annotations, fig 3 needs per-path arrowheads, and fig 5 is a labelled heatmap --
none of which the grammar expresses without contortion.
"""

import json
import glob
import os

import torch   # rho panel: reads the score tensors, not just the run jsons
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from palette import METHOD, OTHER

plt.rcParams.update({
    "font.family": "Inter",
    "mathtext.fontset": "custom", "mathtext.rm": "Inter",
    "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.cal": "Inter:italic", "mathtext.sf": "Inter", "mathtext.tt": "Inter",
    "pdf.fonttype": 42,
    "text.color": "#000000", "axes.labelcolor": "#000000",
    "xtick.color": "#000000", "ytick.color": "#000000",
    "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.linewidth": 0.5,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOTAL = 2293760
SPARS = np.array(sorted(set(float(10 ** x) for x in np.linspace(np.log10(1.0 / TOTAL), 0.0, 24))))
KS = SPARS * TOTAL
LX = np.log10(KS)
CLEAN_MARGIN = 6.68           # F_clean on this cell; the level faithfulness = 1 corresponds to

C_ADAM = METHOD["MAttr"]          # blue
C_SGD = METHOD["MAttr (SGD)"]     # black
C_IG = METHOD["IG"]               # orange
C_SIG = METHOD["Expected Gradients"]     # sand
C_FIX = METHOD["+hard"]           # bluish green -- the "intervention" colour


def style(ax):
    ax.grid(True, lw=0.25, color="#dddddd")
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "plots", f"{name}.{ext}"), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def rd(pattern):
    g = sorted(glob.glob(os.path.join(ROOT, pattern)))
    return json.load(open(g[0])) if g else None


def cpr_of(d):
    """MIB-style CPR from a run json: LINEAR trapezoid of faithfulness over the kept proportion
    p = n_nodes / total (same rule as plot_accauc_vs_faithauc._cpr_of, duplicated here so this
    loader does not import the scatter module). CPR is always this linear AUC (2026-09-17)."""
    p = np.asarray(d["n_nodes"], float) / float(d["total"])
    f = np.asarray(d["faithfulness"], float)
    return float(np.sum((p[1:] - p[:-1]) * (f[1:] + f[:-1]) / 2))


def metric_of(d, key):
    """`key` of a run json, with "cpr" computed from the curve rather than read."""
    return cpr_of(d) if key == "cpr" else d[key]


def auc_of(y):
    y = np.asarray(y, float)
    return float(np.sum((LX[1:] - LX[:-1]) * (y[1:] + y[:-1]) / 2) / (LX[-1] - LX[0]))


def tail_loss(d):
    """Mean training loss over the last quarter of steps. k is drawn log-uniformly on [1, N]
    every step and the eval grid is log-uniform on [1, N] too, so this estimates exactly the
    integral the faith-AUC column reports -- the two ARE the same quantity up to the affine
    faithfulness rescaling. Comparable across runs only within one --loss."""
    ll = d.get("loss_log") or []
    return float(np.mean(ll[int(0.75 * len(ll)):])) if ll else np.nan


# ---------------------------------------------------------------- reference runs
REF = {
    "IG": "results/sva_sweep/addition_llama3_mlp_ig.json",
    "Expected Gradients": "results/adamsgd_mlp/E_steplessig/addition_llama3_mlp_mc_ig_m1_s42.json",
    "MAttr+Adam": "results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.json",
    "MAttr+SGD": "results/sva_mlp_lr/topk_sgd/lr_1.0/addition_llama3_mlp_sufficient_topk_sgd_bs1.json",
}
COL = {"IG": C_IG, "Expected Gradients": C_SIG, "MAttr+Adam": C_ADAM, "MAttr+SGD": C_SGD}


# =================================================================== FIG 1
def fig_curves():
    """The mismatch in one figure: at the SAME sparsity, Adam has the larger mean margin and
    the smaller fraction of decided examples. Left panel is (a monotone function of) the
    TRAINING objective, right panel is the exam."""
    runs = {k: rd(v) for k, v in REF.items()}
    runs = {k: v for k, v in runs.items() if v}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05))
    for ax, key, lab in ((axes[0], "logit_diff", "Mean base–source margin (logits)"),
                         (axes[1], "acc_base", "Fraction of examples decided")):
        for name, d in runs.items():
            y = np.array(d["iso_metrics"][key])
            ax.plot(KS, y, lw=1.1, color=COL[name], label=name,
                    marker="o", ms=1.8, mew=0)
        ax.set_xscale("log")
        ax.set_xlabel("Neurons kept clean, $k$")
        ax.set_ylabel(lab)
        style(ax)
        ax.set_xticks([1e0, 1e2, 1e4, 1e6])
        ax.set_xticklabels(["10⁰", "10²", "10⁴", "10⁶"])
    axes[0].axhline(CLEAN_MARGIN, color="#888888", lw=0.6, ls=(0, (3, 2)))
    axes[0].annotate("full (unmasked) model", xy=(1.4, CLEAN_MARGIN + 1.6), fontsize=5.5,
                     color="#666666")
    # the window where the exam actually discriminates
    for ax in axes:
        ax.axvspan(1e3, 1e5, color="#f2f2f2", zorder=0)
    axes[1].annotate("where the exam\ndiscriminates", xy=(1.1e4, 0.26), fontsize=5.5,
                     color="#666666", ha="center", va="top")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.4,
                   borderpad=0.1, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "adamsgd_curves")


# =================================================================== FIG 2
def collect_cell():
    """Every MAttr run on this cell that logged a training loss, plus the gradient baselines."""
    rows = []
    pats = ["results/sva_sweep/addition_llama3_mlp_sufficient_*.json",
            "results/sva_mlp_lr/*/lr_*/*.json",
            "results/sva_mlp_steps20k/*/lr_*/*.json",
            "results/adamsgd_mlp/A_eps/*/*.json",
            "results/adamsgd_mlp/B_loss/*/*.json",
            "results/adamsgd_mlp/C_fixedk/*/*.json",
            "results/adamsgd_mlp/J_loss_eps/*/*.json",
            "results/adamsgd_mlp/K_match/*/*.json",
            "results/adamsgd_mlp/L_momentum/*/*.json"]
    for pat in pats:
        for f in sorted(glob.glob(os.path.join(ROOT, pat))):
            d = json.load(open(f))
            if "acc_auc" not in d or not (d.get("loss_log") or []):
                continue
            cfg = d.get("config", {})
            rows.append(dict(f=os.path.relpath(f, ROOT), acc=d["acc_auc"], faith=d["faith_auc"],
                             fmax=d["faith_max"], tl=tail_loss(d), loss=d.get("loss"),
                             opt=d.get("optimizer"), var=d.get("variant"),
                             ks=d.get("k_schedule"), steps=len(d["loss_log"]),
                             lr=cfg.get("lr"), eps=cfg.get("adam_eps"),
                             kfrac=cfg.get("fixed_k_frac"), ld_scale=cfg.get("ld_scale")))
    return rows


def fig_scatters():
    """Three ways of asking "is the exam measuring what training optimises". (a) The two exam
    metrics we actually report disagree. (b) They stop disagreeing once faithfulness is clipped
    at 1, i.e. once over-recovery stops counting as extra credit -- so ALL the disagreement is
    over-recovery. (c) Descending the training objective further does not buy exam score.
    Panel (c) is restricted to runs whose loss is the same quantity (logit_diff, log-k,
    sampled k), since a tanh/hinge loss value is not on the same scale."""
    from scipy.stats import spearmanr
    rows = collect_cell()
    for r in rows:
        d = json.load(open(os.path.join(ROOT, r["f"])))
        fa = np.array(d["iso_metrics"]["faithfulness"])
        r["fclip"] = auc_of(np.minimum(fa, 1.0))
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))

    def scat(ax, xk, xlab, sub, subset=None):
        pts = subset if subset is not None else rows
        for opt, mk in (("adam", "o"), ("sgd", "s")):
            r = [x for x in pts if x["opt"] == opt]
            ax.scatter([x[xk] for x in r], [x["acc"] for x in r], s=6,
                       c=(C_ADAM if opt == "adam" else C_SGD), marker=mk, lw=0, alpha=0.8,
                       label=f"MAttr+{'Adam' if opt == 'adam' else 'SGD'}")
        rho = spearmanr([x[xk] for x in pts], [x["acc"] for x in pts]).statistic
        ax.annotate(f"$\\rho$ = {rho:+.2f}", xy=(0.96, 0.96), xycoords="axes fraction",
                    ha="right", va="top", fontsize=6.5)
        ax.set_xlabel(xlab)
        ax.annotate(sub, xy=(0.03, 0.96), xycoords="axes fraction", va="top", fontsize=6.5,
                    fontweight="bold")
        style(ax)

    scat(axes[0], "faith", "Faithfulness AUC", "a")
    scat(axes[1], "fclip", "Faithfulness AUC, clipped at 1", "b")
    # panel (c) only over runs whose loss VALUE is the same quantity -- a tanh/hinge/prob
    # loss is on a different scale and would sit at an unrelated x with no meaning.
    ld = [x for x in rows if x["loss"] == "logit_diff" and x["ks"] == "log" and x["kfrac"] is None]
    scat(axes[2], "tl", "Training loss reached (logit_diff runs)", "c", subset=ld)
    axes[2].invert_xaxis()
    axes[0].set_ylabel("Accuracy AUC (the exam)")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    lo = min(x["acc"] for x in rows) - 0.02
    hi = max(x["acc"] for x in rows) + 0.02
    for ax in axes:
        ax.set_ylim(lo, hi)
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower right", handlelength=1.0,
                   borderpad=0.1, labelspacing=0.2, scatterpoints=1,
                   bbox_to_anchor=(1.02, 0.10))
    fig.subplots_adjust(wspace=0.08)
    save(fig, "adamsgd_scatters")
    return rows


# =================================================================== FIG 3
def fig_trajectories():
    """Is the 2000-step gap just under-training? The 20k step-matched triple, read off the
    training-time probe (64 fixed TRAIN examples, every 500 steps). Adam is above SGD on the
    training objective (right) from step ~2k onward and never catches it on the exam (left)."""
    picks = [
        ("MAttr+Adam lr=0.05", "results/sva_mlp_steps20k/topk_adam/lr_0.05/*.json", C_ADAM, "-"),
        ("MAttr+Adam lr=0.005", "results/sva_mlp_steps20k/topk_adam/lr_0.005/*.json", C_ADAM, (0, (3, 2))),
        ("MAttr+SGD lr=1", "results/sva_mlp_steps20k/topk_sgd/lr_1.0/*.json", C_SGD, "-"),
    ]
    have = [(l, rd(p_), c, ls) for l, p_, c, ls in picks]
    have = [(l, d, c, ls) for l, d, c, ls in have if d and d.get("train_eval_log")]
    if not have:
        print("skip adamsgd_traj (no train_eval_log)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95))
    for ax, key, ylab in ((axes[0], "acc_auc", "Accuracy AUC (the exam)"),
                          (axes[1], "faith_auc", "Faithfulness AUC (≈ the objective)")):
        for lab, d, col, ls in have:
            tl = [r for r in d["train_eval_log"] if r["step"] > 0]
            ax.plot([r["step"] for r in tl], [r[key] for r in tl], lw=1.0, color=col, ls=ls,
                    label=lab)
        ax.set_xscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylab)
        ax.axvline(2000, color="#888888", lw=0.6, ls=(0, (1, 2)))
        style(ax)
    axes[1].axhline(1.0, color="#888888", lw=0.6, ls=(0, (3, 2)))
    axes[1].annotate("full-model recovery", xy=(0.03, 1.06), xycoords=("axes fraction", "data"),
                     fontsize=5.5, color="#666666")
    axes[0].annotate("the 2k budget\nevery other run uses", xy=(1750, 0.05), fontsize=5.5,
                     color="#666666", ha="right")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left", handlelength=1.4,
                   borderpad=0.1, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.28)
    save(fig, "adamsgd_traj")


# =================================================================== FIG 4

def _recall_auc(spath, gt_layer, gt_neurons):
    """Log-AUC of the published-neuron recall curve for one MLP-substrate run.

    Same ranking rule as plots/plot_neuron_recall.py -- max over positions, then rank all L*N
    neurons -- so a cell of the heatmap and a curve of that figure are the same measurement. The
    AUC is taken over LOG k because the curve is read on a log axis and the interesting region is
    the first few decades; a linear AUC would be almost entirely determined by the flat tail.
    """
    m = json.load(open(spath.replace(".scores.pt", ".json")))
    m = m.get("meta", m)
    L, P, N = m["num_layers"], m["seq_len"], m["intermediate_size"]
    sc = torch.load(spath, map_location="cpu").float()
    best = sc.view(L, P, N).max(dim=1).values.reshape(-1)
    order = torch.argsort(best, descending=True).numpy()
    ranks = np.flatnonzero((order // N == gt_layer) & np.isin(order % N, gt_neurons)) + 1
    ks = np.unique(np.round(np.logspace(0, np.log10(L * N), 240)).astype(int))
    rec = np.searchsorted(ranks, ks, side="right") / len(gt_neurons)
    lk = np.log10(ks)
    return float(np.trapezoid(rec, lk) / (lk[-1] - lk[0]))


def eps_grid_matrix(res, key, refs=None, epss=None, lrs=None, gt=None):
    """(len(epss), len(lrs)) matrix of `key` over an eps x lr results tree, NaN where missing.

    EXTRACTED FROM fig_eps_grid so plots/plot_epsgrid_facets.py can build the same numbers
    without a second copy of the three loading rules (json key / rank correlation against a
    reference .scores.pt / published-neuron recall). Panels that disagree about what a cell
    contains would be worse than no faceted figure at all.

    `refs` maps a rho key to a repo-relative reference .scores.pt; returns None if that
    reference is absent, so a caller can skip the panel rather than draw an empty one.
    `gt` is (layer, neurons) for key == "recall_auc".
    """
    epss = epss or ["1e-8", "1e-6", "1e-4", "1e-2", "1e-1", "1e0"]
    lrs = lrs or ["0.005", "0.05", "0.5", "5.0"]
    refs = refs or {}
    ref_ranks = None
    if key in refs:
        from scipy.stats import rankdata
        rp = os.path.join(ROOT, refs[key])
        if not os.path.exists(rp):
            return None
        ref_ranks = rankdata(torch.load(rp, map_location="cpu").float().numpy())
    M = np.full((len(epss), len(lrs)), np.nan)
    for i, e in enumerate(epss):
        for j, lr in enumerate(lrs):
            if key == "recall_auc":
                g = sorted(glob.glob(os.path.join(ROOT, f"{res}/eps_{e}_lr_{lr}/*.scores.pt")))
                if g:
                    M[i, j] = _recall_auc(g[0], gt[0], gt[1])
                continue
            if key == "score_std":
                # Spread of the LEARNED SCORE VECTOR itself, not a benchmark metric -- the one
                # column here that measures the optimiser's output rather than its consequences.
                # float64: at 5.24M elements an fp32 reduction of a vector whose magnitudes span
                # orders of magnitude loses digits (see the .norm() overflow noted in trainer.py).
                g = sorted(glob.glob(os.path.join(ROOT, f"{res}/eps_{e}_lr_{lr}/*.scores.pt")))
                if g:
                    M[i, j] = float(torch.load(g[0], map_location="cpu").double().std())
                continue
            if key in refs:
                from scipy.stats import rankdata
                g = sorted(glob.glob(os.path.join(ROOT, f"{res}/eps_{e}_lr_{lr}/*.scores.pt")))
                if not g:
                    continue
                sc = torch.load(g[0], map_location="cpu").float().numpy()
                if len(sc) != len(ref_ranks):   # different substrate/seq_len: not comparable
                    continue
                M[i, j] = float(np.corrcoef(rankdata(sc), ref_ranks)[0, 1])
                continue
            d = rd(f"{res}/eps_{e}_lr_{lr}/*.json")
            if d:
                M[i, j] = metric_of(d, key)
    return M


def fig_eps_grid(res="results/adamsgd_mlp/A_eps", refs=None, specs=None,
                 epss=None, lrs=None):
    """Arm A: does Adam's eps -- the knob that decides whether the update keeps effect
    MAGNITUDE or degenerates to sign(g) -- move the exam score? Rendered twice: once for the
    exam (accuracy AUC, sequential colormap, higher = better) and once for faithfulness AUC
    (diverging colormap centred at 1 = exact full-model recovery: below 1 under-recovers,
    above 1 over-recovers -- higher is NOT better).

    PARAMETERISED SO OTHER SUBSTRATES REUSE IT RATHER THAN COPY IT (2026-09-02). The SAE
    version of this row (the old plots/plot_sae_epsgrid.py) started life as a second implementation
    of the same 130 lines; two copies of a figure this fiddly drift on the first edit, and the
    whole point of putting the two rows on facing pages is that they are drawn identically.
    `res` is the results tree, `refs` maps a rho panel key to its reference .scores.pt
    (repo-relative), `specs` is the panel list; all three default to the MLP behaviour, so
    calling fig_eps_grid() with no arguments is byte-identical to before."""
    epss = epss or ["1e-8", "1e-6", "1e-4", "1e-2", "1e-1", "1e0"]
    lrs = lrs or ["0.005", "0.05", "0.5", "5.0"]
    # Last field is the canvas WIDTH. The two differ because the colorbar label does: after
    # bbox_inches="tight", "Faithfulness AUC" trims 0.05in wider than "Accuracy AUC" at the same
    # figsize, and two 0.48\textwidth subfigures with unequal bboxes print at different point
    # sizes. Tuned so both land at ~2.77in, not so both share a figsize.
    specs = specs or [("acc_auc", "adamsgd_epsgrid", "Accuracy AUC", "viridis", 0.0, 0.55, None, 1.10, 1.20, (5.5, 5.0, 5.0)),
             ("faith_auc", "adamsgd_epsgrid_faith", "Faithfulness AUC", "RdBu_r", 0.0, 2.0, 1.0, 1.055, 1.20, (5.5, 5.0, 5.0)),
             # Third panel: agreement with IG's RANKING rather than a benchmark score. Computed
             # from the .scores.pt, not read from the json -- see the rho branch below.
             ("rho_ig", "adamsgd_epsgrid_rho", "Spearman $\\rho$ vs. IG", "viridis",
              0.0, 0.30, None, 1.055, 1.20, (5.5, 5.0, 5.0)),
             # Same construction against MAttr+SGD (log k, lr 1.0) instead of IG. The pair is the
             # point: SGD is the SAME METHOD on the same mask path, IG is a different path
             # (input-embedding), so the two panels separate "does eps make Adam agree with the
             # other optimiser" from "...with a gradient attribution". vmax differs because the
             # ranges do (0.394 vs 0.274); a shared scale would flatten the weaker panel.
             ("rho_sgd", "adamsgd_epsgrid_rho_sgd", "Spearman $\\rho$ vs. SGD", "viridis",
              0.0, 0.40, None, 1.055, 1.20, (5.5, 5.0, 5.0)),
             # Log-AUC of the published-neuron recall curve -- the metric of neuron_recall.pdf,
             # reduced to one number per cell. Unlike the other four this is measured against an
             # EXTERNAL ground truth (Feucht et al.'s causally-verified layer-18 neurons) rather
             # than against a benchmark score or another attribution, so it is the only panel
             # here that can say a setting is right rather than merely different.
             # PAIRED WITH neuron_recall AT 0.32 / 0.66 \textwidth, so it is drawn TALL AND
             # NARROW -- ~1.9 x 3.3in, not the four-across row's ~1.3in square. Sized so that at
             # 0.32\textwidth (1.76in) it renders 3.11in tall, matching what neuron_recall
             # --twothirds renders at 0.66\textwidth; subfigures align on their baselines, so a
             # short panel beside a tall one sits in a pool of whitespace.
             #
             # TYPE IS SET FOR ITS NEIGHBOUR, not for the quarter-width row. neuron_recall's
             # --twothirds build uses 7.0pt labels / 5.8pt ticks, and both figures land near 1:1
             # here, so these sit just under that; two subfigures side by side with visibly
             # different label sizes is the first thing a reader notices. Do not "harmonise"
             # this entry with the other four -- they are drawn for a different placement.
             ("recall_auc", "adamsgd_epsgrid_recall", "Neuron-recall log-AUC", "viridis",
              0.0, 0.70, None, 1.90, 4.10, (6.0, 5.0, 5.5))]
    for key, name, lab, cmap, vmin, vmax, _center, fw, fh, fs in specs:
        # SPEARMAN vs IG is a rank correlation over all 2,293,760 units, so it comes from the
        # score TENSORS and not from the run json (which holds benchmark scores only). IG is the
        # same cell -- addition / llama3 / mlp / logit_diff -- from the dir the paper reads, and
        # its ranks are computed ONCE: rankdata on 2.3M elements per cell would otherwise be run
        # 24 times over for the same reference.
        # Reference series for the rho panels. Both live in results/sva_sweep -- the dir the
        # paper reads -- so the comparison is against the shipped run, not a sweep replicate.
        REF = refs if refs is not None else {
            "rho_ig": "results/sva_sweep/addition_llama3_mlp_ig.scores.pt",
            "rho_sgd": "results/sva_sweep/addition_llama3_mlp_sufficient_topk_sgd_bs1.scores.pt"}
        # Published layer-18 neurons for this cell's task, for the recall panel.
        gt_layer = gt_pub = None
        if key == "recall_auc":
            gtj = json.load(open(os.path.join(
                ROOT, "src/matryoshka_attribution/data/arith_wild_l18_neurons.json")))
            gt_layer, gt_pub = gtj["layer"], np.array(gtj["neurons"]["addition"])
        gt = (gt_layer, gt_pub) if key == "recall_auc" else None
        M = eps_grid_matrix(res, key, refs=REF, epss=epss, lrs=lrs, gt=gt)
        if M is None:
            print(f"skip {name} (no reference {REF[key]})")
            continue
        if np.all(np.isnan(M)):
            print(f"skip {name} ({res} empty)")
            continue
        # SIZED FOR A 0.24\textwidth SUBFIGURE -- four across one row. ICLR's \linewidth is
        # 5.5in, so each renders at ~1.32in and the PDF must be drawn AT that size: the previous
        # half-width builds were 2.77in and LaTeX would scale them by 1.32/2.77 = 0.48, printing
        # 5.5pt cell text at 2.6pt. Redrawn instead of rescaled, with the font sizes below
        # chosen so they land at ~4.5-5pt in the compiled PDF at 1:1.
        #
        # WHAT HAD TO GIVE at this width: cell values drop a decimal and the leading zero (.27
        # not 0.267), and the colorbar loses its text label -- there is ~0.7in of horizontal
        # room for four columns once the y-label and colourbar are paid for, and a spelled-out
        # "Faithfulness AUC" alone is wider than that. Name the metric in the \subcaption.
        fig, ax = plt.subplots(figsize=(fw, fh))
        im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        for i in range(len(epss)):
            for j in range(len(lrs)):
                if not np.isnan(M[i, j]):
                    v = (M[i, j] - vmin) / (vmax - vmin)
                    dark = v < 0.6 if cmap == "viridis" else abs(v - 0.5) > 0.3
                    txt = (f"{M[i, j]:.1f}" if key == "faith_auc"
                           else f"{M[i, j]:.2f}".lstrip("0") or "0")
                    ax.text(j, i, txt,
                            ha="center", va="center", fontsize=fs[2],
                            color="#ffffff" if dark else "#000000")
        ax.set_xticks(range(len(lrs)), lrs)
        ax.set_yticks(range(len(epss)),
                      ["10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁻¹", "10⁰"][:len(epss)])
        ax.set_xlabel("Learning rate", fontsize=fs[0])
        ax.set_ylabel("Adam $\\epsilon$", fontsize=fs[0])
        ax.tick_params(labelsize=fs[1], length=1.2, pad=1.0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)
        # COLOURBAR ON TOP, horizontal. At quarter width a side colourbar costs ~0.35in of the
        # ~1.32in available -- a quarter of the figure spent on a scale -- and that width comes
        # straight out of the four data columns. Moving it above buys it back for the grid, which
        # is what lets the cell text sit at 5pt here instead of 4.2.
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", location="top",
                          fraction=0.055, pad=0.06)
        cb.ax.tick_params(labelsize=fs[1], width=0.3, length=1.2, pad=0.8)
        cb.outline.set_linewidth(0.5)
        if key == "faith_auc":
            # AXVLINE, not axhline. The colourbar is HORIZONTAL, so the value runs along x --
            # the old axhline(1.0) drew along the bar's TOP EDGE and marked nothing, which is
            # why the shipped figure had no visible tick at 1. This matters more than it looks:
            # 1.0 is exact full-model recovery and "higher is better" is FALSE on this panel,
            # so the centre of the diverging scale is the one thing a reader must locate.
            cb.ax.axvline(1.0, color="#000000", lw=0.6)
        save(fig, name)


# =================================================================== FIG 5
def fig_interventions():
    """Every intervention tried, best-over-its-own-lr-bracket, against the three references.
    Best-over-bracket is the honest summary here: each loss shape has a different gradient
    scale, so a shared lr would confound 'the fix did nothing' with 'the lr was wrong'."""
    def best(pat, filt=None):
        vals = []
        pats = pat if isinstance(pat, (list, tuple)) else [pat]
        for f in sorted(sum((glob.glob(os.path.join(ROOT, x)) for x in pats), [])):
            d = json.load(open(f))
            if "acc_auc" not in d:
                continue
            if filt and not filt(d):
                continue
            vals.append((d["acc_auc"], f))
        return max(vals) if vals else None

    bars = [
        ("IG (10 steps)", best("results/sva_sweep/addition_llama3_mlp_ig.json"), C_IG),
        ("Expected Gradients (1 draw)", best("results/adamsgd_mlp/E_steplessig/*.json"), C_SIG),
        # "best Adam / best SGD at this budget" = best over EVERY 2000-step run of that
        # optimizer on this cell with the default logit_diff loss, wherever it lives.
        ("MAttr+SGD", best(["results/sva_mlp_lr/topk_sgd/lr_*/*.json",
                            "results/adamsgd_mlp/F_seed/sgd_*/*.json"]), C_SGD),
        ("MAttr+Adam", best(["results/sva_mlp_lr/topk_adam/lr_*/*.json",
                             "results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.json",
                             "results/adamsgd_mlp/A_eps/eps_1e-8_lr_*/*.json",
                             "results/adamsgd_mlp/F_seed/adam_*/*.json"]), C_ADAM),
        ("  + tuned $\\epsilon$", best("results/adamsgd_mlp/A_eps/*/*.json"), C_FIX),
        ("  + bounded loss (tanh)", best("results/adamsgd_mlp/B_loss/ld_tanh*_adam_lr_*/*.json"), C_FIX),
        ("  + hinge loss", best("results/adamsgd_mlp/B_loss/hinge_adam_lr_*/*.json"), C_FIX),
        ("  + bounded loss (prob)", best("results/adamsgd_mlp/B_loss/prob_adam_lr_*/*.json"), C_FIX),
        ("  + fixed sparse $k$", best("results/adamsgd_mlp/C_fixedk/kf_*_adam_lr_*/*.json"), C_FIX),
        ("  + match loss", best("results/adamsgd_mlp/K_match/ld_match*_adam_eps1e-8_*/*.json"), C_FIX),
        ("  + tuned $\\epsilon$, hinge", best("results/adamsgd_mlp/J_loss_eps/hinge*/*.json"), C_FIX),
    ]
    bars = [(n, v, c) for n, v, c in bars if v]
    if len(bars) < 3:
        print("skip adamsgd_interventions (too few arms)")
        return
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ys = np.arange(len(bars))[::-1]
    ax.barh(ys, [v[0] for _, v, _ in bars], height=0.62,
            color=[c for _, _, c in bars], lw=0)
    for y, (_, v, _) in zip(ys, bars):
        ax.text(v[0] + 0.006, y, f"{v[0]:.3f}", va="center", fontsize=5.5)
    ax.set_yticks(ys, [n for n, _, _ in bars])
    ax.set_xlabel("Accuracy AUC (best over the arm's lr bracket)")
    ax.set_xlim(0, 0.60)
    style(ax)
    save(fig, "adamsgd_interventions")
    return bars


# =================================================================== FIG 6
def fig_per_example():
    """Arm D: the margin HISTOGRAM at a fixed k. 'mean 30.8 at 87% accuracy' is consistent
    with gap-padding but does not prove it; this does."""
    D = os.path.join(ROOT, "results/adamsgd_mlp/D_perexample")
    got = {}
    for lab, tag in (("MAttr+Adam", "adam"), ("MAttr+SGD", "sgd"), ("IG", "ig")):
        g = glob.glob(os.path.join(D, f"*xfer_{tag}.json"))
        if g:
            got[lab] = json.load(open(g[0]))
    if len(got) < 2:
        print("skip adamsgd_perexample (arm D not landed)")
        return
    show = [13, 15, 17]                      # k ~ 3.9e3, 1.4e4, 5.0e4
    fig, axes = plt.subplots(1, len(show), figsize=(5.5, 1.75), sharey=True)
    for ax, gi in zip(np.atleast_1d(axes), show):
        for lab, d in got.items():
            v = np.array(d["iso_metrics"]["ld_per_example"][gi])
            ax.hist(v, bins=np.linspace(-20, 70, 46), histtype="step", lw=0.9,
                    color=COL.get(lab, OTHER), label=lab)
        ax.axvline(0, color="#888888", lw=0.6)
        ax.axvline(CLEAN_MARGIN, color="#888888", lw=0.6, ls=(0, (3, 2)))
        ax.set_xlabel("Base–source margin")
        ax.set_title("")
        ax.annotate(f"$k$ = {KS[gi]:,.0f}", xy=(0.97, 0.93), xycoords="axes fraction",
                    ha="right", va="top", fontsize=6)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Examples")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5.5, loc="upper right",
                                  handlelength=1.2, borderpad=0.1, labelspacing=0.2,
                                  bbox_to_anchor=(1.0, 0.86))
    fig.subplots_adjust(wspace=0.1)
    save(fig, "adamsgd_perexample")


# =================================================================== FIG 7
def fig_fixedk():
    """Arm C: train at ONE k instead of sampling it log-uniformly. The eval grid and the train
    k-distribution are already matched (both log-uniform on [1,N]); what is NOT matched is
    where each INTEGRAND has its variance."""
    kfs = ["0.0003", "0.0009", "0.003", "0.01"]
    fig, ax = plt.subplots(figsize=(2.7, 1.9))
    any_pt = False
    for opt, lrs, col, mk in (("adam", ["0.005", "0.05"], C_ADAM, "o"),
                              ("sgd", ["1.0"], C_SGD, "s")):
        xs, ys = [], []
        for kf in kfs:
            best = None
            for lr in lrs:
                d = rd(f"results/adamsgd_mlp/C_fixedk/kf_{kf}_{opt}_lr_{lr}/*.json")
                if d and (best is None or d["acc_auc"] > best):
                    best = d["acc_auc"]
            if best is not None:
                xs.append(float(kf) * TOTAL); ys.append(best)
        if xs:
            any_pt = True
            ax.plot(xs, ys, lw=1.0, color=col, marker=mk, ms=3, mew=0,
                    label=f"MAttr+{'Adam' if opt=='adam' else 'SGD'}, fixed $k$")
    if not any_pt:
        print("skip adamsgd_fixedk (arm C empty)")
        plt.close(fig)
        return
    for lab, pat, col, ls in (("Adam, log-$k$", "results/sva_mlp_lr/topk_adam/lr_*/*.json", C_ADAM, (0, (3, 2))),
                              ("SGD, log-$k$", "results/sva_mlp_lr/topk_sgd/lr_*/*.json", C_SGD, (0, (3, 2)))):
        vals = [json.load(open(f))["acc_auc"] for f in glob.glob(os.path.join(ROOT, pat))]
        if vals:
            ax.axhline(max(vals), color=col, lw=0.7, ls=ls)
            ax.annotate(lab, xy=(0.02, max(vals) + 0.006), xycoords=("axes fraction", "data"),
                        fontsize=5.5, color=col)
    ax.set_xscale("log")
    ax.set_xlabel("Training sparsity $k$ (held fixed)")
    ax.set_ylabel("Accuracy AUC")
    style(ax)
    ax.legend(frameon=False, fontsize=5.5, loc="lower center", handlelength=1.4,
              borderpad=0.1, labelspacing=0.25)
    save(fig, "adamsgd_fixedk")




# =================================================================== FIG 8
EPS_GRID = ["1e-8", "1e-6", "1e-4", "1e-2", "1e-1", "1e0"]
LR_GRID = ["0.005", "0.05", "0.5", "5.0"]


def _eps_scores():
    """(eps, lr) -> (acc_auc, median|s|, top-2082 overlap with IG). Cached-free; the score
    vectors are 9 MB each so this reads ~24 of them."""
    import torch
    igp = os.path.join(ROOT, "results/sva_sweep/addition_llama3_mlp_ig.scores.pt")
    ig = torch.load(igp, map_location="cpu").float().flatten()
    igtop = set(torch.topk(ig, 2082).indices.tolist())
    out = {}
    for e in EPS_GRID:
        for lr in LR_GRID:
            d = rd(f"results/adamsgd_mlp/A_eps/eps_{e}_lr_{lr}/*.json")
            g = sorted(glob.glob(os.path.join(ROOT, f"results/adamsgd_mlp/A_eps/eps_{e}_lr_{lr}/*.scores.pt")))
            if not d or not g:
                continue
            v = torch.load(g[0], map_location="cpu").float().flatten()
            out[(e, lr)] = (d["acc_auc"], float(v.abs().median()),
                            len(set(torch.topk(v, 2082).indices.tolist()) & igtop) / 2082)
    return out


def fig_mechanism():
    """Adam's eps against (a) the exam, (b) whether the update is still magnitude-blind,
    (c) whether the ranking has converged on the gradient-path ranking. One line per lr."""
    S = _eps_scores()
    if len(S) < 6:
        print("skip adamsgd_mechanism (arm A incomplete)")
        return
    xs_eps = [float(e) for e in EPS_GRID]
    shades = ["#a6cee3", "#4a9fd0", "#0072b2", "#00456b"]   # light -> dark with lr
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    for j, lr in enumerate(LR_GRID):
        pts = [(float(e), S[(e, lr)]) for e in EPS_GRID if (e, lr) in S]
        if not pts:
            continue
        x = [p[0] for p in pts]
        for ax, idx in ((axes[0], 0), (axes[1], 1), (axes[2], 2)):
            y = [p[1][idx] for p in pts]
            if idx == 1:      # normalise by the sign-walk prediction lr*sqrt(steps)
                y = [v / (float(lr) * np.sqrt(2000)) for v in y]
            ax.plot(x, y, lw=1.0, marker="o", ms=2.6, mew=0, color=shades[j], label=f"lr {lr}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("Adam $\\epsilon$")
        style(ax)
    axes[1].set_yscale("log")
    sgd = best_acc(["results/sva_mlp_lr/topk_sgd/lr_*/*.json",
                    "results/adamsgd_mlp/F_seed/sgd_*/*.json"])
    ig_ = rd(REF["IG"])
    if sgd:
        axes[0].axhline(sgd, color=C_SGD, lw=0.7, ls=(0, (3, 2)))
        axes[0].annotate("MAttr+SGD", xy=(0.03, sgd + 0.008), xycoords=("axes fraction", "data"),
                         fontsize=5.5, color=C_SGD)
    if ig_:
        axes[0].axhline(ig_["acc_auc"], color=C_IG, lw=0.7, ls=(0, (3, 2)))
        axes[0].annotate("IG", xy=(0.86, ig_["acc_auc"] + 0.008),
                         xycoords=("axes fraction", "data"), fontsize=5.5, color=C_IG)
    axes[1].axhline(1.0, color="#888888", lw=0.6, ls=(0, (3, 2)))
    axes[1].annotate("pure sign(g) walk", xy=(0.03, 1.15), xycoords=("axes fraction", "data"),
                     fontsize=5.5, color="#666666")
    axes[0].set_ylabel("Accuracy AUC")
    axes[1].set_ylabel("median |score| / (lr·√steps)")
    axes[2].set_ylabel("Top-2082 overlap with IG")
    for ax, sub in zip(axes, "abc"):
        ax.annotate(sub, xy=(0.03, 0.96), xycoords="axes fraction", va="top", fontsize=6.5,
                    fontweight="bold")
    axes[2].legend(frameon=False, fontsize=5.5, loc="upper left", handlelength=1.2,
                   borderpad=0.1, labelspacing=0.2, bbox_to_anchor=(0.10, 0.99))
    fig.subplots_adjust(wspace=0.42)
    save(fig, "adamsgd_mechanism")


def best_acc(pats):
    vals = []
    for p_ in (pats if isinstance(pats, (list, tuple)) else [pats]):
        for f in glob.glob(os.path.join(ROOT, p_)):
            d = json.load(open(f))
            if "acc_auc" in d:
                vals.append(d["acc_auc"])
    return max(vals) if vals else None



# =================================================================== FIG 9
def fig_per_example2():
    """Section-D analysis for the FIXED optimizers: per-example margin distributions of the
    rankings that all tie at ~0.49 acc AUC. Does 'fixed' mean the same margin geometry, or
    the same exam score reached different ways?"""
    D2 = os.path.join(ROOT, "results/adamsgd_mlp/D2_perexample")
    D1 = os.path.join(ROOT, "results/adamsgd_mlp/D_perexample")
    series = [("IG", os.path.join(D1, "*xfer_ig.json"), C_IG),
              ("SGD+EMA mom.", os.path.join(D2, "*xfer_ema.json"), C_SGD),
              ("Adam ε=10⁻² lr .5", os.path.join(D2, "*xfer_adameps5.json"), C_ADAM),
              ("hinge, Adam ε=10⁻²", os.path.join(D2, "*xfer_hingeeps.json"), C_FIX)]
    got = []
    for lab, pat, col in series:
        g = glob.glob(pat)
        if g:
            got.append((lab, json.load(open(g[0])), col))
    if len(got) < 3:
        print("skip adamsgd_perexample2 (D2 not landed)")
        return
    show = [13, 15, 17]
    fig, axes = plt.subplots(1, len(show), figsize=(5.5, 1.75), sharey=True)
    for ax, gi in zip(np.atleast_1d(axes), show):
        for lab, d, col in got:
            v = np.array(d["iso_metrics"]["ld_per_example"][gi])
            ax.hist(v, bins=np.linspace(-20, 70, 46), histtype="step", lw=0.9, color=col,
                    label=lab)
        ax.axvline(0, color="#888888", lw=0.6)
        ax.axvline(CLEAN_MARGIN, color="#888888", lw=0.6, ls=(0, (3, 2)))
        ax.set_xlabel("Base–source margin")
        ax.annotate(f"$k$ = {KS[gi]:,.0f}", xy=(0.97, 0.93), xycoords="axes fraction",
                    ha="right", va="top", fontsize=6)
        style(ax)
    np.atleast_1d(axes)[0].set_ylabel("Examples")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=5.5, loc="upper right",
                                  handlelength=1.2, borderpad=0.1, labelspacing=0.2,
                                  bbox_to_anchor=(1.0, 0.86))
    fig.subplots_adjust(wspace=0.1)
    save(fig, "adamsgd_perexample2")



# =================================================================== FIG 10
def fig_efflr():
    """The one-optimizer test: acc AUC and peak over-recovery against EFFECTIVE lr, where
    plain SGD contributes at lr, EMA-momentum SGD at lr (unit steady-state gain), heavy-ball
    at 10*lr (gain 1/(1-mu)), and Adam eps=1e-2 at lr/eps. If big-eps Adam really is EMA-SGD,
    all four series lie on one dose-response curve; systematic departures are what any
    residual sqrt(v)-hat story has to live in."""
    series = [
        ("SGD (plain)", 1.0, "results/sva_mlp_lr/topk_sgd/lr_{lr}/*.json",
         ["0.05", "0.3", "1.0", "3.0", "10.0", "30.0", "100.0", "300.0"], C_SGD, "o"),
        ("SGD + EMA mom.", 1.0, "results/adamsgd_mlp/L_momentum/ema_mu0.9_lr_{lr}/*.json",
         ["0.3", "1.0", "3.0", "10.0", "30.0", "100.0", "300.0"], "#56b4e9", "s"),
        ("SGD + heavy-ball (×10)", 10.0, "results/adamsgd_mlp/L_momentum/hb_mu0.9_lr_{lr}/*.json",
         ["0.03", "0.1", "0.3", "1.0"], "#009e73", "D"),
        ("Adam ε=10⁻² (lr/ε)", 100.0, "results/adamsgd_mlp/A_eps/eps_1e-2_lr_{lr}/*.json",
         ["0.005", "0.05", "0.5", "5.0"], C_ADAM, "^"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    plotted = False
    for lab, gain, pat, lrs, col, mk in series:
        xs_, acc_, fa_, fmax_ = [], [], [], []
        for lr in lrs:
            d = rd(pat.format(lr=lr))
            if d:
                xs_.append(float(lr) * gain)
                acc_.append(d["acc_auc"])
                fa_.append(d["faith_auc"])
                fmax_.append(d["faith_max"])
        if not xs_:
            continue
        plotted = True
        axes[0].plot(xs_, acc_, lw=0.9, marker=mk, ms=2.6, mew=0, color=col, label=lab)
        axes[1].plot(xs_, fa_, lw=0.9, marker=mk, ms=2.6, mew=0, color=col)
        axes[2].plot(xs_, fmax_, lw=0.9, marker=mk, ms=2.6, mew=0, color=col)
    if not plotted:
        print("skip adamsgd_efflr")
        plt.close(fig)
        return
    for ax, ylab in ((axes[0], "Accuracy AUC"), (axes[1], "Faithfulness AUC"),
                     (axes[2], "Peak over-recovery")):
        ax.set_xscale("log")
        ax.set_xlabel("Effective learning rate")
        ax.set_ylabel(ylab)
        style(ax)
    axes[2].set_yscale("log")
    for ax in axes[1:]:
        ax.axhline(1.0, color="#888888", lw=0.6, ls=(0, (3, 2)))
    for ax, sub in zip(axes, "abc"):
        ax.annotate(sub, xy=(0.05, 0.96), xycoords="axes fraction", va="top", fontsize=6.5,
                    fontweight="bold")
    axes[0].legend(frameon=False, fontsize=5, loc="lower left", handlelength=1.2,
                   borderpad=0.1, labelspacing=0.2)
    fig.subplots_adjust(wspace=0.42)
    save(fig, "adamsgd_efflr")



# =================================================================== FIG 11
def fig_acc_vs_faith():
    """Accuracy AUC against faithfulness AUC. The fixed family traces ONE path parameterised
    by effective lr: a clean cluster at (0.5, 0.49), a rightward sweep past faith=1 with
    accuracy roughly flat (padding is free on the exam), then collapse. Broken (default-eps)
    Adam sits BELOW that path at matched faithfulness -- same objective value, worse ranking."""
    series = [
        ("SGD (plain)", 1.0, "results/sva_mlp_lr/topk_sgd/lr_{lr}/*.json",
         ["0.05", "0.3", "1.0", "3.0", "10.0", "30.0", "100.0", "300.0"], C_SGD, "o"),
        ("SGD + EMA mom.", 1.0, "results/adamsgd_mlp/L_momentum/ema_mu0.9_lr_{lr}/*.json",
         ["0.3", "1.0", "3.0", "10.0", "30.0", "100.0", "300.0"], "#56b4e9", "s"),
        ("SGD + heavy-ball", 10.0, "results/adamsgd_mlp/L_momentum/hb_mu0.9_lr_{lr}/*.json",
         ["0.03", "0.1", "0.3", "1.0"], "#009e73", "D"),
        ("Adam ε=10⁻²", 100.0, "results/adamsgd_mlp/A_eps/eps_1e-2_lr_{lr}/*.json",
         ["0.005", "0.05", "0.5", "5.0"], C_ADAM, "^"),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for lab, gain, pat, lrs, col, mk in series:
        pts = []
        for lr in lrs:
            d = rd(pat.format(lr=lr))
            if d:
                pts.append((float(lr) * gain, d["faith_auc"], d["acc_auc"]))
        if not pts:
            continue
        pts.sort()
        ax.plot([p_[1] for p_ in pts], [p_[2] for p_ in pts], lw=0.7, alpha=0.6,
                marker=mk, ms=3, mew=0, color=col, label=lab)
        # label a few effective-lr values so the hairpin (faith comes back DOWN once
        # circuits break) reads as a trajectory, not a tangle
        want = {"SGD (plain)": (1.0, 10.0, 300.0), "Adam ε=10⁻²": (0.5, 50.0, 500.0)}
        for eff, fa_, ac_ in pts:
            if lab in want and any(abs(eff - w) / w < 0.01 for w in want[lab]):
                ax.annotate(f"{eff:g}", xy=(fa_, ac_), xytext=(3, 3),
                            textcoords="offset points", fontsize=5, color=col)
    # broken Adam: default-eps lr sweep + the 1e-6 arm, hollow markers, no path
    bx, by = [], []
    for pat in ("results/sva_mlp_lr/topk_adam/lr_*/*.json",
                "results/sva_sweep/addition_llama3_mlp_sufficient_topk_adam_bs1.json",
                "results/adamsgd_mlp/A_eps/eps_1e-8_lr_*/*.json",
                "results/adamsgd_mlp/A_eps/eps_1e-6_lr_*/*.json"):
        for f in glob.glob(os.path.join(ROOT, pat)):
            d = json.load(open(f))
            bx.append(d["faith_auc"]); by.append(d["acc_auc"])
    ax.scatter(bx, by, s=12, facecolors="none", edgecolors=C_ADAM, lw=0.7,
               label="Adam, default ε (broken)")
    ig = rd(REF["IG"])
    if ig:
        ax.scatter([ig["faith_auc"]], [ig["acc_auc"]], s=45, c=C_IG, marker="*", lw=0,
                   label="IG", zorder=5)
    hg = None
    for f in glob.glob(os.path.join(ROOT, "results/adamsgd_mlp/J_loss_eps/hinge*/*.json")):
        d = json.load(open(f))
        if hg is None or d["acc_auc"] > hg["acc_auc"]:
            hg = d
    if hg:
        ax.scatter([hg["faith_auc"]], [hg["acc_auc"]], s=16, c=C_FIX, marker="P", lw=0,
                   label="hinge · Adam ε=10⁻²", zorder=5)
    ax.axvline(1.0, color="#888888", lw=0.6, ls=(0, (3, 2)))
    ax.annotate("over-recovery →", xy=(1.06, 0.075), fontsize=5.5, color="#666666")
    # effective-lr direction cue along the family path
    ax.annotate("labels = effective lr", xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=5.5, color="#666666")
    ax.set_xlabel("Faithfulness AUC")
    ax.set_ylabel("Accuracy AUC")
    style(ax)
    ax.legend(frameon=False, fontsize=5, loc="lower left", handlelength=1.2, borderpad=0.1,
              labelspacing=0.25, scatterpoints=1)
    save(fig, "adamsgd_accfaith")


if __name__ == "__main__":
    fig_curves()
    fig_scatters()
    fig_trajectories()
    fig_eps_grid()
    fig_interventions()
    fig_per_example()
    fig_fixedk()
    fig_mechanism()
    fig_per_example2()
    fig_efflr()
    fig_acc_vs_faith()
