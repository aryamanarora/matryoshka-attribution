"""One figure: Adam eps x lr, faceted by SUBSTRATE (rows) x METRIC (columns).

REPLACES THREE SEPARATE FOUR-PANEL ROWS (fig:optimiser-eps and its node / SAE twins). Those were
twelve independent PDFs with twelve independent colour scales, which made the one comparison
worth making -- how the SAME sweep behaves on substrates spanning 1,056 to 5.24M units -- the one
comparison a reader could not do. Here each column carries a single scale across all three rows.

*** SO THE SAE rho PANELS ARE NEARLY FLAT, AND THAT IS THE RESULT, NOT A RENDERING FAULT. ***
rho-vs-IG spans 0.002-0.081 on the MLP-output SAE against 0.334-0.744 on nodes; on a shared 0-0.75
scale the SAE row is uniformly dark. The standalone figures gave that row its own 0-0.30 scale and
its internal 40x variation was legible -- real, but a detail. The headline is that SAE rankings
agree with IG essentially not at all while node rankings agree strongly, and a shared scale is the
only way to see it. main() prints every per-row range so nothing is silently lost.

THE NUMBERS COME FROM plot_adamsgd_mlp_diag.eps_grid_matrix, the same loader fig_eps_grid uses,
so a cell here and a cell in the standalone figure cannot disagree.

REFERENCES MUST MATCH EACH ROW'S INTERVENTION. The SAE row correlates against frozen-mode IG and
MAttr+SGD on the same cell, not the results/sva_sweep runs the other two rows use -- those are
`absorb`, and correlating a frozen grid against an absorb reference would fold an intervention
change into a hyperparameter figure. `node` and `mlp` have no SAE error term, so no such choice
arises for them.

WHAT THE THREE ROWS ARE FOR. The eps story is that at ~2.29M mask logits Adam's update degenerates
to sign(g)*lr. `node` is the control at 1,056 units -- 2,170x smaller -- and its accuracy panel is
flat (0.477-0.526, 18 of 24 cells within 0.011 of the best) with no cell above faith 1.0. The two
large substrates span ~0.3 in accuracy and both have a gap-padding corner. Read the rows as
"does the knob matter here", not as a ranking of substrates: the metrics are not comparable
across substrates (different unit counts, different F_patch).

NO resid_sae_span ROW. The frozen intervention is numerically unstable there (4 of 11 completed
runs had non-finite sweep points; IG 13/24), so there is nothing to plot -- see
llama.py:_sae_interchange.

Run:  uv run python plots/plot_epsgrid_facets.py
Out:  plots/epsgrid_facets.pdf  (+ .png)
"""
import os
import sys

import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import palette as P                                    # noqa: E402
import plot_adamsgd_mlp_diag as D                      # eps_grid_matrix  # noqa: E402

EPSS = ["1e-8", "1e-6", "1e-4", "1e-2", "1e-1", "1e0"]
EPS_LAB = ["10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁻¹", "10⁰"]
LRS = ["0.005", "0.05", "0.5", "5.0"]
FERR = "results/sae_ferr/smoke"
SW = "results/sva_sweep"
# (row label, results tree, {rho key: reference .scores.pt}). Row order is by MASK-LOGIT COUNT,
# smallest first, because that is the axis the figure is arguing along: node 1,056; Gemma-2 MLP
# 1,677,312; Llama-3 MLP 2,293,760; Llama-3 MLP-output SAE 5,243,040.
#
# THE GEMMA ROW IS THE CROSS-MODEL CONTROL. Same task, same dataset, same substrate KIND -- only
# the model changes, and with it the architecture, the tokenizer and 4x the parameter count. It
# sits at 1.68M logits against Llama's 2.29M, so if the eps effect tracks the number of logits it
# should appear here too. Labels name the model on every row because three of the four are
# Llama-3 and one is not; a bare "MLP" strip would silently attribute Gemma's row to Llama.
#
# GEMMA IS SAFE TO RUN HERE despite CLAUDE.md's "never evaluate a gemma2 cell in the L2A venv":
# that rule is about eval_mib.py's HookedTransformer path (TL 3.2.1 computes a wrong Gemma-2
# forward). eval_sva loads through AutoModelForCausalLM and never imports transformer_lens --
# checked, not assumed.
ROWS = [
    ("Llama 3 / Node", "results/epslr_node",
     {"rho_ig": f"{SW}/addition_llama3_node_ig.scores.pt",
      "rho_sgd": f"{SW}/addition_llama3_node_sufficient_topk_sgd_bs1.scores.pt"}),
    ("Gemma 2 / MLP", "results/epslr_mlpn_gemma2",
     {"rho_ig": "results/epslr_mlpn_gemma2/refs/addition_gemma2_mlp_ig.scores.pt",
      "rho_sgd": "results/epslr_mlpn_gemma2/refs/"
                 "addition_gemma2_mlp_sufficient_topk_sgd_bs1.scores.pt"}),
    ("Llama 3 / MLP", "results/adamsgd_mlp/A_eps",
     {"rho_ig": f"{SW}/addition_llama3_mlp_ig.scores.pt",
      "rho_sgd": f"{SW}/addition_llama3_mlp_sufficient_topk_sgd_bs1.scores.pt"}),
    ("Llama 3 / SAE", "results/saefrozen_epslr",
     {"rho_ig": f"{FERR}/addition_llama3_mlp_sae_span_ig_ferr.scores.pt",
      "rho_sgd": f"{FERR}/addition_llama3_mlp_sae_span_"
                 "sufficient_topk_sgd_ferr_bs1.scores.pt"}),
]
# (key, column title, cmap, vmin, vmax). ONE SCALE PER COLUMN, spanning all three rows --
# vmax is the global max over the substrates, rounded up. Observed: acc 0.110-0.526,
# faith 0.173-1.833, rho-IG 0.002-0.744, rho-SGD 0.009-0.868.
# (key, column title, cmap, vmin, vmax, transform-or-None)
COLS = [
    ("acc_auc", "Compactness", "viridis", 0.0, 0.55, None),
    # vmax 3.5, NOT 2.0: the Gemma row reaches 3.19 and would clip flat against the old ceiling.
    # A plain linear 0-3.5 would move the diverging map's white point to 1.75, which means
    # nothing -- so this column alone uses TwoSlopeNorm pinned at 1.0 (exact full-model
    # recovery), keeping white where the semantics put it while the red arm stretches.
    ("faith_auc", "Faith log-AUC", "RdBu_r", 0.0, 3.5, None),
    ("rho_ig", "Spearman $\\rho$ vs. IG", "viridis", 0.0, 0.75, None),
    ("rho_sgd", "Spearman $\\rho$ vs. SGD", "viridis", 0.0, 0.90, None),
    # SPREAD OF THE LEARNED SCORE VECTOR -- the only column that measures the optimiser's OUTPUT
    # rather than its consequences, and the one that shows the eps mechanism directly: at small
    # eps Adam's update degenerates to sign(g)*lr, so the score becomes a signed COUNT of steps
    # and its spread collapses.
    #
    # LOG10, because raw sigma spans SIX ORDERS OF MAGNITUDE (6.9e-5 to 143 across the grid, a
    # 1.4-million-fold range within the SAE row alone). On a linear scale 23 of 24 cells in every
    # row would be the same colour.
    #
    # CIVIDIS, not viridis: this is a diagnostic, not a score, and reusing the "higher is better"
    # colormap would invite reading a bright cell as a good one. Nothing here is better or worse.
    #
    # "SD", NOT sigma. This paper spends sigma on the SIGMOID -- sigmoid_topk's gate is sigma(s/T)
    # -- so a column headed "sigma(scores)" reads as the gate applied to the score vector, which
    # is a real quantity in this method and not the one being plotted.
    ("score_std", "log$_{10}$ SD(scores)", "cividis", -4.5, 2.5, np.log10),
]

# Cells that beat a baseline get a corner dot: TOP-LEFT for IG, TOP-RIGHT for MAttr+SGD. The two
# are marked independently rather than nested, because "beats SGD" is NOT a subset of "beats IG":
# on the SAE row SGD (0.416) scores below IG (0.449), so 9 cells beat SGD while only 4 beat IG.
#
# ONE MARK, ONE MEANING: the value is UNDERLINED where the cell beats BOTH IG and MAttr+SGD.
# Earlier passes marked the two baselines separately (corner dots, then coloured outlines, then
# */dagger/double-dagger). All three worked, and all three spent a lot of the reader's attention
# on a distinction that is rarely the one being made -- "did MAttr's tuned cell clear the
# baselines" is a single yes/no. Beating only one is no longer marked at all; main() prints those
# counts so they stay available without being drawn.
#
# DRAWN, NOT TYPESET. matplotlib's mathtext has no \underline (verified: ParseFatalException),
# so the rule is a Line2D placed under each marked value's measured extent after the layout is
# final -- which also lets it inherit the value's own contrast-aware colour, white on dark cells
# and black on light ones.
UNDERLINE_LW, UNDERLINE_PAD = 0.5, 0.0018
#
# *** ON THE FAITH COLUMN "BEATS" MEANS HIGHER AUC, WHICH IS NOT THE SAME AS BETTER. *** 1.0 is
# exact full-model recovery and the diverging map is centred there; a cell at 3.19 (Gemma, eps
# 1e-4, lr 0.05) "beats" every baseline while recovering more than three times the clean logit
# difference, which is the gap-padding signature. The marks are a literal comparison, drawn
# because they were asked for; read them on the IIA column and treat them as a warning sign
# rather than a win wherever the cell is red. For "closest to 1" instead, compare
# abs(M - 1) < abs(ref - 1) in beats().
MARK_KEYS = ("acc_auc", "faith_auc")

FIG_W = 5.5
PANEL_H = 1.06
HEAD, FOOT = 0.62, 0.50     # colourbar band + column titles; x ticks and label
FS_LAB, FS_TICK, FS_CELL, FS_STRIP = 6.5, 5.0, 4.6, 6.0


def baselines(refs, key):
    """(IG, MAttr+SGD) value of `key` for one row, or (None, None) if either json is absent.

    Derived from the rho references rather than listed separately -- those already name the two
    runs, and the eval json sits beside the .scores.pt. One source of truth means a row cannot
    correlate against one IG run while being marked against another.
    """
    out = []
    for k in ("rho_ig", "rho_sgd"):
        p = os.path.join(D.ROOT, refs[k].replace(".scores.pt", ".json"))
        out.append(json.load(open(p))[key] if os.path.exists(p) else None)
    return tuple(out)


# --tag unifk (2026-09-16): the UNIFORM-k twin of the grid -- scripts/sva/launch/
# submit_epslr_unifk_sc.sh, same four substrates and the same 6 x 4 grid with only the
# k-schedule changed (uniform is the headline schedule since 2026-09-15). Trees are the log-k
# ones suffixed _unifk (the Llama MLP row's log-k tree is results/adamsgd_mlp/A_eps; its twin is
# results/epslr_mlpn_unifk). The IG references are k-independent and reused; the SGD reference
# must be uniform-k too, so it is the <tree>/refs run the launcher adds, found by glob -- when
# it has not landed the log-k SGD reference is used and named on stdout.
UNIFK_TREE = {"results/epslr_node": "results/epslr_node_unifk",
              "results/epslr_mlpn_gemma2": "results/epslr_mlpn_gemma2_unifk",
              "results/adamsgd_mlp/A_eps": "results/epslr_mlpn_unifk",
              "results/saefrozen_epslr": "results/saefrozen_epslr_unifk"}


def unifk_rows():
    import glob as _glob
    out = []
    for rlab, res, refs in ROWS:
        tree = UNIFK_TREE[res]
        r = dict(refs)
        sgd = sorted(_glob.glob(os.path.join(ROOT, tree, "refs", "*sgd*.scores.pt")))
        if sgd:
            r["rho_sgd"] = os.path.relpath(sgd[0], ROOT)
        else:
            print(f"  NOTE {rlab}: no uniform-k SGD reference in {tree}/refs yet; using the log-k one")
        out.append((rlab, tree, r))
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, choices=[None, "unifk"],
                    help="unifk: the uniform-k twin (separate trees, separate output file)")
    a = ap.parse_args()
    rows, out_path = ROWS, "plots/epsgrid_facets.pdf"
    if a.tag == "unifk":
        rows, out_path = unifk_rows(), "plots/epsgrid_facets_unifk.pdf"
    M = {}
    for ri, (rlab, res, refs) in enumerate(rows):
        for key, _t, _c, _lo, _hi, tf in COLS:
            m = D.eps_grid_matrix(res, key, refs=refs, epss=EPSS, lrs=LRS)
            if m is None:
                print(f"MISSING reference for {rlab}/{key} -- panel will be blank")
                m = np.full((len(EPSS), len(LRS)), np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                M[(ri, key)] = tf(m) if tf is not None else m

    base = {(ri, k): baselines(refs, k)
            for ri, (_, _, refs) in enumerate(rows) for k in MARK_KEYS}

    underline = []
    plt.rcParams.update(P.RC)
    fh = PANEL_H * len(rows) + HEAD + FOOT
    fig, axes = plt.subplots(len(rows), len(COLS), figsize=(FIG_W, fh),
                             squeeze=False)
    for ci, (key, title, cmap, vmin, vmax, tf) in enumerate(COLS):
        for ri, (rlab, _, _) in enumerate(rows):
            ax = axes[ri][ci]
            m = M[(ri, key)]
            norm = (TwoSlopeNorm(vcenter=1.0, vmin=vmin, vmax=vmax)
                    if key == "faith_auc" else None)
            im = (ax.imshow(m, cmap=cmap, aspect="auto", norm=norm) if norm is not None
                  else ax.imshow(m, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax))
            for i in range(len(EPSS)):
                for j in range(len(LRS)):
                    if np.isnan(m[i, j]):
                        continue
                    # Contrast off the NORMALISED value, so the two-slope faith column picks
                    # the same white-on-dark rule everywhere the colour is actually dark.
                    v = float(norm(m[i, j])) if norm is not None else \
                        (m[i, j] - vmin) / (vmax - vmin)
                    # viridis is dark at the bottom; RdBu_r is dark at BOTH ends.
                    dark = v < 0.6 if cmap == "viridis" else abs(v - 0.5) > 0.3
                    # Two decimals everywhere, leading zero stripped -- so faith reads ".84"
                    # and "1.83", matching the other columns. The standalone fig_eps_grid still
                    # uses one decimal for faith; it is drawn at 0.24\textwidth where the extra
                    # digit does not fit, and its panels are no longer in the paper.
                    # The log column keeps its sign and one decimal; lstrip("0") is for the
                    # 0-1 columns and would turn "0.5" into ".5" here while leaving "-4.2" alone,
                    # i.e. inconsistently.
                    txt = (f"{m[i, j]:.1f}" if key == "score_std"
                           else f"{m[i, j]:.2f}".lstrip("0") or "0")
                    t = ax.text(j, i, txt, ha="center", va="center", fontsize=FS_CELL,
                                color="#ffffff" if dark else "#000000")
                    if key in MARK_KEYS:
                        ig_v, sgd_v = base[(ri, key)]
                        if (ig_v is not None and sgd_v is not None
                                and m[i, j] > ig_v and m[i, j] > sgd_v):
                            underline.append(t)
            ax.set_xticks(range(len(LRS)))
            ax.set_yticks(range(len(EPSS)))
            ax.set_xticklabels(LRS if ri == len(rows) - 1 else [])
            ax.set_yticklabels(EPS_LAB if ci == 0 else [])
            ax.tick_params(labelsize=FS_TICK, length=1.2, pad=1.0)
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if ci == 0:
                ax.set_ylabel("Adam $\\epsilon$", fontsize=FS_LAB, labelpad=1.5)
            if ri == len(rows) - 1:
                ax.set_xlabel("Learning rate", fontsize=FS_LAB, labelpad=1.5)
            if ci == len(COLS) - 1:
                # Facet strip on the right, ggplot-style: the substrate name is ~0.6in at 6pt
                # rotated, which fits a PANEL_H-tall panel where a left-hand label would eat
                # width the four lr columns need.
                ax.text(1.03, 0.5, rlab, transform=ax.transAxes, rotation=270,
                        va="center", ha="left", fontsize=FS_STRIP)

    # Bottom of the rect reserves the marker legend's strip; tight_layout does not know about
    # figure-level legends, so without it the legend lands on top of the "Learning rate" labels.
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.35,
                     rect=(0, 0.20 / fh, 1, 1 - HEAD / fh))
    # ONE COLOURBAR PER COLUMN, above row 0 and spanning that column's width. Horizontal because
    # a vertical bar per column would cost four times its width out of the data area.
    for ci, (key, title, cmap, vmin, vmax, tf) in enumerate(COLS):
        b0 = axes[0][ci].get_position()
        # 92% of the column width, centred: at full width the faith bar's "2.0" tick label
        # collided with the next bar's "0.0" and read as "2.00.0".
        cax = fig.add_axes([b0.x0 + 0.04 * b0.width, 1 - (HEAD - 0.30) / fh,
                            0.92 * b0.width, 0.016])
        cb = fig.colorbar(axes[0][ci].images[0], cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=FS_TICK, width=0.3, length=1.2, pad=0.8)
        cb.ax.xaxis.set_major_locator(MaxNLocator(4, prune="upper"))
        cb.outline.set_linewidth(0.5)
        cax.set_title(title, fontsize=FS_LAB, pad=2.5)
        if key == "faith_auc":
            # 1.0 = exact full-model recovery; "higher is better" is FALSE on this column, so
            # the centre of the diverging scale has to be findable. axvline, not axhline: the
            # bar is horizontal, so value runs along x.
            cb.ax.axvline(1.0, color="#000000", lw=0.6)

    fig.text(0.5, 0.006, "underlined: beats both IG and MAttr+SGD",
             ha="center", va="bottom", fontsize=FS_STRIP)

    # AFTER tight_layout and the colourbar axes, so the measured extents are the final ones.
    # savefig is called without bbox_inches="tight", so nothing moves between here and the write.
    from matplotlib.lines import Line2D
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    for t in underline:
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0, y0), (x1, _) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        fig.add_artist(Line2D([x0, x1], [y0 - UNDERLINE_PAD] * 2, lw=UNDERLINE_LW,
                              color=t.get_color(), transform=fig.transFigure, zorder=5))

    out = out_path
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(f"wrote {out}   {len(rows)} substrates x {len(COLS)} metrics\n")
    # The figure marks only "beats BOTH". These counts keep the one-baseline cases available
    # without spending ink on them -- and they are what the docstring's claim rests on.
    print(f"{'substrate':<16}{'metric':<11}{'>IG':>6}{'>SGD':>7}{'>both':>7}{'cells':>7}")
    for ri, (rlab, _, refs) in enumerate(rows):
        for key in MARK_KEYS:
            m = M[(ri, key)]
            ig_v, sgd_v = base[(ri, key)]
            print(f"{rlab:<16}{key:<11}{int(np.nansum(m > ig_v)):>6}"
                  f"{int(np.nansum(m > sgd_v)):>7}"
                  f"{int(np.nansum((m > ig_v) & (m > sgd_v))):>7}"
                  f"{int(np.isfinite(m).sum()):>7}")
    print()
    print(f"{'substrate':<16}" + "".join(f"{c[0]:>22}" for c in COLS))
    for ri, (rlab, _, _) in enumerate(rows):
        print(f"{rlab:<16}" + "".join(
            f"{np.nanmin(M[(ri, k)]):>10.3f}..{np.nanmax(M[(ri, k)]):<11.3f}"
            for k, *_ in COLS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
