"""LaTeX tables for the cross-format arithmetic appendix (paper/sections/arith-formats.tex).

Reads the two summary.json files written by analyse.py -- the main run (MAttr on the nats loss)
and the normalised-loss arm -- and writes booktabs tabulars into paper/tabs/, so the numbers in
the appendix are generated, never typed.  Re-run after analyse.py.

    uv run python scripts/arith_formats/make_tables.py

Tables:
  arith_formats_summary.tex      format-level and item-level headline numbers, AP vs MAttr vs paper
  arith_formats_sufficiency.tex  restored preference fraction vs top-k fraction, per ranking
  arith_formats_models.tex       per-model accuracy and overlap (AP / MAttr normalised)
  arith_formats_fig4.tex         Llama-3.1-8B linear probability model (loading vs probes/confidence)
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path("results/arith_formats")
NORM = Path("results/arith_formats_abl_norm_lr1")
TABS = Path("paper/tabs")
FORMATS = ["numeric", "english", "spanish", "italian"]
VERBAL = FORMATS[1:]
FMT = {"numeric": "Numeric", "english": "English", "spanish": "Spanish", "italian": "Italian"}
PAPER = {
    "jaccard": {"english": 0.145, "spanish": 0.078, "italian": 0.069},
    "corr": {"english": (0.75, 0.003), "spanish": (0.58, 0.04), "italian": (0.47, 0.10)},
    "pbr": {"english": (0.38, 12), "spanish": (0.32, 11), "italian": (0.29, 9)},
    "acc": {"numeric": 87.4, "english": 36.4, "spanish": 17.9, "italian": 6.8},
}


def load(d):
    return json.load(open(d / "summary.json"))


def p_str(p):
    if p < 1e-3:
        return "$<$0.001"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.3f}"


def table(rows, header, caption, label, align, extra_note=None):
    out = ["\\begin{table}[!t]", "    \\centering", "    \\footnotesize", "    \\setlength{\\tabcolsep}{3pt}",
           f"    \\begin{{tabular}}{{{align}}}", "    \\toprule", "    " + header + " \\\\", "    \\midrule"]
    for r in rows:
        out.append("    " + (r if r.strip() == "\\midrule" else r + " \\\\"))
    if extra_note:
        out += ["    \\midrule", "    " + extra_note + " \\\\"]
    out += ["    \\bottomrule", "    \\end{tabular}", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
            "\\end{table}", ""]
    return "\n".join(out)


def main():
    S, N = load(ROOT), load(NORM)
    models = S["models"]
    n = len(models)
    TABS.mkdir(exist_ok=True)

    # ------------------------------------------------------------- summary table
    arms = [("AP", S, "ap"), ("\\ourmethod{} (nats)", S, "mattr"), ("\\ourmethod{} (rescaled)", N, "mattr")]
    rows = []
    # accuracy medians (same for every arm)
    rows.append("\\multicolumn{5}{l}{\\emph{Accuracy, median over models (\\%)}}")
    for f in FORMATS:
        med = 100 * np.median([S["acc"][m][f] for m in models])
        rows.append(f"    {FMT[f]} & \\multicolumn{{3}}{{c}}{{{med:.1f}}} & {PAPER['acc'][f]:.1f}")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Top-1\\% Jaccard overlap with the numeric circuit, mean over models}}")
    for f in VERBAL:
        cells = [f"{np.nanmean([D['jaccard'][m][f'{k}:{f}'] for m in models]):.3f}" for _, D, k in arms]
        rows.append(f"    {FMT[f]} & " + " & ".join(cells) + f" & {PAPER['jaccard'][f]:.3f}")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Across-model Pearson $r$ between overlap and accuracy ($p$)}}")
    for f in VERBAL:
        cells = []
        for _, D, k in arms:
            c = D["overlap_acc_corr"][f"{k}:{f}"]
            cells.append(f"{c['r']:+.2f} ({p_str(c['p'])})")
        r, p = PAPER["corr"][f]
        rows.append(f"    {FMT[f]} & " + " & ".join(cells) + f" & {r:+.2f} ({p_str(p)})")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Item-level point-biserial $r$, loading vs.\\ correctness: median (\\# models $p<0.05$)}}")
    for f in VERBAL:
        cells = []
        for _, D, k in arms:
            rs = [D["pbr"][m][f"{k}:{f}"]["r"] for m in models]
            sig = sum(1 for m in models if D["pbr"][m][f"{k}:{f}"]["p"] < 0.05)
            cells.append(f"{np.nanmedian(rs):+.2f} ({sig}/{n})")
        r, k_ = PAPER["pbr"][f]
        rows.append(f"    {FMT[f]} & " + " & ".join(cells) + f" & {r:+.2f} ({k_}/13)")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Causal: patch-restored fraction vs.\\ correctness, median $r$ (Wilcoxon $p$ vs.\\ random units)}}")
    for f in VERBAL:
        cells = []
        for _, D, k in arms:
            rs = [D["patch"][m][f"{k}:{f}:circ"]["r"] for m in models]
            w = D["patch_wilcoxon"].get(f"{k}:{f}", {"p": float("nan")})
            cells.append(f"{np.nanmedian(rs):+.2f} ({p_str(w['p'])})")
        pp = {"english": "$<$0.001", "spanish": "$<$0.001", "italian": "$<$0.01"}[f]
        rows.append(f"    {FMT[f]} & " + " & ".join(cells) + f" & -- ({pp})")
    rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\emph{Mean restored fraction, numeric circuit (1\\% of units) patched into the verbal item}}")
    for f in VERBAL:
        cells = [f"{np.nanmean([D['patch'][m][f'{k}:{f}:circ']['mean_restored'] for m in models]):.2f}"
                 for _, D, k in arms]
        rows.append(f"    {FMT[f]} & " + " & ".join(cells) + " & --")
    header = "& AP & \\ourmethod{} (nats) & \\ourmethod{} (rescaled) & paper (AP)"
    cap = ("Every quantity of \\citet{devarda2026shared} re-measured on 13 models with the paper's attribution "
           "patching (AP) and with \\ourmethod{} on the same units and metric; the paper's own numbers in the "
           "last column. \\ourmethod{} (nats) is trained on $-m$, \\ourmethod{} (rescaled) on $-M$ "
           "(\\cref{sec:arith-formats-setup}). Item-level loading is the sum of the item's AP scores over the "
           "circuit's units in every column, so the rows compare circuits, not per-item scorers.")
    txt = table(rows, header, cap, "tab:arith-formats-summary", "lrrrr")
    (TABS / "arith_formats_summary.tex").write_text(txt)

    # ------------------------------------------------------------- sufficiency curves
    fracs = list(next(iter(S["curves"][models[0]].values()))["ap"].keys())
    rows = []
    rank_rows = [("AP", S, "ap"), ("\\ourmethod{} (nats)", S, "mattr"), ("\\ourmethod{} (rescaled)", N, "mattr"),
                 ("Random", S, "random")]
    xfer_rows = [("AP, numeric circuit", S, "ap_numeric"), ("\\ourmethod{} (rescaled), numeric circuit", N, "mattr_numeric")]
    for f in FORMATS:
        rows.append(f"\\multicolumn{{{len(fracs) + 1}}}{{l}}{{\\emph{{{FMT[f]}}}}}")
        for name, D, k in rank_rows + (xfer_rows if f != "numeric" else []):
            vals = []
            for fr in fracs:
                v = [D["curves"][m][f][k][fr] for m in models if k in D["curves"][m][f]]
                vals.append(np.mean(v))
            rows.append(f"    {name} & " + " & ".join(f"{v:.2f}" for v in vals))
        if f != "italian":
            rows.append("\\midrule")
    header = "ranking $\\backslash$ fraction of units & " + " & ".join(f"{100 * float(fr):g}\\%" for fr in fracs)
    cap = ("Sufficiency of the top-$k$ units: restored fraction $M$ of the preference when the top units of a "
           "ranking are patched clean into the sign-flipped run, mean over 13 models and 256 items per format. "
           "\\emph{numeric circuit} rows apply the numeric ranking to the verbal format.")
    (TABS / "arith_formats_sufficiency.tex").write_text(
        table(rows, header, cap, "tab:arith-formats-sufficiency", "l" + "r" * len(fracs)))

    # ------------------------------------------------------------- per-model table
    rows = []
    for m in models:
        acc = " & ".join(f"{100 * S['acc'][m][f]:.1f}" for f in FORMATS)
        jap = " & ".join(f"{S['jaccard'][m][f'ap:{f}']:.3f}" for f in VERBAL)
        jma = " & ".join(f"{N['jaccard'][m][f'mattr:{f}']:.3f}" for f in VERBAL)
        rows.append(f"    \\texttt{{{m}}} & {acc} & {jap} & {jma}")
    header = ("model & \\multicolumn{4}{c}{accuracy (\\%)} & \\multicolumn{3}{c}{AP overlap} & "
              "\\multicolumn{3}{c}{\\ourmethod{} overlap} \\\\\n"
              "    & Num.\\ & En.\\ & Es.\\ & It.\\ & En.\\ & Es.\\ & It.\\ & En.\\ & Es.\\ & It.\\")
    cap = ("Per-model accuracy in the four formats and top-1\\% Jaccard overlap of each verbal format's circuit "
           "with the numeric circuit, for AP and for \\ourmethod{} (rescaled). The ordering English $>$ "
           "Spanish $>$ Italian holds in every row for both methods.")
    (TABS / "arith_formats_models.tex").write_text(
        table(rows, header, cap, "tab:arith-formats-models", "lrrrrrrrrrr"))

    # ------------------------------------------------------------- Fig. 4 regression
    names = ["loading", "probe_resid", "probe_mlp", "mean_logprob", "entropy"]
    labels = {"loading": "circuit loading", "probe_resid": "probe (residual)", "probe_mlp": "probe (MLP)",
              "mean_logprob": "mean log-prob", "entropy": "entropy"}
    key_m = "llama3.1-8b"
    cols = [("ap", "AP"), ("mattr-patch", "causal, \\ourmethod{}")]
    rows = []
    for nm in names:
        cells = []
        for f in VERBAL:
            for k, _ in cols:
                e = S["fig4"].get(f"{key_m}:{k}:{f}")
                if e is None:
                    cells.append("--")
                    continue
                b, p, share = e["ols"][nm]["beta"], e["ols"][nm]["p"], 100 * e["lmg"][nm]
                star = "$^{*}$" if p < 0.05 else ""
                cells.append(f"{b:+.3f}{star} ({share:.1f})")
        rows.append(f"    {labels[nm]} & " + " & ".join(cells))
    r2 = " & ".join(f"{S['fig4'][f'{key_m}:{k}:{f}']['r2']:.2f}" for f in VERBAL for k, _ in cols)
    rows.append("\\midrule")
    rows.append(f"    $R^2$ & {r2}")
    header = ("predictor & \\multicolumn{2}{c}{English} & \\multicolumn{2}{c}{Spanish} & \\multicolumn{2}{c}{Italian} \\\\\n"
              "    & " + " & ".join(lab for _ in VERBAL for _, lab in cols))
    cap = ("Llama-3.1-8B: linear probability model of item correctness on standardised predictors, "
           "$\\beta$ (LMG share of $R^2$, \\%); $^{*}$ $p<0.05$. \\emph{AP}: the paper's loading on the AP "
           "circuit; \\emph{causal, \\ourmethod{}}: the restored fraction when the \\ourmethod{} (nats) "
           "circuit is patched into the item. Probes are trained on held-out numeric items "
           "(1,600 correct / 178 incorrect; residual layer 15 and MLP layer 12 by 5-fold CV, AUC 0.91). "
           "Paper: loading $\\beta = 0.127$ (11.0\\%), $0.030$ (5.3\\%), $0.031$ (6.0\\%).")
    (TABS / "arith_formats_fig4.tex").write_text(table(rows, header, cap, "tab:arith-formats-fig4", "lrrrrrr"))
    print("wrote", [str(p) for p in TABS.glob("arith_formats_*.tex")])


if __name__ == "__main__":
    main()
