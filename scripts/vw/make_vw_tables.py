"""LaTeX tables for the interference-weights appendix (paper/sections/interference-lm.tex).

Reads the pruning sweeps `vw_prune.py` leaves in results/vw/base/ and writes one booktabs
tabular per cut into paper/tabs/, so the numbers in the appendix are generated, never typed.
Re-run after any pruning sweep is refreshed (e.g. stage J rewrites all four Tokens->Logits
JSONs) and the tables follow.

    uv run python scripts/vw/make_vw_tables.py [--run results/vw/base]

Every cell is dL = L(pruned) - L(full model) on the held-out shard, in nats; "all removed" is
the same quantity with the whole family ablated. Best cell per column in bold.
"""
import argparse
import json
from pathlib import Path

DENSITIES = [0.55, 0.30, 0.15, 0.05, 0.02, 0.01, 0.002]

# (key on disk, label in the table). The MAttr arms named here are the ones the appendix
# discusses; the untuned default (lr 0.05, eps 1e-2) is kept in every table so the size of
# the tuning effect is visible next to the tuned arm.
TOK_ROWS = [
    ("helpfulness", r"Helpfulness (oracle)"),
    ("fisher", r"Fisher effectiveness"),
    ("era", r"Expected attribution"),
    ("weight_abs", r"Virtual weight magnitude"),
    ("ig_long", r"Expected Gradients"),
    ("ixg_s3000", r"I$\times$G"),
    ("mattr_adam_lr0.05_eps0.01", r"\ourmethod{}+Adam (lr $0.05$, $\epsilon=10^{-2}$, $b=8$)"),
    ("mattr_adam_lr0.05_eps0.01_row", r"\ourmethod{}+Adam (row granularity)"),
    ("mattr_sgd_lr1.0_eps1e-08", r"\ourmethod{}+SGD (lr $1.0$)"),
    ("mattr_adam_lr0.01_eps1e-08", r"\ourmethod{}+Adam (lr $0.01$, $\epsilon=10^{-8}$, $b=8$)"),
    ("E_eps1e-8_lr0.01_s12000_b32", r"\ourmethod{}+Adam (lr $0.01$, $\epsilon=10^{-8}$, $b=32$)"),
    ("random_s0", r"Random"),
]
FEAT_ROWS = [
    ("helpfulness", r"Helpfulness (oracle)"),
    ("fisher", r"Fisher effectiveness"),
    ("era", r"Expected attribution"),
    ("weight_abs", r"Virtual weight magnitude"),
    ("ig_s3000", r"Expected Gradients"),
    ("ixg_s3000", r"I$\times$G"),
    ("mattr_adam_lr0.05_eps0.01", r"\ourmethod{}+Adam (lr $0.05$, $\epsilon=10^{-2}$, $b=8$)"),
    ("mattr_sgd_lr1.0_eps0.01", r"\ourmethod{}+SGD (lr $1.0$)"),
    ("mattr_adam_lr0.01_eps1e-08", r"\ourmethod{}+Adam (lr $0.01$, $\epsilon=10^{-8}$, $b=8$)"),
    ("H_eps1e-2_lr0.05_s12000_b32", r"\ourmethod{}+Adam (lr $0.05$, $\epsilon=10^{-2}$, $b=32$)"),
    ("H_eps1e-8_lr0.01_s12000_b32", r"\ourmethod{}+Adam (lr $0.01$, $\epsilon=10^{-8}$, $b=32$)"),
    ("random_s0", r"Random"),
]
# Stage J arms, drawn only once they are on disk (the script never invents a row).
J_ROWS = [
    ("J_eps1e-8_lr0.01_s48000_b32", r"\ourmethod{}+Adam (lr $0.01$, $\epsilon=10^{-8}$, $b=32$, 48k)"),
    ("J_eps1e-8_lr0.005_s48000_b32", r"\ourmethod{}+Adam (lr $0.005$, $\epsilon=10^{-8}$, $b=32$, 48k)"),
    ("J_eps1e-8_lr0.01_s24000_b64", r"\ourmethod{}+Adam (lr $0.01$, $\epsilon=10^{-8}$, $b=64$, 24k)"),
]


def fmt(v, best):
    s = f"{v:+.3f}"
    if abs(v) < 5e-4:
        s = f"{v:+.4f}"
    return rf"\textbf{{{s}}}" if best else s


def block(d, rows):
    x, full, curves = d["densities"], d["full_loss"], d["curves"]
    present = [(k, lab) for k, lab in rows if k in curves]
    vals = {k: [curves[k][x.index(dd)] - full for dd in DENSITIES] for k, _ in present}
    best = [min(vals[k][j] for k, _ in present if k != "random_s0") for j in range(len(DENSITIES))]
    out = []
    for k, lab in present:
        cells = [fmt(v, abs(v - b) < 1e-9) for v, b in zip(vals[k], best)]
        out.append(f"    {lab} & " + " & ".join(cells) + r" \\")
    return out, d["empty_loss"] - full


def table(run, fname, rows, caption, label, mid=None):
    d = json.loads((run / fname).read_text())
    # stage-J arms sit with the other MAttr arms, above the Random row
    lines, empty = block(d, rows[:-1] + J_ROWS + rows[-1:])
    head = " & ".join(f"${dd:g}$" for dd in DENSITIES)
    body = [
        r"\begin{table}[!t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \begin{tabular}{l" + "r" * len(DENSITIES) + "}",
        r"    \toprule",
        rf"    ranking $\backslash$ density & {head} \\",
        r"    \midrule",
        *lines,
        r"    \midrule",
        rf"    \multicolumn{{{len(DENSITIES) + 1}}}{{l}}{{all {d['n_weights']:,} weights removed: ${empty:+.3f}$}} \\",
        r"    \bottomrule",
        r"    \end{tabular}",
        rf"    \caption{{{caption}}}",
        rf"    \label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(body) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path("results/vw/base"))
    ap.add_argument("--out", type=Path, default=Path("paper/tabs"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = [
        ("prune_tok_mean.json", TOK_ROWS, "vw_prune_tok_mean.tex",
         r"Tokens$\to$Logits, \textbf{controlled}: only the $16{,}777{,}216$ token-row weights are "
         r"pruned (the $1{,}024$ fixed position rows stay on) and removed weights are replaced by "
         r"their corpus-mean contribution rather than zero. $\Delta$ held-out loss (nats) at each "
         r"density; lower is better, best per column in bold.",
         "tab:vw-tok-mean"),
        ("prune_tok.json", TOK_ROWS, "vw_prune_tok.tex",
         r"Tokens$\to$Logits, token rows only, \textbf{zero} ablation (the note's own ablation).",
         "tab:vw-tok-zero"),
        ("prune.json", TOK_ROWS, "vw_prune_full.tex",
         r"Tokens$\to$Logits, all $20{,}971{,}520$ weights (token and position rows), zero "
         r"ablation. This is the note's own object; the position rows (20\% of weights, 98\% of "
         r"positive helpfulness mass) and the zero-ablation constant both inflate every gap.",
         "tab:vw-full-zero"),
        ("prune_mean.json", TOK_ROWS, "vw_prune_full_mean.tex",
         r"Tokens$\to$Logits, all weights, mean ablation.",
         "tab:vw-full-mean"),
        ("prune_fl_mean.json", FEAT_ROWS, "vw_prune_feat_mean.tex",
         r"Features$\to$Logits (transcoder features, $16{,}777{,}216$ weights), \textbf{mean} "
         r"ablation. Evaluated on $256$ held-out sequences.",
         "tab:vw-feat-mean"),
        ("prune_fl.json", FEAT_ROWS, "vw_prune_feat.tex",
         r"Features$\to$Logits, zero ablation.",
         "tab:vw-feat-zero"),
    ]
    for fname, rows, out, cap, lab in specs:
        if not (args.run / fname).exists():
            print(f"skip {fname} (missing)")
            continue
        (args.out / out).write_text(table(args.run, fname, rows, cap, lab))
        print(f"-> {args.out / out}")
    if (args.run / "audit.json").exists():
        audit_table(args.run, args.out / "vw_audit.tex")


# ---------------------------------------------------------------------------------------------
# The claim-audit table (scripts/vw/vw_audit.py -> results/vw/base/audit.json). One block per
# family; one column per ranking; one row per statistic the note's figures rest on.
AUDIT_COLS = {
    "tl_tok": [("fisher", "Fisher"), ("weight_abs", r"$|w|$"), ("ig_long", "IG"),
               ("mattr_adam_lr0.05_eps0.01", r"\ourmethod{} dflt"),
               ("J_eps1e-8_lr0.01_s24000_b64", r"\ourmethod{} best"), ("helpfulness", "oracle")],
    "fl": [("fisher", "Fisher"), ("weight_abs", r"$|w|$"), ("ig_s3000", "IG"),
           ("mattr_adam_lr0.05_eps0.01", r"\ourmethod{} dflt"),
           ("H_eps1e-8_lr0.01_s12000_b32", r"\ourmethod{} best"), ("helpfulness", "oracle")],
}
AUDIT_TITLES = {"tl_tok": r"Tokens$\to$Logits (token rows)", "fl": r"Features$\to$Logits"}


def audit_table(run, out):
    a = json.loads((run / "audit.json").read_text())
    lines = [r"\begin{table}[!t]", r"    \centering", r"    \small", r"    \setlength{\tabcolsep}{3.5pt}",
             r"    \begin{tabular}{lrrrrrr}", r"    \toprule"]
    for fam in ("tl_tok", "fl"):
        if fam not in a:
            continue
        cols = [(k, lab) for k, lab in AUDIT_COLS[fam] if k in a[fam]["along"]]
        lines.append(rf"    \multicolumn{{7}}{{l}}{{\textbf{{{AUDIT_TITLES[fam]}}}: "
                     rf"{a[fam]['base']['n']:,} weights, {100*a[fam]['base']['frac_sig_helpful']:.1f}\% significantly helpful, "
                     rf"{100*a[fam]['base']['frac_sig_harmful']:.1f}\% significantly harmful}} \\")
        lines.append("    statistic & " + " & ".join(lab for _, lab in cols) + r" \\")
        lines.append(r"    \midrule")

        def row(name, f, fmt="{:.3f}", lower_better=True):
            vals = [f(a[fam], k) for k, _ in cols]
            best = (min if lower_better else max)(v for v in vals if v is not None)
            cells = [(r"\textbf{" + fmt.format(v) + "}") if v == best else fmt.format(v) for v in vals]
            lines.append(f"    {name} & " + " & ".join(cells) + r" \\")

        row(r"harmful share of each node's top-10", lambda A, k: A["per_node"][k]["10"]["harmful"])
        row(r"nodes with a harmful weight in their top-10", lambda A, k: A["per_node"][k]["10"]["rows_with_harmful"])
        row(r"harmful share of the top 1\% of the family", lambda A, k: A["along"][k]["harmful_in_top_0.01"])
        row(r"harmful share of the top 10\%", lambda A, k: A["along"][k]["harmful_in_top_0.1"])
        row(r"density for 90\% of positive helpfulness mass", lambda A, k: A["along"][k]["density_for_90pct_mass"], "{:.4f}")
        row(r"density for 99\% of positive helpfulness mass", lambda A, k: A["along"][k]["density_for_99pct_mass"], "{:.3f}")
        row(r"share of helpful weights kept at density 0.15", lambda A, k: A["along"][k]["at_density"]["0.15"]["helpful_kept"], lower_better=False)
        row(r"positive mass kept at density 0.15", lambda A, k: A["along"][k]["at_density"]["0.15"]["pos_mass"], lower_better=False)
        row(r"$\Delta$ loss at density 0.15, mean ablation", lambda A, k: A["along"][k]["dl_mean"].get("0.15"), "{:+.3f}")
        lines.append(r"    \midrule")
    lines[-1] = r"    \bottomrule"
    lines += [r"    \end{tabular}",
              r"    \caption{Statistics behind the note's figures, recomputed for every ranking over all weights of "
              r"each family. ``Harmful'' and ``helpful'' mean the 95\% CI on mean helpfulness excludes zero. "
              r"Best per row in bold (the oracle sorts by helpfulness itself, so it is the ceiling on the "
              r"first four rows by construction).}",
              r"    \label{tab:vw-audit}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
